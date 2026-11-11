#define _GNU_SOURCE

#include <endian.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../include/vecfs_disk.h"

#define FSCK_BLOCK_SIZE 4096U

static uint32_t crc_table[256];

static void crc_init(void)
{
    uint32_t i;

    for (i = 0; i < 256; i++) {
        uint32_t value = i;
        unsigned int bit;

        for (bit = 0; bit < 8; bit++)
            value = (value & 1) ?
                UINT32_C(0xedb88320) ^
                    (value >> 1) :
                value >> 1;

        crc_table[i] = value;
    }
}

static uint32_t crc32_buffer(
    const void *buffer,
    size_t length)
{
    const uint8_t *bytes = buffer;
    uint32_t crc = UINT32_MAX;
    size_t i;

    for (i = 0; i < length; i++)
        crc = crc_table[
                (crc ^ bytes[i]) & 0xffU] ^
            (crc >> 8);

    return crc ^ UINT32_MAX;
}

static int pread_full(
    int fd,
    void *buffer,
    size_t length,
    off_t offset)
{
    uint8_t *bytes = buffer;
    size_t done = 0;

    while (done < length) {
        ssize_t count = pread(
            fd,
            bytes + done,
            length - done,
            offset + (off_t)done);

        if (count < 0) {
            if (errno == EINTR)
                continue;

            return -1;
        }

        if (!count) {
            errno = EIO;
            return -1;
        }

        done += count;
    }

    return 0;
}

static int bitmap_test(
    const uint8_t *bitmap,
    uint64_t bit,
    uint64_t bytes)
{
    uint64_t byte = bit >> 3;

    if (byte >= bytes)
        return -1;

    return !!(bitmap[byte] &
              (1U << (bit & 7)));
}

static int validate_super(
    const struct vecfs_super_disk_v2 *super)
{
    struct vecfs_super_disk_v2 copy;
    uint32_t stored_crc;
    uint32_t calculated_crc;

    if (le32toh(super->magic) !=
        VECFS_SUPER_MAGIC)
        return -EINVAL;

    if (le16toh(super->major) !=
        VECFS_SUPER_MAJOR)
        return -EPROTONOSUPPORT;

    if (le32toh(super->header_size) !=
        sizeof(*super))
        return -EINVAL;

    if (le32toh(super->block_size) !=
        FSCK_BLOCK_SIZE)
        return -EINVAL;

    if (le32toh(super->inode_size) !=
        VECFS_DISK_INODE_SIZE)
        return -EINVAL;

    copy = *super;

    stored_crc =
        le32toh(copy.checksum);

    copy.checksum =
        htole32(0);

    calculated_crc =
        crc32_buffer(
            &copy,
            sizeof(copy));

    if (stored_crc != calculated_crc) {
        fprintf(stderr,
                "superblock CRC mismatch: "
                "stored=%08x calculated=%08x\n",
                stored_crc,
                calculated_crc);
        return -EBADMSG;
    }

    return 0;
}

static int validate_meta(
    const struct vecfs_ann_meta_disk *meta,
    struct vecfs_ann_config *config)
{
    struct vecfs_ann_meta_disk copy;
    uint32_t stored_crc;
    uint32_t calculated_crc;

    if (le32toh(meta->magic) !=
        VECFS_ANN_META_MAGIC)
        return -EINVAL;

    if (le16toh(meta->major) !=
        VECFS_ANN_META_MAJOR)
        return -EPROTONOSUPPORT;

    if (le32toh(meta->header_size) !=
        sizeof(*meta))
        return -EINVAL;

    copy = *meta;

    stored_crc =
        le32toh(copy.checksum);

    copy.checksum =
        htole32(0);

    calculated_crc =
        crc32_buffer(
            &copy,
            sizeof(copy));

    if (stored_crc != calculated_crc)
        return -EBADMSG;

    if (config) {
        memset(config, 0, sizeof(*config));

        config->dimension =
            le32toh(meta->dimension);
        config->element_type =
            le16toh(meta->element_type);
        config->distance_type =
            le16toh(meta->distance_type);
        config->m =
            le16toh(meta->m);
        config->m0 =
            le16toh(meta->m0);
        config->max_levels =
            le16toh(meta->max_levels);
        config->current_max_level =
            le16toh(meta->current_max_level);
        config->ef_construction =
            le32toh(meta->ef_construction);
        config->default_ef_search =
            le32toh(meta->default_ef_search);
        config->object_stride =
            le64toh(meta->object_stride);
        config->vector_bytes =
            le64toh(meta->vector_bytes);
    }

    return 0;
}

static int validate_region(
    const uint8_t *bitmap,
    uint64_t bitmap_bytes,
    uint64_t start,
    uint64_t count,
    const char *name)
{
    uint64_t block;

    for (block = start;
         block < start + count;
         block++) {
        int used = bitmap_test(
            bitmap,
            block,
            bitmap_bytes);

        if (used <= 0) {
            fprintf(stderr,
                    "%s block %" PRIu64
                    " is not marked allocated\n",
                    name,
                    block);
            return -EUCLEAN;
        }
    }

    return 0;
}

static int fsck_check(
    const char *path)
{
    struct vecfs_super_disk_v2 super;
    struct vecfs_ann_meta_disk meta_a;
    struct vecfs_ann_meta_disk meta_b;
    struct vecfs_ann_config config_a;
    struct vecfs_ann_config config_b;
    uint8_t *bmap = NULL;
    uint8_t *imap = NULL;
    uint64_t blocks;
    uint64_t inodes;
    uint64_t bmap_bytes;
    uint64_t imap_bytes;
    uint64_t inode_table_blocks;
    uint64_t fixed_end;
    uint64_t meta_a_block;
    uint64_t meta_b_block;
    int valid_a;
    int valid_b;
    int fd = -1;
    int ret = -1;

    fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        fprintf(stderr,
                "cannot open %s: %s\n",
                path,
                strerror(errno));
        return -1;
    }

    if (pread_full(
            fd,
            &super,
            sizeof(super),
            0) < 0)
        goto out;

    if (validate_super(&super)) {
        fprintf(stderr,
                "invalid VecFS superblock\n");
        goto out;
    }

    blocks =
        le64toh(super.blocks_count);
    inodes =
        le64toh(super.inodes_count);

    if (!blocks ||
        blocks > UINT32_MAX ||
        !inodes ||
        inodes > UINT32_MAX) {
        fprintf(stderr,
                "invalid block or inode count\n");
        goto out;
    }

    bmap_bytes =
        (uint64_t)le32toh(
            super.bmap_blocks) *
        FSCK_BLOCK_SIZE;

    imap_bytes =
        (uint64_t)le32toh(
            super.imap_blocks) *
        FSCK_BLOCK_SIZE;

    bmap = calloc(1, bmap_bytes);
    imap = calloc(1, imap_bytes);

    if (!bmap || !imap) {
        fprintf(stderr,
                "out of memory\n");
        goto out;
    }

    if (pread_full(
            fd,
            bmap,
            bmap_bytes,
            (off_t)le64toh(
                super.bmap_start) *
                FSCK_BLOCK_SIZE) < 0)
        goto out;

    if (pread_full(
            fd,
            imap,
            imap_bytes,
            (off_t)le64toh(
                super.imap_start) *
                FSCK_BLOCK_SIZE) < 0)
        goto out;

    inode_table_blocks =
        (inodes * VECFS_DISK_INODE_SIZE +
         FSCK_BLOCK_SIZE - 1) /
        FSCK_BLOCK_SIZE;

    fixed_end =
        le64toh(super.data_start);

    if (fixed_end >= blocks) {
        fprintf(stderr,
                "data_start is outside device\n");
        goto out;
    }

    ret = validate_region(
        bmap,
        bmap_bytes,
        0,
        fixed_end,
        "metadata");
    if (ret)
        goto out;

    if (le64toh(super.inode_table) +
            inode_table_blocks >
        fixed_end) {
        fprintf(stderr,
                "inode table overlaps data area\n");
        goto out;
    }

    meta_a_block =
        le64toh(super.ann_meta_a);
    meta_b_block =
        le64toh(super.ann_meta_b);

    if (meta_a_block >= fixed_end ||
        meta_b_block >= fixed_end ||
        meta_a_block == meta_b_block) {
        fprintf(stderr,
                "ANN metadata blocks are invalid\n");
        goto out;
    }

    if (pread_full(
            fd,
            &meta_a,
            sizeof(meta_a),
            (off_t)meta_a_block *
                FSCK_BLOCK_SIZE) < 0)
        goto out;

    if (pread_full(
            fd,
            &meta_b,
            sizeof(meta_b),
            (off_t)meta_b_block *
                FSCK_BLOCK_SIZE) < 0)
        goto out;

    valid_a =
        validate_meta(&meta_a, &config_a) == 0;

    valid_b =
        validate_meta(&meta_b, &config_b) == 0;

    if (!valid_a && !valid_b) {
        fprintf(stderr,
                "both ANN metadata copies are invalid\n");
        goto out;
    }

    if (valid_a && valid_b) {
        if (config_a.dimension !=
                config_b.dimension ||
            config_a.element_type !=
                config_b.element_type ||
            config_a.distance_type !=
                config_b.distance_type ||
            config_a.m != config_b.m ||
            config_a.m0 != config_b.m0 ||
            config_a.max_levels !=
                config_b.max_levels ||
            config_a.object_stride !=
                config_b.object_stride ||
            config_a.vector_bytes !=
                config_b.vector_bytes) {
            fprintf(stderr,
                    "ANN metadata A/B configuration mismatch\n");
            goto out;
        }
    }

    printf(
        "VecFS fsck basic check passed\n"
        "  blocks:       %" PRIu64 "\n"
        "  inodes:       %" PRIu64 "\n"
        "  data start:   %" PRIu64 "\n"
        "  metadata A:   %s\n"
        "  metadata B:   %s\n",
        blocks,
        inodes,
        fixed_end,
        valid_a ? "valid" : "invalid",
        valid_b ? "valid" : "invalid");

    ret = 0;

out:
    if (ret)
        fprintf(stderr,
                "VecFS fsck basic check failed: %s\n",
                strerror(errno));

    free(imap);
    free(bmap);

    if (fd >= 0)
        close(fd);

    return ret;
}

static void usage(
    const char *program)
{
    fprintf(stderr,
            "Usage: %s <device-or-image>\n",
            program);
}

int main(
    int argc,
    char **argv)
{
    int ret;

    if (argc != 2) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    crc_init();

    ret = fsck_check(argv[1]);

    return ret ?
        EXIT_FAILURE :
        EXIT_SUCCESS;
}
