#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <linux/fs.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

#include "../include/vecfs_disk.h"

#define DEFAULT_INODE_RATIO 16ULL

struct mkfs_options {
    const char *device;

    uint32_t dimension;
    uint16_t element_type;
    uint16_t distance_type;

    uint16_t m;
    uint16_t m0;
    uint16_t max_levels;

    uint32_t ef_construction;
    uint32_t ef_search;

    uint64_t random_seed;
    uint64_t inode_count;

    /* 嵌入服务配置（写入 superblock；空 = 未配置，derive 落到内置哈希）。 */
    char embed_url[256];
    char embed_key[256];
    char embed_model[64];
    char embed_protocol[16];
};

struct mkfs_layout {
    uint64_t total_blocks;
    uint64_t inode_count;

    uint64_t imap_start;
    uint32_t imap_blocks;

    uint64_t bmap_start;
    uint32_t bmap_blocks;

    uint64_t inode_table;
    uint64_t inode_table_blocks;

    uint64_t ann_meta_a;
    uint64_t ann_meta_b;

    uint64_t data_start;
};

static uint32_t crc32_table[256];

static void crc32_init(void)
{
    uint32_t i;

    for (i = 0; i < 256; i++) {
        uint32_t c = i;
        int j;

        for (j = 0; j < 8; j++)
            c = (c & 1) ?
                (0xEDB88320U ^ (c >> 1)) :
                (c >> 1);

        crc32_table[i] = c;
    }
}

static uint32_t vecfs_crc32(
    const void *data,
    size_t length)
{
    const uint8_t *p = data;
    uint32_t crc = ~0U;
    size_t i;

    for (i = 0; i < length; i++)
        crc = crc32_table[(crc ^ p[i]) & 0xff] ^
              (crc >> 8);

    return crc ^ ~0U;
}

static uint64_t align_u64(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

static uint32_t element_size(uint16_t type)
{
    switch (type) {
    case VECFS_ANN_ELEM_FP32:
        return 4;
    case VECFS_ANN_ELEM_FP16:
        return 2;
    case VECFS_ANN_ELEM_INT8:
        return 1;
    case VECFS_ANN_ELEM_INT16:
        return 2;
    default:
        return 0;
    }
}

static uint64_t calculate_object_stride(
    const struct mkfs_options *opt)
{
    uint64_t off;
    uint64_t upper_levels;

    upper_levels = opt->max_levels - 1;

    off = sizeof(struct vecfs_ann_object_disk);
    off = align_u64(off, 64);

    off += (uint64_t)opt->m0 * sizeof(uint32_t);
    off = align_u64(off, 64);

    off += (uint64_t)opt->dimension *
           element_size(opt->element_type);
    off = align_u64(off, 64);

    off += upper_levels * sizeof(uint16_t);
    off = align_u64(off, 64);

    off += upper_levels * opt->m * sizeof(uint32_t);
    off = align_u64(off, 64);

    return off;
}

static uint64_t level_mult_q32(uint16_t m)
{
    double value = 1.0 / log((double)m);
    return (uint64_t)(value * 4294967296.0);
}

static uint64_t bitmap_blocks(uint64_t entries)
{
    uint64_t bits_per_block =
        VECFS_DISK_BLOCK_SIZE * 8ULL;

    return (entries + bits_per_block - 1) /
           bits_per_block;
}

static int parse_distance(const char *name, uint16_t *type)
{
    if (!strcmp(name, "l2")) {
        *type = VECFS_ANN_DIST_L2;
        return 0;
    }

    if (!strcmp(name, "ip")) {
        *type = VECFS_ANN_DIST_IP;
        return 0;
    }

    return -1;
}

static int validate_options(const struct mkfs_options *opt)
{
    if (!opt->dimension ||
        opt->dimension > VECFS_ANN_MAX_DIM) {
        fprintf(stderr, "dimension must be in [1, %u]\n",
                VECFS_ANN_MAX_DIM);
        return -1;
    }

    if (opt->element_type != VECFS_ANN_ELEM_FP32) {
        fprintf(stderr,
                "VecFS V2 currently supports FP32 only\n");
        return -1;
    }

    if (opt->distance_type != VECFS_ANN_DIST_L2 &&
        opt->distance_type != VECFS_ANN_DIST_IP) {
        fprintf(stderr, "distance must be l2 or ip\n");
        return -1;
    }

    if (opt->m < 2 || opt->m > VECFS_ANN_MAX_M) {
        fprintf(stderr, "M must be in [2, %u]\n",
                VECFS_ANN_MAX_M);
        return -1;
    }

    if (opt->m0 < opt->m ||
        opt->m0 > VECFS_ANN_MAX_M0) {
        fprintf(stderr, "M0 must be in [M, %u]\n",
                VECFS_ANN_MAX_M0);
        return -1;
    }

    if (!opt->max_levels ||
        opt->max_levels > VECFS_ANN_MAX_LEVELS) {
        fprintf(stderr, "levels must be in [1, %u]\n",
                VECFS_ANN_MAX_LEVELS);
        return -1;
    }

    if (opt->ef_construction < opt->m0 ||
        opt->ef_construction > 4096) {
        fprintf(stderr,
                "ef-construction must be in [M0, 4096]\n");
        return -1;
    }

    if (!opt->ef_search || opt->ef_search > 65535) {
        fprintf(stderr,
                "ef-search must be in [1, 65535]\n");
        return -1;
    }

    return 0;
}

static void calculate_layout(
    uint64_t total_blocks,
    struct mkfs_options *opt,
    struct mkfs_layout *layout)
{
    uint64_t inode_count;

    memset(layout, 0, sizeof(*layout));

    layout->total_blocks = total_blocks;

    inode_count = opt->inode_count;
    if (!inode_count)
        inode_count = total_blocks / DEFAULT_INODE_RATIO;

    if (inode_count < VECFS_FIRST_USER_INO + 1)
        inode_count = VECFS_FIRST_USER_INO + 1;

    if (inode_count > UINT32_MAX)
        inode_count = UINT32_MAX;

    layout->inode_count = inode_count;

    layout->imap_start = 1;
    layout->imap_blocks =
        (uint32_t)bitmap_blocks(inode_count);

    layout->bmap_start =
        layout->imap_start + layout->imap_blocks;
    layout->bmap_blocks =
        (uint32_t)bitmap_blocks(total_blocks);

    layout->inode_table =
        layout->bmap_start + layout->bmap_blocks;
    layout->inode_table_blocks =
        (inode_count * VECFS_DISK_INODE_SIZE +
         VECFS_DISK_BLOCK_SIZE - 1) /
        VECFS_DISK_BLOCK_SIZE;

    layout->ann_meta_a =
        layout->inode_table +
        layout->inode_table_blocks;

    layout->ann_meta_b = layout->ann_meta_a + 1;
    layout->data_start = layout->ann_meta_b + 1;
}

static int pwrite_full(
    int fd,
    const void *buffer,
    size_t size,
    off_t offset)
{
    const uint8_t *p = buffer;

    while (size) {
        ssize_t written = pwrite(fd, p, size, offset);

        if (written < 0) {
            if (errno == EINTR)
                continue;
            return -1;
        }

        if (!written) {
            errno = EIO;
            return -1;
        }

        p += written;
        size -= written;
        offset += written;
    }

    return 0;
}

static int write_zero_blocks(
    int fd,
    uint64_t start,
    uint64_t count)
{
    void *zero;
    uint64_t i;

    zero = calloc(1, VECFS_DISK_BLOCK_SIZE);
    if (!zero)
        return -1;

    for (i = 0; i < count; i++) {
        if (pwrite_full(
                fd,
                zero,
                VECFS_DISK_BLOCK_SIZE,
                (start + i) *
                    (off_t)VECFS_DISK_BLOCK_SIZE)) {
            free(zero);
            return -1;
        }
    }

    free(zero);
    return 0;
}

static void build_ann_meta(
    struct vecfs_ann_meta_disk *meta,
    const struct mkfs_options *opt,
    uint64_t sequence)
{
    uint64_t vector_bytes;
    uint64_t stride;

    vector_bytes =
        (uint64_t)opt->dimension *
        element_size(opt->element_type);

    stride = calculate_object_stride(opt);

    memset(meta, 0, sizeof(*meta));

    meta->magic = htole32(VECFS_ANN_META_MAGIC);
    meta->major = htole16(VECFS_ANN_META_MAJOR);
    meta->minor = htole16(VECFS_ANN_META_MINOR);
    meta->header_size = htole32(sizeof(*meta));

    meta->sequence = htole64(sequence);
    meta->epoch = htole64(1);

    meta->state = htole32(VECFS_ANN_META_EMPTY);
    meta->flags = 0;

    meta->dimension = htole32(opt->dimension);
    meta->element_type = htole16(opt->element_type);
    meta->distance_type =
        htole16(opt->distance_type);

    meta->m = htole16(opt->m);
    meta->m0 = htole16(opt->m0);
    meta->max_levels = htole16(opt->max_levels);
    meta->current_max_level = 0;

    meta->ef_construction =
        htole32(opt->ef_construction);
    meta->default_ef_search =
        htole32(opt->ef_search);

    meta->random_seed =
        htole64(opt->random_seed);
    meta->level_mult_q32 =
        htole64(level_mult_q32(opt->m));

    meta->live_nodes = 0;
    meta->deleted_nodes = 0;
    meta->next_node_id = htole64(VECFS_ANN_FIRST_NODE_ID);
    meta->next_external_id = htole64(1);


    meta->entry_node_id =
        htole32(VECFS_ANN_INVALID_NODE_ID);

    /*
     * 阶段 2 初始化隐藏 arena inode 后再设置。
     */
    meta->active_arena_ino = 0;
    meta->rebuild_arena_ino = 0;
    meta->rebuild_state =
        htole32(VECFS_ANN_REBUILD_DISK_NONE);

    meta->object_stride = htole64(stride);
    meta->vector_bytes = htole64(vector_bytes);
    meta->committed_txid = 0;

    meta->checksum = 0;
    meta->checksum =
        htole32(vecfs_crc32(meta, sizeof(*meta)));
}

static void build_super(
    struct vecfs_super_disk_v2 *super,
    const struct mkfs_layout *layout,
    const struct mkfs_options *opt)
{
    uint64_t reserved_inodes;

    memset(super, 0, sizeof(*super));

    reserved_inodes = VECFS_FIRST_USER_INO;
    if (reserved_inodes > layout->inode_count)
        reserved_inodes = layout->inode_count;

    super->magic = htole32(VECFS_SUPER_MAGIC);
    super->major = htole16(VECFS_SUPER_MAJOR);
    super->minor = htole16(VECFS_SUPER_MINOR);

    super->header_size = htole32(sizeof(*super));

    super->features = htole64(
        VECFS_FEATURE_ANN |
        VECFS_FEATURE_ANN_META_AB |
        VECFS_FEATURE_ANN_SHARED_DATA);

    super->write_time_sec = htole64(time(NULL));
    super->block_size =
        htole32(VECFS_DISK_BLOCK_SIZE);
    super->root_ino = htole32(VECFS_ROOT_INO);

    super->inodes_count =
        htole64(layout->inode_count);
    super->blocks_count =
        htole64(layout->total_blocks);

    super->data_start =
        htole64(layout->data_start);
    super->inode_table =
        htole64(layout->inode_table);

    super->free_inodes =
        htole64(layout->inode_count -
                reserved_inodes);
    super->free_blocks =
        htole64(layout->total_blocks -
                layout->data_start);

    super->imap_start =
        htole64(layout->imap_start);
    super->bmap_start =
        htole64(layout->bmap_start);
    super->imap_blocks =
        htole32(layout->imap_blocks);
    super->bmap_blocks =
        htole32(layout->bmap_blocks);

    super->ann_meta_a =
        htole64(layout->ann_meta_a);
    super->ann_meta_b =
        htole64(layout->ann_meta_b);
    super->ann_active_hint = htole32(0);

    super->inode_size =
        htole32(VECFS_DISK_INODE_SIZE);

    /* 嵌入服务配置（固定字节数组直接拷贝；空 = 未配置）。 */
    memcpy(super->embed_url, opt->embed_url,
           sizeof(super->embed_url));
    memcpy(super->embed_key, opt->embed_key,
           sizeof(super->embed_key));
    memcpy(super->embed_model, opt->embed_model,
           sizeof(super->embed_model));
    memcpy(super->embed_protocol, opt->embed_protocol,
           sizeof(super->embed_protocol));

    /* 新格式化文件系统视为干净卸载（无未落盘的 stale 失效）。 */
    super->clean_magic = htole64(VECFS_SUPER_CLEAN_MAGIC);

    super->checksum = 0;
    super->checksum =
        htole32(vecfs_crc32(super, sizeof(*super)));
}

static int write_bitmaps(
    int fd,
    const struct mkfs_layout *layout)
{
    uint8_t *imap = NULL;
    uint8_t *bmap = NULL;
    size_t imap_bytes;
    size_t bmap_bytes;
    uint64_t i;
    int ret = -1;

    imap_bytes =
        (size_t)layout->imap_blocks *
        VECFS_DISK_BLOCK_SIZE;

    bmap_bytes =
        (size_t)layout->bmap_blocks *
        VECFS_DISK_BLOCK_SIZE;

    imap = calloc(1, imap_bytes);
    bmap = calloc(1, bmap_bytes);

    if (!imap || !bmap)
        goto out;

    /*
     * bitmap bit 0 对应 inode 1。
     * 保留 inode 1..VECFS_FIRST_USER_INO。
     */
    for (i = 0;
         i < VECFS_FIRST_USER_INO &&
         i < layout->inode_count;
         i++)
        imap[i >> 3] |= 1U << (i & 7);

    /*
     * data_start 之前全部为固定元数据。
     */
    for (i = 0; i < layout->data_start; i++)
        bmap[i >> 3] |= 1U << (i & 7);

    if (pwrite_full(
            fd,
            imap,
            imap_bytes,
            layout->imap_start *
                (off_t)VECFS_DISK_BLOCK_SIZE))
        goto out;

    if (pwrite_full(
            fd,
            bmap,
            bmap_bytes,
            layout->bmap_start *
                (off_t)VECFS_DISK_BLOCK_SIZE))
        goto out;

    ret = 0;

out:
    free(imap);
    free(bmap);
    return ret;
}

static void usage(const char *program)
{
    fprintf(stderr,
        "Usage: %s [options] <device>\n"
        "\n"
        "Options:\n"
        "  --dim N                 vector dimension, default 96\n"
        "  --distance l2|ip        default l2\n"
        "  --m N                   upper M, default 16\n"
        "  --m0 N                  level-0 M, default 32\n"
        "  --levels N              maximum levels, default 16\n"
        "  --ef-construction N     default 200\n"
        "  --ef-search N           default 64\n"
        "  --seed N                default 100\n"
        "  --inodes N              inode count\n"
        "  --embed-url URL         embedding service URL, default empty\n"
        "  --embed-key KEY         API key, default empty (local models omit)\n"
        "  --embed-model NAME      model name, default empty\n"
        "  --embed-protocol NAME   default ollama\n"
        "  --help\n",
        program);
}

int main(int argc, char **argv)
{
    static const struct option options[] = {
        { "dim", required_argument, NULL, 'd' },
        { "distance", required_argument, NULL, 'D' },
        { "m", required_argument, NULL, 'm' },
        { "m0", required_argument, NULL, '0' },
        { "levels", required_argument, NULL, 'l' },
        { "ef-construction", required_argument, NULL, 'c' },
        { "ef-search", required_argument, NULL, 's' },
        { "seed", required_argument, NULL, 'r' },
        { "inodes", required_argument, NULL, 'i' },
        { "embed-url", required_argument, NULL, 'u' },
        { "embed-key", required_argument, NULL, 'k' },
        { "embed-model", required_argument, NULL, 'e' },
        { "embed-protocol", required_argument, NULL, 'p' },
        { "help", no_argument, NULL, 'h' },
        { NULL, 0, NULL, 0 },
    };

    struct mkfs_options opt = {
        .dimension = VECFS_ANN_DEFAULT_DIM,
        .element_type = VECFS_ANN_ELEM_FP32,
        .distance_type = VECFS_ANN_DIST_L2,
        .m = VECFS_ANN_DEFAULT_M,
        .m0 = VECFS_ANN_DEFAULT_M0,
        .max_levels = VECFS_ANN_DEFAULT_MAX_LEVELS,
        .ef_construction =
            VECFS_ANN_DEFAULT_EF_CONSTRUCTION,
        .ef_search = VECFS_ANN_DEFAULT_EF_SEARCH,
        .random_seed =
            VECFS_ANN_DEFAULT_RANDOM_SEED,
        .embed_protocol = "ollama",
    };

    struct mkfs_layout layout;
    struct vecfs_super_disk_v2 *super = NULL;
    struct vecfs_ann_meta_disk *meta_a = NULL;
    struct vecfs_ann_meta_disk *meta_b = NULL;
    uint64_t device_bytes;
    uint64_t total_blocks;
    int fd = -1;
    int option;
    int ret = EXIT_FAILURE;

    crc32_init();

    while ((option = getopt_long(
                argc, argv, "d:D:m:0:l:c:s:r:i:u:k:e:p:h",
                options, NULL)) != -1) {
        switch (option) {
        case 'd':
            opt.dimension = strtoul(optarg, NULL, 0);
            break;
        case 'D':
            if (parse_distance(
                    optarg, &opt.distance_type)) {
                fprintf(stderr,
                        "invalid distance: %s\n",
                        optarg);
                return EXIT_FAILURE;
            }
            break;
        case 'm':
            opt.m = strtoul(optarg, NULL, 0);
            break;
        case '0':
            opt.m0 = strtoul(optarg, NULL, 0);
            break;
        case 'l':
            opt.max_levels =
                strtoul(optarg, NULL, 0);
            break;
        case 'c':
            opt.ef_construction =
                strtoul(optarg, NULL, 0);
            break;
        case 's':
            opt.ef_search =
                strtoul(optarg, NULL, 0);
            break;
        case 'r':
            opt.random_seed =
                strtoull(optarg, NULL, 0);
            break;
        case 'i':
            opt.inode_count =
                strtoull(optarg, NULL, 0);
            break;
        case 'u':
            strncpy(opt.embed_url, optarg,
                    sizeof(opt.embed_url) - 1);
            break;
        case 'k':
            strncpy(opt.embed_key, optarg,
                    sizeof(opt.embed_key) - 1);
            break;
        case 'e':
            strncpy(opt.embed_model, optarg,
                    sizeof(opt.embed_model) - 1);
            break;
        case 'p':
            strncpy(opt.embed_protocol, optarg,
                    sizeof(opt.embed_protocol) - 1);
            break;
        case 'h':
            usage(argv[0]);
            return EXIT_SUCCESS;
        default:
            usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (optind + 1 != argc) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    opt.device = argv[optind];

    if (validate_options(&opt))
        return EXIT_FAILURE;

    fd = open(opt.device, O_RDWR | O_EXCL | O_CLOEXEC);
    if (fd < 0) {
        fprintf(stderr, "cannot open %s: %s\n",
                opt.device, strerror(errno));
        goto out;
    }

    if (ioctl(fd, BLKGETSIZE64, &device_bytes) < 0) {
        fprintf(stderr,
                "cannot determine size of %s: %s\n",
                opt.device, strerror(errno));
        goto out;
    }

    total_blocks =
        device_bytes / VECFS_DISK_BLOCK_SIZE;

    /*
     * 磁盘块号和现有 i_block[] 都是 u32。
     */
    if (!total_blocks || total_blocks > UINT32_MAX) {
        fprintf(stderr,
                "device has %llu blocks; VecFS V2 currently "
                "supports at most %u blocks\n",
                (unsigned long long)total_blocks,
                UINT32_MAX);
        goto out;
    }

    calculate_layout(total_blocks, &opt, &layout);

    if (layout.data_start >= layout.total_blocks) {
        fprintf(stderr,
                "metadata requires %llu blocks but device "
                "contains only %llu blocks\n",
                (unsigned long long)layout.data_start,
                (unsigned long long)layout.total_blocks);
        goto out;
    }

    /*
     * 保证格式化后至少有一定的数据空间，避免在极小设备上
     * 产生只能挂载但无法正常使用的文件系统。
     */
    if (layout.total_blocks - layout.data_start < 16) {
        fprintf(stderr,
                "device is too small: only %llu data blocks "
                "would remain\n",
                (unsigned long long)
                    (layout.total_blocks -
                     layout.data_start));
        goto out;
    }

    super = calloc(1, sizeof(*super));
    meta_a = calloc(1, sizeof(*meta_a));
    meta_b = calloc(1, sizeof(*meta_b));

    if (!super || !meta_a || !meta_b) {
        fprintf(stderr, "out of memory\n");
        goto out;
    }

    build_super(super, &layout, &opt);

    /*
     * A 是初始有效副本，sequence=1。
     * B 同样有效，但 sequence=0，因此 mount 会选择 A。
     *
     * 第一次 metadata commit 会写 B，并将 sequence 提升到 2。
     */
    build_ann_meta(meta_a, &opt, 1);
    build_ann_meta(meta_b, &opt, 0);

    /*
     * 清零所有固定元数据块：
     *
     *   superblock
     *   inode bitmap
     *   block bitmap
     *   inode table
     *   ANN metadata A/B
     *
     * 不清零整个块设备，避免格式化时间随设备容量增长。
     */
    if (write_zero_blocks(fd, 0, layout.data_start)) {
        fprintf(stderr,
                "failed to clear metadata region: %s\n",
                strerror(errno));
        goto out;
    }

    if (write_bitmaps(fd, &layout)) {
        fprintf(stderr,
                "failed to initialize bitmaps: %s\n",
                strerror(errno));
        goto out;
    }

    /*
     * 先写 ANN metadata，再写超级块。
     * 如果中途失败，不会留下一个指向未初始化 ANN metadata 的
     * 完整超级块。
     */
    if (pwrite_full(
            fd,
            meta_a,
            sizeof(*meta_a),
            layout.ann_meta_a *
                (off_t)VECFS_DISK_BLOCK_SIZE)) {
        fprintf(stderr,
                "failed to write ANN metadata A: %s\n",
                strerror(errno));
        goto out;
    }

    if (pwrite_full(
            fd,
            meta_b,
            sizeof(*meta_b),
            layout.ann_meta_b *
                (off_t)VECFS_DISK_BLOCK_SIZE)) {
        fprintf(stderr,
                "failed to write ANN metadata B: %s\n",
                strerror(errno));
        goto out;
    }

    if (pwrite_full(fd, super, sizeof(*super), 0)) {
        fprintf(stderr,
                "failed to write superblock: %s\n",
                strerror(errno));
        goto out;
    }

    if (fsync(fd) < 0) {
        fprintf(stderr,
                "failed to flush %s: %s\n",
                opt.device, strerror(errno));
        goto out;
    }

    printf("VecFS V2 formatted successfully\n");
    printf("  device:             %s\n", opt.device);
    printf("  block size:         %u\n",
           VECFS_DISK_BLOCK_SIZE);
    printf("  total blocks:       %llu\n",
           (unsigned long long)layout.total_blocks);
    printf("  inode count:        %llu\n",
           (unsigned long long)layout.inode_count);
    printf("  data start:         %llu\n",
           (unsigned long long)layout.data_start);
    printf("  free data blocks:   %llu\n",
           (unsigned long long)
               (layout.total_blocks -
                layout.data_start));
    printf("  ANN metadata A:     %llu\n",
           (unsigned long long)layout.ann_meta_a);
    printf("  ANN metadata B:     %llu\n",
           (unsigned long long)layout.ann_meta_b);
    printf("  vector dimension:   %u\n", opt.dimension);
    printf("  vector type:        fp32\n");
    printf("  distance:           %s\n",
           opt.distance_type == VECFS_ANN_DIST_L2 ?
               "l2" : "ip");
    printf("  M:                  %u\n", opt.m);
    printf("  M0:                 %u\n", opt.m0);
    printf("  maximum levels:     %u\n", opt.max_levels);
    printf("  ef construction:    %u\n",
           opt.ef_construction);
    printf("  default ef search:  %u\n",
           opt.ef_search);
    printf("  object stride:      %llu bytes\n",
           (unsigned long long)
               calculate_object_stride(&opt));

    ret = EXIT_SUCCESS;

out:
    free(meta_b);
    free(meta_a);
    free(super);

    if (fd >= 0)
        close(fd);

    return ret;
}

