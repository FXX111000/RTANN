#define _GNU_SOURCE

#include <endian.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "../include/vecfs_ioctl.h"
#include "exchange-format.h"

struct crc32_context {
    uint32_t value;
};

struct exchange_node {
    struct vecfs_exchange_node_header header;

    void *vector;
    uint32_t *level0_links;
    uint16_t *upper_counts;
    uint32_t *upper_links;

    uint32_t level0_count;
    uint32_t upper_count_count;
    uint32_t upper_link_count;

    size_t record_bytes;
    void *record_buffer;
};

static uint32_t crc32_table[256];

static void crc32_initialize_table(void)
{
    uint32_t i;

    for (i = 0; i < 256; i++) {
        uint32_t value = i;
        unsigned int bit;

        for (bit = 0; bit < 8; bit++) {
            if (value & 1)
                value =
                    UINT32_C(0xedb88320) ^
                    (value >> 1);
            else
                value >>= 1;
        }

        crc32_table[i] = value;
    }
}

static void crc32_begin(
    struct crc32_context *context)
{
    context->value = UINT32_MAX;
}

static void crc32_update(
    struct crc32_context *context,
    const void *buffer,
    size_t length)
{
    const uint8_t *bytes = buffer;
    size_t i;

    for (i = 0; i < length; i++) {
        context->value =
            crc32_table[
                (context->value ^ bytes[i]) &
                UINT8_MAX] ^
            (context->value >> 8);
    }
}

static uint32_t crc32_end(
    const struct crc32_context *context)
{
    return context->value ^ UINT32_MAX;
}

static uint32_t crc32_buffer(
    const void *buffer,
    size_t length)
{
    struct crc32_context context;

    crc32_begin(&context);
    crc32_update(&context, buffer, length);

    return crc32_end(&context);
}

static int checked_add_size(
    size_t left,
    size_t right,
    size_t *result)
{
    if (left > SIZE_MAX - right)
        return -1;

    *result = left + right;
    return 0;
}

static int checked_mul_size(
    size_t left,
    size_t right,
    size_t *result)
{
    if (left && right > SIZE_MAX / left)
        return -1;

    *result = left * right;
    return 0;
}

static int read_full(
    int fd,
    void *buffer,
    size_t length)
{
    uint8_t *bytes = buffer;
    size_t done = 0;

    while (done < length) {
        ssize_t count = read(
            fd,
            bytes + done,
            length - done);

        if (count < 0) {
            if (errno == EINTR)
                continue;

            return -1;
        }

        if (!count) {
            errno = ENODATA;
            return -1;
        }

        done += (size_t)count;
    }

    return 0;
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
            errno = ENODATA;
            return -1;
        }

        done += (size_t)count;
    }

    return 0;
}

static int write_full(
    int fd,
    const void *buffer,
    size_t length)
{
    const uint8_t *bytes = buffer;
    size_t done = 0;

    while (done < length) {
        ssize_t count = write(
            fd,
            bytes + done,
            length - done);

        if (count < 0) {
            if (errno == EINTR)
                continue;

            return -1;
        }

        if (!count) {
            errno = EIO;
            return -1;
        }

        done += (size_t)count;
    }

    return 0;
}

static int pwrite_full(
    int fd,
    const void *buffer,
    size_t length,
    off_t offset)
{
    const uint8_t *bytes = buffer;
    size_t done = 0;

    while (done < length) {
        ssize_t count = pwrite(
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

        done += (size_t)count;
    }

    return 0;
}

static int seek_absolute(
    int fd,
    uint64_t offset)
{
    if (offset > (uint64_t)INT64_MAX) {
        errno = EOVERFLOW;
        return -1;
    }

    if (lseek(fd, (off_t)offset, SEEK_SET) < 0)
        return -1;

    return 0;
}

static int get_file_size(
    int fd,
    uint64_t *size)
{
    struct stat status;

    if (fstat(fd, &status) < 0)
        return -1;

    if (status.st_size < 0) {
        errno = EINVAL;
        return -1;
    }

    *size = (uint64_t)status.st_size;
    return 0;
}

static void exchange_node_destroy(
    struct exchange_node *node)
{
    if (!node)
        return;

    free(node->record_buffer);

    memset(node, 0, sizeof(*node));
}

static int exchange_header_crc_valid(
    const struct vecfs_exchange_file_header *header)
{
    struct vecfs_exchange_file_header copy;
    uint32_t stored_crc;
    uint32_t calculated_crc;

    copy = *header;

    stored_crc = le32toh(copy.header_crc);
    copy.header_crc = htole32(0);

    calculated_crc =
        crc32_buffer(&copy, sizeof(copy));

    return stored_crc == calculated_crc;
}

static void exchange_header_update_crc(
    struct vecfs_exchange_file_header *header)
{
    header->header_crc = htole32(0);

    header->header_crc = htole32(
        crc32_buffer(header, sizeof(*header)));
}

static int exchange_validate_header(
    const struct vecfs_exchange_file_header *header,
    uint64_t file_size)
{
    uint64_t records_offset;
    uint64_t records_bytes;
    uint64_t node_count;
    uint32_t capacity;
    uint16_t m;
    uint16_t m0;
    uint16_t max_levels;
    uint16_t current_max_level;

    if (le64toh(header->magic) !=
        VECFS_EXCHANGE_FILE_MAGIC) {
        fprintf(stderr,
                "invalid exchange magic\n");
        return -1;
    }

    if (le16toh(header->major) !=
        VECFS_EXCHANGE_FILE_MAJOR) {
        fprintf(stderr,
                "unsupported exchange major version: %u\n",
                le16toh(header->major));
        return -1;
    }

    if (le16toh(header->minor) >
        VECFS_EXCHANGE_FILE_MINOR) {
        fprintf(stderr,
                "unsupported exchange minor version: %u\n",
                le16toh(header->minor));
        return -1;
    }

    if (!exchange_header_crc_valid(header)) {
        fprintf(stderr,
                "exchange header CRC mismatch\n");
        return -1;
    }

    if (le16toh(header->element_type) !=
        VECFS_UAPI_ELEM_FP32) {
        fprintf(stderr,
                "only FP32 exchange files are supported\n");
        return -1;
    }

    if (le16toh(header->distance_type) !=
            VECFS_UAPI_DIST_L2 &&
        le16toh(header->distance_type) !=
            VECFS_UAPI_DIST_IP) {
        fprintf(stderr,
                "invalid exchange distance type\n");
        return -1;
    }

    if (!le32toh(header->dimension)) {
        fprintf(stderr,
                "invalid zero vector dimension\n");
        return -1;
    }

    m = le16toh(header->m);
    m0 = le16toh(header->m0);
    max_levels = le16toh(header->max_levels);
    current_max_level =
        le16toh(header->current_max_level);

    if (!m || !m0 || m0 < m) {
        fprintf(stderr,
                "invalid M/M0 values\n");
        return -1;
    }

    if (!max_levels ||
        current_max_level >= max_levels) {
        fprintf(stderr,
                "invalid HNSW level configuration\n");
        return -1;
    }

    node_count = le64toh(header->node_count);
    capacity = le32toh(header->node_capacity);

    if (capacity < 2) {
        fprintf(stderr,
                "invalid node capacity\n");
        return -1;
    }

    if (node_count >
        (uint64_t)capacity - 1) {
        fprintf(stderr,
                "node count exceeds capacity\n");
        return -1;
    }

    records_offset =
        le64toh(header->node_records_offset);

    records_bytes =
        le64toh(header->node_records_bytes);

    if (records_offset < sizeof(*header) ||
        records_offset > file_size ||
        records_bytes > file_size - records_offset) {
        fprintf(stderr,
                "invalid exchange record region\n");
        return -1;
    }

    if (records_offset + records_bytes !=
        file_size) {
        fprintf(stderr,
                "unexpected trailing data in exchange file\n");
        return -1;
    }

    if (!node_count &&
        le32toh(header->entry_node_id) !=
            UINT32_MAX) {
        fprintf(stderr,
                "empty graph has a valid entry node\n");
        return -1;
    }

    return 0;
}

static int exchange_validate_records_crc(
    int fd,
    const struct vecfs_exchange_file_header *header)
{
    struct crc32_context context;
    uint8_t *buffer;
    uint64_t remaining;
    off_t offset;
    uint32_t expected;
    uint32_t actual;
    int result = -1;

    remaining =
        le64toh(header->node_records_bytes);

    offset = (off_t)
        le64toh(header->node_records_offset);

    expected =
        le32toh(header->records_crc);

    buffer = malloc(1024U * 1024U);
    if (!buffer)
        return -1;

    crc32_begin(&context);

    while (remaining) {
        size_t chunk = remaining >
                1024U * 1024U ?
            1024U * 1024U :
            (size_t)remaining;

        if (pread_full(
                fd,
                buffer,
                chunk,
                offset) < 0)
            goto out;

        crc32_update(
            &context,
            buffer,
            chunk);

        offset += (off_t)chunk;
        remaining -= chunk;
    }

    actual = crc32_end(&context);

    if (actual != expected) {
        fprintf(stderr,
                "exchange records CRC mismatch: "
                "expected=%08x actual=%08x\n",
                expected,
                actual);

        errno = EBADMSG;
        goto out;
    }

    result = 0;

out:
    free(buffer);
    return result;
}

static int exchange_calculate_record_size(
    uint32_t vector_bytes,
    uint16_t level0_count,
    uint16_t level,
    uint16_t m,
    size_t *record_size)
{
    size_t total =
        sizeof(struct vecfs_exchange_node_header);

    size_t value;

    if (checked_add_size(
            total,
            vector_bytes,
            &total))
        return -1;

    if (checked_mul_size(
            level0_count,
            sizeof(uint32_t),
            &value) ||
        checked_add_size(
            total,
            value,
            &total))
        return -1;

    if (checked_mul_size(
            level,
            sizeof(uint16_t),
            &value) ||
        checked_add_size(
            total,
            value,
            &total))
        return -1;

    if (checked_mul_size(
            level,
            m,
            &value) ||
        checked_mul_size(
            value,
            sizeof(uint32_t),
            &value) ||
        checked_add_size(
            total,
            value,
            &total))
        return -1;

    if (total >
        VECFS_EXCHANGE_MAX_RECORD_BYTES)
        return -1;

    *record_size = total;
    return 0;
}

static int exchange_record_crc_valid(
    void *record,
    size_t record_size)
{
    struct vecfs_exchange_node_header *header =
        record;

    uint32_t stored_crc =
        le32toh(header->record_crc);

    uint32_t calculated_crc;

    header->record_crc = htole32(0);

    calculated_crc =
        crc32_buffer(record, record_size);

    header->record_crc =
        htole32(stored_crc);

    return stored_crc == calculated_crc;
}

static int exchange_read_node(
    int fd,
    const struct vecfs_exchange_file_header *file_header,
    struct exchange_node *node)
{
    struct vecfs_exchange_node_header disk_header;
    uint32_t vector_bytes;
    uint16_t level0_count;
    uint16_t level;
    uint16_t upper_level_count;
    uint16_t m;
    uint64_t disk_record_bytes;
    size_t expected_record_bytes;
    uint8_t *cursor;
    size_t level0_bytes;
    size_t upper_count_bytes;
    size_t upper_link_bytes;

    memset(node, 0, sizeof(*node));

    if (read_full(
            fd,
            &disk_header,
            sizeof(disk_header)) < 0)
        return -1;

    vector_bytes =
        le32toh(disk_header.vector_bytes);

    level0_count =
        le16toh(disk_header.level0_count);

    level =
        le16toh(disk_header.level);

    upper_level_count =
        le16toh(disk_header.upper_level_count);

    m = le16toh(file_header->m);

    if (upper_level_count != level) {
        fprintf(stderr,
                "node %u has inconsistent upper level count\n",
                le32toh(disk_header.node_id));

        errno = EINVAL;
        return -1;
    }

    if (level0_count >
        le16toh(file_header->m0)) {
        fprintf(stderr,
                "node %u level-0 count exceeds M0\n",
                le32toh(disk_header.node_id));

        errno = EINVAL;
        return -1;
    }

    if (level >=
        le16toh(file_header->max_levels)) {
        fprintf(stderr,
                "node %u level exceeds max levels\n",
                le32toh(disk_header.node_id));

        errno = EINVAL;
        return -1;
    }

    if (vector_bytes !=
        le32toh(file_header->dimension) *
            sizeof(float)) {
        fprintf(stderr,
                "node %u has invalid vector size\n",
                le32toh(disk_header.node_id));

        errno = EINVAL;
        return -1;
    }

    if (exchange_calculate_record_size(
            vector_bytes,
            level0_count,
            level,
            m,
            &expected_record_bytes)) {
        errno = EOVERFLOW;
        return -1;
    }

    disk_record_bytes =
        le64toh(disk_header.record_bytes);

    if (disk_record_bytes !=
        expected_record_bytes) {
        fprintf(stderr,
                "node %u record size mismatch\n",
                le32toh(disk_header.node_id));

        errno = EINVAL;
        return -1;
    }

    node->record_buffer =
        malloc(expected_record_bytes);

    if (!node->record_buffer)
        return -1;

    memcpy(
        node->record_buffer,
        &disk_header,
        sizeof(disk_header));

    if (read_full(
            fd,
            (uint8_t *)node->record_buffer +
                sizeof(disk_header),
            expected_record_bytes -
                sizeof(disk_header)) < 0)
        goto fail;

    if (!exchange_record_crc_valid(
            node->record_buffer,
            expected_record_bytes)) {
        fprintf(stderr,
                "node %u record CRC mismatch\n",
                le32toh(disk_header.node_id));

        errno = EBADMSG;
        goto fail;
    }

    node->header =
        *(struct vecfs_exchange_node_header *)
            node->record_buffer;

    node->record_bytes =
        expected_record_bytes;

    cursor =
        (uint8_t *)node->record_buffer +
        sizeof(struct vecfs_exchange_node_header);

    node->vector = cursor;
    cursor += vector_bytes;

    node->level0_links =
        (uint32_t *)cursor;

    if (checked_mul_size(
            level0_count,
            sizeof(uint32_t),
            &level0_bytes))
        goto overflow;

    cursor += level0_bytes;

    node->upper_counts =
        (uint16_t *)cursor;

    if (checked_mul_size(
            level,
            sizeof(uint16_t),
            &upper_count_bytes))
        goto overflow;

    cursor += upper_count_bytes;

    node->upper_links =
        (uint32_t *)cursor;

    if (checked_mul_size(
            level,
            m,
            &upper_link_bytes) ||
        checked_mul_size(
            upper_link_bytes,
            sizeof(uint32_t),
            &upper_link_bytes))
        goto overflow;

    cursor += upper_link_bytes;

    if (cursor !=
        (uint8_t *)node->record_buffer +
            expected_record_bytes) {
        errno = EINVAL;
        goto fail;
    }

    node->level0_count =
        level0_count;

    node->upper_count_count =
        level;

    node->upper_link_count =
        (uint32_t)level * m;

    return 0;

overflow:
    errno = EOVERFLOW;

fail:
    exchange_node_destroy(node);
    return -1;
}

static int export_node_record(
    int vecfs_fd,
    int output_fd,
    const struct vecfs_ioctl_exchange_info *info,
    uint32_t node_id,
    struct crc32_context *records_crc,
    uint64_t *record_bytes,
    uint64_t *exported_nodes)
{
    struct vecfs_ioctl_export_node request;
    struct vecfs_exchange_node_header *node_header;
    uint8_t *record = NULL;
    uint8_t *cursor;
    float *vector = NULL;
    uint32_t *level0 = NULL;
    uint16_t *upper_counts = NULL;
    uint32_t *upper_links = NULL;
    uint16_t level;
    uint16_t level0_count;
    size_t record_size;
    size_t level0_bytes;
    size_t upper_count_bytes;
    size_t upper_link_bytes;
    uint32_t i;
    int result = -1;

    vector = malloc((size_t)info->vector_bytes);

    level0 = calloc(
        info->m0,
        sizeof(*level0));

    upper_counts = calloc(
        info->max_levels,
        sizeof(*upper_counts));

    upper_links = calloc(
        (size_t)info->max_levels * info->m,
        sizeof(*upper_links));

    if (!vector ||
        !level0 ||
        !upper_counts ||
        !upper_links)
        goto out;

    memset(&request, 0, sizeof(request));

    request.node_id = node_id;

    request.flags =
        VECFS_EXPORT_F_INCLUDE_VECTORS |
        VECFS_EXPORT_F_INCLUDE_GRAPH;

    request.vector_ptr =
        (uint64_t)(uintptr_t)vector;

    request.vector_capacity =
        info->vector_bytes;

    request.level0_links_ptr =
        (uint64_t)(uintptr_t)level0;

    request.level0_capacity =
        info->m0;

    request.upper_counts_ptr =
        (uint64_t)(uintptr_t)upper_counts;

    request.upper_counts_capacity =
        info->max_levels;

    request.upper_links_ptr =
        (uint64_t)(uintptr_t)upper_links;

    request.upper_links_capacity =
        (uint32_t)info->max_levels * info->m;

    if (ioctl(
            vecfs_fd,
            VECFS_IOC_EXPORT_NODE,
            &request) < 0) {
        /*
         * node ID 空洞和 tombstone 已被内核 export 接口隐藏。
         */
        if (errno == ENOENT) {
            result = 1;
            goto out;
        }

        fprintf(stderr,
                "EXPORT_NODE(%u) failed: %s\n",
                node_id,
                strerror(errno));

        goto out;
    }

    level = request.level;
    level0_count =
        request.level0_count_out;

    if (level >= info->max_levels ||
        level0_count > info->m0 ||
        request.vector_bytes !=
            info->vector_bytes ||
        request.upper_counts_count_out !=
            level ||
        request.upper_links_count_out !=
            (uint32_t)level * info->m) {
        fprintf(stderr,
                "kernel returned invalid node %u metadata\n",
                node_id);

        errno = EPROTO;
        goto out;
    }

    if (exchange_calculate_record_size(
            request.vector_bytes,
            level0_count,
            level,
            info->m,
            &record_size)) {
        errno = EOVERFLOW;
        goto out;
    }

    record = calloc(1, record_size);
    if (!record)
        goto out;

    node_header =
        (struct vecfs_exchange_node_header *)
            record;

    node_header->node_id =
        htole32(node_id);

    node_header->level =
        htole16(level);

    node_header->flags =
        htole16(VECFS_EXCHANGE_NODE_F_NONE);

    node_header->external_id =
        htole64(request.external_id);

    node_header->vector_bytes =
        htole32(request.vector_bytes);

    node_header->level0_count =
        htole16(level0_count);

    node_header->upper_level_count =
        htole16(level);

    node_header->record_bytes =
        htole64(record_size);

    node_header->record_crc =
        htole32(0);

    cursor = record + sizeof(*node_header);

    memcpy(
        cursor,
        vector,
        request.vector_bytes);

    cursor += request.vector_bytes;

    if (checked_mul_size(
            level0_count,
            sizeof(uint32_t),
            &level0_bytes))
        goto overflow;

    for (i = 0; i < level0_count; i++) {
        uint32_t disk_id =
            htole32(level0[i]);

        memcpy(
            cursor + (size_t)i * sizeof(disk_id),
            &disk_id,
            sizeof(disk_id));
    }

    cursor += level0_bytes;

    if (checked_mul_size(
            level,
            sizeof(uint16_t),
            &upper_count_bytes))
        goto overflow;

    for (i = 0; i < level; i++) {
        uint16_t disk_count =
            htole16(upper_counts[i]);

        if (upper_counts[i] > info->m) {
            errno = EPROTO;
            goto out;
        }

        memcpy(
            cursor + (size_t)i *
                sizeof(disk_count),
            &disk_count,
            sizeof(disk_count));
    }

    cursor += upper_count_bytes;

    if (checked_mul_size(
            level,
            info->m,
            &upper_link_bytes) ||
        checked_mul_size(
            upper_link_bytes,
            sizeof(uint32_t),
            &upper_link_bytes))
        goto overflow;

    for (i = 0;
         i < (uint32_t)level * info->m;
         i++) {
        uint32_t disk_id =
            htole32(upper_links[i]);

        memcpy(
            cursor + (size_t)i * sizeof(disk_id),
            &disk_id,
            sizeof(disk_id));
    }

    cursor += upper_link_bytes;

    if (cursor != record + record_size) {
        errno = EINVAL;
        goto out;
    }

    node_header->record_crc =
        htole32(0);

    node_header->record_crc =
        htole32(
            crc32_buffer(record, record_size));

    if (write_full(
            output_fd,
            record,
            record_size) < 0)
        goto out;

    crc32_update(
        records_crc,
        record,
        record_size);

    if (*record_bytes >
        UINT64_MAX - record_size) {
        errno = EOVERFLOW;
        goto out;
    }

    *record_bytes += record_size;
    (*exported_nodes)++;

    result = 0;
    goto out;

overflow:
    errno = EOVERFLOW;

out:
    free(record);
    free(upper_links);
    free(upper_counts);
    free(level0);
    free(vector);

    return result;
}

static int command_export(
    const char *vecfs_path,
    const char *exchange_path)
{
    struct vecfs_ioctl_exchange_info info;
    struct vecfs_exchange_file_header header;
    struct crc32_context records_crc;
    uint64_t record_bytes = 0;
    uint64_t exported_nodes = 0;
    uint32_t node_id;
    int vecfs_fd = -1;
    int output_fd = -1;
    int result = -1;

    vecfs_fd = open(
        vecfs_path,
        O_RDONLY | O_CLOEXEC);

    if (vecfs_fd < 0) {
        fprintf(stderr,
                "cannot open VecFS path %s: %s\n",
                vecfs_path,
                strerror(errno));

        goto out;
    }

    memset(&info, 0, sizeof(info));

    if (ioctl(
            vecfs_fd,
            VECFS_IOC_EXPORT_INFO,
            &info) < 0) {
        fprintf(stderr,
                "EXPORT_INFO failed: %s\n",
                strerror(errno));

        goto out;
    }

    if (info.exchange_major !=
            VECFS_EXCHANGE_MAJOR ||
        info.element_type !=
            VECFS_UAPI_ELEM_FP32 ||
        !info.dimension ||
        !info.m ||
        !info.m0 ||
        info.m0 < info.m ||
        info.node_capacity < 2) {
        fprintf(stderr,
                "kernel returned unsupported exchange configuration\n");

        errno = EPROTO;
        goto out;
    }

    output_fd = open(
        exchange_path,
        O_WRONLY |
        O_CREAT |
        O_TRUNC |
        O_CLOEXEC,
        0644);

    if (output_fd < 0) {
        fprintf(stderr,
                "cannot create %s: %s\n",
                exchange_path,
                strerror(errno));

        goto out;
    }

    memset(&header, 0, sizeof(header));

    /*
     * 先写占位 header，导出完成后回写 node count、CRC 和长度。
     */
    if (write_full(
            output_fd,
            &header,
            sizeof(header)) < 0)
        goto out;

    crc32_begin(&records_crc);

    for (node_id = 1;
         node_id < info.node_capacity;
         node_id++) {
        int export_result =
            export_node_record(
                vecfs_fd,
                output_fd,
                &info,
                node_id,
                &records_crc,
                &record_bytes,
                &exported_nodes);

        if (export_result < 0)
            goto out;
    }

    if (exported_nodes !=
        info.node_count) {
        fprintf(stderr,
                "exported node count mismatch: "
                "expected=%" PRIu64 " actual=%" PRIu64 "\n",
                (uint64_t)info.node_count,
                (uint64_t)exported_nodes);

        errno = EIO;
        goto out;
    }

    memset(&header, 0, sizeof(header));

    header.magic =
        htole64(VECFS_EXCHANGE_FILE_MAGIC);

    header.major =
        htole16(VECFS_EXCHANGE_FILE_MAJOR);

    header.minor =
        htole16(VECFS_EXCHANGE_FILE_MINOR);

    header.element_type =
        htole16(info.element_type);

    header.distance_type =
        htole16(info.distance_type);

    header.dimension =
        htole32(info.dimension);

    header.m =
        htole16(info.m);

    header.m0 =
        htole16(info.m0);

    header.max_levels =
        htole16(info.max_levels);

    header.current_max_level =
        htole16(info.current_max_level);

    header.ef_construction =
        htole32(info.ef_construction);

    header.default_ef_search =
        htole32(info.default_ef_search);

    header.node_count =
        htole64(exported_nodes);

    header.node_capacity =
        htole32(info.node_capacity);

    header.entry_node_id =
        htole32(info.entry_node_id);

    header.random_seed =
        htole64(info.random_seed);

    header.node_records_offset =
        htole64(sizeof(header));

    header.node_records_bytes =
        htole64(record_bytes);

    header.records_crc =
        htole32(crc32_end(&records_crc));

    exchange_header_update_crc(&header);

    if (pwrite_full(
            output_fd,
            &header,
            sizeof(header),
            0) < 0)
        goto out;

    if (fsync(output_fd) < 0)
        goto out;

    printf(
        "exported %" PRIu64 " nodes to %s\n"
        "  dimension:       %u\n"
        "  distance:        %s\n"
        "  M/M0:            %u/%u\n"
        "  max levels:      %u\n"
        "  node capacity:   %" PRIu64 "\n"
        "  entry node:      %u\n"
        "  records bytes:   %" PRIu64 "\n",
        (uint64_t)exported_nodes,
        exchange_path,
        info.dimension,
        info.distance_type ==
            VECFS_UAPI_DIST_L2 ?
            "l2" : "ip",
        info.m,
        info.m0,
        info.max_levels,
        (uint64_t)info.node_capacity,
        info.entry_node_id,
        (uint64_t)record_bytes);

    result = 0;

out:
    if (output_fd >= 0)
        close(output_fd);

    if (vecfs_fd >= 0)
        close(vecfs_fd);

    if (result)
        unlink(exchange_path);

    return result;
}

static int import_node_record(
    int vecfs_fd,
    uint64_t transaction_id,
    const struct vecfs_exchange_file_header *header,
    const struct exchange_node *node)
{
    struct vecfs_ioctl_import_node request;
    uint16_t level =
        le16toh(node->header.level);

    uint16_t level0_count =
        le16toh(node->header.level0_count);

    uint16_t m =
        le16toh(header->m);

    uint32_t *level0_host = NULL;
    uint16_t *upper_counts_host = NULL;
    uint32_t *upper_links_host = NULL;
    uint32_t i;
    int result = -1;

    if (level0_count) {
        level0_host = calloc(
            level0_count,
            sizeof(*level0_host));

        if (!level0_host)
            goto out;

        for (i = 0; i < level0_count; i++)
            level0_host[i] =
                le32toh(node->level0_links[i]);
    }

    if (level) {
        upper_counts_host = calloc(
            level,
            sizeof(*upper_counts_host));

        upper_links_host = calloc(
            (size_t)level * m,
            sizeof(*upper_links_host));

        if (!upper_counts_host ||
            !upper_links_host)
            goto out;

        for (i = 0; i < level; i++) {
            upper_counts_host[i] =
                le16toh(node->upper_counts[i]);

            if (upper_counts_host[i] > m) {
                errno = EINVAL;
                goto out;
            }
        }

        for (i = 0;
             i < (uint32_t)level * m;
             i++)
            upper_links_host[i] =
                le32toh(node->upper_links[i]);
    }

    memset(&request, 0, sizeof(request));

    request.transaction_id =
        transaction_id;

    request.node_id =
        le32toh(node->header.node_id);

    request.level =
        level;

    request.external_id =
        le64toh(node->header.external_id);

    request.vector_ptr =
        (uint64_t)(uintptr_t)node->vector;

    request.vector_bytes =
        le32toh(node->header.vector_bytes);

    request.level0_count =
        level0_count;

    request.upper_level_count =
        level;

    request.level0_links_ptr =
        (uint64_t)(uintptr_t)level0_host;

    request.upper_counts_ptr =
        (uint64_t)(uintptr_t)upper_counts_host;

    request.upper_links_ptr =
        (uint64_t)(uintptr_t)upper_links_host;

    if (ioctl(
            vecfs_fd,
            VECFS_IOC_IMPORT_NODE,
            &request) < 0) {
        fprintf(stderr,
                "IMPORT_NODE(%u) failed: %s\n",
                request.node_id,
                strerror(errno));

        goto out;
    }

    result = 0;

out:
    free(upper_links_host);
    free(upper_counts_host);
    free(level0_host);

    return result;
}

static int command_import(
    const char *exchange_path,
    const char *vecfs_path)
{
    struct vecfs_exchange_file_header header;
    struct vecfs_ioctl_import_begin begin;
    struct vecfs_ioctl_import_commit commit;
    struct vecfs_ioctl_import_abort abort_request;
    uint64_t file_size;
    uint64_t node_count;
    uint64_t record_bytes;
    uint64_t consumed_bytes = 0;
    uint64_t imported = 0;
    uint64_t transaction_id = 0;
    int input_fd = -1;
    int vecfs_fd = -1;
    int result = -1;

    input_fd = open(
        exchange_path,
        O_RDONLY | O_CLOEXEC);

    if (input_fd < 0) {
        fprintf(stderr,
                "cannot open %s: %s\n",
                exchange_path,
                strerror(errno));

        goto out;
    }

    if (get_file_size(
            input_fd,
            &file_size) < 0)
        goto out;

    if (file_size < sizeof(header)) {
        fprintf(stderr,
                "exchange file is too small\n");

        errno = EINVAL;
        goto out;
    }

    if (pread_full(
            input_fd,
            &header,
            sizeof(header),
            0) < 0)
        goto out;

    if (exchange_validate_header(
            &header,
            file_size))
        goto out;

    if (exchange_validate_records_crc(
            input_fd,
            &header) < 0)
        goto out;

    if (seek_absolute(
            input_fd,
            le64toh(
                header.node_records_offset)) < 0)
        goto out;

    vecfs_fd = open(
        vecfs_path,
        O_RDWR | O_CLOEXEC);

    if (vecfs_fd < 0) {
        fprintf(stderr,
                "cannot open VecFS path %s: %s\n",
                vecfs_path,
                strerror(errno));

        goto out;
    }

    memset(&begin, 0, sizeof(begin));

    begin.info.exchange_major =
        VECFS_EXCHANGE_MAJOR;

    begin.info.exchange_minor =
        VECFS_EXCHANGE_MINOR;

    begin.info.element_type =
        le16toh(header.element_type);

    begin.info.distance_type =
        le16toh(header.distance_type);

    begin.info.dimension =
        le32toh(header.dimension);

    begin.info.m =
        le16toh(header.m);

    begin.info.m0 =
        le16toh(header.m0);

    begin.info.max_levels =
        le16toh(header.max_levels);

    begin.info.current_max_level =
        le16toh(header.current_max_level);

    begin.info.ef_construction =
        le32toh(header.ef_construction);

    begin.info.default_ef_search =
        le32toh(header.default_ef_search);

    begin.expected_nodes =
        le64toh(header.node_count);

    begin.node_capacity =
        le32toh(header.node_capacity);

    begin.flags =
        VECFS_IMPORT_F_PRESERVE_NODE_IDS;

    if (ioctl(
            vecfs_fd,
            VECFS_IOC_IMPORT_BEGIN,
            &begin) < 0) {
        fprintf(stderr,
                "IMPORT_BEGIN failed: %s\n",
                strerror(errno));

        goto out;
    }

    transaction_id =
        begin.transaction_id;

    node_count =
        le64toh(header.node_count);

    record_bytes =
        le64toh(header.node_records_bytes);

    while (imported < node_count) {
        struct exchange_node node;

        if (consumed_bytes >= record_bytes) {
            fprintf(stderr,
                    "record region ended before node count\n");

            errno = ENODATA;
            goto abort_import;
        }

        if (exchange_read_node(
                input_fd,
                &header,
                &node) < 0)
            goto abort_import;

        if (node.record_bytes >
            record_bytes - consumed_bytes) {
            exchange_node_destroy(&node);

            fprintf(stderr,
                    "node record exceeds record region\n");

            errno = EINVAL;
            goto abort_import;
        }

        if (import_node_record(
                vecfs_fd,
                transaction_id,
                &header,
                &node) < 0) {
            exchange_node_destroy(&node);
            goto abort_import;
        }

        consumed_bytes +=
            node.record_bytes;

        imported++;

        exchange_node_destroy(&node);
    }

    if (consumed_bytes != record_bytes) {
        fprintf(stderr,
                "record bytes mismatch: "
                "consumed=%" PRIu64 " expected=%" PRIu64 "\n",
                consumed_bytes,
                record_bytes);

        errno = EINVAL;
        goto abort_import;
    }

    memset(&commit, 0, sizeof(commit));

    commit.transaction_id =
        transaction_id;

    commit.entry_node_id =
        le32toh(header.entry_node_id);

    commit.current_max_level =
        le16toh(header.current_max_level);

    if (ioctl(
            vecfs_fd,
            VECFS_IOC_IMPORT_COMMIT,
            &commit) < 0) {
        fprintf(stderr,
                "IMPORT_COMMIT failed: %s\n",
                strerror(errno));

        goto abort_import;
    }

    printf(
        "imported %" PRIu64 " nodes into %s\n"
        "  epoch:       %" PRIu64 "\n"
        "  entry node:  %u\n"
        "  max level:   %u\n",
        (uint64_t)commit.imported_nodes,
        vecfs_path,
        (uint64_t)commit.new_epoch,
        commit.entry_node_id,
        commit.current_max_level);

    transaction_id = 0;
    result = 0;
    goto out;

abort_import:
    if (transaction_id) {
        memset(
            &abort_request,
            0,
            sizeof(abort_request));

        abort_request.transaction_id =
            transaction_id;

        if (ioctl(
                vecfs_fd,
                VECFS_IOC_IMPORT_ABORT,
                &abort_request) < 0)
            fprintf(stderr,
                    "IMPORT_ABORT failed: %s\n",
                    strerror(errno));
    }

out:
    if (vecfs_fd >= 0)
        close(vecfs_fd);

    if (input_fd >= 0)
        close(input_fd);

    return result;
}

static const char *element_type_name(
    uint16_t type)
{
    switch (type) {
    case VECFS_UAPI_ELEM_FP32:
        return "fp32";
    case VECFS_UAPI_ELEM_FP16:
        return "fp16";
    case VECFS_UAPI_ELEM_INT8:
        return "int8";
    case VECFS_UAPI_ELEM_INT16:
        return "int16";
    default:
        return "unknown";
    }
}

static const char *distance_type_name(
    uint16_t type)
{
    switch (type) {
    case VECFS_UAPI_DIST_L2:
        return "l2";
    case VECFS_UAPI_DIST_IP:
        return "ip";
    default:
        return "unknown";
    }
}

static int command_info(
    const char *exchange_path)
{
    struct vecfs_exchange_file_header header;
    uint64_t file_size;
    int fd;
    int result = -1;

    fd = open(
        exchange_path,
        O_RDONLY | O_CLOEXEC);

    if (fd < 0) {
        fprintf(stderr,
                "cannot open %s: %s\n",
                exchange_path,
                strerror(errno));

        return -1;
    }

    if (get_file_size(fd, &file_size) < 0)
        goto out;

    if (file_size < sizeof(header)) {
        errno = EINVAL;
        goto out;
    }

    if (pread_full(
            fd,
            &header,
            sizeof(header),
            0) < 0)
        goto out;

    if (exchange_validate_header(
            &header,
            file_size))
        goto out;

    if (exchange_validate_records_crc(
            fd,
            &header) < 0)
        goto out;

    printf(
        "VecFS exchange file: %s\n"
        "  version:           %u.%u\n"
        "  element type:      %s\n"
        "  distance:          %s\n"
        "  dimension:         %u\n"
        "  M/M0:              %u/%u\n"
        "  max levels:        %u\n"
        "  current max level: %u\n"
        "  ef construction:   %u\n"
        "  default ef search: %u\n"
        "  node count:        %" PRIu64 "\n"
        "  node capacity:     %u\n"
        "  entry node:        %u\n"
        "  records bytes:     %" PRIu64 "\n",
        exchange_path,
        le16toh(header.major),
        le16toh(header.minor),
        element_type_name(
            le16toh(header.element_type)),
        distance_type_name(
            le16toh(header.distance_type)),
        le32toh(header.dimension),
        le16toh(header.m),
        le16toh(header.m0),
        le16toh(header.max_levels),
        le16toh(header.current_max_level),
        le32toh(header.ef_construction),
        le32toh(header.default_ef_search),
        le64toh(header.node_count),
        le32toh(header.node_capacity),
        le32toh(header.entry_node_id),
        le64toh(header.node_records_bytes));

    result = 0;

out:
    if (result)
        fprintf(stderr,
                "failed to inspect %s: %s\n",
                exchange_path,
                strerror(errno));

    close(fd);
    return result;
}

static void usage(
    const char *program)
{
    fprintf(
        stderr,
        "Usage:\n"
        "  %s export <vecfs-path> <exchange-file>\n"
        "  %s import <exchange-file> <vecfs-path>\n"
        "  %s info   <exchange-file>\n",
        program,
        program,
        program);
}

int main(
    int argc,
    char **argv)
{
    crc32_initialize_table();

    if (argc < 2) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (!strcmp(argv[1], "export")) {
        if (argc != 4) {
            usage(argv[0]);
            return EXIT_FAILURE;
        }

        return command_export(
            argv[2],
            argv[3]) ?
            EXIT_FAILURE :
            EXIT_SUCCESS;
    }

    if (!strcmp(argv[1], "import")) {
        if (argc != 4) {
            usage(argv[0]);
            return EXIT_FAILURE;
        }

        return command_import(
            argv[2],
            argv[3]) ?
            EXIT_FAILURE :
            EXIT_SUCCESS;
    }

    if (!strcmp(argv[1], "info")) {
        if (argc != 3) {
            usage(argv[0]);
            return EXIT_FAILURE;
        }

        return command_info(argv[2]) ?
            EXIT_FAILURE :
            EXIT_SUCCESS;
    }

    usage(argv[0]);
    return EXIT_FAILURE;
}
