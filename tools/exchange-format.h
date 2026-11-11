#ifndef _VECFS_EXCHANGE_FORMAT_H
#define _VECFS_EXCHANGE_FORMAT_H

#include <stdint.h>

#define VECFS_EXCHANGE_FILE_MAGIC \
    0x5645434653455843ULL /* "VECFSEXC" */

#define VECFS_EXCHANGE_FILE_MAJOR 1
#define VECFS_EXCHANGE_FILE_MINOR 0

/* flags for struct vecfs_exchange_node_header */
#define VECFS_EXCHANGE_NODE_F_NONE         0

/* maximum bytes for a single node record */
#define VECFS_EXCHANGE_MAX_RECORD_BYTES    (16ULL * 1024 * 1024)

#pragma pack(push, 1)

struct vecfs_exchange_file_header {
    uint64_t magic;

    uint16_t major;
    uint16_t minor;

    uint16_t element_type;
    uint16_t distance_type;

    uint32_t dimension;

    uint16_t m;
    uint16_t m0;
    uint16_t max_levels;
    uint16_t current_max_level;

    uint32_t ef_construction;
    uint32_t default_ef_search;

    uint64_t node_count;
    uint32_t node_capacity;
    uint32_t entry_node_id;

    uint64_t random_seed;

    uint64_t node_records_offset;
    uint64_t node_records_bytes;

    uint32_t header_crc;
    uint32_t records_crc;

    uint8_t reserved[64];
};

struct vecfs_exchange_node_header {
    uint32_t node_id;
    uint16_t level;
    uint16_t flags;

    uint64_t external_id;

    uint32_t vector_bytes;
    uint16_t level0_count;
    uint16_t upper_level_count;

    /*
     * 本 node record 的总字节数，包括 header。
     */
    uint64_t record_bytes;

    uint32_t record_crc;
    uint32_t reserved32;
};

/*
 * node record payload：
 *
 * [node header]
 * [vector_bytes]
 * [level0_count * uint32_t]
 * [level * uint16_t upper counts]
 * [level * M * uint32_t upper links]
 */
#pragma pack(pop)

#endif
