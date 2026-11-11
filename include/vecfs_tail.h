#ifndef _VECFS_TAIL_H
#define _VECFS_TAIL_H

#include <linux/fs.h>
#include <linux/types.h>

#define VECFS_TAIL_V1_MAGIC        "VECFSVEC"
#define VECFS_TAIL_V1_MAGIC_LEN    8

#define VECFS_TAIL_V2_MAGIC        "VECFSV2\0"
#define VECFS_TAIL_V2_MAGIC_LEN    8

#define VECFS_TAIL_V1_VERSION      1
#define VECFS_TAIL_V2_MAJOR        2
#define VECFS_TAIL_V2_MINOR        0

#define VECFS_TAIL_MAX_VECTORS     4096U
#define VECFS_TAIL_MAX_BYTES       (64U * 1024U * 1024U)

#define VECFS_TAIL_DESC_F_HAS_RANGE      (1U << 0)
#define VECFS_TAIL_DESC_F_CONTENT_STALE  (1U << 1)

struct vecfs_tail_header_v1 {
    __u8 magic[VECFS_TAIL_V1_MAGIC_LEN];

    __le32 version;
    __le32 vector_dimension;

    __le16 element_size;
    __le16 reserved16;

    __le32 vector_count;

    __le64 payload_size;
    __le64 content_size;

    __le32 crc32;
    __le32 reserved32;
} __packed;

struct vecfs_tail_footer_v2 {
    __u8 magic[VECFS_TAIL_V2_MAGIC_LEN];

    __le16 major;
    __le16 minor;

    __le32 flags;

    __le64 visible_size;
    __le64 payload_offset;
    __le64 descriptor_offset;

    __le32 vector_count;
    __le32 descriptor_size;

    __le32 dimension;
    __le16 element_type;
    __le16 distance_type;

    __le64 transaction_id;

    __le32 payload_crc;
    __le32 descriptor_crc;
    __le32 footer_crc;
    __le32 reserved32;
} __packed;

struct vecfs_tail_vector_desc_v2 {
    __le64 external_id;

    __le32 ordinal;
    __le32 flags;

    __le64 vector_offset;
    __le64 vector_size;

    __le64 content_offset;
    __le64 content_length;
} __packed;

int vecfs_process_embedded_vectors(
    struct file *file);

#endif
