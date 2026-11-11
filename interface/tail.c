// SPDX-License-Identifier: GPL-2.0

#include <linux/crc32.h>
#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/pagemap.h>
#include <linux/slab.h>
#include <linux/uio.h>
#include <linux/vmalloc.h>

#include "../include/vecfs.h"
#include "../include/vecfs_ann_batch.h"
#include "../include/vecfs_ann_indirect.h"
#include "../include/vecfs_ann_mutation.h"
#include "../include/vecfs_tail.h"

struct vecfs_tail_transaction {
    struct vecfs_ann_batch_item *items;
    void **vectors;

    u32 count;
    loff_t visible_size;
};

static u32 vecfs_tail_crc32(
    const void *buffer,
    size_t length)
{
    return crc32_le(
               ~0U,
               buffer,
               length) ^ ~0U;
}

static int vecfs_tail_read_exact(
    struct file *file,
    void *buffer,
    size_t length,
    loff_t position)
{
    size_t copied = 0;

    while (copied < length) {
        loff_t pos =
            position + copied;

        ssize_t ret = kernel_read(
            file,
            (u8 *)buffer + copied,
            length - copied,
            &pos);

        if (ret < 0)
            return ret;

        if (!ret)
            return -ENODATA;

        copied += ret;
    }

    return 0;
}

static int vecfs_tail_read_alloc(
    struct file *file,
    loff_t position,
    size_t length,
    void **result)
{
    void *buffer;
    int ret;

    if (!length || !result)
        return -EINVAL;

    if (length > VECFS_TAIL_MAX_BYTES)
        return -E2BIG;

    buffer = kvmalloc(
        length,
        GFP_NOFS);
    if (!buffer)
        return -ENOMEM;

    ret = vecfs_tail_read_exact(
        file,
        buffer,
        length,
        position);
    if (ret) {
        kvfree(buffer);
        return ret;
    }

    *result = buffer;
    return 0;
}

static void vecfs_tail_transaction_destroy(
    struct vecfs_tail_transaction *transaction)
{
    u32 i;

    if (!transaction)
        return;

    if (transaction->vectors) {
        for (i = 0;
             i < transaction->count;
             i++)
            kvfree(transaction->vectors[i]);
    }

    kvfree(transaction->vectors);
    kvfree(transaction->items);

    memset(transaction, 0, sizeof(*transaction));
}

static int vecfs_tail_transaction_init(
    struct vecfs_tail_transaction *transaction,
    u32 count)
{
    if (!transaction || !count)
        return -EINVAL;

    if (count > VECFS_TAIL_MAX_VECTORS)
        return -E2BIG;

    memset(transaction, 0, sizeof(*transaction));

    transaction->items = kvcalloc(
        count,
        sizeof(*transaction->items),
        GFP_NOFS);

    transaction->vectors = kvcalloc(
        count,
        sizeof(*transaction->vectors),
        GFP_NOFS);

    if (!transaction->items ||
        !transaction->vectors) {
        vecfs_tail_transaction_destroy(
            transaction);

        return -ENOMEM;
    }

    transaction->count = count;
    return 0;
}

static bool vecfs_tail_ordinal_reserved(
    const struct vecfs_tail_transaction *transaction,
    u32 count,
    u32 ordinal)
{
    u32 i;

    for (i = 0; i < count; i++) {
        if (transaction->items[i].
                request.vector_ordinal ==
            ordinal)
            return true;
    }

    return false;
}

static int vecfs_tail_allocate_ordinal(
    struct inode *inode,
    const struct vecfs_tail_transaction *transaction,
    u32 initialized,
    u32 *cursor,
    u32 *ordinal)
{
    u32 value = *cursor;

    while (value < VECFS_ANN_ORDINAL_AUTO) {
        u32 existing;
        int ret;

        if (vecfs_tail_ordinal_reserved(
                transaction,
                initialized,
                value)) {
            value++;
            continue;
        }

        ret = vecfs_ann_get_node(
            inode,
            value,
            &existing);

        if (ret == -ENOENT) {
            *ordinal = value;
            *cursor = value + 1;
            return 0;
        }

        if (ret)
            return ret;

        value++;
    }

    return -ENOSPC;
}

static void vecfs_tail_fill_file_request(
    struct inode *inode,
    struct vecfs_ann_batch_item *item,
    void *vector,
    u64 external_id,
    u32 ordinal,
    u32 flags,
    u64 content_offset,
    u64 content_length)
{
    memset(item, 0, sizeof(*item));

    item->request.vector = vector;
    item->request.vector_bytes =
        VECFS_SB(inode->i_sb)->
            s_ann.config.vector_bytes;

    item->request.external_id =
        external_id;

    item->request.owner_ino =
        inode->i_ino;
    item->request.owner_generation =
        inode->i_generation;
    item->request.vector_ordinal =
        ordinal;

    item->request.content_offset =
        content_offset;
    item->request.content_length =
        content_length;

    item->request.owner_type =
        VECFS_ANN_OWNER_FILE;
    item->request.flags = flags;
}

static int vecfs_tail_parse_v1(
    struct file *file,
    const struct vecfs_tail_header_v1 *header,
    loff_t file_size,
    struct vecfs_tail_transaction *transaction)
{
    struct inode *inode = file_inode(file);
    struct vecfs_sb_info *sbi =
        VECFS_SB(inode->i_sb);
    u64 payload_size;
    u64 content_size;
    u64 expected_payload;
    u32 vector_count;
    u32 vector_bytes;
    u32 cursor = 0;
    void *payload = NULL;
    u32 i;
    int ret;

    if (le32_to_cpu(header->version) !=
        VECFS_TAIL_V1_VERSION)
        return -EPROTONOSUPPORT;

    if (le32_to_cpu(
            header->vector_dimension) !=
        sbi->s_ann.config.dimension)
        return -EINVAL;

    if (le16_to_cpu(header->element_size) !=
        sizeof(float))
        return -EINVAL;

    vector_count =
        le32_to_cpu(header->vector_count);

    if (!vector_count ||
        vector_count > VECFS_TAIL_MAX_VECTORS)
        return -E2BIG;

    vector_bytes =
        sbi->s_ann.config.vector_bytes;

    if (check_mul_overflow(
            (u64)vector_count,
            (u64)vector_bytes,
            &expected_payload))
        return -EOVERFLOW;

    payload_size =
        le64_to_cpu(header->payload_size);

    content_size =
        le64_to_cpu(header->content_size);

    if (payload_size != expected_payload)
        return -EINVAL;

    if (payload_size >
        VECFS_TAIL_MAX_BYTES)
        return -E2BIG;

    if (content_size >
        (u64)file_size)
        return -EINVAL;

    if (content_size + payload_size +
            sizeof(*header) !=
        (u64)file_size)
        return -EINVAL;

    ret = vecfs_tail_transaction_init(
        transaction,
        vector_count);
    if (ret)
        return ret;

    transaction->visible_size =
        content_size;

    ret = vecfs_tail_read_alloc(
        file,
        content_size,
        payload_size,
        &payload);
    if (ret)
        goto fail;

    if (le32_to_cpu(header->crc32) &&
        vecfs_tail_crc32(
            payload,
            payload_size) !=
        le32_to_cpu(header->crc32)) {
        ret = -EBADMSG;
        goto fail;
    }

    for (i = 0; i < vector_count; i++) {
        u32 ordinal;

        transaction->vectors[i] =
            kvmemdup(
                (u8 *)payload +
                    (size_t)i * vector_bytes,
                vector_bytes,
                GFP_NOFS);

        if (!transaction->vectors[i]) {
            ret = -ENOMEM;
            goto fail;
        }

        ret = vecfs_tail_allocate_ordinal(
            inode,
            transaction,
            i,
            &cursor,
            &ordinal);
        if (ret)
            goto fail;

        vecfs_tail_fill_file_request(
            inode,
            &transaction->items[i],
            transaction->vectors[i],
            VECFS_ANN_EXTERNAL_ID_AUTO,
            ordinal,
            0,
            0,
            0);
    }

    kvfree(payload);
    return 0;

fail:
    kvfree(payload);
    vecfs_tail_transaction_destroy(
        transaction);

    return ret;
}

static int vecfs_tail_validate_footer_v2(
    struct super_block *sb,
    const struct vecfs_tail_footer_v2 *footer,
    loff_t file_size)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_tail_footer_v2 copy;
    u64 visible_size;
    u64 payload_offset;
    u64 descriptor_offset;
    u64 descriptor_bytes;
    u32 count;

    if (le16_to_cpu(footer->major) !=
            VECFS_TAIL_V2_MAJOR ||
        le16_to_cpu(footer->minor) >
            VECFS_TAIL_V2_MINOR)
        return -EPROTONOSUPPORT;

    if (le32_to_cpu(footer->dimension) !=
        sbi->s_ann.config.dimension)
        return -EINVAL;

    if (le16_to_cpu(footer->element_type) !=
        sbi->s_ann.config.element_type)
        return -EINVAL;

    if (le16_to_cpu(footer->distance_type) !=
        sbi->s_ann.config.distance_type)
        return -EINVAL;

    if (le32_to_cpu(footer->descriptor_size) !=
        sizeof(struct vecfs_tail_vector_desc_v2))
        return -EINVAL;

    count = le32_to_cpu(
        footer->vector_count);

    if (!count ||
        count > VECFS_TAIL_MAX_VECTORS)
        return -E2BIG;

    if (check_mul_overflow(
            (u64)count,
            (u64)sizeof(
                struct vecfs_tail_vector_desc_v2),
            &descriptor_bytes))
        return -EOVERFLOW;

    visible_size =
        le64_to_cpu(footer->visible_size);

    payload_offset =
        le64_to_cpu(footer->payload_offset);

    descriptor_offset =
        le64_to_cpu(footer->descriptor_offset);

    if (visible_size > payload_offset ||
        payload_offset > descriptor_offset)
        return -EINVAL;

    if (descriptor_offset + descriptor_bytes +
            sizeof(*footer) !=
        (u64)file_size)
        return -EINVAL;

    if (descriptor_offset - payload_offset >
        VECFS_TAIL_MAX_BYTES)
        return -E2BIG;

    copy = *footer;
    copy.footer_crc = 0;

    if (le32_to_cpu(footer->footer_crc) &&
        vecfs_tail_crc32(
            &copy,
            sizeof(copy)) !=
        le32_to_cpu(footer->footer_crc))
        return -EBADMSG;

    return 0;
}

static int vecfs_tail_parse_v2(
    struct file *file,
    const struct vecfs_tail_footer_v2 *footer,
    loff_t file_size,
    struct vecfs_tail_transaction *transaction)
{
    struct inode *inode = file_inode(file);
    struct vecfs_tail_vector_desc_v2 *descriptors = NULL;
    void *payload = NULL;
    u64 payload_offset;
    u64 descriptor_offset;
    u64 descriptor_bytes;
    u64 payload_bytes;
    u32 count;
    u32 cursor = 0;
    u32 i;
    int ret;

    ret = vecfs_tail_validate_footer_v2(
        inode->i_sb,
        footer,
        file_size);
    if (ret)
        return ret;

    count = le32_to_cpu(
        footer->vector_count);

    payload_offset =
        le64_to_cpu(footer->payload_offset);

    descriptor_offset =
        le64_to_cpu(footer->descriptor_offset);

    payload_bytes =
        descriptor_offset - payload_offset;

    descriptor_bytes =
        (u64)count *
        sizeof(*descriptors);

    ret = vecfs_tail_transaction_init(
        transaction,
        count);
    if (ret)
        return ret;

    transaction->visible_size =
        le64_to_cpu(footer->visible_size);

    ret = vecfs_tail_read_alloc(
        file,
        descriptor_offset,
        descriptor_bytes,
        (void **)&descriptors);
    if (ret)
        goto fail;

    if (le32_to_cpu(
            footer->descriptor_crc) &&
        vecfs_tail_crc32(
            descriptors,
            descriptor_bytes) !=
        le32_to_cpu(
            footer->descriptor_crc)) {
        ret = -EBADMSG;
        goto fail;
    }

    if (payload_bytes) {
        ret = vecfs_tail_read_alloc(
            file,
            payload_offset,
            payload_bytes,
            &payload);
        if (ret)
            goto fail;

        if (le32_to_cpu(
                footer->payload_crc) &&
            vecfs_tail_crc32(
                payload,
                payload_bytes) !=
            le32_to_cpu(
                footer->payload_crc)) {
            ret = -EBADMSG;
            goto fail;
        }
    }

    for (i = 0; i < count; i++) {
        struct vecfs_tail_vector_desc_v2 *desc =
            &descriptors[i];

        u64 vector_offset =
            le64_to_cpu(desc->vector_offset);

        u64 vector_size =
            le64_to_cpu(desc->vector_size);

        u64 content_offset =
            le64_to_cpu(desc->content_offset);

        u64 content_length =
            le64_to_cpu(desc->content_length);

        u64 external_id =
            le64_to_cpu(desc->external_id);

        u32 ordinal =
            le32_to_cpu(desc->ordinal);

        u32 descriptor_flags =
            le32_to_cpu(desc->flags);

        u16 object_flags = 0;

        if (vector_size !=
            VECFS_SB(inode->i_sb)->
                s_ann.config.vector_bytes) {
            ret = -EINVAL;
            goto fail;
        }

        if (vector_offset < payload_offset ||
            vector_offset >
                descriptor_offset - vector_size) {
            ret = -EINVAL;
            goto fail;
        }

        if (descriptor_flags &
            ~(VECFS_TAIL_DESC_F_HAS_RANGE |
              VECFS_TAIL_DESC_F_CONTENT_STALE)) {
            ret = -EINVAL;
            goto fail;
        }

        if ((descriptor_flags &
             VECFS_TAIL_DESC_F_HAS_RANGE) &&
            content_offset >
                U64_MAX - content_length) {
            ret = -EOVERFLOW;
            goto fail;
        }

        if (ordinal ==
            VECFS_ANN_ORDINAL_AUTO) {
            ret = vecfs_tail_allocate_ordinal(
                inode,
                transaction,
                i,
                &cursor,
                &ordinal);
            if (ret)
                goto fail;
        }

        if (vecfs_tail_ordinal_reserved(
                transaction,
                i,
                ordinal)) {
            ret = -EEXIST;
            goto fail;
        }

        transaction->vectors[i] =
            kvmemdup(
                (u8 *)payload +
                    (vector_offset -
                     payload_offset),
                vector_size,
                GFP_NOFS);

        if (!transaction->vectors[i]) {
            ret = -ENOMEM;
            goto fail;
        }

        if (descriptor_flags &
            VECFS_TAIL_DESC_F_HAS_RANGE)
            object_flags |=
                VECFS_ANN_OBJ_F_HAS_CONTENT_RANGE;

        if (descriptor_flags &
            VECFS_TAIL_DESC_F_CONTENT_STALE)
            object_flags |=
                VECFS_ANN_OBJ_F_CONTENT_STALE;

        vecfs_tail_fill_file_request(
            inode,
            &transaction->items[i],
            transaction->vectors[i],
            external_id,
            ordinal,
            object_flags,
            content_offset,
            content_length);
    }

    kvfree(payload);
    kvfree(descriptors);
    return 0;

fail:
    kvfree(payload);
    kvfree(descriptors);

    vecfs_tail_transaction_destroy(
        transaction);

    return ret;
}

static int vecfs_tail_detect_and_parse(
    struct file *file,
    struct vecfs_tail_transaction *transaction)
{
    loff_t file_size =
        i_size_read(file_inode(file));

    struct vecfs_tail_footer_v2 footer_v2;
    struct vecfs_tail_header_v1 header_v1;
    int ret;

    memset(transaction, 0, sizeof(*transaction));

    if (file_size >=
        sizeof(footer_v2)) {
        ret = vecfs_tail_read_exact(
            file,
            &footer_v2,
            sizeof(footer_v2),
            file_size -
                sizeof(footer_v2));
        if (ret)
            return ret;

        if (!memcmp(
                footer_v2.magic,
                VECFS_TAIL_V2_MAGIC,
                VECFS_TAIL_V2_MAGIC_LEN))
            return vecfs_tail_parse_v2(
                file,
                &footer_v2,
                file_size,
                transaction);
    }

    if (file_size >=
        sizeof(header_v1)) {
        ret = vecfs_tail_read_exact(
            file,
            &header_v1,
            sizeof(header_v1),
            file_size -
                sizeof(header_v1));
        if (ret)
            return ret;

        if (!memcmp(
                header_v1.magic,
                VECFS_TAIL_V1_MAGIC,
                VECFS_TAIL_V1_MAGIC_LEN))
            return vecfs_tail_parse_v1(
                file,
                &header_v1,
                file_size,
                transaction);
    }

    /*
     * 没有尾部协议不是错误。
     */
    return -ENODATA;
}

static int vecfs_tail_commit(
    struct file *file,
    struct vecfs_tail_transaction *transaction)
{
    struct inode *inode = file_inode(file);
    loff_t old_size = i_size_read(inode);
    int ret;

    ret = vecfs_ann_batch_insert_file(
        inode,
        transaction->items,
        transaction->count);
    if (ret)
        return ret;

    /*
     * ANN 已成功提交后移除尾部。truncate_setsize() 会先调整
     * page cache 和 inode size，随后 fsync 路径负责写回。
     */
    ret = inode_newsize_ok(
        inode,
        transaction->visible_size);
    if (ret)
        goto rollback_vectors;

    truncate_setsize(
        inode,
        transaction->visible_size);

    inode_set_mtime_to_ts(
        inode,
        inode_set_ctime_current(inode));

    mark_inode_dirty(inode);

    return 0;

rollback_vectors:
    /*
     * 极少数 truncate 失败场景下，将本事务节点 mark-delete 并清除
     * inode ordinal 映射。邻接表中指向这些 tombstone 的边仍可安全
     * 导航，之后 rebuild 会彻底移除。
     */
    {
        u32 i;

        for (i = 0;
             i < transaction->count;
             i++) {
            vecfs_ann_file_mark_delete(
                inode,
                transaction->items[i].
                    request.vector_ordinal);

            vecfs_ann_clear_node(
                inode,
                transaction->items[i].
                    request.vector_ordinal);
        }
    }

    i_size_write(
        inode,
        old_size);

    mark_inode_dirty(inode);
    return ret;
}

int vecfs_process_embedded_vectors(
    struct file *file)
{
    struct inode *inode;
    struct vecfs_tail_transaction transaction;
    int ret;

    if (!file)
        return -EINVAL;

    inode = file_inode(file);

    if (!S_ISREG(inode->i_mode))
        return 0;

    /*
     * 同一文件的 write/fsync/release 尾部提交串行化。
     */
    inode_lock(inode);

    ret = vecfs_tail_detect_and_parse(
        file,
        &transaction);

    if (ret == -ENODATA) {
        inode_unlock(inode);
        return 0;
    }

    if (ret) {
        inode_unlock(inode);
        return ret;
    }

    ret = vecfs_tail_commit(
        file,
        &transaction);

    vecfs_tail_transaction_destroy(
        &transaction);

    inode_unlock(inode);
    return ret;
}
