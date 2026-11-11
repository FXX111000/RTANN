// SPDX-License-Identifier: GPL-2.0

#include <linux/crc32.h>
#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>

#include "../../include/vecfs.h"
#include "../../include/vecfs_ann_meta.h"
#include "../../include/vecfs_ann_store.h"
#include "internal.h"

static loff_t vecfs_ann_object_position(
    struct vecfs_ann_store *store,
    u32 node_id)
{
    return (loff_t)((u64)node_id *
                    store->layout.stride);
}

u32 vecfs_ann_vector_crc(
    struct vecfs_ann_store *store,
    const void *vector)
{
    return crc32_le(
               ~0U,
               vector,
               store->layout.vector_bytes) ^ ~0U;
}

u32 vecfs_ann_object_crc(
    struct vecfs_ann_store *store,
    void *object)
{
    struct vecfs_ann_object_disk *disk = object;
    __le32 saved;
    u32 crc;

    saved = disk->object_crc;
    disk->object_crc = 0;

    crc = crc32_le(
              ~0U,
              object,
              store->layout.stride) ^ ~0U;

    disk->object_crc = saved;
    return crc;
}

static void vecfs_ann_initialize_links(
    struct vecfs_ann_store *store,
    void *object)
{
    __le32 *level0;
    __le16 *counts;
    __le32 *upper;
    u64 upper_slots;
    unsigned int i;

    level0 = object + store->layout.level0_offset;
    for (i = 0; i < store->layout.m0; i++)
        level0[i] =
            cpu_to_le32(VECFS_ANN_INVALID_NODE_ID);

    counts = object +
             store->layout.upper_counts_offset;

    for (i = 0;
         i < store->layout.max_levels - 1;
         i++)
        counts[i] = 0;

    upper = object +
            store->layout.upper_links_offset;

    upper_slots =
        (u64)(store->layout.max_levels - 1) *
        store->layout.m;

    for (i = 0; i < upper_slots; i++)
        upper[i] =
            cpu_to_le32(VECFS_ANN_INVALID_NODE_ID);
}

bool vecfs_ann_object_valid(
    struct vecfs_ann_store *store,
    u32 expected_id,
    const void *object)
{
    struct vecfs_ann_object_disk *disk;
    void *mutable_object = (void *)object;
    const void *vector;
    u32 state;
    u32 stored_crc;
    u32 calculated_crc;
    u32 stored_vector_crc;
    u16 level;
    u16 owner_type;

    if (!store || !object)
        return false;

    disk = mutable_object;

    if (le32_to_cpu(disk->magic) !=
        VECFS_ANN_OBJECT_MAGIC)
        return false;

    if (le16_to_cpu(disk->version) !=
        VECFS_ANN_OBJECT_VERSION)
        return false;

    if (le16_to_cpu(disk->header_size) !=
        sizeof(*disk))
        return false;

    if (le32_to_cpu(disk->node_id) !=
        expected_id)
        return false;

    state = le32_to_cpu(disk->state);

    if (state != VECFS_ANN_OBJ_FREE &&
        state != VECFS_ANN_OBJ_PREPARED &&
        state != VECFS_ANN_OBJ_LIVE &&
        state != VECFS_ANN_OBJ_DELETED &&
        state != VECFS_ANN_OBJ_CORRUPT)
        return false;

    stored_crc = le32_to_cpu(
        disk->object_crc);

    calculated_crc =
        vecfs_ann_object_crc(
            store,
            mutable_object);

    if (stored_crc != calculated_crc)
        return false;

    /*
     * FREE object 只用于保留稳定的 node ID 槽位。
     *
     * 它不包含有效向量、owner、external ID 和邻居。
     */
    if (state == VECFS_ANN_OBJ_FREE)
        return true;

    level = le16_to_cpu(disk->level);

    if (level >= store->layout.max_levels)
        return false;

    if (le16_to_cpu(disk->level0_count) >
        store->layout.m0)
        return false;

    owner_type =
        le16_to_cpu(disk->owner_type);

    if (owner_type !=
            VECFS_ANN_OWNER_STANDALONE &&
        owner_type !=
            VECFS_ANN_OWNER_FILE &&
        owner_type !=
            VECFS_ANN_OWNER_IMPORTED)
        return false;

    if (owner_type ==
            VECFS_ANN_OWNER_FILE &&
        !le64_to_cpu(disk->owner_ino))
        return false;

    vector = (const u8 *)object +
             store->layout.vector_offset;

    stored_vector_crc =
        le32_to_cpu(disk->vector_crc);

    if (stored_vector_crc !=
        vecfs_ann_vector_crc(
            store,
            vector))
        return false;

    return true;
}


int vecfs_ann_object_load_locked(
    struct vecfs_ann_store *store,
    u32 node_id,
    void *object)
{
    loff_t position;
    int ret;

    position = vecfs_ann_object_position(
        store, node_id);

    ret = vecfs_ann_arena_read_locked(
        store,
        position,
        object,
        store->layout.stride);
    if (ret)
        return ret;

    if (!vecfs_ann_object_valid(
            store, node_id, object))
        return -EBADMSG;

    return 0;
}

int vecfs_ann_object_store_locked(
    struct vecfs_ann_store *store,
    u32 node_id,
    void *object,
    bool sync)
{
    struct vecfs_ann_object_disk *disk = object;
    void *vector;
    loff_t position;

    vector = (u8 *)object +
             store->layout.vector_offset;

    disk->vector_crc = cpu_to_le32(
        vecfs_ann_vector_crc(store, vector));

    disk->object_crc = 0;
    disk->object_crc = cpu_to_le32(
        vecfs_ann_object_crc(store, object));

    position = vecfs_ann_object_position(
        store, node_id);

    return vecfs_ann_arena_write_locked(
        store,
        position,
        object,
        store->layout.stride,
        sync);
}

void vecfs_ann_object_to_info(
    const struct vecfs_ann_object_disk *disk,
    struct vecfs_ann_object_info *info)
{
    memset(info, 0, sizeof(*info));

    info->node_id =
        le32_to_cpu(disk->node_id);
    info->state =
        le32_to_cpu(disk->state);

    info->external_id =
        le64_to_cpu(disk->external_id);

    info->owner_ino =
        le64_to_cpu(disk->owner_ino);
    info->owner_generation =
        le32_to_cpu(disk->owner_generation);
    info->vector_ordinal =
        le32_to_cpu(disk->vector_ordinal);

    info->content_offset =
        le64_to_cpu(disk->content_offset);
    info->content_length =
        le64_to_cpu(disk->content_length);

    info->level =
        le16_to_cpu(disk->level);
    info->level0_count =
        le16_to_cpu(disk->level0_count);
    info->owner_type =
        le16_to_cpu(disk->owner_type);
    info->flags =
        le16_to_cpu(disk->flags);

    info->object_epoch =
        le64_to_cpu(disk->object_epoch);
}

int vecfs_ann_object_read(
    struct super_block *sb,
    u32 node_id,
    void *object,
    size_t object_size)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ann_store *store;
    int ret;

    if (!sbi || !object)
        return -EINVAL;

    store = sbi->s_ann_store;
    if (!store)
        return -ENODEV;

    if (object_size < store->layout.stride)
        return -ENOSPC;

    down_read(&store->io_lock);

    ret = vecfs_ann_object_load_locked(
        store, node_id, object);

    up_read(&store->io_lock);
    return ret;
}

int vecfs_ann_object_read_info(
    struct super_block *sb,
    u32 node_id,
    struct vecfs_ann_object_info *info)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ann_store *store;
    void *object;
    int ret;

    if (!sbi || !info)
        return -EINVAL;

    store = sbi->s_ann_store;
    if (!store)
        return -ENODEV;

    object = kvmalloc(
        store->layout.stride, GFP_KERNEL);
    if (!object)
        return -ENOMEM;

    ret = vecfs_ann_object_read(
        sb,
        node_id,
        object,
        store->layout.stride);
    if (!ret)
        vecfs_ann_object_to_info(object, info);

    kvfree(object);
    return ret;
}

int vecfs_ann_object_prepare(
    struct super_block *sb,
    const struct vecfs_ann_object_create *create,
    u32 *node_id)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ann_store *store;
    struct vecfs_ann_object_disk *disk;
    void *object;
    u64 external_id;
    u64 next_id;
    int ret;

    if (!sbi || !create || !node_id)
        return -EINVAL;

    store = sbi->s_ann_store;
    if (!store)
        return -ENODEV;

    if (!create->vector ||
        create->vector_bytes !=
            store->layout.vector_bytes)
        return -EINVAL;

    if (create->level >=
        store->layout.max_levels)
        return -EINVAL;

    if (create->owner_type !=
            VECFS_ANN_OWNER_STANDALONE &&
        create->owner_type !=
            VECFS_ANN_OWNER_FILE &&
        create->owner_type !=
            VECFS_ANN_OWNER_IMPORTED)
        return -EINVAL;

    if (create->owner_type ==
            VECFS_ANN_OWNER_FILE &&
        !create->owner_ino)
        return -EINVAL;

    if ((create->flags &
         VECFS_ANN_OBJ_F_HAS_CONTENT_RANGE) &&
        create->content_offset >
            U64_MAX - create->content_length)
        return -EOVERFLOW;

    object = kvzalloc(
        store->layout.stride, GFP_KERNEL);
    if (!object)
        return -ENOMEM;

    mutex_lock(&store->allocation_lock);

    next_id = sbi->s_ann.next_node_id;

    if (next_id < VECFS_ANN_FIRST_NODE_ID)
        next_id = VECFS_ANN_FIRST_NODE_ID;

    if (next_id >= VECFS_ANN_INVALID_NODE_ID) {
        ret = -ENOSPC;
        goto out_unlock;
    }

    

    external_id = create->external_id;

    if (external_id ==
        VECFS_ANN_EXTERNAL_ID_AUTO) {
        external_id =
            sbi->s_ann.next_external_id++;

        if (external_id ==
            VECFS_ANN_EXTERNAL_ID_AUTO) {
            ret = -ENOSPC;
            goto out_unlock;
        }
    }

    disk = object;

    disk->magic =
        cpu_to_le32(VECFS_ANN_OBJECT_MAGIC);
    disk->version =
        cpu_to_le16(VECFS_ANN_OBJECT_VERSION);
    disk->header_size =
        cpu_to_le16(sizeof(*disk));

    disk->node_id = cpu_to_le32(next_id);
    disk->state =
        cpu_to_le32(VECFS_ANN_OBJ_PREPARED);

    disk->external_id =
        cpu_to_le64(external_id);

    disk->owner_ino =
        cpu_to_le64(create->owner_ino);
    disk->owner_generation =
        cpu_to_le32(create->owner_generation);
    disk->vector_ordinal =
        cpu_to_le32(create->vector_ordinal);

    disk->content_offset =
        cpu_to_le64(create->content_offset);
    disk->content_length =
        cpu_to_le64(create->content_length);

    disk->level =
        cpu_to_le16(create->level);
    disk->level0_count = 0;
    disk->owner_type =
        cpu_to_le16(create->owner_type);
    disk->flags =
        cpu_to_le16(create->flags);

    disk->object_epoch =
        cpu_to_le64(sbi->s_ann.epoch);

    vecfs_ann_initialize_links(store, object);

    memcpy(
        (u8 *)object +
            store->layout.vector_offset,
        create->vector,
        store->layout.vector_bytes);

    down_write(&store->io_lock);

    ret = vecfs_ann_object_store_locked(
        store,
        next_id,
        object,
        false);

    up_write(&store->io_lock);

    if (ret)
        goto out_unlock;

    sbi->s_ann.next_node_id = next_id + 1;
    sbi->s_ann.dirty = true;

    *node_id = next_id;

out_unlock:
    mutex_unlock(&store->allocation_lock);
    kvfree(object);
    return ret;
}

static int vecfs_ann_object_set_state(
    struct super_block *sb,
    u32 node_id,
    u32 new_state,
    bool sync)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ann_store *store;
    struct vecfs_ann_object_disk *disk;
    void *object;
    u32 old_state;
    int ret;

    if (!sbi)
        return -EINVAL;

    store = sbi->s_ann_store;
    if (!store)
        return -ENODEV;

    object = kvmalloc(
        store->layout.stride,
        GFP_NOFS);
    if (!object)
        return -ENOMEM;

    down_write(&store->io_lock);

    ret = vecfs_ann_object_load_locked(
        store,
        node_id,
        object);
    if (ret)
        goto out;

    disk = object;
    old_state = le32_to_cpu(disk->state);

    if (old_state == new_state) {
        ret = -EALREADY;
        goto out;
    }

    switch (new_state) {
    case VECFS_ANN_OBJ_LIVE:
        if (old_state != VECFS_ANN_OBJ_PREPARED &&
            old_state != VECFS_ANN_OBJ_DELETED) {
            ret = -EINVAL;
            goto out;
        }
        break;

    case VECFS_ANN_OBJ_DELETED:
        if (old_state != VECFS_ANN_OBJ_LIVE &&
            old_state != VECFS_ANN_OBJ_PREPARED) {
            ret = -EINVAL;
            goto out;
        }
        break;

    default:
        ret = -EINVAL;
        goto out;
    }

    disk->state = cpu_to_le32(new_state);

    ret = vecfs_ann_object_store_locked(
        store,
        node_id,
        object,
        sync);
    if (ret)
        goto out;

    mutex_lock(&sbi->s_ann.lock);

    if (old_state == VECFS_ANN_OBJ_LIVE &&
        new_state == VECFS_ANN_OBJ_DELETED) {
        if (sbi->s_ann.live_nodes)
            sbi->s_ann.live_nodes--;

        sbi->s_ann.deleted_nodes++;
    } else if (old_state ==
                   VECFS_ANN_OBJ_DELETED &&
               new_state ==
                   VECFS_ANN_OBJ_LIVE) {
        if (sbi->s_ann.deleted_nodes)
            sbi->s_ann.deleted_nodes--;

        sbi->s_ann.live_nodes++;
    } else if (old_state ==
                   VECFS_ANN_OBJ_PREPARED &&
               new_state ==
                   VECFS_ANN_OBJ_LIVE) {
        sbi->s_ann.live_nodes++;
    } else if (old_state ==
                   VECFS_ANN_OBJ_PREPARED &&
               new_state ==
                   VECFS_ANN_OBJ_DELETED) {
        sbi->s_ann.deleted_nodes++;
    }

    sbi->s_ann.dirty = true;

    mutex_unlock(&sbi->s_ann.lock);

out:
    up_write(&store->io_lock);
    kvfree(object);

    return ret;
}

int vecfs_ann_object_publish(
    struct super_block *sb,
    u32 node_id,
    bool sync)
{
    return vecfs_ann_object_set_state(
        sb,
        node_id,
        VECFS_ANN_OBJ_LIVE,
        sync);
}

int vecfs_ann_object_mark_deleted(
    struct super_block *sb,
    u32 node_id)
{
    return vecfs_ann_object_set_state(
        sb,
        node_id,
        VECFS_ANN_OBJ_DELETED,
        true);
}

int vecfs_ann_object_unmark_deleted(
    struct super_block *sb,
    u32 node_id)
{
    return vecfs_ann_object_set_state(
        sb,
        node_id,
        VECFS_ANN_OBJ_LIVE,
        true);
}


int vecfs_ann_object_read_vector(
    struct super_block *sb,
    u32 node_id,
    void *vector,
    size_t vector_size)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ann_store *store;
    void *object;
    int ret;

    if (!sbi || !vector)
        return -EINVAL;

    store = sbi->s_ann_store;
    if (!store)
        return -ENODEV;

    if (vector_size < store->layout.vector_bytes)
        return -ENOSPC;

    object = kvmalloc(
        store->layout.stride, GFP_KERNEL);
    if (!object)
        return -ENOMEM;

    ret = vecfs_ann_object_read(
        sb, node_id, object,
        store->layout.stride);
    if (!ret)
        memcpy(
            vector,
            (u8 *)object +
                store->layout.vector_offset,
            store->layout.vector_bytes);

    kvfree(object);
    return ret;
}

int vecfs_ann_object_write_vector(
    struct super_block *sb,
    u32 node_id,
    const void *vector,
    size_t vector_size)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ann_store *store;
    struct vecfs_ann_object_disk *disk;
    void *object;
    int ret;

    if (!sbi || !vector)
        return -EINVAL;

    store = sbi->s_ann_store;
    if (!store)
        return -ENODEV;

    if (vector_size != store->layout.vector_bytes)
        return -EINVAL;

    object = kvmalloc(
        store->layout.stride, GFP_KERNEL);
    if (!object)
        return -ENOMEM;

    down_write(&store->io_lock);

    ret = vecfs_ann_object_load_locked(
        store, node_id, object);
    if (ret)
        goto out;

    disk = object;

    if (le32_to_cpu(disk->state) !=
        VECFS_ANN_OBJ_LIVE) {
        ret = -ENOENT;
        goto out;
    }

    memcpy(
        (u8 *)object +
            store->layout.vector_offset,
        vector,
        store->layout.vector_bytes);

    ret = vecfs_ann_object_store_locked(
        store, node_id, object, false);

out:
    up_write(&store->io_lock);
    kvfree(object);
    return ret;
}

static int vecfs_ann_validate_neighbors(
    const u32 *neighbors,
    u16 count,
    u32 self)
{
    u16 i;
    u16 j;

    for (i = 0; i < count; i++) {
        if (neighbors[i] ==
                VECFS_ANN_INVALID_NODE_ID) {
            pr_err(
                "vecfs: invalid ANN neighbor "
                "self=%u index=%u invalid-id\n",
                self, i);
            return -EUCLEAN;
        }

        if (neighbors[i] == self) {
            pr_err(
                "vecfs: ANN self-loop "
                "node=%u index=%u count=%u\n",
                self, i, count);
            return -EUCLEAN;
        }

        for (j = 0; j < i; j++) {
            if (neighbors[j] ==
                neighbors[i]) {
                pr_err(
                    "vecfs: duplicate ANN neighbor "
                    "self=%u neighbor=%u "
                    "indexes=%u,%u count=%u\n",
                    self,
                    neighbors[i],
                    j, i, count);
                return -EUCLEAN;
            }
        }
    }

    return 0;
}


int vecfs_ann_object_read_level0(
    struct super_block *sb,
    u32 node_id,
    u32 *neighbors,
    u16 capacity,
    u16 *count)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ann_store *store;
    struct vecfs_ann_object_disk *disk;
    __le32 *links;
    void *object;
    u16 stored_count;
    u16 i;
    int ret;

    if (!sbi || !neighbors || !count)
        return -EINVAL;

    store = sbi->s_ann_store;
    if (!store)
        return -ENODEV;

    object = kvmalloc(
        store->layout.stride, GFP_KERNEL);
    if (!object)
        return -ENOMEM;

    ret = vecfs_ann_object_read(
        sb, node_id, object,
        store->layout.stride);
    if (ret)
        goto out;

    disk = object;
    stored_count =
        le16_to_cpu(disk->level0_count);

    if (stored_count > capacity) {
        ret = -ENOSPC;
        goto out;
    }

    links = object + store->layout.level0_offset;

    for (i = 0; i < stored_count; i++)
        neighbors[i] = le32_to_cpu(links[i]);

    *count = stored_count;

out:
    kvfree(object);
    return ret;
}

int vecfs_ann_object_write_level0(
    struct super_block *sb,
    u32 node_id,
    const u32 *neighbors,
    u16 count)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ann_store *store;
    struct vecfs_ann_object_disk *disk;
    __le32 *links;
    void *object;
    u16 i;
    int ret;

    if (!sbi || (!neighbors && count))
        return -EINVAL;

    store = sbi->s_ann_store;
    if (!store)
        return -ENODEV;

    if (count > store->layout.m0)
        return -EINVAL;

    ret = vecfs_ann_validate_neighbors(
        neighbors, count, node_id);
    if (ret)
        return ret;

    object = kvmalloc(
        store->layout.stride, GFP_KERNEL);
    if (!object)
        return -ENOMEM;

    down_write(&store->io_lock);

    ret = vecfs_ann_object_load_locked(
        store, node_id, object);
    if (ret)
        goto out;

    disk = object;
    links = object + store->layout.level0_offset;

    for (i = 0; i < store->layout.m0; i++)
        links[i] = cpu_to_le32(
            VECFS_ANN_INVALID_NODE_ID);

    for (i = 0; i < count; i++)
        links[i] = cpu_to_le32(neighbors[i]);

    disk->level0_count = cpu_to_le16(count);

    ret = vecfs_ann_object_store_locked(
        store, node_id, object, false);

out:
    up_write(&store->io_lock);
    kvfree(object);
    return ret;
}

static __le16 *vecfs_ann_upper_counts(
    struct vecfs_ann_store *store,
    void *object)
{
    return object +
           store->layout.upper_counts_offset;
}

static __le32 *vecfs_ann_store_upper_links(
    struct vecfs_ann_store *store,
    void *object,
    u16 level)
{
    u64 index = (u64)(level - 1) *
                store->layout.m;

    return object +
           store->layout.upper_links_offset +
           index * sizeof(__le32);
}

int vecfs_ann_object_read_upper(
    struct super_block *sb,
    u32 node_id,
    u16 level,
    u32 *neighbors,
    u16 capacity,
    u16 *count)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ann_store *store;
    struct vecfs_ann_object_disk *disk;
    __le16 *counts;
    __le32 *links;
    void *object;
    u16 stored_count;
    u16 i;
    int ret;

    if (!sbi || !neighbors || !count)
        return -EINVAL;

    store = sbi->s_ann_store;
    if (!store)
        return -ENODEV;

    if (!level ||
        level >= store->layout.max_levels)
        return -EINVAL;

    object = kvmalloc(
        store->layout.stride, GFP_KERNEL);
    if (!object)
        return -ENOMEM;

    ret = vecfs_ann_object_read(
        sb, node_id, object,
        store->layout.stride);
    if (ret)
        goto out;

    disk = object;

    if (level > le16_to_cpu(disk->level)) {
        ret = -ENOENT;
        goto out;
    }

    counts = vecfs_ann_upper_counts(store, object);
    stored_count = le16_to_cpu(counts[level - 1]);

    if (stored_count > capacity) {
        ret = -ENOSPC;
        goto out;
    }

    links = vecfs_ann_store_upper_links(
        store, object, level);

    for (i = 0; i < stored_count; i++)
        neighbors[i] = le32_to_cpu(links[i]);

    *count = stored_count;

out:
    kvfree(object);
    return ret;
}

int vecfs_ann_object_write_upper(
    struct super_block *sb,
    u32 node_id,
    u16 level,
    const u32 *neighbors,
    u16 count)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ann_store *store;
    struct vecfs_ann_object_disk *disk;
    __le16 *counts;
    __le32 *links;
    void *object;
    u16 i;
    int ret;

    if (!sbi || (!neighbors && count))
        return -EINVAL;

    store = sbi->s_ann_store;
    if (!store)
        return -ENODEV;

    if (!level ||
        level >= store->layout.max_levels ||
        count > store->layout.m)
        return -EINVAL;

    ret = vecfs_ann_validate_neighbors(
        neighbors, count, node_id);
    if (ret)
        return ret;

    object = kvmalloc(
        store->layout.stride, GFP_KERNEL);
    if (!object)
        return -ENOMEM;

    down_write(&store->io_lock);

    ret = vecfs_ann_object_load_locked(
        store, node_id, object);
    if (ret)
        goto out;

    disk = object;

    if (level > le16_to_cpu(disk->level)) {
        ret = -EINVAL;
        goto out;
    }

    counts = vecfs_ann_upper_counts(store, object);
    links = vecfs_ann_store_upper_links(
        store, object, level);

    for (i = 0; i < store->layout.m; i++)
        links[i] = cpu_to_le32(
            VECFS_ANN_INVALID_NODE_ID);

    for (i = 0; i < count; i++)
        links[i] = cpu_to_le32(neighbors[i]);

    counts[level - 1] = cpu_to_le16(count);

    ret = vecfs_ann_object_store_locked(
        store, node_id, object, false);

out:
    up_write(&store->io_lock);
    kvfree(object);
    return ret;
}

int vecfs_ann_object_update_flags(
    struct super_block *sb,
    u32 node_id,
    u16 clear_mask,
    u16 set_mask)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_ann_store *store;
    struct vecfs_ann_object_disk *disk;
    void *object;
    u32 state;
    u16 old_flags;
    u16 new_flags;
    int ret;

    if (!sb)
        return -EINVAL;

    sbi = VECFS_SB(sb);
    if (!sbi)
        return -EINVAL;

    store = sbi->s_ann_store;
    if (!store)
        return -ENODEV;

    /*
     * 当前只允许更新 object flag 区域中已定义的标志。
     */
    if ((clear_mask | set_mask) &
        ~(VECFS_ANN_OBJ_F_HAS_CONTENT_RANGE |
          VECFS_ANN_OBJ_F_CONTENT_STALE |
          VECFS_ANN_OBJ_F_EXTERNAL_ID_AUTO))
        return -EINVAL;

    object = kvmalloc(
        store->layout.stride,
        GFP_NOFS);
    if (!object)
        return -ENOMEM;

    down_write(&store->io_lock);

    ret = vecfs_ann_object_load_locked(
        store,
        node_id,
        object);
    if (ret)
        goto out_unlock;

    disk = object;
    state = le32_to_cpu(disk->state);

    if (state != VECFS_ANN_OBJ_LIVE &&
        state != VECFS_ANN_OBJ_DELETED) {
        ret = -ENOENT;
        goto out_unlock;
    }

    old_flags = le16_to_cpu(disk->flags);

    new_flags =
        (old_flags & ~clear_mask) |
        set_mask;

    if (new_flags == old_flags) {
        ret = 0;
        goto out_unlock;
    }

    disk->flags =
        cpu_to_le16(new_flags);

    ret = vecfs_ann_object_store_locked(
        store,
        node_id,
        object,
        true);

out_unlock:
    up_write(&store->io_lock);
    kvfree(object);

    return ret;
}

int vecfs_ann_object_update_binding(
    struct super_block *sb,
    u32 node_id,
    u64 content_offset,
    u64 content_length,
    u16 flags)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_ann_store *store;
    struct vecfs_ann_object_disk *disk;
    void *object;
    u32 state;
    int ret;

    if (!sb)
        return -EINVAL;

    if ((flags &
         ~(VECFS_ANN_OBJ_F_HAS_CONTENT_RANGE |
           VECFS_ANN_OBJ_F_CONTENT_STALE |
           VECFS_ANN_OBJ_F_EXTERNAL_ID_AUTO)))
        return -EINVAL;

    if ((flags &
         VECFS_ANN_OBJ_F_HAS_CONTENT_RANGE) &&
        content_offset >
            U64_MAX - content_length)
        return -EOVERFLOW;

    sbi = VECFS_SB(sb);
    if (!sbi)
        return -EINVAL;

    store = sbi->s_ann_store;
    if (!store)
        return -ENODEV;

    object = kvmalloc(
        store->layout.stride,
        GFP_NOFS);
    if (!object)
        return -ENOMEM;

    down_write(&store->io_lock);

    ret = vecfs_ann_object_load_locked(
        store,
        node_id,
        object);
    if (ret)
        goto out_unlock;

    disk = object;
    state = le32_to_cpu(disk->state);

    if (state != VECFS_ANN_OBJ_LIVE &&
        state != VECFS_ANN_OBJ_DELETED) {
        ret = -ENOENT;
        goto out_unlock;
    }

    disk->content_offset =
        cpu_to_le64(content_offset);

    disk->content_length =
        cpu_to_le64(content_length);

    disk->flags =
        cpu_to_le16(flags);

    ret = vecfs_ann_object_store_locked(
        store,
        node_id,
        object,
        true);

out_unlock:
    up_write(&store->io_lock);
    kvfree(object);

    return ret;
}

int vecfs_ann_object_read_raw(
    struct super_block *sb,
    u32 node_id,
    void *object,
    size_t object_size)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_ann_store *store;
    int ret;

    if (!sb || !object)
        return -EINVAL;

    sbi = VECFS_SB(sb);
    if (!sbi || !sbi->s_ann_store)
        return -ENODEV;

    store = sbi->s_ann_store;

    if (object_size < store->layout.stride)
        return -ENOSPC;

    down_read(&store->io_lock);

    ret = vecfs_ann_arena_read_locked(
        store,
        (loff_t)((u64)node_id *
                 store->layout.stride),
        object,
        store->layout.stride);

    up_read(&store->io_lock);

    return ret;
}

int vecfs_ann_object_restore_raw(
    struct super_block *sb,
    u32 node_id,
    const void *object,
    size_t object_size)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_ann_store *store;
    void *copy;
    int ret;

    if (!sb || !object)
        return -EINVAL;

    sbi = VECFS_SB(sb);
    if (!sbi || !sbi->s_ann_store)
        return -ENODEV;

    store = sbi->s_ann_store;

    if (object_size < store->layout.stride)
        return -ENOSPC;

    copy = kvmemdup(
        object,
        store->layout.stride,
        GFP_NOFS);
    if (!copy)
        return -ENOMEM;

    if (!vecfs_ann_object_valid(
            store,
            node_id,
            copy)) {
        kvfree(copy);
        return -EBADMSG;
    }

    down_write(&store->io_lock);

    ret = vecfs_ann_arena_write_locked(
        store,
        (loff_t)((u64)node_id *
                 store->layout.stride),
        copy,
        store->layout.stride,
        true);

    up_write(&store->io_lock);

    kvfree(copy);
    return ret;
}
