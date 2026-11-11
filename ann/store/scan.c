// SPDX-License-Identifier: GPL-2.0

#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/kernel.h>
#include <linux/sched.h>
#include <linux/vmalloc.h>

#include "../../include/vecfs.h"
#include "../../include/vecfs_ann_meta.h"
#include "../../include/vecfs_ann_store.h"
#include "internal.h"

static bool vecfs_ann_object_is_zero(
    const void *object,
    size_t size)
{
    const unsigned long *words = object;
    size_t word_count = size / sizeof(*words);
    size_t i;
    const u8 *tail;

    for (i = 0; i < word_count; i++) {
        if (words[i])
            return false;
    }

    tail = (const u8 *)(words + word_count);

    for (i = word_count * sizeof(*words);
         i < size;
         i++) {
        if (tail[i -
                 word_count * sizeof(*words)])
            return false;
    }

    return true;
}

static bool vecfs_ann_links_sane(
    struct vecfs_ann_store *store,
    u32 node_id,
    u64 slot_count,
    void *object)
{
    struct vecfs_ann_object_disk *disk = object;
    __le32 *level0;
    __le16 *counts;
    __le32 *upper;
    u16 level0_count;
    u16 node_level;
    u16 i;
    u16 level;

    node_level = le16_to_cpu(disk->level);
    level0_count =
        le16_to_cpu(disk->level0_count);

    if (level0_count > store->layout.m0)
        return false;

    level0 = object + store->layout.level0_offset;

    for (i = 0; i < level0_count; i++) {
        u32 neighbor = le32_to_cpu(level0[i]);

        if (neighbor ==
                VECFS_ANN_INVALID_NODE_ID ||
            neighbor == node_id ||
            neighbor >= slot_count)
            return false;
    }

    counts = object +
             store->layout.upper_counts_offset;
    upper = object +
            store->layout.upper_links_offset;

    for (level = 1;
         level < store->layout.max_levels;
         level++) {
        u16 count =
            le16_to_cpu(counts[level - 1]);
        u64 base =
            (u64)(level - 1) * store->layout.m;

        if (level > node_level) {
            if (count)
                return false;
            continue;
        }

        if (count > store->layout.m)
            return false;

        for (i = 0; i < count; i++) {
            u32 neighbor =
                le32_to_cpu(upper[base + i]);

            if (neighbor ==
                    VECFS_ANN_INVALID_NODE_ID ||
                neighbor == node_id ||
                neighbor >= slot_count)
                return false;
        }
    }

    return true;
}

int vecfs_ann_object_quarantine_locked(
    struct vecfs_ann_store *store,
    u32 node_id)
{
    struct vecfs_sb_info *sbi =
        VECFS_SB(store->sb);
    struct vecfs_ann_object_disk *disk;
    void *object;
    int ret;

    if (sb_rdonly(store->sb))
        return -EROFS;

    object = kvzalloc(
        store->layout.stride, GFP_KERNEL);
    if (!object)
        return -ENOMEM;

    disk = object;

    disk->magic =
        cpu_to_le32(VECFS_ANN_OBJECT_MAGIC);
    disk->version =
        cpu_to_le16(VECFS_ANN_OBJECT_VERSION);
    disk->header_size =
        cpu_to_le16(sizeof(*disk));

    disk->node_id = cpu_to_le32(node_id);
    disk->state =
        cpu_to_le32(VECFS_ANN_OBJ_CORRUPT);

    disk->external_id =
        cpu_to_le64(VECFS_ANN_EXTERNAL_ID_AUTO);

    disk->owner_type =
        cpu_to_le16(VECFS_ANN_OWNER_STANDALONE);

    disk->object_epoch =
        cpu_to_le64(sbi->s_ann.epoch);

    ret = vecfs_ann_object_store_locked(
        store, node_id, object, false);

    kvfree(object);
    return ret;
}

static int vecfs_ann_recover_prepared_locked(
    struct vecfs_ann_store *store,
    u32 node_id,
    void *object)
{
    struct vecfs_ann_object_disk *disk = object;

    if (sb_rdonly(store->sb))
        return 0;

    disk->state =
        cpu_to_le32(VECFS_ANN_OBJ_DELETED);

    return vecfs_ann_object_store_locked(
        store, node_id, object, false);
}

int vecfs_ann_store_scan(struct super_block *sb)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ann_store *store;
    loff_t arena_size;
    u64 slot_count;
    u64 live_nodes = 0;
    u64 deleted_nodes = 0;
    u64 corrupt_nodes = 0;
    u32 entry_node = VECFS_ANN_INVALID_NODE_ID;
    u16 current_max_level = 0;
    void *object;
    u64 id;
    bool metadata_changed = false;
    int ret = 0;

    if (!sbi || !sbi->s_ann_store)
        return -ENODEV;

    store = sbi->s_ann_store;

    if (!store->arena_inode)
        return -ENODEV;

    arena_size = i_size_read(store->arena_inode);

    u64 max_arena_bytes;
    u64 max_slots_by_device;
    u64 max_slots_by_addressing;

    max_arena_bytes =
        (u64)VECFS_SB(sb)->s_blocks_count *
        sb->s_blocksize;

    if ((u64)arena_size > max_arena_bytes)
        return -EUCLEAN;

    max_slots_by_device =
        div64_u64(
            max_arena_bytes,
            store->layout.stride);

    max_slots_by_addressing =
        VECFS_ANN_INVALID_NODE_ID - 1;

    if (slot_count > max_slots_by_device ||
        slot_count > max_slots_by_addressing)
        return -EUCLEAN;


    if (arena_size < 0)
        return -EUCLEAN;

    if (arena_size % store->layout.stride) {
        pr_warn(
            "vecfs: ANN arena has a partial trailing object\n");

        slot_count = div64_u64(
            arena_size,
            store->layout.stride);
    } else {
        slot_count = div64_u64(
            arena_size,
            store->layout.stride);
    }

    if (slot_count >= VECFS_ANN_INVALID_NODE_ID)
        return -EFBIG;

    object = kvmalloc(
        store->layout.stride, GFP_KERNEL);
    if (!object)
        return -ENOMEM;

    down_write(&store->io_lock);

    for (id = VECFS_ANN_FIRST_NODE_ID; id < slot_count; id++) {
        struct vecfs_ann_object_disk *disk;
        u32 state;
        u16 level;

        memset(
            object, 0, store->layout.stride);

        ret = vecfs_ann_arena_read_locked(
            store,
            id * store->layout.stride,
            object,
            store->layout.stride);
        if (ret)
            goto out_unlock;

        if (vecfs_ann_object_is_zero(
                object,
                store->layout.stride)) {
            /*
             * 固定 ID 当前不复用。中间全零槽位视为损坏，
             * 尾部全零对象也不能自动降低 next_node_id，
             * 以免重用可能已经写入邻居表的 ID。
             */
            corrupt_nodes++;

            if (!sb_rdonly(sb)) {
                ret =
                    vecfs_ann_object_quarantine_locked(
                        store, id);
                if (ret)
                    goto out_unlock;
            }

            continue;
        }

        if (!vecfs_ann_object_valid(
                store, id, object)) {
            corrupt_nodes++;

            if (!sb_rdonly(sb)) {
                ret =
                    vecfs_ann_object_quarantine_locked(
                        store, id);
                if (ret)
                    goto out_unlock;
            }

            continue;
        }

        disk = object;
        state = le32_to_cpu(disk->state);
        level = le16_to_cpu(disk->level);

        if (!vecfs_ann_links_sane(
                store, id, slot_count, object)) {
            corrupt_nodes++;

            if (!sb_rdonly(sb)) {
                ret =
                    vecfs_ann_object_quarantine_locked(
                        store, id);
                if (ret)
                    goto out_unlock;
            }

            continue;
        }

        switch (state) {
        case VECFS_ANN_OBJ_FREE:
            /*
            * rebuild 后删除节点的位置会成为 FREE。
            * node ID 不复用，因此该槽位只作为稳定 ID 空洞。
            */
            break;

        case VECFS_ANN_OBJ_PREPARED:
            ret = vecfs_ann_recover_prepared_locked(
                store,
                id,
                object);
            if (ret)
                goto out_unlock;

            deleted_nodes++;
            metadata_changed = true;
            break;

        case VECFS_ANN_OBJ_LIVE:
            live_nodes++;

            if (entry_node ==
                    VECFS_ANN_INVALID_NODE_ID ||
                level > current_max_level) {
                entry_node = id;
                current_max_level = level;
            }
            break;

        case VECFS_ANN_OBJ_DELETED:
        case VECFS_ANN_OBJ_CORRUPT:
            deleted_nodes++;
            break;

        default:
            corrupt_nodes++;
            break;
        }

        if (!(id & 16383))
            schedule();
        else if (!(id & 63))
            cond_resched();
    }

out_unlock:
    up_write(&store->io_lock);
    kvfree(object);

    if (ret)
        return ret;

    store->scanned_slots = slot_count;
    store->corrupt_objects = corrupt_nodes;

    mutex_lock(&sbi->s_ann.lock);
    u64 next_node_id = max_t(u64, slot_count, VECFS_ANN_FIRST_NODE_ID);
    if (sbi->s_ann.next_node_id != next_node_id) {
        sbi->s_ann.next_node_id = next_node_id;
        metadata_changed = true;
    }

    if (sbi->s_ann.live_nodes != live_nodes) {
        sbi->s_ann.live_nodes = live_nodes;
        metadata_changed = true;
    }

    if (sbi->s_ann.deleted_nodes !=
        deleted_nodes + corrupt_nodes) {
        sbi->s_ann.deleted_nodes =
            deleted_nodes + corrupt_nodes;
        metadata_changed = true;
    }

    if (sbi->s_ann.entry_node_id != entry_node) {
        sbi->s_ann.entry_node_id = entry_node;
        metadata_changed = true;
    }

    if (sbi->s_ann.config.current_max_level !=
        current_max_level) {
        sbi->s_ann.config.current_max_level =
            current_max_level;
        metadata_changed = true;
    }

    if (!live_nodes &&
        sbi->s_ann.state != VECFS_ANN_META_EMPTY) {
        sbi->s_ann.state = VECFS_ANN_META_EMPTY;
        metadata_changed = true;
    } else if (live_nodes &&
               sbi->s_ann.state ==
                   VECFS_ANN_META_EMPTY) {
        sbi->s_ann.state = VECFS_ANN_META_ACTIVE;
        metadata_changed = true;
    }

    if (metadata_changed)
        sbi->s_ann.dirty = true;

    mutex_unlock(&sbi->s_ann.lock);

    if (metadata_changed && !sb_rdonly(sb)) {
        ret = vecfs_ann_meta_commit(sb, true);
        if (ret)
            return ret;
    }

    pr_info(
        "vecfs: ANN arena scan slots=%llu live=%llu "
        "deleted=%llu corrupt=%llu entry=%u max_level=%u\n",
        slot_count,
        live_nodes,
        deleted_nodes,
        corrupt_nodes,
        entry_node,
        current_max_level);

    return 0;
}
