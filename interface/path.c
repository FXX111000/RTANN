// SPDX-License-Identifier: GPL-2.0

#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/string.h>

#include "../include/vecfs.h"
#include "../include/vecfs_path.h"

struct vecfs_path_component {
    const char *name;
    u16 length;
};

int vecfs_result_relative_path(
    struct super_block *sb,
    u64 ino,
    u32 generation,
    char *destination,
    size_t capacity,
    size_t *length)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_path_cache *cache;
    struct vecfs_path_component *components;
    struct vecfs_path_entry *entry;
    u64 current_ino;
    size_t component_capacity;
    size_t component_count = 0;
    size_t required = 0;
    size_t output_offset = 0;
    size_t i;
    int ret = 0;

    if (!sb ||
        !destination ||
        !capacity ||
        !length)
        return -EINVAL;

    *length = 0;

    sbi = VECFS_SB(sb);
    if (!sbi)
        return -EINVAL;

    if (!ino ||
        ino > sbi->s_inodes_count)
        return -ENOENT;

    cache = &sbi->s_path_cache;

    if (!cache->initialized)
        return -ENODEV;

    /*
     * parent 链不可能合法地超过 inode 总数。使用该上限同时检测
     * 路径缓存中的环。
     */
    component_capacity =
        min_t(
            size_t,
            sbi->s_inodes_count,
            PATH_MAX / 2);

    if (!component_capacity)
        return -ENOENT;

    components = kvmalloc_array(
        component_capacity,
        sizeof(*components),
        GFP_NOFS);
    if (!components)
        return -ENOMEM;

    mutex_lock(&cache->lock);

    current_ino = ino;

    while (current_ino !=
           sbi->s_root_ino) {
        entry = xa_load(
            &cache->entries,
            current_ino);

        if (!entry) {
            ret = -ENOENT;
            goto out_unlock;
        }

        if (current_ino == ino &&
            generation &&
            entry->inode_generation !=
                generation) {
            ret = -ESTALE;
            goto out_unlock;
        }

        if (!entry->name_length ||
            entry->name_length >
                VECFS_MAX_NAME_LEN ||
            !entry->parent_ino ||
            entry->parent_ino ==
                current_ino) {
            ret = -EUCLEAN;
            goto out_unlock;
        }

        if (component_count >=
            component_capacity) {
            ret = -ELOOP;
            goto out_unlock;
        }

        components[component_count].name =
            entry->name;

        components[component_count].length =
            entry->name_length;

        component_count++;

        if (required >
            SIZE_MAX - entry->name_length) {
            ret = -EOVERFLOW;
            goto out_unlock;
        }

        required +=
            entry->name_length;

        if (component_count > 1) {
            if (required == SIZE_MAX) {
                ret = -EOVERFLOW;
                goto out_unlock;
            }

            required++;
        }

        if (required >= PATH_MAX) {
            ret = -ENAMETOOLONG;
            goto out_unlock;
        }

        current_ino =
            entry->parent_ino;
    }

    if (!component_count) {
        if (capacity < 2) {
            ret = -ENOSPC;
            goto out_unlock;
        }

        destination[0] = '.';
        destination[1] = '\0';
        *length = 1;

        goto out_unlock;
    }

    if (required + 1 > capacity) {
        ret = -ENOSPC;
        goto out_unlock;
    }

    /*
     * parent 链是 leaf -> root，输出时反向遍历。
     */
    for (i = component_count;
         i > 0;
         i--) {
        const struct vecfs_path_component *component =
            &components[i - 1];

        if (output_offset) {
            destination[output_offset] = '/';
            output_offset++;
        }

        memcpy(
            destination + output_offset,
            component->name,
            component->length);

        output_offset +=
            component->length;
    }

    destination[output_offset] = '\0';
    *length = output_offset;

out_unlock:
    mutex_unlock(&cache->lock);

    kvfree(components);
    return ret;
}
