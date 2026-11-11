// SPDX-License-Identifier: GPL-2.0

#include <linux/compat.h>
#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/vmalloc.h>

#include "../include/vecfs.h"
#include "../include/vecfs_ann_build.h"
#include "../include/vecfs_ann_mutation.h"
#include "../include/vecfs_ann_runtime.h"
#include "../include/vecfs_ann_search.h"
#include "../include/vecfs_ann_sched.h"
#include "../include/vecfs_interface.h"
#include "../include/vecfs_ioctl.h"
#include "../include/vecfs_path.h"

#include "../include/vecfs_ann_info.h"
#include "../include/vecfs_ann_maintenance.h"
#include "../include/vecfs_ann_validate.h"
#include "../include/vecfs_ann_rebuild.h"
#include "../include/vecfs_ann_exchange.h"
#include "../include/vecfs_ann_indirect.h"
#include "../include/vecfs_ann_info.h"

struct vecfs_ioctl_file_list_context {
    struct vecfs_ioctl_vector_info *entries;

    u32 start_ordinal;
    u32 capacity;
    u32 count;

    u32 next_ordinal;
    bool done;
    bool include_deleted;
};

#define VECFS_FILE_LIST_STOP 1

static void vecfs_ioctl_fill_vector_info(
    const struct vecfs_ann_object_info *source,
    struct vecfs_ioctl_vector_info *destination);

static int vecfs_ioctl_file_list_callback(
    struct inode *inode,
    u32 ordinal,
    u32 node_id,
    void *private_data)
{
    struct vecfs_ioctl_file_list_context *context =
        private_data;

    struct vecfs_ann_object_info info;
    int ret;

    if (ordinal <
        context->start_ordinal)
        return 0;

    if (context->count >=
        context->capacity) {
        context->next_ordinal =
            ordinal;

        context->done = false;

        return VECFS_FILE_LIST_STOP;
    }

    ret = vecfs_ann_object_read_info(
        inode->i_sb,
        node_id,
        &info);
    if (ret == -ENOENT ||
        ret == -EBADMSG ||
        ret == -ENODATA) {
        /*
         * 损坏映射由 validate 处理，list 跳过。
         */
        return 0;
    }

    if (ret)
        return ret;

    if (info.owner_type !=
            VECFS_ANN_OWNER_FILE ||
        info.owner_ino !=
            inode->i_ino ||
        info.owner_generation !=
            inode->i_generation ||
        info.vector_ordinal != ordinal)
        return 0;

    if (info.state ==
            VECFS_ANN_OBJ_DELETED &&
        !context->include_deleted)
        return 0;

    if (info.state !=
            VECFS_ANN_OBJ_LIVE &&
        info.state !=
            VECFS_ANN_OBJ_DELETED)
        return 0;

    vecfs_ioctl_fill_vector_info(
        &info,
        &context->entries[
            context->count]);

    context->count++;

    if (ordinal == U32_MAX) {
        context->done = true;
    } else {
        context->next_ordinal =
            ordinal + 1;
    }

    return 0;
}

static long vecfs_ioctl_ann_rebuild(
    struct file *file,
    void __user *argument)
{
    struct vecfs_ioctl_rebuild user;
    struct vecfs_ann_rebuild_report report;
    u64 request_id;
    int ret;

    if (copy_from_user(
            &user,
            argument,
            sizeof(user)))
        return -EFAULT;

    if (user.flags &
        ~(VECFS_REBUILD_F_SYNC |
          VECFS_REBUILD_F_PRESERVE_LEVELS |
          VECFS_REBUILD_F_ASYNC |
          VECFS_REBUILD_F_PARALLEL))
        return -EINVAL;

    if ((user.flags &
         VECFS_REBUILD_F_SYNC) &&
        (user.flags &
         VECFS_REBUILD_F_ASYNC))
        return -EINVAL;

    if (user.flags &
        VECFS_REBUILD_F_ASYNC) {
        ret = vecfs_ann_rebuild_submit(
            file_inode(file)->i_sb,
            user.flags,
            &request_id);
        if (ret)
            return ret;

        user.request_id = request_id;
        user.status = -EINPROGRESS;

        if (copy_to_user(
                argument,
                &user,
                sizeof(user)))
            return -EFAULT;

        return 0;
    }

    memset(&report, 0, sizeof(report));

    ret = vecfs_ann_rebuild(
        file_inode(file)->i_sb,
        user.flags,
        &report);

    user.status = ret;

    user.source_live_nodes =
        report.source_live_nodes;
    user.source_deleted_nodes =
        report.source_deleted_nodes;
    user.rebuilt_nodes =
        report.rebuilt_nodes;
    user.removed_nodes =
        report.removed_nodes;

    user.old_epoch =
        report.old_epoch;
    user.new_epoch =
        report.new_epoch;

    user.old_arena_inode =
        report.old_arena_inode;
    user.new_arena_inode =
        report.new_arena_inode;

    user.duration_ns =
        report.duration_ns;

    if (copy_to_user(
            argument,
            &user,
            sizeof(user)))
        return -EFAULT;

    return ret;
}



static long vecfs_ioctl_file_vector_list(
    struct file *file,
    void __user *argument)
{
    struct inode *inode =
        file_inode(file);

    struct vecfs_ioctl_file_vector_list user;
    struct vecfs_ioctl_file_list_context context;
    struct vecfs_ioctl_vector_info *entries = NULL;
    int ret;

    if (!S_ISREG(inode->i_mode))
        return -EINVAL;

    if (copy_from_user(
            &user,
            argument,
            sizeof(user)))
        return -EFAULT;

    if (!user.entry_capacity ||
        user.entry_capacity >
            VECFS_IOCTL_MAX_K)
        return -EINVAL;

    if (!user.entries_ptr)
        return -EINVAL;

    if (user.flags &
        ~VECFS_FILE_LIST_F_INCLUDE_DELETED)
        return -EINVAL;

    entries = kvcalloc(
        user.entry_capacity,
        sizeof(*entries),
        GFP_NOFS);

    if (!entries)
        return -ENOMEM;

    memset(&context, 0, sizeof(context));

    context.entries = entries;
    context.start_ordinal =
        user.start_ordinal;
    context.capacity =
        user.entry_capacity;
    context.done = true;
    context.include_deleted =
        user.flags &
        VECFS_FILE_LIST_F_INCLUDE_DELETED;

    ret = vecfs_ann_for_each_node(
        inode,
        vecfs_ioctl_file_list_callback,
        &context);

    if (ret == VECFS_FILE_LIST_STOP)
        ret = 0;

    if (ret)
        goto out;

    if (context.count &&
        copy_to_user(
            u64_to_user_ptr(
                user.entries_ptr),
            entries,
            context.count *
                sizeof(*entries))) {
        ret = -EFAULT;
        goto out;
    }

    user.entry_count =
        context.count;

    user.next_ordinal =
        context.next_ordinal;

    user.done =
        context.done ? 1 : 0;

    if (copy_to_user(
            argument,
            &user,
            sizeof(user))) {
        ret = -EFAULT;
        goto out;
    }

    ret = 0;

out:
    kvfree(entries);
    return ret;
}

static void vecfs_ioctl_fill_validate_report(
    const struct vecfs_ann_validate_report *source,
    struct vecfs_ioctl_validate_report *destination)
{
    memset(destination, 0, sizeof(*destination));

    destination->objects_checked =
        source->objects_checked;

    destination->live_objects =
        source->live_objects;

    destination->deleted_objects =
        source->deleted_objects;

    destination->corrupt_objects =
        source->corrupt_objects;

    destination->file_objects =
        source->file_objects;

    destination->standalone_objects =
        source->standalone_objects;

    destination->imported_objects =
        source->imported_objects;

    destination->inode_mappings_checked =
        source->inode_mappings_checked;

    destination->missing_inodes =
        source->missing_inodes;

    destination->generation_mismatches =
        source->generation_mismatches;

    destination->missing_inode_mappings =
        source->missing_inode_mappings;

    destination->stale_inode_mappings =
        source->stale_inode_mappings;

    destination->owner_mismatches =
        source->owner_mismatches;

    destination->ordinal_mismatches =
        source->ordinal_mismatches;

    destination->duplicate_external_ids =
        source->duplicate_external_ids;

    destination->invalid_neighbors =
        source->invalid_neighbors;

    destination->objects_marked_deleted =
        source->objects_marked_deleted;

    destination->mappings_cleared =
        source->mappings_cleared;

    destination->metadata_repairs =
        source->metadata_repairs;

    destination->recovered_entry_node =
        source->recovered_entry_node;

    destination->recovered_max_level =
        source->recovered_max_level;
}

static long vecfs_ioctl_ann_validate(
    struct file *file,
    void __user *argument)
{
    struct vecfs_ioctl_validate user;
    struct vecfs_ann_validate_report report;
    u32 internal_flags = 0;
    int ret;

    if (copy_from_user(
            &user,
            argument,
            sizeof(user)))
        return -EFAULT;

    if (user.flags &
        ~(VECFS_VALIDATE_F_REPAIR |
          VECFS_VALIDATE_F_STRICT |
          VECFS_VALIDATE_F_CHECK_LINKS))
        return -EINVAL;

    if (user.flags &
        VECFS_VALIDATE_F_REPAIR)
        internal_flags |=
            VECFS_ANN_VALIDATE_F_REPAIR;

    if (user.flags &
        VECFS_VALIDATE_F_STRICT)
        internal_flags |=
            VECFS_ANN_VALIDATE_F_STRICT;

    if (user.flags &
        VECFS_VALIDATE_F_CHECK_LINKS)
        internal_flags |=
            VECFS_ANN_VALIDATE_F_CHECK_LINKS;

    memset(&report, 0, sizeof(report));

    ret = vecfs_ann_validate_and_refresh(
        file_inode(file)->i_sb,
        internal_flags,
        &report);

    /*
     * 无论 validate 成功还是发现 EUCLEAN，都向用户返回报告。
     */
    user.status = ret;

    vecfs_ioctl_fill_validate_report(
        &report,
        &user.report);

    if (copy_to_user(
            argument,
            &user,
            sizeof(user)))
        return -EFAULT;

    return ret;
}


static int vecfs_copy_vector_from_user(
    struct super_block *sb,
    u64 pointer,
    u32 bytes,
    void **vector)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    void __user *user_pointer =
        u64_to_user_ptr(pointer);
    void *buffer;

    /*pr_err("vecfs: copy_vector_from_user ENTER ptr=%llx bytes=%u\n",
           (unsigned long long)pointer, bytes);*/

    if (!pointer || !vector) {
        pr_err("vecfs: copy_vector_from_user: pointer=%llx vector=%p -> EINVAL\n",
               (unsigned long long)pointer, vector);
        return -EINVAL;
    }

    if (bytes !=
        sbi->s_ann.config.vector_bytes) {
        pr_err("vecfs: copy_vector_from_user: bytes=%u expected=%u -> EINVAL\n",
               bytes, sbi->s_ann.config.vector_bytes);
        return -EINVAL;
    }

    buffer = memdup_user(
        user_pointer,
        bytes);
    if (IS_ERR(buffer)) {
        pr_err("vecfs: copy_vector_from_user: memdup_user=%ld\n",
               PTR_ERR(buffer));
        return PTR_ERR(buffer);
    }

    //pr_err("vecfs: copy_vector_from_user OK vector=%p\n", buffer);
    *vector = buffer;
    return 0;
}

static int vecfs_allocate_ordinal(
    struct inode *inode,
    u32 *ordinal)
{
    u32 value;

    if (!inode || !ordinal)
        return -EINVAL;

    for (value = 0;
         value < VECFS_ANN_ORDINAL_AUTO;
         value++) {
        u32 node_id;
        int ret;

        ret = vecfs_ann_get_node(
            inode,
            value,
            &node_id);

        if (ret == -ENOENT) {
            *ordinal = value;
            return 0;
        }

        if (ret)
            return ret;

        if (unlikely(value == U32_MAX - 1))
            break;

        cond_resched();
    }

    return -ENOSPC;
}

static long vecfs_ioctl_get_info(
    struct super_block *sb,
    void __user *argument)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ioctl_info info;

    memset(&info, 0, sizeof(info));

    info.abi_major = VECFS_IOCTL_ABI_MAJOR;
    info.abi_minor = VECFS_IOCTL_ABI_MINOR;

    info.fs_major = sbi->s_disk_major;
    info.fs_minor = sbi->s_disk_minor;

    mutex_lock(&sbi->s_ann.lock);

    info.dimension =
        sbi->s_ann.config.dimension;
    info.element_type =
        sbi->s_ann.config.element_type;
    info.distance_type =
        sbi->s_ann.config.distance_type;

    info.m = sbi->s_ann.config.m;
    info.m0 = sbi->s_ann.config.m0;
    info.max_levels =
        sbi->s_ann.config.max_levels;
    info.current_max_level =
        sbi->s_ann.config.current_max_level;

    info.ef_construction =
        sbi->s_ann.config.ef_construction;

    info.live_nodes =
        sbi->s_ann.live_nodes;
    info.deleted_nodes =
        sbi->s_ann.deleted_nodes;
    info.next_node_id =
        sbi->s_ann.next_node_id;

    info.entry_node_id =
        sbi->s_ann.entry_node_id;

    info.object_stride =
        sbi->s_ann.config.object_stride;
    info.vector_bytes =
        sbi->s_ann.config.vector_bytes;

    mutex_unlock(&sbi->s_ann.lock);

    info.default_ef_search =
        atomic_read(
            &sbi->s_ann_runtime.default_ef_search);

    info.simd_mode =
        sbi->s_ann_distance.simd_mode;

    if (copy_to_user(
            argument,
            &info,
            sizeof(info)))
        return -EFAULT;

    return 0;
}

static long vecfs_ioctl_get_runtime(
    struct super_block *sb,
    void __user *argument)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ioctl_runtime runtime;

    memset(&runtime, 0, sizeof(runtime));

    runtime.ef_search =
        atomic_read(
            &sbi->s_ann_runtime.default_ef_search);

    runtime.maximum_queries =
        atomic_read(
            &sbi->s_ann_runtime.maximum_queries);

    if (copy_to_user(
            argument,
            &runtime,
            sizeof(runtime)))
        return -EFAULT;

    return 0;
}

static long vecfs_ioctl_set_runtime(
    struct super_block *sb,
    void __user *argument)
{
    struct vecfs_ioctl_runtime runtime;

    if (copy_from_user(
            &runtime,
            argument,
            sizeof(runtime)))
        return -EFAULT;

    if (runtime.flags &
        ~VECFS_RUNTIME_F_PERSIST_EF)
        return -EINVAL;

    /*
     * 更新只影响之后开始的查询。
     */
    return vecfs_ann_runtime_set_ef(
        sb,
        runtime.ef_search,
        runtime.flags &
            VECFS_RUNTIME_F_PERSIST_EF);
}

static long vecfs_ioctl_vector_add(
    struct file *file,
    void __user *argument)
{
    struct vecfs_ioctl_vector_add user;
    struct vecfs_ann_insert_request request;
    struct vecfs_ann_insert_result result;
    void *vector;
    int ret;

    if (copy_from_user(
            &user,
            argument,
            sizeof(user)))
        return -EFAULT;
    /*
    pr_err("vecfs: ioctl_vector_add: vptr=%llx vbytes=%u eid=%llu\n",
           (unsigned long long)user.vector_ptr,
           user.vector_bytes,
           (unsigned long long)user.external_id);
*/
    ret = vecfs_copy_vector_from_user(
        file_inode(file)->i_sb,
        user.vector_ptr,
        user.vector_bytes,
        &vector);
    if (ret) {
        pr_err("vecfs: ioctl_vector_add: copy_vector FAIL ret=%d\n", ret);
        return ret;
    }

    memset(&request, 0, sizeof(request));

    request.vector = vector;
    request.vector_bytes =
        user.vector_bytes;
    request.external_id =
        user.external_id;
    request.owner_type =
        VECFS_ANN_OWNER_STANDALONE;

    ret = vecfs_ann_insert(
        file_inode(file)->i_sb,
        &request,
        &result);

    kfree(vector);

    if (ret) {
        pr_err("vecfs: ioctl_vector_add: ann_insert FAIL ret=%d\n", ret);
        return ret;
    }

    user.result_external_id =
        result.external_id;
    user.result_node_id =
        result.node_id;
    user.result_level =
        result.level;

    if (copy_to_user(
            argument,
            &user,
            sizeof(user)))
        return -EFAULT;

    /*pr_err("vecfs: ioctl_vector_add: SUCCESS nid=%llu\n",
           (unsigned long long)result.node_id);*/
    return 0;
}

static long vecfs_ioctl_vector_update(
    struct file *file,
    void __user *argument)
{
    struct vecfs_ioctl_vector_update user;
    struct vecfs_ann_update_request request;
    void *vector;
    int ret;

    if (copy_from_user(
            &user,
            argument,
            sizeof(user)))
        return -EFAULT;

    ret = vecfs_copy_vector_from_user(
        file_inode(file)->i_sb,
        user.vector_ptr,
        user.vector_bytes,
        &vector);
    if (ret)
        return ret;

    request.vector = vector;
    request.vector_bytes =
        user.vector_bytes;

    request.neighbor_probability_ppm =
        user.neighbor_probability_ppm ?
            user.neighbor_probability_ppm :
            1000000;

    ret = vecfs_ann_update_by_label(
        file_inode(file)->i_sb,
        user.external_id,
        &request,
        VECFS_ANN_ACCESS_STANDALONE);

    kfree(vector);
    return ret;
}

static long vecfs_ioctl_vector_delete(
    struct file *file,
    void __user *argument,
    bool undelete)
{
    struct vecfs_ioctl_vector_label user;

    if (copy_from_user(
            &user,
            argument,
            sizeof(user)))
        return -EFAULT;

    if (undelete)
        return vecfs_ann_unmark_delete_by_label(
            file_inode(file)->i_sb,
            user.external_id,
            VECFS_ANN_ACCESS_STANDALONE);

    return vecfs_ann_mark_delete_by_label(
        file_inode(file)->i_sb,
        user.external_id,
        VECFS_ANN_ACCESS_STANDALONE);
}

static long vecfs_ioctl_file_vector_add(
    struct file *file,
    void __user *argument)
{
    struct inode *inode = file_inode(file);
    struct vecfs_ioctl_file_vector_add user;
    struct vecfs_ann_insert_result result;
    void *vector;
    u32 ordinal;
    int ret;

    if (!S_ISREG(inode->i_mode))
        return -EINVAL;

    if (copy_from_user(
            &user,
            argument,
            sizeof(user)))
        return -EFAULT;

    if (user.flags &
        ~(VECFS_VECTOR_F_HAS_CONTENT_RANGE |
          VECFS_VECTOR_F_CONTENT_STALE))
        return -EINVAL;

    if ((user.flags &
         VECFS_VECTOR_F_HAS_CONTENT_RANGE) &&
        user.content_offset >
            U64_MAX - user.content_length)
        return -EOVERFLOW;

    ordinal = user.ordinal;

    if (ordinal == VECFS_ORDINAL_AUTO) {
        ret = vecfs_allocate_ordinal(
            inode,
            &ordinal);
        if (ret)
            return ret;
    }

    ret = vecfs_copy_vector_from_user(
        inode->i_sb,
        user.vector_ptr,
        user.vector_bytes,
        &vector);
    if (ret)
        return ret;

    ret = vecfs_ann_insert_file_vector(
        inode,
        ordinal,
        vector,
        user.vector_bytes,
        user.external_id,
        user.content_offset,
        user.content_length,
        user.flags,
        &result);

    kfree(vector);

    if (ret)
        return ret;

    user.result_external_id =
        result.external_id;
    user.result_node_id =
        result.node_id;
    user.result_ordinal =
        ordinal;
    user.result_level =
        result.level;

    if (copy_to_user(
            argument,
            &user,
            sizeof(user)))
        return -EFAULT;

    return 0;
}

static long vecfs_ioctl_file_vector_update(
    struct file *file,
    void __user *argument)
{
    struct inode *inode =
        file_inode(file);

    struct vecfs_ioctl_file_vector_update user;
    struct vecfs_ann_vector_record rec;
    struct vecfs_ann_update_request request;
    void *vector = NULL;
    int ret;

    if (!S_ISREG(inode->i_mode))
        return -EINVAL;

    if (copy_from_user(
            &user,
            argument,
            sizeof(user)))
        return -EFAULT;

    if (user.flags &
        ~(VECFS_VECTOR_F_HAS_CONTENT_RANGE |
          VECFS_VECTOR_F_CONTENT_STALE))
        return -EINVAL;

    if ((user.flags &
         VECFS_VECTOR_F_HAS_CONTENT_RANGE) &&
        user.content_offset >
            U64_MAX - user.content_length)
        return -EOVERFLOW;

    memset(&rec, 0, sizeof(rec));

    ret = vecfs_ann_get_file_vector(
        inode,
        user.ordinal,
        true,
        &rec);
    if (ret)
        return ret;

    /*
     * 未显式传入新的 binding 时，保留旧 binding。
     * 这样只更新向量的旧调用者不会丢失 content range。
     */
    if (!(user.flags &
          VECFS_VECTOR_F_HAS_CONTENT_RANGE)) {
        user.content_offset =
            rec.info.content_offset;

        user.content_length =
            rec.info.content_length;

        if (rec.info.flags &
            VECFS_ANN_OBJ_F_HAS_CONTENT_RANGE)
            user.flags |=
                VECFS_VECTOR_F_HAS_CONTENT_RANGE;
    }

    /*
     * 没有显式 INCLUDE stale 时，重新生成向量默认清除 stale。
     */
    if (!(user.flags &
          VECFS_VECTOR_F_CONTENT_STALE))
        user.flags &=
            ~VECFS_VECTOR_F_CONTENT_STALE;

    vecfs_ann_vector_record_destroy(
        &rec);

    ret = vecfs_copy_vector_from_user(
        inode->i_sb,
        user.vector_ptr,
        user.vector_bytes,
        &vector);
    if (ret)
        return ret;

    memset(&request, 0, sizeof(request));

    request.vector = vector;
    request.vector_bytes =
        user.vector_bytes;

    request.neighbor_probability_ppm =
        user.neighbor_probability_ppm ?
            user.neighbor_probability_ppm :
            1000000;

    request.update_binding = true;
    request.binding_flags =
        (user.flags &
         VECFS_VECTOR_F_HAS_CONTENT_RANGE) ?
            VECFS_ANN_OBJ_F_HAS_CONTENT_RANGE :
            0;

    if (user.flags &
        VECFS_VECTOR_F_CONTENT_STALE)
        request.binding_flags |=
            VECFS_ANN_OBJ_F_CONTENT_STALE;

    request.content_offset =
        user.content_offset;
    request.content_length =
        user.content_length;

    ret = vecfs_ann_file_update(
        inode,
        user.ordinal,
        &request);

    kvfree(vector);
    return ret;
}



static long vecfs_ioctl_file_vector_delete(
    struct file *file,
    void __user *argument)
{
    struct inode *inode = file_inode(file);
    struct vecfs_ioctl_file_vector_delete user;

    if (!S_ISREG(inode->i_mode))
        return -EINVAL;

    if (copy_from_user(
            &user,
            argument,
            sizeof(user)))
        return -EFAULT;

    return vecfs_ann_file_mark_delete(
        inode,
        user.ordinal);
}

static bool __maybe_unused vecfs_result_allowed(
    const struct vecfs_ann_search_result *result,
    u32 flags)
{
    if ((flags & VECFS_QUERY_F_FILE_ONLY) &&
        result->owner_type !=
            VECFS_ANN_OWNER_FILE)
        return false;

    if ((flags &
         VECFS_QUERY_F_STANDALONE_ONLY) &&
        result->owner_type ==
            VECFS_ANN_OWNER_FILE)
        return false;

    if (!(flags &
          VECFS_QUERY_F_INCLUDE_STALE) &&
        (result->flags &
         VECFS_ANN_OBJ_F_CONTENT_STALE))
        return false;

    return true;
}

static int vecfs_fill_user_result(
    struct super_block *sb,
    const struct vecfs_ann_search_result *source,
    struct vecfs_ioctl_knn_result *destination,
    char *paths,
    u32 path_capacity,
    u32 *path_bytes)
{
    size_t path_length = 0;
    int ret;

    memset(destination, 0, sizeof(*destination));

    destination->external_id =
        source->external_id;

    destination->ino =
        source->owner_ino;
    destination->inode_generation =
        source->owner_generation;
    destination->vector_ordinal =
        source->vector_ordinal;

    destination->content_offset =
        source->content_offset;
    destination->content_length =
        source->content_length;

    destination->owner_type =
        source->owner_type;

    memcpy(
        &destination->distance_bits,
        &source->distance,
        sizeof(source->distance));

    if (source->flags &
        VECFS_ANN_OBJ_F_HAS_CONTENT_RANGE)
        destination->flags |=
            VECFS_RESULT_F_HAS_CONTENT_RANGE;

    if (source->flags &
        VECFS_ANN_OBJ_F_CONTENT_STALE)
        destination->flags |=
            VECFS_RESULT_F_CONTENT_STALE;

    if (source->flags &
        VECFS_ANN_OBJ_F_CONTENT_INVALID)
        destination->flags |=
            VECFS_RESULT_F_CONTENT_INVALID;

    if (source->owner_type !=
        VECFS_ANN_OWNER_FILE)
        return 0;

    if (!paths || *path_bytes >= path_capacity)
        return 0;

    ret = vecfs_result_relative_path(
        sb,
        source->owner_ino,
        source->owner_generation,
        paths + *path_bytes,
        path_capacity - *path_bytes,
        &path_length);

    /*
     * 路径解析失败不使整个 KNN 查询失败。
     */
    if (ret)
        return 0;

    destination->path_offset =
        *path_bytes;
    destination->path_length =
        path_length;
    destination->flags |=
        VECFS_RESULT_F_HAS_PATH;

    *path_bytes += path_length + 1;
    return 0;
}

static long vecfs_ioctl_knn(
    struct file *file,
    void __user *argument)
{
    struct super_block *sb =
        file_inode(file)->i_sb;

    struct vecfs_ioctl_knn user;
    struct vecfs_ann_search_request request;

    struct vecfs_ann_search_result *internal_results =
        NULL;

    struct vecfs_ioctl_knn_result *user_results =
        NULL;

    void *query = NULL;
    char *paths = NULL;

    u32 internal_count = 0;
    u32 path_bytes = 0;
    u32 internal_flags = 0;
    u32 i;
    int ret;

    if (copy_from_user(
            &user,
            argument,
            sizeof(user)))
        return -EFAULT;

    if (!user.k ||
        user.k > VECFS_IOCTL_MAX_K)
        return -EINVAL;

    if (user.result_capacity <
        user.k)
        return -ENOSPC;

    if (!user.results_ptr)
        return -EINVAL;

    if (user.flags &
        ~(VECFS_QUERY_F_FILE_ONLY |
          VECFS_QUERY_F_STANDALONE_ONLY |
          VECFS_QUERY_F_INCLUDE_STALE))
        return -EINVAL;

    if ((user.flags &
         VECFS_QUERY_F_FILE_ONLY) &&
        (user.flags &
         VECFS_QUERY_F_STANDALONE_ONLY))
        return -EINVAL;

    if (user.path_buffer_capacity >
        VECFS_IOCTL_MAX_PATH_BUFFER)
        return -EINVAL;

    if (user.path_buffer_capacity &&
        !user.path_buffer_ptr)
        return -EINVAL;

    ret = vecfs_copy_vector_from_user(
        sb,
        user.query_ptr,
        user.query_bytes,
        &query);
    if (ret)
        return ret;

    internal_results = kvmalloc_array(
        user.k,
        sizeof(*internal_results),
        GFP_KERNEL);

    user_results = kvcalloc(
        user.k,
        sizeof(*user_results),
        GFP_KERNEL);

    if (!internal_results ||
        !user_results) {
        ret = -ENOMEM;
        goto out;
    }

    if (user.path_buffer_capacity) {
        paths = kvzalloc(
            user.path_buffer_capacity,
            GFP_KERNEL);
        if (!paths) {
            ret = -ENOMEM;
            goto out;
        }
    }

    if (user.flags &
        VECFS_QUERY_F_FILE_ONLY)
        internal_flags |=
            VECFS_ANN_SEARCH_F_FILE_ONLY;

    if (user.flags &
        VECFS_QUERY_F_STANDALONE_ONLY)
        internal_flags |=
            VECFS_ANN_SEARCH_F_STANDALONE_ONLY;

    if (user.flags &
        VECFS_QUERY_F_INCLUDE_STALE)
        internal_flags |=
            VECFS_ANN_SEARCH_F_INCLUDE_STALE;

    memset(&request, 0, sizeof(request));

    request.query_vector = query;
    request.query_bytes =
        user.query_bytes;

    request.k = user.k;

    /*
     * vecfs_ann_search() 在开始时捕获一次 ef_search。
     */
    request.ef_search =
        user.ef_search;

    request.flags =
        internal_flags;

    ret = vecfs_ann_search(
        sb,
        &request,
        internal_results,
        user.k,
        &internal_count);
    if (ret)
        goto out;

    for (i = 0;
         i < internal_count;
         i++) {
        ret = vecfs_fill_user_result(
            sb,
            &internal_results[i],
            &user_results[i],
            paths,
            user.path_buffer_capacity,
            &path_bytes);
        if (ret)
            goto out;
    }

    if (internal_count &&
        copy_to_user(
            u64_to_user_ptr(
                user.results_ptr),
            user_results,
            internal_count *
                sizeof(*user_results))) {
        ret = -EFAULT;
        goto out;
    }

    if (path_bytes &&
        copy_to_user(
            u64_to_user_ptr(
                user.path_buffer_ptr),
            paths,
            path_bytes)) {
        ret = -EFAULT;
        goto out;
    }

    user.result_count =
        internal_count;

    user.path_buffer_bytes =
        path_bytes;

    if (copy_to_user(
            argument,
            &user,
            sizeof(user))) {
        ret = -EFAULT;
        goto out;
    }

    vecfs_shield_count_query(sb, internal_count);

    ret = 0;

out:
    kvfree(paths);
    kvfree(user_results);
    kvfree(internal_results);
    kfree(query);

    return ret;
}

/*
 * RTANN 批量检索：一次提交 n_queries 个查询，调度器并发执行后同步返回。
 * 检索路径是纯净 bare HNSW（返回 node id + 距离 + 距离计算次数 + 调度轨迹）。
 */
static long vecfs_ioctl_knn_batch(
    struct file *file,
    void __user *argument)
{
    struct super_block *sb = file_inode(file)->i_sb;
    struct vecfs_ioctl_knn_batch user;
    struct vecfs_ann_sched_req req;
    size_t queries_bytes, results_bytes, distances_bytes, dist_counts_bytes, trace_bytes;
    float *queries = NULL;
    u32 *results = NULL;
    u32 *distances = NULL;
    u64 *dist_counts = NULL;
    struct vecfs_ioctl_sched_trace *trace_buf = NULL;
    int ret;

    if (copy_from_user(&user, argument, sizeof(user)))
        return -EFAULT;
    if (user.flags)
        return -EINVAL;
    if (user.sched_mode > VECFS_SCHED_MODE_FULL)
        return -EINVAL;
    if (!user.queries_ptr || !user.results_ptr)
        return -EINVAL;

    queries_bytes = (size_t)user.n_queries * user.dim * sizeof(float);
    results_bytes = (size_t)user.n_queries * user.k * sizeof(u32);
    distances_bytes = user.distances_ptr ? (size_t)user.n_queries * user.k * sizeof(u32) : 0;
    dist_counts_bytes = (size_t)user.n_queries * sizeof(u64);
    trace_bytes = user.trace_ptr ? (size_t)user.n_queries * sizeof(struct vecfs_ioctl_sched_trace) : 0;

    queries = kvmalloc(queries_bytes, GFP_KERNEL);
    results = kvmalloc(results_bytes, GFP_KERNEL);
    dist_counts = kvmalloc(dist_counts_bytes, GFP_KERNEL);
    if (!queries || !results || !dist_counts) {
        ret = -ENOMEM;
        goto out;
    }
    if (distances_bytes) {
        distances = kvmalloc(distances_bytes, GFP_KERNEL);
        if (!distances) {
            ret = -ENOMEM;
            goto out;
        }
    }
    if (trace_bytes) {
        trace_buf = kvmalloc(
            trace_bytes, GFP_KERNEL);
        if (!trace_buf) {
            ret = -ENOMEM;
            goto out;
        }
    }

    if (copy_from_user(queries, u64_to_user_ptr(user.queries_ptr), queries_bytes)) {
        ret = -EFAULT;
        goto out;
    }

    memset(&req, 0, sizeof(req));
    req.n_queries = user.n_queries;
    req.dim = user.dim;
    req.k = user.k;
    req.ef = user.ef_search;
    req.sched_mode = user.sched_mode;
    req.queries = queries;
    req.results = results;
    req.distances = distances;
    req.dist_counts = dist_counts;
    req.traces = trace_buf;

    ret = vecfs_ann_sched_submit(sb, &req);
    if (ret)
        goto out;

    if (copy_to_user(u64_to_user_ptr(user.results_ptr), results, results_bytes)) {
        ret = -EFAULT;
        goto out;
    }
    if (distances && copy_to_user(u64_to_user_ptr(user.distances_ptr), distances, distances_bytes)) {
        ret = -EFAULT;
        goto out;
    }
    if (user.dist_counts_ptr && copy_to_user(u64_to_user_ptr(user.dist_counts_ptr), dist_counts, dist_counts_bytes)) {
        ret = -EFAULT;
        goto out;
    }
    if (trace_buf && copy_to_user(u64_to_user_ptr(user.trace_ptr), trace_buf, trace_bytes)) {
        ret = -EFAULT;
        goto out;
    }

    ret = 0;
out:
    kvfree(trace_buf);
    kvfree(dist_counts);
    kvfree(distances);
    kvfree(results);
    kvfree(queries);
    return ret;
}

static long vecfs_ioctl_ann_rebuild_status(
    struct file *file,
    void __user *argument)
{
    struct vecfs_ioctl_rebuild_status user;
    struct vecfs_ann_rebuild_report report;
    enum vecfs_ann_rebuild_state state;
    int status;
    int ret;

    if (copy_from_user(
            &user,
            argument,
            sizeof(user)))
        return -EFAULT;

    memset(&report, 0, sizeof(report));

    ret = vecfs_ann_rebuild_get_status(
        file_inode(file)->i_sb,
        user.request_id,
        &state,
        &status,
        &report);
    if (ret)
        return ret;

    user.state = state;
    user.status = status;

    user.source_live_nodes =
        report.source_live_nodes;
    user.source_deleted_nodes =
        report.source_deleted_nodes;
    user.rebuilt_nodes =
        report.rebuilt_nodes;
    user.removed_nodes =
        report.removed_nodes;

    user.old_epoch =
        report.old_epoch;
    user.new_epoch =
        report.new_epoch;

    user.old_arena_inode =
        report.old_arena_inode;
    user.new_arena_inode =
        report.new_arena_inode;

    user.duration_ns =
        report.duration_ns;

    if (copy_to_user(
            argument,
            &user,
            sizeof(user)))
        return -EFAULT;

    return 0;
}

static long vecfs_ioctl_ann_rebuild_cancel(
    struct file *file,
    void __user *argument)
{
    struct vecfs_ioctl_rebuild_cancel user;

    if (copy_from_user(
            &user,
            argument,
            sizeof(user)))
        return -EFAULT;

    return vecfs_ann_rebuild_cancel(
        file_inode(file)->i_sb,
        user.request_id);
}




static long vecfs_ioctl_vector_get(
    struct file *file,
    void __user *argument);

static long vecfs_ioctl_file_vector_get(
    struct file *file,
    void __user *argument);

static long vecfs_ioctl_export_info(
    struct file *file,
    void __user *argument);

static long vecfs_ioctl_export_node(
    struct file *file,
    void __user *argument);

static long vecfs_ioctl_import_begin(
    struct file *file,
    void __user *argument);

static long vecfs_ioctl_import_node(
    struct file *file,
    void __user *argument);

static long vecfs_ioctl_import_commit(
    struct file *file,
    void __user *argument);

static long vecfs_ioctl_import_abort(
    struct file *file,
    void __user *argument);

static long vecfs_ioctl_shield_set_mode(
    struct super_block *sb,
    void __user *argument)
{
    struct vecfs_ioctl_shield_set_mode user;

    if (copy_from_user(&user, argument, sizeof(user)))
        return -EFAULT;

    if (user.flags)
        return -EINVAL;

    return vecfs_shield_set_mode(sb, user.mode, user.window_us);
}

static long vecfs_ioctl_shield_get_stats(
    struct super_block *sb,
    void __user *argument)
{
    struct vecfs_ioctl_shield_stats user;
    struct vecfs_shield_stats snapshot;
    u32 flags;

    if (copy_from_user(&user, argument, sizeof(user)))
        return -EFAULT;

    flags = user.flags;

    if (flags & ~VECFS_SHIELD_STATS_F_RESET)
        return -EINVAL;

    vecfs_shield_stats_snapshot(sb, &snapshot);

    memset(&user, 0, sizeof(user));

    user.writes = snapshot.writes;
    user.tasks_begun = snapshot.tasks_begun;
    user.tasks_ended = snapshot.tasks_ended;
    user.tasks_expired = snapshot.tasks_expired;
    user.publishes_rejected = snapshot.publishes_rejected;
    user.queries = snapshot.queries;
    user.results_returned = snapshot.results_returned;
    user.chunks_embedded = snapshot.chunks_embedded;
    user.chunks_reused = snapshot.chunks_reused;
    user.chunks_redundant = snapshot.chunks_redundant;
    user.files_skipped = snapshot.files_skipped;

    user.stale_sync_writes = snapshot.stale_sync_writes;
    user.stale_deferred = snapshot.stale_deferred;
    user.clone_ns = snapshot.clone_ns;
    user.clone_count = snapshot.clone_count;
    user.upcall_wait_ns = snapshot.upcall_wait_ns;
    user.upcall_count = snapshot.upcall_count;
    user.persist_ns = snapshot.persist_ns;
    user.persist_count = snapshot.persist_count;
    user.build_ns = snapshot.build_ns;
    user.build_count = snapshot.build_count;
    user.publish_ns = snapshot.publish_ns;
    user.publish_count = snapshot.publish_count;

    if (copy_to_user(argument, &user, sizeof(user)))
        return -EFAULT;

    if (flags & VECFS_SHIELD_STATS_F_RESET)
        vecfs_shield_stats_reset(sb);

    return 0;
}

static long vecfs_ioctl_shield_set_strategy(
    struct super_block *sb,
    void __user *argument)
{
    struct vecfs_ioctl_shield_strategy user;

    if (copy_from_user(&user, argument, sizeof(user)))
        return -EFAULT;

    return vecfs_shield_set_strategy(
        sb, user.strategy_id, user.flags, user.chunk_size);
}

static long vecfs_ioctl_shield_task_begin(
    struct file *file,
    void __user *argument)
{
    struct vecfs_ioctl_shield_task_begin user;
    int ret;

    if (copy_from_user(&user, argument, sizeof(user)))
        return -EFAULT;

    ret = vecfs_shield_task_begin(
        file_inode(file), &user.target_version);
    if (ret)
        return ret;

    if (copy_to_user(argument, &user, sizeof(user)))
        return -EFAULT;

    return 0;
}

static long vecfs_ioctl_shield_task_end(
    struct file *file,
    void __user *argument)
{
    struct vecfs_ioctl_shield_task_end user;
    bool expired;
    int ret;

    if (copy_from_user(&user, argument, sizeof(user)))
        return -EFAULT;

    if (user.flags)
        return -EINVAL;

    ret = vecfs_shield_task_end(
        file_inode(file), user.target_version, &expired);
    if (ret)
        return ret;

    user.expired = expired ? 1 : 0;

    if (copy_to_user(argument, &user, sizeof(user)))
        return -EFAULT;

    return 0;
}

static long vecfs_ioctl_shield_file_state(
    struct file *file,
    void __user *argument)
{
    struct vecfs_ioctl_shield_file_state user;
    struct vecfs_shield_file_state_info info;
    int ret;

    if (copy_from_user(&user, argument, sizeof(user)))
        return -EFAULT;

    ret = vecfs_shield_file_state_get(file_inode(file), &info);
    if (ret)
        return ret;

    memset(&user, 0, sizeof(user));

    user.state = info.state;
    user.shield_version = info.shield_version;
    user.target_version = info.target_version;
    user.semantic_version = info.semantic_version;
    user.first_change = info.first_change;
    user.last_change = info.last_change;

    if (copy_to_user(argument, &user, sizeof(user)))
        return -EFAULT;

    return 0;
}

long vecfs_ioctl(
    struct file *file,
    unsigned int command,
    unsigned long argument)
{
    struct inode *inode;
    void __user *user_argument =
        (void __user *)argument;

    if (!file)
        return -EINVAL;

    inode = file_inode(file);

    if (!inode ||
        !inode->i_sb ||
        inode->i_sb->s_magic !=
            VECFS_SUPER_MAGIC)
        return -ENOTTY;

    switch (command) {
    case VECFS_IOC_GET_INFO:
        return vecfs_ioctl_get_info(
            inode->i_sb,
            user_argument);

    case VECFS_IOC_GET_RUNTIME:
        return vecfs_ioctl_get_runtime(
            inode->i_sb,
            user_argument);

    case VECFS_IOC_SET_RUNTIME:
        return vecfs_ioctl_set_runtime(
            inode->i_sb,
            user_argument);

    case VECFS_IOC_VECTOR_ADD:
        return vecfs_ioctl_vector_add(
            file,
            user_argument);

    case VECFS_IOC_VECTOR_UPDATE:
        return vecfs_ioctl_vector_update(
            file,
            user_argument);

    case VECFS_IOC_VECTOR_DELETE:
        return vecfs_ioctl_vector_delete(
            file,
            user_argument,
            false);

    case VECFS_IOC_VECTOR_UNDELETE:
        return vecfs_ioctl_vector_delete(
            file,
            user_argument,
            true);

    case VECFS_IOC_VECTOR_GET:
        return vecfs_ioctl_vector_get(
            file,
            user_argument);

    case VECFS_IOC_FILE_VECTOR_ADD:
        return vecfs_ioctl_file_vector_add(
            file,
            user_argument);

    case VECFS_IOC_FILE_VECTOR_UPDATE:
        return vecfs_ioctl_file_vector_update(
            file,
            user_argument);

    case VECFS_IOC_FILE_VECTOR_DELETE:
        return vecfs_ioctl_file_vector_delete(
            file,
            user_argument);

    case VECFS_IOC_FILE_VECTOR_GET:
        return vecfs_ioctl_file_vector_get(
            file,
            user_argument);

    case VECFS_IOC_FILE_VECTOR_LIST:
        return vecfs_ioctl_file_vector_list(
            file,
            user_argument);

    case VECFS_IOC_KNN_SEARCH:
        return vecfs_ioctl_knn(
            file,
            user_argument);

    case VECFS_IOC_KNN_SEARCH_BATCH:
        return vecfs_ioctl_knn_batch(
            file,
            user_argument);

    case VECFS_IOC_ANN_VALIDATE:
        return vecfs_ioctl_ann_validate(
            file,
            user_argument);

    case VECFS_IOC_ANN_REBUILD:
        return vecfs_ioctl_ann_rebuild(
            file,
            user_argument);

    case VECFS_IOC_EXPORT_INFO:
        return vecfs_ioctl_export_info(
            file,
            user_argument);

    case VECFS_IOC_EXPORT_NODE:
        return vecfs_ioctl_export_node(
            file,
            user_argument);

    case VECFS_IOC_IMPORT_BEGIN:
        return vecfs_ioctl_import_begin(
            file,
            user_argument);

    case VECFS_IOC_IMPORT_NODE:
        return vecfs_ioctl_import_node(
            file,
            user_argument);

    case VECFS_IOC_IMPORT_COMMIT:
        return vecfs_ioctl_import_commit(
            file,
            user_argument);

    case VECFS_IOC_IMPORT_ABORT:
        return vecfs_ioctl_import_abort(
            file,
            user_argument);
    case VECFS_IOC_ANN_REBUILD_STATUS:
        return vecfs_ioctl_ann_rebuild_status(
            file,
            user_argument);

    case VECFS_IOC_ANN_REBUILD_CANCEL:
        return vecfs_ioctl_ann_rebuild_cancel(
            file,
            user_argument);

    case VECFS_IOC_SHIELD_SET_MODE:
        return vecfs_ioctl_shield_set_mode(
            inode->i_sb,
            user_argument);

    case VECFS_IOC_SHIELD_GET_STATS:
        return vecfs_ioctl_shield_get_stats(
            inode->i_sb,
            user_argument);

    case VECFS_IOC_SHIELD_TASK_BEGIN:
        return vecfs_ioctl_shield_task_begin(
            file,
            user_argument);

    case VECFS_IOC_SHIELD_TASK_END:
        return vecfs_ioctl_shield_task_end(
            file,
            user_argument);

    case VECFS_IOC_SHIELD_FILE_STATE:
        return vecfs_ioctl_shield_file_state(
            file,
            user_argument);

    case VECFS_IOC_SHIELD_SET_STRATEGY:
        return vecfs_ioctl_shield_set_strategy(
            inode->i_sb,
            user_argument);

    default:
        return -ENOTTY;
    }
}



#ifdef CONFIG_COMPAT
long vecfs_compat_ioctl(
    struct file *file,
    unsigned int command,
    unsigned long argument)
{
    /*
     * UAPI 只使用固定宽度整数和 __u64 用户指针，
     * 32/64 位 ABI 布局一致。
     */
    return vecfs_ioctl(
        file,
        command,
        (unsigned long)compat_ptr(argument));
}

#endif /* CONFIG_COMPAT */

static void vecfs_ioctl_fill_vector_info(
    const struct vecfs_ann_object_info *source,
    struct vecfs_ioctl_vector_info *destination)
{
    memset(destination, 0, sizeof(*destination));

    destination->external_id =
        source->external_id;

    destination->ino =
        source->owner_ino;

    destination->inode_generation =
        source->owner_generation;

    destination->vector_ordinal =
        source->vector_ordinal;

    destination->content_offset =
        source->content_offset;

    destination->content_length =
        source->content_length;

    destination->node_id =
        source->node_id;

    destination->state =
        source->state;

    destination->level =
        source->level;

    destination->owner_type =
        source->owner_type;

    destination->flags =
        source->flags;
}

static int vecfs_ioctl_copy_vector_out(
    u64 user_pointer,
    u32 capacity,
    const struct vecfs_ann_vector_record *record,
    u32 *output_bytes)
{
    if (!record || !output_bytes)
        return -EINVAL;

    *output_bytes =
        record->vector_bytes;

    /*
     * pointer == 0 表示只查询所需大小和 metadata。
     */
    if (!user_pointer)
        return 0;

    if (capacity <
        record->vector_bytes)
        return -ENOSPC;

    if (copy_to_user(
            u64_to_user_ptr(user_pointer),
            record->vector,
            record->vector_bytes))
        return -EFAULT;

    return 0;
}

static long vecfs_ioctl_vector_get(
    struct file *file,
    void __user *argument)
{
    struct vecfs_ioctl_vector_get user;
    struct vecfs_ann_vector_record record;
    bool include_deleted;
    int ret;

    if (copy_from_user(
            &user,
            argument,
            sizeof(user)))
        return -EFAULT;

    if (user.flags &
        ~VECFS_VECTOR_GET_F_INCLUDE_DELETED)
        return -EINVAL;

    include_deleted =
        user.flags &
        VECFS_VECTOR_GET_F_INCLUDE_DELETED;

    memset(&record, 0, sizeof(record));

    /*
     * standalone API 禁止读取 FILE owner 节点。
     * IMPORTED 节点按 standalone API 管理。
     */
    ret = vecfs_ann_get_by_label(
        file_inode(file)->i_sb,
        user.external_id,
        include_deleted,
        false,
        &record);
    if (ret)
        return ret;

    ret = vecfs_ioctl_copy_vector_out(
        user.vector_ptr,
        user.vector_capacity,
        &record,
        &user.vector_bytes);
    if (ret)
        goto out;

    vecfs_ioctl_fill_vector_info(
        &record.info,
        &user.info);

    if (copy_to_user(
            argument,
            &user,
            sizeof(user))) {
        ret = -EFAULT;
        goto out;
    }

    ret = 0;

out:
    vecfs_ann_vector_record_destroy(
        &record);

    return ret;
}

static long vecfs_ioctl_file_vector_get(
    struct file *file,
    void __user *argument)
{
    struct inode *inode =
        file_inode(file);

    struct vecfs_ioctl_file_vector_get user;
    struct vecfs_ann_vector_record record;
    bool include_deleted;
    int ret;

    if (!S_ISREG(inode->i_mode))
        return -EINVAL;

    if (copy_from_user(
            &user,
            argument,
            sizeof(user)))
        return -EFAULT;

    if (user.flags &
        ~VECFS_VECTOR_GET_F_INCLUDE_DELETED)
        return -EINVAL;

    include_deleted =
        user.flags &
        VECFS_VECTOR_GET_F_INCLUDE_DELETED;

    memset(&record, 0, sizeof(record));

    ret = vecfs_ann_get_file_vector(
        inode,
        user.ordinal,
        include_deleted,
        &record);
    if (ret)
        return ret;

    ret = vecfs_ioctl_copy_vector_out(
        user.vector_ptr,
        user.vector_capacity,
        &record,
        &user.vector_bytes);
    if (ret)
        goto out;

    vecfs_ioctl_fill_vector_info(
        &record.info,
        &user.info);

    if (copy_to_user(
            argument,
            &user,
            sizeof(user))) {
        ret = -EFAULT;
        goto out;
    }

    ret = 0;

out:
    vecfs_ann_vector_record_destroy(
        &record);

    return ret;
}

static long vecfs_ioctl_export_info(
    struct file *file,
    void __user *argument)
{
    struct super_block *sb =
        file_inode(file)->i_sb;

    struct vecfs_sb_info *sbi =
        VECFS_SB(sb);

    struct vecfs_ioctl_exchange_info info;

    u64 node_count;
    u32 capacity;
    u32 entry_node;
    u16 max_level;
    u64 epoch;
    int ret;

    memset(&info, 0, sizeof(info));

    ret = vecfs_ann_export_info(
        sb,
        &node_count,
        &capacity,
        &entry_node,
        &max_level,
        &epoch);
    if (ret)
        return ret;

    info.exchange_major =
        VECFS_EXCHANGE_MAJOR;

    info.exchange_minor =
        VECFS_EXCHANGE_MINOR;

    info.element_type =
        sbi->s_ann.config.element_type;

    info.distance_type =
        sbi->s_ann.config.distance_type;

    info.dimension =
        sbi->s_ann.config.dimension;

    info.m =
        sbi->s_ann.config.m;

    info.m0 =
        sbi->s_ann.config.m0;

    info.max_levels =
        sbi->s_ann.config.max_levels;

    info.current_max_level =
        max_level;

    info.ef_construction =
        sbi->s_ann.config.ef_construction;

    info.default_ef_search =
        atomic_read(
            &sbi->s_ann_runtime.default_ef_search);

    info.node_count =
        node_count;

    info.node_capacity =
        capacity;

    info.entry_node_id =
        entry_node;

    info.epoch =
        epoch;

    info.random_seed =
        sbi->s_ann.config.random_seed;

    info.vector_bytes =
        sbi->s_ann.config.vector_bytes;

    if (copy_to_user(
            argument,
            &info,
            sizeof(info)))
        return -EFAULT;

    return 0;
}

static long vecfs_ioctl_export_node(
    struct file *file,
    void __user *argument)
{
    struct vecfs_ioctl_export_node user;
    struct vecfs_ann_export_node node;
    struct vecfs_sb_info *sbi =
        VECFS_SB(file_inode(file)->i_sb);

    u32 upper_counts_required;
    u32 upper_links_required;
    int ret;

    if (copy_from_user(
            &user,
            argument,
            sizeof(user)))
        return -EFAULT;

    if (user.flags &
        ~(VECFS_EXPORT_F_INCLUDE_VECTORS |
          VECFS_EXPORT_F_INCLUDE_GRAPH))
        return -EINVAL;

    memset(&node, 0, sizeof(node));

    ret = vecfs_ann_export_node(
        file_inode(file)->i_sb,
        user.node_id,
        &node);
    if (ret)
        return ret;

    upper_counts_required =
        node.level;

    upper_links_required =
        (u32)node.level *
        sbi->s_ann.config.m;

    user.external_id =
        node.external_id;

    user.level =
        node.level;

    user.level0_count =
        node.level0_count;

    user.upper_level_count =
        node.level;

    user.vector_bytes =
        node.vector_bytes;

    user.level0_count_out =
        node.level0_count;

    user.upper_counts_count_out =
        upper_counts_required;

    user.upper_links_count_out =
        upper_links_required;

    if (user.flags &
        VECFS_EXPORT_F_INCLUDE_VECTORS) {
        if (!user.vector_ptr ||
            user.vector_capacity <
                node.vector_bytes) {
            ret = -ENOSPC;
            goto out;
        }

        if (copy_to_user(
                u64_to_user_ptr(
                    user.vector_ptr),
                node.vector,
                node.vector_bytes)) {
            ret = -EFAULT;
            goto out;
        }
    }

    if (user.flags &
        VECFS_EXPORT_F_INCLUDE_GRAPH) {
        if (node.level0_count) {
            if (!user.level0_links_ptr ||
                user.level0_capacity <
                    node.level0_count) {
                ret = -ENOSPC;
                goto out;
            }

            if (copy_to_user(
                    u64_to_user_ptr(
                        user.level0_links_ptr),
                    node.level0_links,
                    (size_t)node.level0_count *
                        sizeof(u32))) {
                ret = -EFAULT;
                goto out;
            }
        }

        if (upper_counts_required) {
            if (!user.upper_counts_ptr ||
                user.upper_counts_capacity <
                    upper_counts_required ||
                !user.upper_links_ptr ||
                user.upper_links_capacity <
                    upper_links_required) {
                ret = -ENOSPC;
                goto out;
            }

            if (copy_to_user(
                    u64_to_user_ptr(
                        user.upper_counts_ptr),
                    node.upper_counts,
                    (size_t)upper_counts_required *
                        sizeof(u16))) {
                ret = -EFAULT;
                goto out;
            }

            if (copy_to_user(
                    u64_to_user_ptr(
                        user.upper_links_ptr),
                    node.upper_links,
                    (size_t)upper_links_required *
                        sizeof(u32))) {
                ret = -EFAULT;
                goto out;
            }
        }
    }

    if (copy_to_user(
            argument,
            &user,
            sizeof(user))) {
        ret = -EFAULT;
        goto out;
    }

    ret = 0;

out:
    kvfree(
        (void *)node.upper_links);

    kvfree(
        (void *)node.upper_counts);

    kvfree(
        (void *)node.level0_links);

    kvfree(
        (void *)node.vector);

    return ret;
}

static long vecfs_ioctl_import_begin(
    struct file *file,
    void __user *argument)
{
    struct vecfs_ioctl_import_begin user;
    struct vecfs_ann_import_begin request;
    u64 transaction_id;
    int ret;

    if (copy_from_user(
            &user,
            argument,
            sizeof(user)))
        return -EFAULT;

    if (user.info.exchange_major !=
        VECFS_EXCHANGE_MAJOR)
        return -EPROTONOSUPPORT;

    if (user.info.exchange_minor >
        VECFS_EXCHANGE_MINOR)
        return -EPROTONOSUPPORT;

    memset(&request, 0, sizeof(request));

    request.element_type =
        user.info.element_type;

    request.distance_type =
        user.info.distance_type;

    request.dimension =
        user.info.dimension;

    request.m =
        user.info.m;

    request.m0 =
        user.info.m0;

    request.max_levels =
        user.info.max_levels;

    request.expected_nodes =
        user.expected_nodes;

    request.node_capacity =
        user.node_capacity;

    request.flags =
        user.flags;

    ret = vecfs_ann_import_begin(
        file_inode(file)->i_sb,
        &request,
        &transaction_id);
    if (ret)
        return ret;

    user.transaction_id =
        transaction_id;

    if (copy_to_user(
            argument,
            &user,
            sizeof(user))) {
        vecfs_ann_import_abort(
            file_inode(file)->i_sb,
            transaction_id);

        return -EFAULT;
    }

    return 0;
}

static long vecfs_ioctl_import_node(
    struct file *file,
    void __user *argument)
{
    struct vecfs_ioctl_import_node user;
    struct vecfs_ann_import_node node;
    struct vecfs_sb_info *sbi =
        VECFS_SB(file_inode(file)->i_sb);

    void *vector = NULL;
    u32 *level0 = NULL;
    u16 *upper_counts = NULL;
    u32 *upper_links = NULL;

    size_t upper_slots;
    int ret;

    if (copy_from_user(
            &user,
            argument,
            sizeof(user)))
        return -EFAULT;

    if (user.vector_bytes !=
        sbi->s_ann.config.vector_bytes)
        return -EINVAL;

    if (user.level0_count >
        sbi->s_ann.config.m0)
        return -EINVAL;

    if (user.level >=
        sbi->s_ann.config.max_levels)
        return -EINVAL;

    if (user.upper_level_count !=
        user.level)
        return -EINVAL;

    vector = memdup_user(
        u64_to_user_ptr(
            user.vector_ptr),
        user.vector_bytes);

    if (IS_ERR(vector))
        return PTR_ERR(vector);

    if (user.level0_count) {
        level0 = memdup_user(
            u64_to_user_ptr(
                user.level0_links_ptr),
            (size_t)user.level0_count *
                sizeof(u32));

        if (IS_ERR(level0)) {
            ret = PTR_ERR(level0);
            level0 = NULL;
            goto out;
        }
    }

    if (user.level) {
        upper_counts = memdup_user(
            u64_to_user_ptr(
                user.upper_counts_ptr),
            (size_t)user.level *
                sizeof(u16));

        if (IS_ERR(upper_counts)) {
            ret = PTR_ERR(upper_counts);
            upper_counts = NULL;
            goto out;
        }

        upper_slots =
            (size_t)user.level *
            sbi->s_ann.config.m;

        upper_links = memdup_user(
            u64_to_user_ptr(
                user.upper_links_ptr),
            upper_slots * sizeof(u32));

        if (IS_ERR(upper_links)) {
            ret = PTR_ERR(upper_links);
            upper_links = NULL;
            goto out;
        }
    }

    memset(&node, 0, sizeof(node));

    node.node_id =
        user.node_id;

    node.external_id =
        user.external_id;

    node.level =
        user.level;

    node.vector =
        vector;

    node.vector_bytes =
        user.vector_bytes;

    node.level0_links =
        level0;

    node.level0_count =
        user.level0_count;

    node.upper_counts =
        upper_counts;

    node.upper_links =
        upper_links;

    ret = vecfs_ann_import_node(
        file_inode(file)->i_sb,
        user.transaction_id,
        &node);

out:
    kvfree(upper_links);
    kvfree(upper_counts);
    kvfree(level0);
    kvfree(vector);

    return ret;
}

static long vecfs_ioctl_import_commit(
    struct file *file,
    void __user *argument)
{
    struct vecfs_ioctl_import_commit user;
    u64 imported_nodes = 0;
    u64 new_epoch = 0;
    int ret;

    if (copy_from_user(
            &user,
            argument,
            sizeof(user)))
        return -EFAULT;

    ret = vecfs_ann_import_commit(
        file_inode(file)->i_sb,
        user.transaction_id,
        user.entry_node_id,
        user.current_max_level,
        &imported_nodes,
        &new_epoch);

    user.status = ret;
    user.imported_nodes =
        imported_nodes;
    user.new_epoch =
        new_epoch;

    if (copy_to_user(
            argument,
            &user,
            sizeof(user)))
        return -EFAULT;

    return ret;
}

static long vecfs_ioctl_import_abort(
    struct file *file,
    void __user *argument)
{
    struct vecfs_ioctl_import_abort user;

    if (copy_from_user(
            &user,
            argument,
            sizeof(user)))
        return -EFAULT;

    return vecfs_ann_import_abort(
        file_inode(file)->i_sb,
        user.transaction_id);
}
