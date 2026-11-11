// SPDX-License-Identifier: GPL-2.0

#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>

#include "../include/vecfs.h"
#include "../include/vecfs_ann_search.h"

int vecfs_ann_search_context_init(
    struct vecfs_ann_search_context *context,
    u32 node_capacity,
    u32 result_capacity)
{
    int ret;

    if (!context || !node_capacity || !result_capacity)
        return -EINVAL;

    memset(context, 0, sizeof(*context));
    INIT_LIST_HEAD(&context->pool_node);

    ret = vecfs_ann_visited_init(
        &context->visited,
        node_capacity,
        GFP_KERNEL);
    if (ret)
        return ret;

    ret = vecfs_ann_heap_init(
        &context->candidates,
        node_capacity,
        VECFS_ANN_HEAP_MIN,
        GFP_KERNEL);
    if (ret)
        goto fail_visited;

    ret = vecfs_ann_heap_init(
        &context->results,
        result_capacity,
        VECFS_ANN_HEAP_MAX,
        GFP_KERNEL);
    if (ret)
        goto fail_candidates;

    context->sorted_results = kvmalloc_array(
        result_capacity,
        sizeof(*context->sorted_results),
        GFP_KERNEL);
    if (!context->sorted_results) {
        ret = -ENOMEM;
        goto fail_results;
    }

    context->node_capacity = node_capacity;
    context->result_capacity = result_capacity;
    context->sorted_capacity = result_capacity;

    return 0;

fail_results:
    vecfs_ann_heap_destroy(&context->results);
fail_candidates:
    vecfs_ann_heap_destroy(&context->candidates);
fail_visited:
    vecfs_ann_visited_destroy(&context->visited);
    return ret;
}

void vecfs_ann_search_context_destroy(
    struct vecfs_ann_search_context *context)
{
    if (!context)
        return;

    kvfree(context->sorted_results);

    vecfs_ann_heap_destroy(&context->results);
    vecfs_ann_heap_destroy(&context->candidates);
    vecfs_ann_visited_destroy(&context->visited);

    memset(context, 0, sizeof(*context));
}

int vecfs_ann_search_context_prepare(
    struct vecfs_ann_search_context *context,
    u32 node_capacity,
    u32 result_capacity)
{
    struct vecfs_ann_search_context replacement;
    int ret;

    if (!context)
        return -EINVAL;

    if (context->node_capacity >= node_capacity &&
        context->result_capacity >= result_capacity)
        return 0;

    node_capacity = max(
        node_capacity, context->node_capacity);

    result_capacity = max(
        result_capacity, context->result_capacity);

    ret = vecfs_ann_search_context_init(
        &replacement,
        node_capacity,
        result_capacity);
    if (ret)
        return ret;

    vecfs_ann_search_context_destroy(context);

    *context = replacement;
    INIT_LIST_HEAD(&context->pool_node);

    return 0;
}

void vecfs_ann_search_context_reset(
    struct vecfs_ann_search_context *context)
{
    if (!context)
        return;

    vecfs_ann_visited_reset(&context->visited);
    vecfs_ann_heap_reset(&context->candidates);
    vecfs_ann_heap_reset(&context->results);
}

int vecfs_ann_search_pool_init(
    struct super_block *sb,
    u32 maximum_contexts)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ann_search_pool *pool;

    if (!sbi || !maximum_contexts)
        return -EINVAL;

    pool = kzalloc(sizeof(*pool), GFP_KERNEL);
    if (!pool)
        return -ENOMEM;

    mutex_init(&pool->lock);
    init_waitqueue_head(&pool->wait);
    INIT_LIST_HEAD(&pool->free_contexts);

    pool->maximum_contexts = maximum_contexts;

    sbi->s_ann_search_pool = pool;
    return 0;
}

void vecfs_ann_search_pool_destroy(
    struct super_block *sb)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ann_search_pool *pool;
    struct vecfs_ann_search_context *context;
    struct vecfs_ann_search_context *temporary;

    if (!sbi)
        return;

    pool = sbi->s_ann_search_pool;
    if (!pool)
        return;

    mutex_lock(&pool->lock);
    pool->shutting_down = true;
    mutex_unlock(&pool->lock);

    wake_up_all(&pool->wait);

    mutex_lock(&pool->lock);

    list_for_each_entry_safe(
        context,
        temporary,
        &pool->free_contexts,
        pool_node) {
        list_del_init(&context->pool_node);
        vecfs_ann_search_context_destroy(context);
        kfree(context);
    }

    mutex_unlock(&pool->lock);

    mutex_destroy(&pool->lock);
    sbi->s_ann_search_pool = NULL;
    kfree(pool);
}

int vecfs_ann_search_context_get(
    struct super_block *sb,
    u32 node_capacity,
    u32 result_capacity,
    struct vecfs_ann_search_context **result)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ann_search_pool *pool;
    struct vecfs_ann_search_context *context;
    int ret;

    if (!sbi || !result)
        return -EINVAL;

    pool = sbi->s_ann_search_pool;
    if (!pool)
        return -ENODEV;

retry:
    mutex_lock(&pool->lock);

    if (pool->shutting_down) {
        mutex_unlock(&pool->lock);
        return -ESHUTDOWN;
    }

    if (!list_empty(&pool->free_contexts)) {
        context = list_first_entry(
            &pool->free_contexts,
            struct vecfs_ann_search_context,
            pool_node);

        list_del_init(&context->pool_node);
        pool->free_count--;

        mutex_unlock(&pool->lock);

        ret = vecfs_ann_search_context_prepare(
            context,
            node_capacity,
            result_capacity);
        if (ret) {
            vecfs_ann_search_context_put(
                sb, context);
            return ret;
        }

        vecfs_ann_search_context_reset(context);
        *result = context;
        return 0;
    }

    if (pool->total_contexts <
        pool->maximum_contexts) {
        pool->total_contexts++;
        mutex_unlock(&pool->lock);

        context = kzalloc(
            sizeof(*context), GFP_KERNEL);
        if (!context) {
            ret = -ENOMEM;
            goto undo_count;
        }

        ret = vecfs_ann_search_context_init(
            context,
            node_capacity,
            result_capacity);
        if (ret) {
            kfree(context);
            goto undo_count;
        }

        vecfs_ann_search_context_reset(context);
        *result = context;
        return 0;
    }

    mutex_unlock(&pool->lock);

    ret = wait_event_interruptible(
        pool->wait,
        READ_ONCE(pool->shutting_down) ||
        READ_ONCE(pool->free_count));

    if (ret)
        return ret;

    goto retry;

undo_count:
    mutex_lock(&pool->lock);
    pool->total_contexts--;
    mutex_unlock(&pool->lock);
    wake_up(&pool->wait);
    return ret;
}

void vecfs_ann_search_context_put(
    struct super_block *sb,
    struct vecfs_ann_search_context *context)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ann_search_pool *pool;

    if (!sbi || !context)
        return;

    pool = sbi->s_ann_search_pool;
    if (!pool) {
        vecfs_ann_search_context_destroy(context);
        kfree(context);
        return;
    }

    mutex_lock(&pool->lock);

    if (pool->shutting_down) {
        if (pool->total_contexts)
            pool->total_contexts--;

        mutex_unlock(&pool->lock);

        vecfs_ann_search_context_destroy(context);
        kfree(context);
        wake_up_all(&pool->wait);
        return;
    }

    list_add(
        &context->pool_node,
        &pool->free_contexts);

    pool->free_count++;

    mutex_unlock(&pool->lock);
    wake_up(&pool->wait);
}
