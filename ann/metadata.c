#include <linux/buffer_head.h>
#include <linux/crc32.h>
#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/kernel.h>
#include <linux/math64.h>
#include <linux/slab.h>

#include "../include/vecfs.h"
#include "../include/vecfs_ann_meta.h"

#define VECFS_ANN_ALIGN 64ULL

static u64 vecfs_align_u64(u64 value, u64 alignment)
{
    return ALIGN(value, alignment);
}

static u32 vecfs_element_size(u16 type)
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

u64 vecfs_ann_calculate_object_stride(
    u32 dimension,
    u16 element_type,
    u16 m,
    u16 m0,
    u16 max_levels)
{
    u64 off;
    u64 vector_bytes;
    u64 upper_levels;
    u32 elem_size;

    elem_size = vecfs_element_size(element_type);
    if (!elem_size || !dimension || !max_levels)
        return 0;

    vector_bytes = (u64)dimension * elem_size;
    upper_levels = max_levels - 1;

    /*
     * Object layout:
     *
     * [object header]
     * [padding to 64]
     * [level-0 neighbor IDs: m0 * u32]
     * [padding to 64]
     * [vector payload]
     * [padding to 64]
     * [upper neighbor counts: (levels - 1) * u16]
     * [padding to 64]
     * [upper neighbor IDs: (levels - 1) * m * u32]
     * [padding to 64]
     */
    off = sizeof(struct vecfs_ann_object_disk);
    off = vecfs_align_u64(off, VECFS_ANN_ALIGN);

    off += (u64)m0 * sizeof(__le32);
    off = vecfs_align_u64(off, VECFS_ANN_ALIGN);

    off += vector_bytes;
    off = vecfs_align_u64(off, VECFS_ANN_ALIGN);

    off += upper_levels * sizeof(__le16);
    off = vecfs_align_u64(off, VECFS_ANN_ALIGN);

    off += upper_levels * m * sizeof(__le32);
    off = vecfs_align_u64(off, VECFS_ANN_ALIGN);

    return off;
}

int vecfs_ann_validate_config(const struct vecfs_ann_config *cfg)
{
    u32 elem_size;
    u64 expected_stride;

    if (!cfg)
        return -EINVAL;

    if (!cfg->dimension || cfg->dimension > VECFS_ANN_MAX_DIM)
        return -EINVAL;

    /*
     * V2 第一版只开放 FP32。
     */
    if (cfg->element_type != VECFS_ANN_ELEM_FP32)
        return -EOPNOTSUPP;

    if (cfg->distance_type != VECFS_ANN_DIST_L2 &&
        cfg->distance_type != VECFS_ANN_DIST_IP)
        return -EINVAL;

    if (cfg->m < 2 || cfg->m > VECFS_ANN_MAX_M)
        return -EINVAL;

    if (cfg->m0 < cfg->m || cfg->m0 > VECFS_ANN_MAX_M0)
        return -EINVAL;

    if (!cfg->max_levels ||
        cfg->max_levels > VECFS_ANN_MAX_LEVELS)
        return -EINVAL;

    if (cfg->current_max_level >= cfg->max_levels &&
        cfg->current_max_level != 0)
        return -EINVAL;

    if (cfg->ef_construction < cfg->m0 ||
        cfg->ef_construction > 4096)
        return -EINVAL;

    if (!cfg->default_ef_search ||
        cfg->default_ef_search > 65535)
        return -EINVAL;

    elem_size = vecfs_element_size(cfg->element_type);
    if (!elem_size)
        return -EINVAL;

    if (cfg->vector_bytes != (u64)cfg->dimension * elem_size)
        return -EINVAL;

    expected_stride = vecfs_ann_calculate_object_stride(
        cfg->dimension,
        cfg->element_type,
        cfg->m,
        cfg->m0,
        cfg->max_levels);

    if (!expected_stride || cfg->object_stride != expected_stride)
        return -EINVAL;

    return 0;
}

static u32 vecfs_ann_meta_crc(const struct vecfs_ann_meta_disk *disk)
{
    struct vecfs_ann_meta_disk *copy;
    u32 crc;

    copy = kmemdup(disk, sizeof(*copy), GFP_KERNEL);
    if (!copy)
        return 0;

    copy->checksum = 0;
    crc = crc32_le(~0U, (const u8 *)copy, sizeof(*copy)) ^ ~0U;

    kfree(copy);
    return crc;
}

static bool vecfs_ann_meta_disk_valid(
    const struct vecfs_ann_meta_disk *disk,
    struct vecfs_ann_config *cfg)
{
    u32 stored_crc;
    u32 calculated_crc;

    if (le32_to_cpu(disk->magic) != VECFS_ANN_META_MAGIC)
        return false;

    if (le16_to_cpu(disk->major) != VECFS_ANN_META_MAJOR)
        return false;

    if (le32_to_cpu(disk->header_size) != sizeof(*disk))
        return false;

    stored_crc = le32_to_cpu(disk->checksum);
    calculated_crc = vecfs_ann_meta_crc(disk);

    if (!calculated_crc || stored_crc != calculated_crc)
        return false;

    memset(cfg, 0, sizeof(*cfg));

    cfg->dimension = le32_to_cpu(disk->dimension);
    cfg->element_type = le16_to_cpu(disk->element_type);
    cfg->distance_type = le16_to_cpu(disk->distance_type);
    cfg->m = le16_to_cpu(disk->m);
    cfg->m0 = le16_to_cpu(disk->m0);
    cfg->max_levels = le16_to_cpu(disk->max_levels);
    cfg->current_max_level =
        le16_to_cpu(disk->current_max_level);
    cfg->ef_construction =
        le32_to_cpu(disk->ef_construction);
    cfg->default_ef_search =
        le32_to_cpu(disk->default_ef_search);
    cfg->random_seed = le64_to_cpu(disk->random_seed);
    cfg->level_mult_q32 =
        le64_to_cpu(disk->level_mult_q32);
    cfg->object_stride =
        le64_to_cpu(disk->object_stride);
    cfg->vector_bytes =
        le64_to_cpu(disk->vector_bytes);

    return vecfs_ann_validate_config(cfg) == 0;
}

static int vecfs_ann_read_copy(
    struct super_block *sb,
    sector_t block,
    struct vecfs_ann_meta_disk **out)
{
    struct buffer_head *bh;
    struct vecfs_ann_meta_disk *copy;

    bh = sb_bread(sb, block);
    if (!bh)
        return -EIO;

    copy = kmemdup(bh->b_data, sizeof(*copy), GFP_KERNEL);
    brelse(bh);

    if (!copy)
        return -ENOMEM;

    *out = copy;
    return 0;
}

static void vecfs_ann_meta_from_disk(
    struct vecfs_ann_meta *meta,
    const struct vecfs_ann_meta_disk *disk,
    const struct vecfs_ann_config *cfg,
    u8 copy)
{
    meta->config = *cfg;

    meta->sequence = le64_to_cpu(disk->sequence);
    meta->epoch = le64_to_cpu(disk->epoch);

    meta->state = le32_to_cpu(disk->state);
    meta->flags = le32_to_cpu(disk->flags);

    meta->live_nodes = le64_to_cpu(disk->live_nodes);
    meta->deleted_nodes = le64_to_cpu(disk->deleted_nodes);
    meta->next_node_id = le64_to_cpu(disk->next_node_id);
    meta->next_external_id =
        le64_to_cpu(disk->next_external_id);

    meta->entry_node_id =
        le32_to_cpu(disk->entry_node_id);
    meta->active_arena_ino =
        le32_to_cpu(disk->active_arena_ino);

    meta->rebuild_arena_ino =
        le32_to_cpu(disk->rebuild_arena_ino);
    meta->rebuild_state =
        le32_to_cpu(disk->rebuild_state);

    meta->committed_txid =
        le64_to_cpu(disk->committed_txid);

    meta->active_copy = copy;
    meta->dirty = false;
}

int vecfs_ann_meta_load(struct super_block *sb)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ann_meta_disk *a = NULL;
    struct vecfs_ann_meta_disk *b = NULL;
    struct vecfs_ann_config cfg_a;
    struct vecfs_ann_config cfg_b;
    bool valid_a = false;
    bool valid_b = false;
    int ret_a;
    int ret_b;

    mutex_init(&sbi->s_ann.lock);

    ret_a = vecfs_ann_read_copy(
        sb, sbi->s_ann_meta_block[0], &a);
    ret_b = vecfs_ann_read_copy(
        sb, sbi->s_ann_meta_block[1], &b);

    if (!ret_a)
        valid_a = vecfs_ann_meta_disk_valid(a, &cfg_a);

    if (!ret_b)
        valid_b = vecfs_ann_meta_disk_valid(b, &cfg_b);

    if (!valid_a && !valid_b) {
        pr_err("vecfs: both ANN metadata copies are invalid\n");
        kfree(a);
        kfree(b);
        mutex_destroy(&sbi->s_ann.lock);
        return -EUCLEAN;
    }

    if (valid_a && valid_b) {
        u64 seq_a = le64_to_cpu(a->sequence);
        u64 seq_b = le64_to_cpu(b->sequence);

        if (seq_b > seq_a)
            vecfs_ann_meta_from_disk(
                &sbi->s_ann, b, &cfg_b, 1);
        else
            vecfs_ann_meta_from_disk(
                &sbi->s_ann, a, &cfg_a, 0);
    } else if (valid_a) {
        vecfs_ann_meta_from_disk(
            &sbi->s_ann, a, &cfg_a, 0);
    } else {
        vecfs_ann_meta_from_disk(
            &sbi->s_ann, b, &cfg_b, 1);
    }

    pr_info("vecfs: ANN metadata copy=%c seq=%llu epoch=%llu "
            "dim=%u type=%u distance=%u M=%u M0=%u levels=%u "
            "stride=%llu state=%u\n",
            sbi->s_ann.active_copy ? 'B' : 'A',
            sbi->s_ann.sequence,
            sbi->s_ann.epoch,
            sbi->s_ann.config.dimension,
            sbi->s_ann.config.element_type,
            sbi->s_ann.config.distance_type,
            sbi->s_ann.config.m,
            sbi->s_ann.config.m0,
            sbi->s_ann.config.max_levels,
            sbi->s_ann.config.object_stride,
            sbi->s_ann.state);

    kfree(a);
    kfree(b);
    return 0;
}

static void vecfs_ann_meta_to_disk(
    const struct vecfs_ann_meta *meta,
    struct vecfs_ann_meta_disk *disk,
    u64 sequence)
{
    const struct vecfs_ann_config *cfg = &meta->config;

    memset(disk, 0, sizeof(*disk));

    disk->magic = cpu_to_le32(VECFS_ANN_META_MAGIC);
    disk->major = cpu_to_le16(VECFS_ANN_META_MAJOR);
    disk->minor = cpu_to_le16(VECFS_ANN_META_MINOR);
    disk->header_size = cpu_to_le32(sizeof(*disk));

    disk->sequence = cpu_to_le64(sequence);
    disk->epoch = cpu_to_le64(meta->epoch);

    disk->state = cpu_to_le32(meta->state);
    disk->flags = cpu_to_le32(meta->flags);

    disk->dimension = cpu_to_le32(cfg->dimension);
    disk->element_type = cpu_to_le16(cfg->element_type);
    disk->distance_type = cpu_to_le16(cfg->distance_type);

    disk->m = cpu_to_le16(cfg->m);
    disk->m0 = cpu_to_le16(cfg->m0);
    disk->max_levels = cpu_to_le16(cfg->max_levels);
    disk->current_max_level =
        cpu_to_le16(cfg->current_max_level);

    disk->ef_construction =
        cpu_to_le32(cfg->ef_construction);
    disk->default_ef_search =
        cpu_to_le32(cfg->default_ef_search);

    disk->random_seed = cpu_to_le64(cfg->random_seed);
    disk->level_mult_q32 =
        cpu_to_le64(cfg->level_mult_q32);

    disk->live_nodes = cpu_to_le64(meta->live_nodes);
    disk->deleted_nodes =
        cpu_to_le64(meta->deleted_nodes);
    disk->next_node_id =
        cpu_to_le64(meta->next_node_id);
    disk->next_external_id =
        cpu_to_le64(meta->next_external_id);

    disk->entry_node_id =
        cpu_to_le32(meta->entry_node_id);
    disk->active_arena_ino =
        cpu_to_le32(meta->active_arena_ino);

    disk->rebuild_arena_ino =
        cpu_to_le32(meta->rebuild_arena_ino);
    disk->rebuild_state =
        cpu_to_le32(meta->rebuild_state);

    disk->object_stride =
        cpu_to_le64(cfg->object_stride);
    disk->vector_bytes =
        cpu_to_le64(cfg->vector_bytes);

    disk->committed_txid =
        cpu_to_le64(meta->committed_txid);

    disk->checksum = 0;
    disk->checksum =
        cpu_to_le32(vecfs_ann_meta_crc(disk));
}

int vecfs_ann_meta_commit(struct super_block *sb, bool sync)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ann_meta *meta = &sbi->s_ann;
    struct vecfs_ann_meta_disk *disk;
    struct buffer_head *bh;
    sector_t block;
    u8 target_copy;
    u64 new_sequence;
    int ret = 0;

    disk = kzalloc(sizeof(*disk), GFP_NOFS);
    if (!disk)
        return -ENOMEM;

    mutex_lock(&meta->lock);

    target_copy = meta->active_copy ^ 1;
    block = sbi->s_ann_meta_block[target_copy];
    new_sequence = meta->sequence + 1;

    vecfs_ann_meta_to_disk(meta, disk, new_sequence);

    bh = sb_getblk(sb, block);
    if (!bh) {
        ret = -ENOMEM;
        goto out_unlock;
    }

    lock_buffer(bh);
    memcpy(bh->b_data, disk, sizeof(*disk));
    set_buffer_uptodate(bh);
    unlock_buffer(bh);

    mark_buffer_dirty(bh);

    if (sync) {
        sync_dirty_buffer(bh);
        if (buffer_write_io_error(bh))
            ret = -EIO;
    }

    brelse(bh);

    if (!ret) {
        meta->sequence = new_sequence;
        meta->active_copy = target_copy;
        /*
         * 仅 sync 落盘后清 dirty；sync==false 时保留 dirty=true，
         * 让卸载 vecfs_sync_metadata() 据此做最后一次 sync 提交。
         */
        if (sync)
            meta->dirty = false;
    }

out_unlock:
    mutex_unlock(&meta->lock);
    kfree(disk);
    return ret;
}

void vecfs_ann_meta_destroy(struct super_block *sb)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);

    mutex_destroy(&sbi->s_ann.lock);
}
