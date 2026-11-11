#ifndef _VECFS_ANN_MAINTENANCE_H
#define _VECFS_ANN_MAINTENANCE_H

#include <linux/types.h>

#include "vecfs_ann_validate.h"

struct super_block;

int vecfs_ann_validate_and_refresh(
    struct super_block *sb,
    u32 flags,
    struct vecfs_ann_validate_report *report);

#endif
