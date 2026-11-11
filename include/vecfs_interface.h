#ifndef _VECFS_INTERFACE_H
#define _VECFS_INTERFACE_H

#include <linux/fs.h>

long vecfs_ioctl(
    struct file *file,
    unsigned int command,
    unsigned long argument);

#ifdef CONFIG_COMPAT
long vecfs_compat_ioctl(
    struct file *file,
    unsigned int command,
    unsigned long argument);
#endif

#endif
