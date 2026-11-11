#!/usr/bin/env bash
# 从 hnswlib .bin 一步导入到 VecFS 磁盘设备：
#   bin -> (读头参数) -> mkfs + 挂载 -> hnswlib-converter -> vecfs-index import
#
# 用法:
#   ./import_hnsw.sh <bin文件> <设备名> [ip|l2] [挂载点]
#
# 例子:
#   ./import_hnsw.sh data/gist-960-euclidean-M16-M0-32-efc400.bin /dev/nvme1n1p2 l2 /mnt/vecfs-gist
#   ./import_hnsw.sh data/deep-image-96-angular-M16-M0-32-efc400.bin /dev/nvme1n1p1 ip /mnt/vecfs-deepimage
#
# 说明:
#   - dim/M/M0/ef_construction/levels 自动从 .bin 头读出；distance 不在头里，需指定
#     （省略时按文件名猜：gist/euclid -> l2，deep/angular/cosine -> ip）。
#   - 设备会被 mkfs.vecfs 重新格式化（不可逆），执行前会打印摘要并二次确认。
#   - 挂载类型跟随 Makefile 的 MODNAME（当前 vecfs2）。
set -euo pipefail

BIN="${1:-}"
DEV="${2:-}"
DIST="${3:-}"
MNT="${4:-}"

if [[ -z "$BIN" || -z "$DEV" ]]; then
    echo "用法: $0 <bin文件> <设备名> [ip|l2] [挂载点]" >&2
    exit 1
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOOLS="$REPO/tools"

[[ -f "$BIN" ]] || { echo "[fatal] bin 文件不存在: $BIN" >&2; exit 1; }
[[ -b "$DEV" ]] || { echo "[fatal] 不是块设备: $DEV" >&2; exit 1; }

# 挂载类型跟随 Makefile 的 MODNAME
MODNAME="$(sed -n 's/^MODNAME[[:space:]]*?=[[:space:]]*//p' "$REPO/Makefile" | head -1)"
MODNAME="${MODNAME:-vecfs}"
MNT="${MNT:-/mnt/vecfs-$(basename "$DEV")}"
EXCHANGE="${BIN%.bin}.exchange"

# ---- 从 hnswlib .bin 头读参数（与 tools/hnswlib-converter.cpp from-hnsw 一致） ----
read -r DIM M M0 EFC NATLVL < <(
    python3 - "$BIN" <<'PY'
import struct, sys
with open(sys.argv[1], "rb") as f:
    h = f.read(96)
off0, maxe, cur, szdpe, laboff, dataoff = struct.unpack("<6Q", h[:48])
natlvl = struct.unpack("<i", h[48:52])[0]
nentry = struct.unpack("<I", h[52:56])[0]
maxm, maxm0, m = struct.unpack("<3Q", h[56:80])
efc = struct.unpack("<Q", h[88:96])[0]
dim = (laboff - dataoff) // 4
print(dim, m, maxm0, efc, natlvl)
PY
)
LEVELS=$(( NATLVL + 1 ))
[[ "$LEVELS" -lt 2 ]] && LEVELS=2
EFS="${EFS:-200}"

# ---- distance：显式优先，否则按文件名猜 ----
if [[ -z "$DIST" ]]; then
    base="$(basename "$BIN")"
    case "$base" in
        *gist*|*euclid*)  DIST=l2 ;;
        *deep*|*angular*|*cosine*) DIST=ip ;;
        *) echo "[fatal] 无法从文件名判断 distance，请显式指定 ip|l2" >&2; exit 1 ;;
    esac
fi
[[ "$DIST" == "ip" || "$DIST" == "l2" ]] || { echo "[fatal] distance 必须是 ip 或 l2" >&2; exit 1; }

echo "=================================================="
echo "  bin         : $BIN"
echo "  设备        : $DEV  (将被 mkfs.vecfs 覆盖！)"
echo "  挂载点      : $MNT"
echo "  参数        : dim=$DIM distance=$DIST M=$M M0=$M0 levels=$LEVELS efc=$EFC efs=$EFS"
echo "=================================================="
if [[ "${YES:-0}" != "1" ]]; then
    read -r -p "确认格式化 $DEV 并导入？[y/N] " ans
    [[ "$ans" == "y" || "$ans" == "Y" ]] || { echo "已取消"; exit 0; }
fi

# ---- 设备不应处于挂载状态 ----
if mount | grep -qF " $DEV "; then
    echo "[fatal] $DEV 仍在挂载，请先 umount 再重试" >&2
    exit 1
fi

# ---- 模块加载（挂载需要） ----
if ! grep -qw "$MODNAME" /proc/modules; then
    echo "[module] $MODNAME 未加载，尝试 insmod $REPO/$MODNAME.ko"
    sudo insmod "$REPO/$MODNAME.ko"
fi

# ---- 格式化 + 挂载 ----
echo "[mkfs] tools/mkfs.vecfs --dim $DIM --distance $DIST --m $M --m0 $M0 --levels $LEVELS --ef-construction $EFC --ef-search $EFS $DEV"
sudo "$TOOLS/mkfs.vecfs" \
    --dim "$DIM" --distance "$DIST" --m "$M" --m0 "$M0" --levels "$LEVELS" \
    --ef-construction "$EFC" --ef-search "$EFS" "$DEV"

sudo mkdir -p "$MNT"
echo "[mount] mount -t $MODNAME $DEV $MNT"
sudo mount -t "$MODNAME" "$DEV" "$MNT"

# ---- bin -> exchange ----
echo "[convert] from-hnsw --distance $DIST --max-levels $LEVELS --ef-search $EFS -> $EXCHANGE"
"$TOOLS/hnswlib-converter" from-hnsw \
    --distance "$DIST" --max-levels "$LEVELS" --ef-search "$EFS" \
    "$BIN" "$EXCHANGE"

"$TOOLS/vecfs-index" info "$EXCHANGE"

# ---- 导入（ioctl 锚点：任意挂载点内普通文件） ----
printf 'control' | sudo tee "$MNT/control" >/dev/null
echo "[import] vecfs-index import $EXCHANGE $MNT/control"
sudo "$TOOLS/vecfs-index" import "$EXCHANGE" "$MNT/control"

echo
echo "完成。查询时 --dev_path 指向挂载点目录: $MNT"
