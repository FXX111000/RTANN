// SPDX-License-Identifier: GPL-2.0
//
// vecfs-embed-helper —— 内核 SHIELD 派生时的用户态嵌入助手（批量）。
//
// 由内核 call_usermodehelper 拉起（配置经 env 传入，key 不进 cmdline）：
//   VECFS_EMBED_URL     嵌入服务基地址，默认 http://127.0.0.1:11434
//   VECFS_EMBED_KEY     API key（本地模型留空，可选）
//   VECFS_EMBED_MODEL   模型名（必需）
//   VECFS_EMBED_PROTO   协议（当前仅 ollama）
//   VECFS_EMBED_DEVICE  字符设备路径，默认 /dev/vecfs-embed
//
// 协议（与 include/vecfs_embed_upcall.h 对齐，批量）：
//   读一次 = 一整条请求：16B 头(id,nchunks,dim) + nchunks×[le32 len + data]。
//   一次 HTTP 调用 Ollama /api/embed（input = nchunks 个文本）→ N×dim 向量，
//   逐行 L2 归一化后回写：16B 头(id,status,nchunks) + nchunks×dim fp32。
//   读返回 0（EOF，内核 shutdown）时干净退出。
//
// 错误映射：HTTP 4xx / 维度不符 / JSON schema 不符 → -EINVAL（永久）；
//           网络 / 5xx → -EAGAIN（内核 requeue 重试）。

#include <curl/curl.h>
#include <endian.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <poll.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "../include/vecfs_embed_upcall.h"

#define HEADER_SIZE (sizeof(struct vecfs_embed_request))
#define MAX_WIRE (VECFS_EMBED_UP_MAX_CHUNKS * 4 + VECFS_EMBED_UP_MAX_DATA)

/* 调试日志：由内核 call_usermodehelper 拉起时 stderr 被丢弃，故写到
 * /tmp/vecfs-embed-helper.log。默认关闭（编译期门控），调试时给 helper 加
 * -DVECFS_EMBED_DEBUG 重编即可；helper 由 usermodehelper 用固定 envp 拉起，
 * 普通 getenv 拿不到运行时开关，故用编译期开关而非 env。 */
#ifdef VECFS_EMBED_DEBUG
static void dbg(const char *fmt, ...)
{
    static FILE *log;
    va_list ap;
    char ts[32];
    struct timespec now;
    struct tm tm;

    if (!log) {
        log = fopen("/tmp/vecfs-embed-helper.log", "a");
        if (!log)
            return;
    }
    clock_gettime(CLOCK_REALTIME, &now);
    localtime_r(&now.tv_sec, &tm);
    strftime(ts, sizeof(ts), "%H:%M:%S", &tm);
    fprintf(log, "%s.%03ld ", ts, now.tv_nsec / 1000000L);
    va_start(ap, fmt);
    vfprintf(log, fmt, ap);
    va_end(ap);
    fputc('\n', log);
    fflush(log);
}
#else
#define dbg(...) ((void)0)
#endif

/* 动态字符串缓冲（请求体构建与响应体累积共用）。 */
struct strbuf {
    char *data;
    size_t len;
    size_t cap;
};

static int sb_reserve(struct strbuf *b, size_t extra)
{
    size_t need, nc;
    char *p;

    if (b->len + extra + 1 <= b->cap)
        return 0;
    need = b->len + extra + 1;
    nc = b->cap ? b->cap * 2 : 256;
    while (nc < need)
        nc *= 2;
    p = realloc(b->data, nc);
    if (!p)
        return -1;
    b->data = p;
    b->cap = nc;
    return 0;
}

static int sb_append(struct strbuf *b, const char *s, size_t n)
{
    if (sb_reserve(b, n))
        return -1;
    memcpy(b->data + b->len, s, n);
    b->len += n;
    b->data[b->len] = '\0';
    return 0;
}

static int sb_append_cstr(struct strbuf *b, const char *s)
{
    return sb_append(b, s, strlen(s));
}

/* 追加带引号与 JSON 转义的字符串。 */
static int sb_append_json_str(struct strbuf *b, const char *s, size_t n)
{
    size_t i;

    if (sb_append(b, "\"", 1))
        return -1;
    for (i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        char esc[8];
        int elen = 0;

        switch (c) {
        case '"':
            esc[0] = '\\'; esc[1] = '"'; elen = 2; break;
        case '\\':
            esc[0] = '\\'; esc[1] = '\\'; elen = 2; break;
        case '\n':
            esc[0] = '\\'; esc[1] = 'n'; elen = 2; break;
        case '\r':
            esc[0] = '\\'; esc[1] = 'r'; elen = 2; break;
        case '\t':
            esc[0] = '\\'; esc[1] = 't'; elen = 2; break;
        default:
            if (c < 0x20) {
                snprintf(esc, sizeof(esc), "\\u%04x", c);
                elen = 6;
            } else {
                esc[0] = (char)c;
                elen = 1;
            }
        }
        if (sb_append(b, esc, (size_t)elen))
            return -1;
    }
    return sb_append(b, "\"", 1);
}

static size_t curl_write_cb(char *ptr, size_t size, size_t nmemb, void *ud)
{
    struct strbuf *b = ud;

    if (sb_append(b, ptr, size * nmemb))
        return 0; /* 触发 CURL 写入错误 */
    return size * nmemb;
}

static int write_all(int fd, const void *buf, size_t n)
{
    size_t off = 0;

    while (off < n) {
        ssize_t w = write(fd, (const char *)buf + off, n - off);

        if (w < 0) {
            if (errno == EINTR)
                continue;
            return -1;
        }
        off += (size_t)w;
    }
    return 0;
}

/* ---------------------------------------------------------------------------
 * 聚焦 JSON 解析：只解析 Ollama /api/embed 响应里的 embeddings（N×dim double）。
 * schema 与协议绑定，任何不符返回 -EINVAL。
 * ------------------------------------------------------------------------- */

static const char *skip_ws(const char *p)
{
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')
        p++;
    return p;
}

/* 从指向 '[' 的位置解析 double 数组到 out[0..max)，*count=个数。 */
static const char *parse_double_array(const char *p, float *out,
                                      uint32_t max, uint32_t *count)
{
    uint32_t n = 0;

    p = skip_ws(p);
    if (*p != '[')
        return NULL;
    p = skip_ws(p + 1);
    if (*p == ']') {
        *count = 0;
        return p + 1;
    }
    for (;;) {
        char *end;
        double v;

        if (n >= max)
            return NULL;
        v = strtod(p, &end);
        if (end == p)
            return NULL;
        out[n++] = (float)v;
        p = skip_ws(end);
        if (*p == ',') {
            p = skip_ws(p + 1);
            continue;
        }
        if (*p == ']') {
            *count = n;
            return p + 1;
        }
        return NULL;
    }
}

static int parse_embeddings(const char *json, uint32_t nchunks, uint32_t dim,
                            float *out)
{
    const char *p;
    uint32_t i;

    p = strstr(json, "\"embeddings\"");
    if (!p)
        return -EINVAL;
    p = skip_ws(p + strlen("\"embeddings\""));
    if (*p != ':')
        return -EINVAL;
    p = skip_ws(p + 1);
    if (*p != '[')
        return -EINVAL;
    p = skip_ws(p + 1);

    for (i = 0; i < nchunks; i++) {
        uint32_t cnt = 0;

        p = parse_double_array(p, out + (size_t)i * dim, dim, &cnt);
        if (!p || cnt != dim)
            return -EINVAL;
        p = skip_ws(p);
        if (i + 1 < nchunks) {
            if (*p != ',')
                return -EINVAL;
            p = skip_ws(p + 1);
        }
    }
    return 0;
}

static void l2_normalize(float *v, uint32_t dim)
{
    double sum = 0.0;
    float norm, inv;
    uint32_t i;

    for (i = 0; i < dim; i++)
        sum += (double)v[i] * (double)v[i];
    norm = (float)sqrt(sum);
    if (norm <= 0.0f)
        return;
    inv = 1.0f / norm;
    for (i = 0; i < dim; i++)
        v[i] *= inv;
}

static int send_error(int fd, uint64_t id, int status)
{
    struct vecfs_embed_response hdr;

    hdr.id = id;
    hdr.status = (__s32)status;
    hdr.nchunks = 0;
    return write_all(fd, &hdr, sizeof(hdr));
}

/*
 * 并发执行模型：helper 由「串行一请求一请求」改为「单事件循环 + curl_multi
 * 多路并发」。内核派生最多 ~256 个 worker 同时各提交一条批量嵌入请求，串行
 * 处理只能 ~4 req/s，30s 求值窗口内仅能服务 ~100 条，其余 worker 超时 requeue
 * → 派生停摆、GPU 掉零。并发后单进程同时驱动 VECFS_EMBED_CONCURRENCY 路传输，
 * 吞吐随并发线性提升；数据/控制两 fd 一并纳入 poll，做到「有新请求就接、
 * 有取消就中止」。单线程事件循环内所有 read/write 串行，无需额外加锁。
 */
#define VECFS_EMBED_CONCURRENCY 32

/* 一个在途嵌入请求：easy 加入 multi 后由事件循环统一驱动，传输完成（或
 * 取消/超时/出错）再回写应答并释放槽位。easy 预分配复用（每轮 reset）。 */
struct embed_slot {
    CURL *easy;
    int used;
    uint64_t id;
    uint32_t nchunks;
    uint32_t dim;
    struct strbuf body;     /* 请求体（POSTFIELDS 引用到传输完成） */
    struct strbuf respbody; /* 响应累积（WRITEDATA 引用） */
    float *vecs;
    uint32_t *lens;
    const char **texts;
};

static void slot_release(struct embed_slot *s)
{
    if (s->easy)
        curl_easy_reset(s->easy);
    free(s->body.data);
    s->body.data = NULL;
    s->body.len = s->body.cap = 0;
    free(s->respbody.data);
    s->respbody.data = NULL;
    s->respbody.len = s->respbody.cap = 0;
    free(s->vecs);
    s->vecs = NULL;
    free(s->lens);
    s->lens = NULL;
    free(s->texts);
    s->texts = NULL;
    s->used = 0;
}

/* 解析一条请求并启动该槽位的 HTTP 传输（加入 multi）。
 * 返回 0：传输中；-1：已就地应答（协议/解析错误），调用方须 slot_release 复位。 */
static int slot_start(struct embed_slot *s, CURLM *multi, uint8_t *buf,
                      ssize_t n, const char *api_url, const char *model,
                      struct curl_slist *hdrs, int fd)
{
    struct vecfs_embed_request req;
    size_t off = 0, total_data = 0;
    uint32_t j;

    memcpy(&req, buf, sizeof(req));
    if (req.nchunks == 0 || req.nchunks > VECFS_EMBED_UP_MAX_CHUNKS ||
        req.dim == 0 || req.dim > VECFS_EMBED_UP_MAX_DIM) {
        dbg("bad nchunks/dim -> -EINVAL");
        send_error(fd, req.id, -EINVAL);
        return -1;
    }

    s->lens = malloc((size_t)req.nchunks * sizeof(*s->lens));
    s->texts = malloc((size_t)req.nchunks * sizeof(*s->texts));
    if (!s->lens || !s->texts) {
        dbg("alloc lens/texts fail -> -EAGAIN");
        send_error(fd, req.id, -EAGAIN);
        return -1;
    }

    off = sizeof(req);
    for (j = 0; j < req.nchunks; j++) {
        uint32_t len;

        if (off + 4 > (size_t)n) {
            send_error(fd, req.id, -EINVAL);
            return -1;
        }
        memcpy(&len, buf + off, 4);
        off += 4;
        len = le32toh(len);
        if (len > VECFS_EMBED_UP_MAX_DATA ||
            total_data > (size_t)VECFS_EMBED_UP_MAX_DATA - len) {
            send_error(fd, req.id, -EINVAL);
            return -1;
        }
        if (off + len > (size_t)n) {
            send_error(fd, req.id, -EINVAL);
            return -1;
        }
        s->lens[j] = len;
        s->texts[j] = (const char *)(buf + off);
        off += len;
        total_data += len;
    }

    if (sb_append_cstr(&s->body, "{\"model\":") ||
        sb_append_json_str(&s->body, model, strlen(model)) ||
        sb_append_cstr(&s->body, ",\"input\":[")) {
        send_error(fd, req.id, -EAGAIN);
        return -1;
    }
    for (j = 0; j < req.nchunks; j++) {
        if ((j && sb_append_cstr(&s->body, ",")) ||
            sb_append_json_str(&s->body, s->texts[j], s->lens[j])) {
            send_error(fd, req.id, -EAGAIN);
            return -1;
        }
    }
    if (sb_append_cstr(&s->body, "]}")) {
        send_error(fd, req.id, -EAGAIN);
        return -1;
    }

    s->id = req.id;
    s->nchunks = req.nchunks;
    s->dim = req.dim;
    s->respbody.data = NULL;
    s->respbody.len = s->respbody.cap = 0;

    curl_easy_setopt(s->easy, CURLOPT_URL, api_url);
    curl_easy_setopt(s->easy, CURLOPT_POST, 1L);
    curl_easy_setopt(s->easy, CURLOPT_HTTPHEADER, hdrs);
    curl_easy_setopt(s->easy, CURLOPT_POSTFIELDS, s->body.data);
    curl_easy_setopt(s->easy, CURLOPT_POSTFIELDSIZE, (long)s->body.len);
    curl_easy_setopt(s->easy, CURLOPT_WRITEFUNCTION, curl_write_cb);
    curl_easy_setopt(s->easy, CURLOPT_WRITEDATA, &s->respbody);
    /*
     * 传输硬超时（略小于内核 30s 的 wait 超时）：后端一旦对某请求无响应，
     * 该路传输 25s 后 CURLE_OPERATION_TIMEDOUT → 回 -EAGAIN，worker requeue，
     * 该槽位随即空出服务后续请求，不拖垮并发流水。
     */
    curl_easy_setopt(s->easy, CURLOPT_TIMEOUT_MS, 25000L);

    if (curl_multi_add_handle(multi, s->easy) != CURLM_OK) {
        dbg("add_handle fail id=%llu -> -EAGAIN",
            (unsigned long long)req.id);
        send_error(fd, req.id, -EAGAIN);
        return -1;
    }
    s->used = 1;
    dbg("start id=%llu nchunks=%u dim=%u", (unsigned long long)req.id,
        req.nchunks, req.dim);
    return 0;
}

/* 传输完成：查 http 码 / 解析 / 归一化 / 回写应答，然后释放槽位。 */
static void slot_finish(struct embed_slot *s, int fd, CURLcode crc)
{
    long http_code = 0;
    uint32_t j;

    if (crc != CURLE_OK) {
        dbg("curl error id=%llu: %s -> -EAGAIN",
            (unsigned long long)s->id, curl_easy_strerror(crc));
        send_error(fd, s->id, -EAGAIN);
        slot_release(s);
        return;
    }
    curl_easy_getinfo(s->easy, CURLINFO_RESPONSE_CODE, &http_code);
    if (http_code != 200) {
        int status = (http_code >= 500) ? -EAGAIN : -EINVAL;

        dbg("http %ld id=%llu -> %d", http_code,
            (unsigned long long)s->id, status);
        send_error(fd, s->id, status);
        slot_release(s);
        return;
    }

    s->vecs = malloc((size_t)s->nchunks * s->dim * sizeof(float));
    if (!s->vecs) {
        send_error(fd, s->id, -EAGAIN);
        slot_release(s);
        return;
    }
    if (parse_embeddings(s->respbody.data, s->nchunks, s->dim, s->vecs) != 0) {
        dbg("parse_embeddings failed id=%llu (body %zu bytes) -> -EINVAL",
            (unsigned long long)s->id, s->respbody.len);
        send_error(fd, s->id, -EINVAL);
        slot_release(s);
        return;
    }

    for (j = 0; j < s->nchunks; j++)
        l2_normalize(s->vecs + (size_t)j * s->dim, s->dim);

    /* 内核 write() 要求「一次 write = 16B 头 + nchunks×dim×4B 载荷」，拼成
     * 连续缓冲单次 write_all；单线程下各槽位回写天然串行。 */
    {
        struct vecfs_embed_response resp;
        size_t pay = (size_t)s->nchunks * s->dim * sizeof(float);
        uint8_t *out = malloc(sizeof(resp) + pay);

        resp.id = s->id;
        resp.status = 0;
        resp.nchunks = s->nchunks;
        if (!out) {
            send_error(fd, s->id, -EAGAIN);
            slot_release(s);
            return;
        }
        memcpy(out, &resp, sizeof(resp));
        memcpy(out + sizeof(resp), s->vecs, pay);
        if (write_all(fd, out, sizeof(resp) + pay)) {
            dbg("write response failed id=%llu: %s",
                (unsigned long long)s->id, strerror(errno));
            /* 内核可能已超时摘除该 id：正常，忽略并继续。 */
        }
        free(out);
    }
    dbg("OK id=%llu nchunks=%u dim=%u", (unsigned long long)s->id,
        s->nchunks, s->dim);
    slot_release(s);
}

int main(void)
{
    const char *url = getenv("VECFS_EMBED_URL");
    const char *key = getenv("VECFS_EMBED_KEY");
    const char *model = getenv("VECFS_EMBED_MODEL");
    const char *device = getenv("VECFS_EMBED_DEVICE");
    char api_url[1024];
    uint8_t *buf = NULL;
    int fd = -1;
    int ctrl_fd = -1;
    CURLM *multi = NULL;
    struct curl_slist *hdrs = NULL;
    struct embed_slot *slots = NULL;
    int rc = 1;
    uint32_t i;

    if (!device || !*device)
        device = "/dev/vecfs-embed";
    if (!url || !*url)
        url = "http://127.0.0.1:11434";
    if (!model || !*model) {
        fprintf(stderr, "vecfs-embed-helper: VECFS_EMBED_MODEL missing\n");
        return 1;
    }
    snprintf(api_url, sizeof(api_url), "%s/api/embed", url);

    fd = open(device, O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        fprintf(stderr, "vecfs-embed-helper: open %s: %s\n",
                device, strerror(errno));
        return 1;
    }

    /* 控制通道（只读，收内核下发的 CANCEL 事件）。 */
    ctrl_fd = open(device, O_RDONLY | O_CLOEXEC);
    if (ctrl_fd < 0) {
        fprintf(stderr, "vecfs-embed-helper: open %s (ctrl): %s\n",
                device, strerror(errno));
        close(fd);
        return 1;
    }

    curl_global_init(CURL_GLOBAL_DEFAULT);
    multi = curl_multi_init();
    if (!multi) {
        fprintf(stderr, "vecfs-embed-helper: curl_multi_init failed\n");
        goto out;
    }

    hdrs = curl_slist_append(hdrs, "Content-Type: application/json");
    if (key && *key) {
        char auth[512];
        snprintf(auth, sizeof(auth), "Authorization: Bearer %s", key);
        hdrs = curl_slist_append(hdrs, auth);
    }

    buf = malloc((size_t)HEADER_SIZE + MAX_WIRE);
    if (!buf)
        goto out;

    slots = calloc(VECFS_EMBED_CONCURRENCY, sizeof(*slots));
    if (!slots)
        goto out;
    for (i = 0; i < VECFS_EMBED_CONCURRENCY; i++) {
        slots[i].easy = curl_easy_init();
        if (!slots[i].easy) {
            fprintf(stderr, "vecfs-embed-helper: curl_easy_init #%u failed\n",
                    i);
            goto out;
        }
    }

    dbg("started device=%s url=%s model=%s conc=%d", device, url, model,
        VECFS_EMBED_CONCURRENCY);

    for (;;) {
        /* 1) 填满空闲槽位：数据 fd 有 PENDING 请求就接起一路传输。 */
        for (i = 0; i < VECFS_EMBED_CONCURRENCY; i++) {
            struct pollfd pfd;
            ssize_t n;

            if (slots[i].used)
                continue;
            pfd.fd = fd;
            pfd.events = POLLIN;
            pfd.revents = 0;
            if (poll(&pfd, 1, 0) <= 0)
                break; /* 暂无更多请求 */

            n = read(fd, buf, (size_t)HEADER_SIZE + MAX_WIRE);
            if (n == 0) {
                dbg("EOF: kernel shutdown");
                rc = 0;
                goto out;
            }
            if (n < 0) {
                if (errno == EINTR)
                    continue;
                dbg("read error: %s", strerror(errno));
                goto out;
            }
            if (n < (ssize_t)HEADER_SIZE) {
                dbg("short header n=%zd", n);
                goto out;
            }
            {
                struct vecfs_embed_request hdr;
                memcpy(&hdr, buf, sizeof(hdr));
                dbg("read id=%llu nchunks=%u n=%zd", (unsigned long long)hdr.id,
                    hdr.nchunks, n);
            }
            if (slot_start(&slots[i], multi, buf, n, api_url, model,
                           hdrs, fd) < 0)
                slot_release(&slots[i]); /* 已就地应答，槽位复位 */
        }

        /* 2) 驱动在途传输，并监听数据/控制两通道。 */
        {
            struct curl_waitfd extra[2];
            int numfds = 0;
            int running = 0;

            extra[0].fd = fd;
            extra[0].events = CURL_WAIT_POLLIN;
            extra[0].revents = 0;
            extra[1].fd = ctrl_fd;
            extra[1].events = CURL_WAIT_POLLIN;
            extra[1].revents = 0;

            curl_multi_poll(multi, extra, 2, 500, &numfds);
            curl_multi_perform(multi, &running);

            /* 3) 控制通道：CANCEL(id) → 中止对应传输（不回写应答）。 */
            if (extra[1].revents & CURL_WAIT_POLLIN) {
                struct vecfs_embed_control c;
                ssize_t cn = read(ctrl_fd, &c, sizeof(c));

                if (cn == (ssize_t)sizeof(c) &&
                    c.opcode == VECFS_EMBED_CTRL_CANCEL) {
                    for (i = 0; i < VECFS_EMBED_CONCURRENCY; i++) {
                        if (slots[i].used && slots[i].id == c.id) {
                            dbg("cancelled id=%llu",
                                (unsigned long long)c.id);
                            curl_multi_remove_handle(multi, slots[i].easy);
                            slot_release(&slots[i]);
                            break;
                        }
                    }
                }
            }
        }

        /* 4) 完成传输 → 回写应答 → 释放槽位。 */
        for (;;) {
            CURLMsg *msg;
            int msgs_left = 0;
            CURL *done_easy;
            CURLcode done_crc;
            int found = 0;

            msg = curl_multi_info_read(multi, &msgs_left);
            if (!msg)
                break;
            if (msg->msg != CURLMSG_DONE)
                continue;
            done_easy = msg->easy_handle;
            done_crc = msg->data.result;

            for (i = 0; i < VECFS_EMBED_CONCURRENCY; i++) {
                if (slots[i].used && slots[i].easy == done_easy) {
                    curl_multi_remove_handle(multi, done_easy);
                    slot_finish(&slots[i], fd, done_crc);
                    found = 1;
                    break;
                }
            }
            if (!found)
                curl_multi_remove_handle(multi, done_easy);
        }
    }

out:
    if (slots) {
        for (i = 0; i < VECFS_EMBED_CONCURRENCY; i++) {
            if (slots[i].used && slots[i].easy)
                curl_multi_remove_handle(multi, slots[i].easy);
            if (slots[i].easy)
                curl_easy_cleanup(slots[i].easy);
            free(slots[i].body.data);
            free(slots[i].respbody.data);
            free(slots[i].vecs);
            free(slots[i].lens);
            free(slots[i].texts);
        }
        free(slots);
    }
    free(buf);
    if (hdrs)
        curl_slist_free_all(hdrs);
    if (multi)
        curl_multi_cleanup(multi);
    curl_global_cleanup();
    if (ctrl_fd >= 0)
        close(ctrl_fd);
    if (fd >= 0)
        close(fd);
    return rc;
}
