CC ?= cc
CXX ?= c++

CPPFLAGS += -I..
CPPFLAGS += -I../include

CFLAGS ?= -O2 -g
CFLAGS += -std=gnu11
CFLAGS += -Wall
CFLAGS += -Wextra
CFLAGS += -Wpedantic
CFLAGS += -Werror
CFLAGS += -D_FILE_OFFSET_BITS=64

CXXFLAGS ?= -O2 -g
CXXFLAGS += -std=c++17
CXXFLAGS += -Wall
CXXFLAGS += -Wextra
CXXFLAGS += -Wpedantic
CXXFLAGS += -Werror
CXXFLAGS += -D_FILE_OFFSET_BITS=64

LDFLAGS ?=

# libcurl（vecfs-embed-helper 依赖）。优先 pkg-config；未安装开发包时可用
# 覆盖变量（例：本机无系统 dev 头、用 conda 头 + 系统运行库）：
#   make -C tools \
#     CURL_CFLAGS="-I/opt/anaconda3/include" \
#     CURL_LIBS="-L/usr/lib/x86_64-linux-gnu -l:libcurl.so.4"
CURL_CFLAGS ?= $(shell pkg-config --cflags libcurl 2>/dev/null)
CURL_LIBS   ?= $(shell pkg-config --libs libcurl 2>/dev/null)
ifeq ($(strip $(CURL_LIBS)),)
  CURL_LIBS = -lcurl
endif

TOOLS := \
	mkfs.vecfs \
	vecfs-index \
	hnswlib-converter \
	fsck.vecfs \
	vecfs-embed-helper

.PHONY: all clean

all: $(TOOLS)

fsck.vecfs: fsck.vecfs.o
	$(CC) $(LDFLAGS) -o $@ $^

fsck.vecfs.o: \
	fsck.vecfs.c \
	../include/vecfs_disk.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

mkfs.vecfs: mkfs.vecfs.o
	$(CC) $(LDFLAGS) -o $@ $^ -lm

vecfs-embed-helper: vecfs-embed-helper.o
	$(CC) $(LDFLAGS) -o $@ $^ $(CURL_LIBS) -lm

vecfs-index: vecfs-index.o
	$(CC) $(LDFLAGS) -o $@ $^

hnswlib-converter: hnswlib-converter.o
	$(CXX) $(LDFLAGS) -o $@ $^

mkfs.vecfs.o: \
	mkfs.vecfs.c \
	../include/vecfs_disk.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

vecfs-embed-helper.o: \
	vecfs-embed-helper.c \
	../include/vecfs_embed_upcall.h
	$(CC) $(CPPFLAGS) $(CURL_CFLAGS) $(CFLAGS) -c -o $@ $<

vecfs-index.o: \
	vecfs-index.c \
	exchange-format.h \
	../include/vecfs_ioctl.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

hnswlib-converter.o: \
	hnswlib-converter.cpp \
	exchange-format.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f \
		$(TOOLS) \
		mkfs.vecfs.o \
		vecfs-index.o \
		hnswlib-converter.o \
		vecfs-embed-helper.o
