#ifndef __UTILS__
#define __UTILS__

#define sqr(x) (x * x)
#define intceildiv(x, y) (((x) + (y) - 1) / (y))
#define set_block_thread(x) intceildiv(x, MAX_THREADS), MAX_THREADS

#define coor_init(r, c, k)                                      \
    const int r = (blockIdx.x * MAX_THREADS + threadIdx.x) / k; \
    const int c = (blockIdx.x * MAX_THREADS + threadIdx.x) % k

#define FULL_MASK 0xffffffff

#endif // __UTILS__