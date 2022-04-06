#ifndef __UTILS__
#define __UTILS__

#define sqr(x) (x * x)
#define intceildiv(x, y) ((x + y - 1) / y)
#define block_cnt(x) intceildiv(x, MAX_THREADS)

#endif // __UTILS__