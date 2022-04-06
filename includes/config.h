#ifndef __CONFIG__
#define __CONFIG__

#define WARP_SIZE 32
#define MAX_THREADS 512
#define MAX_NUMBER_OF_DIMENSIONS 50
#define MAX_NUMBER_OF_NEIGHBOURS 100
#define BLOCK_SIZE 1000

#define sqr(x) (x * x)
#define intceildiv(x, y) ((x + y - 1) / y)
#define block_cnt(x) intceildiv(x, MAX_THREADS)

#define __DEBUG__

#endif // __CONFIG__