/**
 * @file config.h
 * @author Khau Dang Nhat Minh (khaudangnhatminh@gmail.com)
 * @brief Sets the configuration of both problem and solver models
 * @version 0.1
 * @date 2022-04-06
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef __CONFIG__
#define __CONFIG__

// Problem constraints
#define MAX_NUMBER_OF_DIMENSIONS 50
#define MAX_NUMBER_OF_NEIGHBOURS 100

// GPU architecture
#define WARP_SIZE 32
#define MAX_THREADS 512

// Solver model
#define BLOCK_SIZE 8192

// Flag for development stage, to be removed in production
#define __DEBUG__

#endif // __CONFIG__