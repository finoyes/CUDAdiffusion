#ifndef COMMON_H
#define COMMON_H

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "[CUDA ERROR] %s:%d: '%s' failed: %s (%d)\n",    \
                    __FILE__, __LINE__, #call,                               \
                    cudaGetErrorString(err__), (int)err__);                  \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

#define CUDA_CHECK_LAST()                                                    \
    do {                                                                     \
        cudaError_t err__ = cudaGetLastError();                             \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "[CUDA KERNEL LAUNCH ERROR] %s:%d: %s (%d)\n",   \
                    __FILE__, __LINE__, cudaGetErrorString(err__), (int)err__);\
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

#define IMG_SIZE        32
#define IMG_CHANNELS    3
#define IMG_ELEMS       (IMG_CHANNELS * IMG_SIZE * IMG_SIZE)

#define HIDDEN_CH       32
#define CONV_K          3
#define CONV_PAD        1

#define TIME_EMB_DIM    32

#define TIMESTEPS       300
#define BETA_START      1e-4f
#define BETA_END        0.02f

#define LEARNING_RATE   2e-4f
#define ADAM_BETA1      0.9f
#define ADAM_BETA2      0.999f
#define ADAM_EPS        1e-8f

#define TRAIN_ITERS_DEFAULT 5000
#define LOG_EVERY           250

#define TILE_DIM        16
#define HALO_DIM        (TILE_DIM + CONV_K - 1)

#endif // COMMON_H
