// https://arxiv.org/html/2006.11239v2#S2

#ifndef DIFFUSION_H
#define DIFFUSION_H

#include "common.h"

struct DiffusionSchedule {
    float beta[TIMESTEPS];
    float alpha[TIMESTEPS];
    float alpha_bar[TIMESTEPS];                 // cumulative product of alpha
    float sqrt_alpha_bar[TIMESTEPS];             // sqrt(alpha_bar_t)
    float sqrt_one_minus_alpha_bar[TIMESTEPS];   // sqrt(1 - alpha_bar_t)
};

// Fills `sched` with a linear beta schedule from BETA_START to BETA_END and
// derives alpha_t = 1 - beta_t, alpha_bar_t = prod_{s<=t} alpha_s, and their
// square roots. Pure host-side arithmetic (TIMESTEPS scalar ops).
void build_diffusion_schedule(DiffusionSchedule* sched);

// Builds a (TIMESTEPS x TIME_EMB_DIM) sinusoidal timestep-embedding table
// (the classic Transformer/DDPM positional encoding) on the host and
// uploads it once to a freshly-allocated device buffer. The embedding of
// timestep t lives at d_table + t*TIME_EMB_DIM. Caller owns the returned
// pointer and must cudaFree it.
float* build_time_embedding_table_device();
 