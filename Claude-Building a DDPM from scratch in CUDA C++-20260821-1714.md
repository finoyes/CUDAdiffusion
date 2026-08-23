# Building a DDPM from scratch in CUDA C++

**Created:** 7/31/2026 13:06:47  
**Updated:** 8/21/2026 16:07:59  
**Exported:** 8/21/2026 17:14:37  
**Link:** [https://claude.ai/chat/9fe9937b-4328-4cd5-9ca6-353781998a2b](https://claude.ai/chat/9fe9937b-4328-4cd5-9ca6-353781998a2b)  

## User:

> 7/31/2026 13:06:48

Act as an Expert HPC AI Engineer and CUDA C++ Developer. Your task is to provide an exhaustive, step-by-step guide to building a fully functional, end-to-end Denoising Diffusion Probabilistic Model (DDPM) entirely from scratch in CUDA C++. 
The goal of this project is to take a real image dataset (e.g., PNG/JPG), train a minimal diffusion model on it (overfitting to a few images or a single image for demonstration), and generate a new output image based on that training. No external deep learning frameworks (like PyTorch or TensorFlow) are allowed. You may only use standard C++ libraries, CUDA runtime, and `stb_image.h`/`stb_image_write.h` for image I/O.
I need you to provide the COMPLETE code for every single file with NO placeholders (do not use "// ... code goes here"). Furthermore, you must explain EVERY logical block and crucial line of code in detail, breaking down the math and the HPC hardware mapping.
Please structure your response exactly as follows, implementing a micro-UNet or simple CNN backbone to ensure the code fits in this context window:
1. PROJECT SETUP & DIRECTORY STRUCTURE
- Define the exact directory tree (e.g., `Makefile`, `src/main.cu`, `src/diffusion.cu`, `src/network.cu`, `include/stb_image.h`, etc.).
- Provide the `Makefile` with the correct `nvcc` compilation flags for optimization (e.g., `-O3 -arch=sm_80`).
2. IMAGE I/O & MEMORY MANAGEMENT (`utils.cu` / `utils.h`)
- Provide the complete code to load real images into flat 1D arrays using `stb_image`.
- Write the host-to-device and device-to-host memory transfer functions.
- Explain line-by-line how the memory is aligned and transferred.
3. FORWARD DIFFUSION & SCHEDULER (`diffusion.cu`)
- Implement the CUDA kernels for the forward process: adding Gaussian noise to the image at timestep `t` using `curand` or a custom seeded RNG inside the kernel.
- Explain how you achieve coalesced memory access in this element-wise kernel.
4. NEURAL NETWORK KERNELS (`network.cu`)
- Implement the custom CUDA kernels for the model's forward and backward passes. Include at least:
  - A 2D Convolution kernel utilizing `__shared__` memory (tiling) to reduce global memory bandwidth bottlenecks.
  - An activation function kernel (e.g., SiLU/Swish).
  - A Loss calculation kernel (MSE) using Parallel Reduction to compute the error across the grid.
- Explain the block/grid dimensions, warp synchronization, and shared memory mapping line-by-line.
5. BACKPROPAGATION & OPTIMIZER (`train.cu`)
- Implement the manual gradient calculation (backward pass kernels).
- Implement an Adam optimizer kernel to update the network weights.
- Explain the chain rule application in your CUDA code.
6. REVERSE DIFFUSION (GENERATION) (`generate.cu`)
- Implement the generation loop starting from pure Gaussian noise, iteratively calling the network to predict and subtract noise to recover an image.
- Include the code to save the final generated output back to a real image file using `stb_image_write.h`.
7. ORCHESTRATION (`main.cu`)
- Tie it all together in `main.cu`. 
- Initialize memory, load the image, run the training loop for `N` epochs, and trigger the image generation function.
CRITICAL RULES FOR YOUR RESPONSE:
- Write PRODUCTION-READY, compilable CUDA C++ code.
- Do NOT skip any steps or provide partial implementations. If the code is long, prioritize a minimal network architecture (e.g., 3-4 layers) so it fits, but make it mathematically correct for DDPM.
- After every code block, provide a markdown section titled "Line-by-Line Explanation" that dissects the pointers, memory fences, math operations, and HPC design choices.

## Assistant:

> 7/31/2026 13:21:28

> Validated convolution implementation, architected project structure, commenced CUDA development.
>
> - Architected comprehensive CUDA DDPM implementation with meticulous mathematical validation.
> - Check whether CUDA toolkit, GPU, Python/numpy, and g++ are available in this sandbox
> - Create a scratch directory for numerical verification of the backprop math before writing CUDA
> - Architected NumPy verification framework mirroring CUDA implementation with gradient checks.
> - Numpy reference implementation mirroring the planned CUDA kernel math, with finite-difference gradient checking to validate the manual backprop formulas before translating them into CUDA C++
> - Run the finite-difference gradient check for the full 3-conv-layer network with time embeddings
> - Validated mathematical derivations and verified diffusion process consistency numerically.
> - Verify the reverse-diffusion (sampling) mean formula against the closed-form DDPM posterior mean using two independent derivations
> - Validated backward pass logic, pivoting toward CUDA implementation with tiled convolution verification.
> - Emulate the exact block/thread/shared-memory indexing of the planned tiled CUDA conv2d kernel in Python, to catch halo/boundary indexing bugs before writing CUDA (numpy conv2d_forward already verified as ground truth)
> - Run the tiled shared-memory convolution indexing emulation against the verified reference conv2d
> - Validated convolution implementation, architected project structure, commenced CUDA development.
> - **Done**

Math verified — all analytic gradients match finite differences exactly, and the tiled shared-memory convolution indexing matches the reference convolution exactly (including boundary tiles). Now I'll build the actual project.

> Architected CUDA hyperparameters and finalized common header structure.
>
> - Fetch the genuine, unmodified public-domain stb_image.h / stb_image_write.h single-header libraries from the official repository
> - Diagnose curl failure when fetching stb_image.h
> - Check current directory state and permissions
> - Diagnosed shell incompatibility and command chaining issues.
> - Recreate directory structure with explicit mkdir calls and absolute paths, then fetch the stb headers
> - Check if working directory persists across separate tool calls
> - Architected CUDA hyperparameters and finalized common header structure.
> - **Done**

Now let's write the source files. Starting with the shared configuration header.

> Architected CUDA network structs, kernels, and memory management strategy.
>
> - Shared hyperparameters, tensor-size macros, and CUDA error-checking macro used by every translation unit in the project
> - Architected utility functions for image I/O and GPU memory transfers.
> - Header declaring image I/O and host/device memory-transfer helper functions
> - Architected image loading, resizing, normalization, and CHW conversion utilities.
> - Implementation of image loading/saving (via stb_image) and host/device memory helpers
> - Architected CUDA kernels for diffusion scheduling, noise generation, and reverse sampling.
> - Header for the noise scheduler, sinusoidal time-embedding table, and forward/reverse diffusion kernel launchers
> - Implementation of the diffusion scheduler, time-embedding table, and the forward/reverse diffusion CUDA kernels using in-kernel curand
> - Architected CUDA network structs, kernels, and memory management strategy.
> - Header defining the network's layer structs and the public forward-pass API
> - **Done**

## User:

> 8/1/2026 10:40:32

continue this response

## Assistant:

> 8/1/2026 10:57:15

> Architected CUDA kernels and optimized gradient buffer reuse.
>
> - Orchestrated CUDA kernel implementations and subsequent module architecture.
> - Implementation of the forward-pass CUDA kernels (tiled conv2d, linear, time-bias broadcast, SiLU) and the network init/forward/free host functions
> - Architected CUDA kernels and optimized gradient buffer reuse.
> - **Done**

Now I realize I can simplify the network struct — since the conv-bias gradient and time-bias gradient are mathematically identical (verified in the numpy check), I don't need separate buffers. Let me clean that up before writing `train.cu`.

> Architected GPU-accelerated diffusion model training pipeline orchestration.
>
> - Remove the redundant d_bias1/d_bias2 buffers: conv1.db/conv2.db already hold the exact same gradient value (proved in the numpy verification) and can be reused directly as the input to the time-layer backward pass
> - Remove allocation of the now-eliminated d_bias1/d_bias2 buffers from network_init
> - Remove freeing of the now-eliminated d_bias1/d_bias2 buffers from network_free
> - Architected CUDA kernel implementation for training module.
> - Header declaring the loss, backward-pass, and Adam optimizer entry points
> - Architected backward pass kernels and host wrappers.
> - Implementation of the backward-pass CUDA kernels (conv2d weight/input gradients, channel-sum bias reduction, SiLU backward, linear backward), MSE loss with parallel reduction, and the Adam optimizer kernel
> - Architected reverse diffusion sampling with pointer optimization and memory management.
> - Header for the reverse-diffusion image generation entry point
> - Implementation of the reverse-diffusion sampling loop that turns pure Gaussian noise into a generated image
> - Architected GPU-accelerated diffusion model training pipeline orchestration.
> - **Done**

Now the orchestration file that ties everything together.

> Deliberated CUDA validation strategy, settled on compile-link checking.
>
> - Main entry point: parses CLI args, loads the image dataset, builds the diffusion schedule, initializes the network, runs the training loop, and triggers generation
> - Deliberated CUDA compiler flags and build configuration decisions.
> - Makefile with nvcc optimization flags, configurable GPU architecture, and a convenience 'run' target
> - Deliberated CUDA validation strategy, settled on compile-link checking.
> - **Done**

Now let me try to catch any C++/structural bugs by compiling a CUDA-runtime "shim" version of the whole project with g++ — this can't validate GPU semantics, but it will catch typos, signature mismatches between headers and implementations, missing includes, and other structural errors before handing over "production-ready" code.

> Diagnosed shim limitations and strategized preprocessing fixes.
>
> - Set up a scratch copy of the project for host-compiler structural verification (not touching the real deliverable files)
> - Architected CUDA integration framework and transformation pipeline.
> - Fake CUDA runtime/curand shim so g++ can parse and link the project's C++ structure (not a semantic GPU emulation, just enough type/symbol surface to catch typos, signature mismatches, and missing includes)
> - Architected CUDA file transformation pipeline with header substitution and syntax stripping.
> - Script to transform the real .cu sources into g++-parseable .cpp copies (swap CUDA runtime header for the shim, strip <<<...>>> kernel-launch syntax) for structural verification only
> - Run the transform script and sanity-check that all kernel launch syntax was stripped and headers swapped
> - Compile each shimmed source file individually to catch per-file syntax/type errors
> - Diagnosed shim limitations and strategized preprocessing fixes.
> - **Done**

The two errors are both limitations of my *shim*, not bugs in the real code — `dim3` needs a real constructor, and `extern __shared__` (CUDA's syntax for dynamically-sized shared memory, which I used correctly) needs special handling since a shim can't emulate "size determined by kernel launch." Let me fix the shim.

> Validated compilation, verified CUDA syntax, planned runtime testing.
>
> - Give Dim3shim a real (x,y,z) constructor so dim3 block(16,16) syntax parses, matching real CUDA's dim3
> - Strategized code modification for specialized array declarations.
> - Add a substitution for the extern __shared__ dynamic-shared-memory idiom, which is valid, correct CUDA but can't be macro-shimmed generically
> - Recompile all shimmed sources after fixing the shim's dim3 constructor and extern __shared__ handling
> - Confirm the remaining warnings are only the benign #pragma unroll notices (nvcc-specific pragma unknown to g++)
> - Link all compiled translation units together to catch cross-file signature mismatches, missing definitions, or duplicate symbols
> - Validated compilation, verified CUDA syntax, planned runtime testing.
> - **Done**

The project links cleanly — no missing symbols or signature mismatches across all 6 files. Let me also do a runtime smoke test to catch any memory bugs (segfaults, buffer overruns) that only show up when the program actually executes its full control flow.

> Validated implementation and prepared comprehensive project review.
>
> - Generate a small synthetic RGB test PNG using the real (unshimmed) stb_image_write to feed into the smoke test
> - Run the full shimmed pipeline end-to-end (load image -> train loop -> reverse-diffusion generation -> save PNG) to check for crashes, segfaults, or hangs
> - Verify the generated output file is a structurally valid PNG
> - Validated implementation and prepared comprehensive project review.
> - **Done**

All verification passes: math is correct (finite-difference gradient check), the tiled-convolution indexing is correct (matches reference exactly, including boundaries), the C++ structure compiles and links cleanly across all 6 files with zero real warnings, and a full end-to-end run (image load → train loop → 300-step reverse diffusion → PNG save) completes without crashing and produces a valid PNG. Let me do a final review pass of the real deliverable files.

> Verified plan and prepared deliverable package.
>
> - Final directory listing and line-count summary of the real deliverable project
> - Get the final, current content of network.h after edits
> - Get the final, current content of network.cu after edits
> - Verified plan and prepared deliverable package.
> - **Done**

Everything checks out. Now let me package the complete project for download.

> Orchestrated comprehensive HPC project delivery with rigorous verification methodology.
>
> - Add placeholder README notes for the data/output directories and zip the complete, verified project for the user to download
> - Orchestrated comprehensive HPC project delivery with rigorous verification methodology.
> - **Done**

All verified and packaged. Here is the complete, from-scratch CUDA C++ DDPM implementation.

Before diving in: since this sandbox has no GPU/`nvcc`, I didn't just write this from memory — I derived every backward-pass formula, checked it against **finite-difference gradient checking in NumPy** (all 10 parameter tensors matched analytic vs. numerical gradients to 1e-7), emulated the tiled shared-memory kernel's exact indexing logic in Python and confirmed it matches direct convolution (including boundary tiles), verified the reverse-sampling coefficients against the closed-form DDPM posterior, and finally compiled+linked+**ran** the entire real project end-to-end against a CUDA-runtime shim with g++ (no crashes, valid PNG produced) to catch structural bugs. Details are in the verification section at the end.

# 1. Project Setup & Directory Structure

```
ddpm_cuda/
├── Makefile
├── include/
│   ├── common.h              # shared macros, hyperparameters, CUDA_CHECK
│   ├── utils.h                # image I/O + host/device memory helpers
│   ├── diffusion.h            # noise scheduler + time embeddings
│   ├── network.h              # ConvLayer/LinearLayer/Network structs
│   ├── train.h                # loss, backward pass, Adam optimizer
│   ├── generate.h             # reverse-diffusion sampling
│   ├── stb_image.h            # third-party, public domain (image decode)
│   └── stb_image_write.h      # third-party, public domain (image encode)
├── src/
│   ├── utils.cu
│   ├── diffusion.cu
│   ├── network.cu
│   ├── train.cu
│   ├── generate.cu
│   └── main.cu
├── data/                       # put your input PNG/JPG here
└── output/                     # generated PNG lands here
```

`stb_image.h`/`stb_image_write.h` are Sean Barrett's widely-used single-header libraries (~7,000 and ~1,700 lines respectively) — I pulled the genuine, unmodified files directly from `https://raw.githubusercontent.com/nothings/stb/master/` rather than retyping them inline; they're included verbatim in the project download at the end. Everything else below (13 files, ~1,670 lines) is original and written for this project.

```makefile
# ==============================================================================
# Makefile -- from-scratch CUDA C++ DDPM
#
# Usage:
#   make                        # build with default ARCH=sm_80 (Ampere/A100)
#   make ARCH=sm_86              # build for a different GPU (e.g. RTX 30xx)
#   make run IMG=data/cat.png    # build (if needed), then train + generate
#   make clean                   # remove build artifacts
#
# Common ARCH values (GPU compute capability -- check yours with
# `nvidia-smi --query-gpu=compute_cap --format=csv`):
#   sm_70  Volta    (V100)
#   sm_75  Turing   (RTX 20xx, T4)
#   sm_80  Ampere   (A100)
#   sm_86  Ampere   (RTX 30xx)
#   sm_89  Ada      (RTX 40xx, L4)
#   sm_90  Hopper   (H100)
# ==============================================================================

NVCC := nvcc
ARCH ?= sm_80

INCLUDES := -Iinclude

# -O3                 optimize host-side C++ (arg parsing, schedule/embedding
#                     precompute, the training-loop driver code)
# -arch=$(ARCH)       target GPU compute capability; selects which SASS
#                     variant every __global__ kernel below is assembled for
# -std=c++17          host-code language standard (std::vector/string/mt19937)
# --use_fast_math     lets device math (expf, sqrtf, ...) use faster
#                     approximate intrinsics; a good trade for a demo that
#                     already only targets float32 image data. Drop this flag
#                     if you need bit-exact IEEE math.
# -Xptxas -O3          also raise the optimization level of the PTX->SASS
#                     assembler stage (the second half of nvcc's two-stage
#                     compile pipeline), not just the front-end
# -lineinfo           keep source line info in the binary so `cuda-gdb` /
#                     `compute-sanitizer` / `nsight compute` can map crashes
#                     and profiler hotspots back to this source
NVCC_FLAGS := -O3 -arch=$(ARCH) -std=c++17 --use_fast_math -Xptxas -O3 -lineinfo $(INCLUDES)

# NOTE ON LINKING: we deliberately do NOT pass -lcurand. Every random number
# this project generates uses curand's DEVICE API (curand_init/curand_normal,
# called from inside __global__ kernels via <curand_kernel.h>), which is a
# header-only implementation compiled directly into each kernel. -lcurand is
# only required if you call the HOST API (curandCreateGenerator, etc.). We
# also do not pass -rdc=true (relocatable device code): every kernel is only
# ever launched from the same .cu file that defines it, so each translation
# unit can be compiled as a fully self-contained program -- simpler and
# slightly faster to build than enabling cross-file device linking we don't
# need.
LD_FLAGS :=

SRC_DIR := src
OBJ_DIR := build
BIN_DIR := bin

SOURCES := $(wildcard $(SRC_DIR)/*.cu)
OBJECTS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(SOURCES))
TARGET  := $(BIN_DIR)/ddpm

.PHONY: all clean run dirs

all: dirs $(TARGET)

dirs:
	@mkdir -p $(OBJ_DIR) $(BIN_DIR) output

$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $(OBJECTS) -o $@ $(LD_FLAGS)
	@echo "Built $(TARGET)"

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | dirs
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Convenience target: `make run IMG=data/your_image.png ITERS=5000`
IMG   ?= data/input.png
ITERS ?= 5000
run: all
	$(BIN_DIR)/ddpm $(IMG) --iters $(ITERS) --out output/generated.png

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
```

### Line-by-Line Explanation

- **`ARCH ?= sm_80`** — `?=` means "set only if not already set," so `make ARCH=sm_86` overrides it from the command line without editing the file. `sm_80` targets Ampere (A100); pick the value matching your card via `nvidia-smi --query-gpu=compute_cap`.
- **`-arch=$(ARCH)`** is the single most important flag: it tells `nvcc` which **SASS** (Streaming Assembler) instruction set to generate for every `__global__` kernel. Get it wrong and the binary either won't run at all or silently falls back to a much slower JIT-compiled PTX path.
- **`--use_fast_math`** swaps IEEE-precise `expf`/`sqrtf`/division for faster approximate hardware intrinsics (`__expf`, reciprocal-sqrt instructions). SiLU and Adam use `expf`/`sqrtf` in the hot path, so this is a real, meaningful speedup here — at the cost of a few ULPs of precision, irrelevant for float32 image data.
- **`-Xptxas -O3`** — `nvcc` compiles in two stages: its own frontend emits PTX (a virtual ISA), then `ptxas` assembles PTX into real SASS. `-O3` alone mostly governs the frontend/host side; `-Xptxas -O3` forwards the flag through to the assembler stage too.
- **`-lineinfo`** embeds source-line mappings in the binary (small size cost) so `compute-sanitizer` (CUDA's memcheck/racecheck tool) and `cuda-gdb` can point at exact source lines instead of raw addresses — invaluable the first time a kernel segfaults.
- **No `-lcurand`, no `-rdc=true`** — explained in the comment block itself: we only use curand's *device* API (header-only, compiled directly into each kernel body, no separate `.so` needed), and every kernel is launched only from the same file that defines it, so whole-program (non-relocatable) device-code compilation is sufficient and simpler.
- **`$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu`** is a classic GNU Make pattern rule: for any `build/foo.o` target, look for `src/foo.cu` as the prerequisite. `$<` expands to that prerequisite, `$@` to the target — this is what lets the Makefile compile all 6 `.cu` files without repeating the compile command 6 times.
- **`| dirs`** (order-only prerequisite) ensures `build/`, `bin/`, `output/` exist before compilation, without forcing recompilation every time (a normal prerequisite would trigger a rebuild whenever the directory's mtime changed).

# 2. Image I/O & Memory Management (`utils.h` / `utils.cu`)

```cpp
#ifndef UTILS_H
#define UTILS_H

#include <cstddef>

// ----------------------------------------------------------------------------
// Image I/O. Both functions work in "CHW, float, [-1, 1]" layout:
//   index(c, y, x) = c * IMG_SIZE * IMG_SIZE + y * IMG_SIZE + x
// which is the layout every kernel in network.cu / diffusion.cu assumes
// (channel-major so each channel is one contiguous IMG_SIZE*IMG_SIZE plane).
// ----------------------------------------------------------------------------

// Loads an image file (PNG/JPG/BMP/... - anything stb_image supports),
// forces it to 3 channels, nearest-neighbor resizes to IMG_SIZE x IMG_SIZE,
// and returns a freshly malloc'd host float array of length IMG_ELEMS holding
// CHW data normalized to [-1, 1]. Caller owns the returned pointer (free()).
// Returns nullptr on failure (bad path, unsupported format, decode error).
float* load_image_chw(const char* path);

// Writes a CHW float array (values expected in roughly [-1, 1], clamped
// defensively) of length IMG_ELEMS out to `path` as an 8-bit PNG.
// Returns true on success.
bool save_image_chw(const char* path, const float* chw);

// ----------------------------------------------------------------------------
// Host <-> device transfer helpers. Thin, explicit wrappers around
// cudaMalloc/cudaMemcpy/cudaFree so every allocation site in the project
// goes through one error-checked choke point (see CUDA_CHECK in common.h).
// ----------------------------------------------------------------------------

// Allocates `n_floats` floats of device global memory. Aborts on failure.
float* device_alloc(size_t n_floats);

// Allocates `n_floats` floats of device memory and zero-initializes them
// (used for gradients and Adam moment buffers, which must start at 0).
float* device_alloc_zero(size_t n_floats);

// Copies `n_floats` floats from a host array into an existing device array.
void copy_to_device(float* d_dst, const float* h_src, size_t n_floats);

// Copies `n_floats` floats from an existing device array into a host array.
void copy_to_host(float* h_dst, const float* d_src, size_t n_floats);

// Convenience: allocate a device buffer AND immediately upload `h_src` into it.
float* device_alloc_from_host(const float* h_src, size_t n_floats);

#endif // UTILS_H
```

```cpp
// utils.cu
//
// Image I/O (via stb_image / stb_image_write) and host<->device memory
// transfer helpers. This is the ONLY translation unit that defines
// STB_IMAGE_IMPLEMENTATION / STB_IMAGE_WRITE_IMPLEMENTATION -- those macros
// tell the stb single-header libraries to emit their function BODIES here
// (every other file that includes stb_image.h only sees the declarations).

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "common.h"
#include "utils.h"

// ============================================================================
// load_image_chw
// ============================================================================
float* load_image_chw(const char* path) {
    int src_w, src_h, src_c;
    // Force 3 output channels regardless of source format (grayscale, RGBA,
    // palette, ...); stb_image will do the channel conversion for us.
    unsigned char* pixels = stbi_load(path, &src_w, &src_h, &src_c, 3);
    if (!pixels) {
        fprintf(stderr, "[utils] failed to load image '%s': %s\n",
                path, stbi_failure_reason());
        return nullptr;
    }

    // --- Step 1: nearest-neighbor resize from (src_w x src_h) to
    //     (IMG_SIZE x IMG_SIZE), still interleaved HWC uint8. We implement
    //     resizing by hand (no stb_image_resize allowed per project scope):
    //     for every destination pixel, map back to the nearest source pixel.
    unsigned char* resized = (unsigned char*)malloc((size_t)IMG_SIZE * IMG_SIZE * 3);
    for (int y = 0; y < IMG_SIZE; y++) {
        int sy = (int)((long long)y * src_h / IMG_SIZE);
        if (sy >= src_h) sy = src_h - 1;
        for (int x = 0; x < IMG_SIZE; x++) {
            int sx = (int)((long long)x * src_w / IMG_SIZE);
            if (sx >= src_w) sx = src_w - 1;
            const unsigned char* srcpx = &pixels[(sy * src_w + sx) * 3];
            unsigned char* dstpx = &resized[(y * IMG_SIZE + x) * 3];
            dstpx[0] = srcpx[0];
            dstpx[1] = srcpx[1];
            dstpx[2] = srcpx[2];
        }
    }
    stbi_image_free(pixels);

    // --- Step 2: interleaved HWC uint8 [0,255]  ->  planar CHW float [-1,1].
    // Planar (channel-major) layout is what every conv kernel expects:
    // index(c,y,x) = c*IMG_SIZE*IMG_SIZE + y*IMG_SIZE + x, so a whole channel
    // is one contiguous run of IMG_SIZE*IMG_SIZE floats.
    float* chw = (float*)malloc((size_t)IMG_ELEMS * sizeof(float));
    const int plane = IMG_SIZE * IMG_SIZE;
    for (int y = 0; y < IMG_SIZE; y++) {
        for (int x = 0; x < IMG_SIZE; x++) {
            const unsigned char* srcpx = &resized[(y * IMG_SIZE + x) * 3];
            for (int c = 0; c < 3; c++) {
                float v01 = srcpx[c] / 255.0f;          // [0, 1]
                chw[c * plane + y * IMG_SIZE + x] = v01 * 2.0f - 1.0f; // [-1, 1]
            }
        }
    }
    free(resized);
    return chw;
}

// ============================================================================
// save_image_chw
// ============================================================================
bool save_image_chw(const char* path, const float* chw) {
    unsigned char* hwc = (unsigned char*)malloc((size_t)IMG_SIZE * IMG_SIZE * 3);
    const int plane = IMG_SIZE * IMG_SIZE;
    for (int y = 0; y < IMG_SIZE; y++) {
        for (int x = 0; x < IMG_SIZE; x++) {
            for (int c = 0; c < 3; c++) {
                float v = chw[c * plane + y * IMG_SIZE + x]; // ~[-1,1]
                v = (v + 1.0f) * 0.5f;                        // -> ~[0,1]
                v = std::min(1.0f, std::max(0.0f, v));        // clamp (generation
                                                                // can overshoot)
                int iv = (int)std::lround(v * 255.0f);
                hwc[(y * IMG_SIZE + x) * 3 + c] = (unsigned char)iv;
            }
        }
    }
    int ok = stbi_write_png(path, IMG_SIZE, IMG_SIZE, 3, hwc, IMG_SIZE * 3);
    free(hwc);
    if (!ok) {
        fprintf(stderr, "[utils] failed to write image '%s'\n", path);
        return false;
    }
    return true;
}

// ============================================================================
// Device memory helpers
// ============================================================================
float* device_alloc(size_t n_floats) {
    float* ptr = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&ptr, n_floats * sizeof(float)));
    return ptr;
}

float* device_alloc_zero(size_t n_floats) {
    float* ptr = device_alloc(n_floats);
    CUDA_CHECK(cudaMemset(ptr, 0, n_floats * sizeof(float)));
    return ptr;
}

void copy_to_device(float* d_dst, const float* h_src, size_t n_floats) {
    CUDA_CHECK(cudaMemcpy(d_dst, h_src, n_floats * sizeof(float),
                           cudaMemcpyHostToDevice));
}

void copy_to_host(float* h_dst, const float* d_src, size_t n_floats) {
    CUDA_CHECK(cudaMemcpy(h_dst, d_src, n_floats * sizeof(float),
                           cudaMemcpyDeviceToHost));
}

float* device_alloc_from_host(const float* h_src, size_t n_floats) {
    float* d_ptr = device_alloc(n_floats);
    copy_to_device(d_ptr, h_src, n_floats);
    return d_ptr;
}
```

### Line-by-Line Explanation

**Common header (also in `common.h`, used everywhere):**
- `CUDA_CHECK(call)` wraps every runtime call, checks the returned `cudaError_t`, and `exit()`s with file/line/message on failure. Without this, a failed `cudaMalloc` returns a null pointer *silently*, and the crash you eventually see is in some unrelated kernel three calls later — this macro turns invisible corruption into an immediate, localized, readable error.

**`load_image_chw`:**
- `stbi_load(path, &src_w, &src_h, &src_c, 3)` — the trailing `3` is `desired_channels`; stb_image transparently converts grayscale/RGBA/palette source images into 3-channel RGB output for us, so downstream code never branches on source format.
- **Resize loop**: `sy = y * src_h / IMG_SIZE` is the nearest-neighbor backward mapping — for each *destination* pixel we compute which *source* pixel is closest, rather than forward-scattering source pixels (which would leave gaps). Using `long long` for the intermediate product avoids `int` overflow for large source images before dividing.
- **CHW conversion loop**: `chw[c * plane + y * IMG_SIZE + x]` — this is the layout every single kernel downstream assumes. Getting this indexing formula wrong here would silently scramble channels and every conv kernel would learn on nonsense; it's worth memorizing: **channel-major, then row-major within a channel**.
- Normalization `v01 * 2.0f - 1.0f` maps `[0,1] -> [-1,1]`, matching the original DDPM paper's preprocessing (a network predicting `N(0,1)`-scale noise trains far better against a zero-centered signal than a `[0,1]`-only one).

**`save_image_chw`:**
- Inverts normalization: `(v + 1)*0.5` then clamps to `[0,1]` before scaling to `[0,255]` — the clamp matters because a partially-trained network's generated samples can legitimately overshoot slightly past `±1`.
- `std::lround` rounds to nearest integer (rather than truncating, which would introduce a small negative bias) before the narrowing cast to `unsigned char`.

**Device memory helpers:**
- `device_alloc_zero` is `cudaMalloc` + `cudaMemset(...,0,...)` — used for every gradient and Adam moment buffer, which must start at exactly zero (Adam's bias-correction math assumes `m_0 = v_0 = 0`).
- `copy_to_device` / `copy_to_host` are one-line `cudaMemcpy` wrappers, but centralizing them means every transfer in the project is `CUDA_CHECK`-guarded without repeating that boilerplate at 20+ call sites.
- `device_alloc_from_host` is the common "allocate + immediately upload a freshly-computed host array" pattern (used for every weight tensor and the time-embedding table) — one call instead of three.

# 3. Forward Diffusion & Scheduler (`diffusion.h` / `diffusion.cu`)

**The math first.** DDPM (Ho et al., 2020) defines a fixed forward Markov chain that gradually destroys an image with Gaussian noise over `T` steps, and trains a network to reverse it one step at a time.

Forward (closed form, given a linear noise schedule `β_t`):

$$q(x_t\mid x_0)=\mathcal N\!\big(x_t;\ \sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I\big)\ \Longrightarrow\ x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\quad \epsilon\sim\mathcal N(0,I)$$

where $\alpha_t = 1-\beta_t$ and $\bar\alpha_t=\prod_{s\le t}\alpha_s$. Training objective (the "simple" loss): sample a random $t$, sample $\epsilon$, form $x_t$, and train a network $\epsilon_\theta(x_t,t)$ to predict $\epsilon$ via MSE. That single equation is *all* of section 3 and *all* of the training loop in `main.cu`.

```cpp
#ifndef DIFFUSION_H
#define DIFFUSION_H

#include "common.h"

// ----------------------------------------------------------------------------
// DiffusionSchedule: the beta/alpha/alpha_bar tables that define the forward
// (noising) Markov chain q(x_1..x_T | x_0). Lives entirely on the HOST.
// Kernels never index into these arrays directly -- for a given timestep t
// we just look up the handful of scalars we need (e.g. sched.alpha[t]) in
// host code and pass them BY VALUE as kernel launch arguments. This avoids
// needing a device copy of the schedule at all, since we only ever need one
// timestep's worth of coefficients per kernel launch.
// ----------------------------------------------------------------------------
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

// ----------------------------------------------------------------------------
// Forward process q(x_t | x_0):  x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 -
// alpha_bar_t) * epsilon,  epsilon ~ N(0, I).
// Generates epsilon with an in-kernel curand generator (freshly seeded per
// launch via `seed`), writes the noisy sample to d_xt AND writes the sampled
// epsilon to d_noise_out (this epsilon is the network's training target).
// All three buffers (d_x0, d_xt, d_noise_out) are device pointers of length
// IMG_ELEMS.
// ----------------------------------------------------------------------------
void diffusion_forward_sample(const float* d_x0, float* d_xt, float* d_noise_out,
                               const DiffusionSchedule& sched, int t,
                               unsigned long long seed);

// ----------------------------------------------------------------------------
// Reverse process p_theta(x_{t-1} | x_t): one denoising step.
//   mean   = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1-alpha_bar_t)) * eps_pred)
//   x_{t-1} = mean + sqrt(beta_t) * z,   z ~ N(0,I) if t > 0, else z = 0
// d_xt, d_eps_pred, d_x_prev are device pointers of length IMG_ELEMS.
// ----------------------------------------------------------------------------
void diffusion_reverse_step(const float* d_xt, const float* d_eps_pred,
                             float* d_x_prev, const DiffusionSchedule& sched,
                             int t, unsigned long long seed);

// Fills a device buffer of length IMG_ELEMS with iid standard-normal samples
// (x_T ~ N(0, I), the starting point of the reverse/generation chain).
void diffusion_sample_prior(float* d_x, unsigned long long seed);

#endif // DIFFUSION_H
```

```cpp
// diffusion.cu
//
// Everything specific to the DIFFUSION PROCESS itself, as opposed to the
// neural network that learns to reverse it:
//   - the noise schedule (beta_t, alpha_t, alpha_bar_t)
//   - the sinusoidal timestep embedding table
//   - the forward kernel that corrupts an image with noise (training)
//   - the reverse kernel that removes a predicted amount of noise (sampling)
//
// Both stochastic kernels use curand's DEVICE API (curand_init +
// curand_normal called from inside __global__ code), as requested, rather
// than pre-generating noise on the host. See the write-up for the
// performance tradeoff this implies.

#include <cmath>
#include <curand_kernel.h>

#include "common.h"
#include "diffusion.h"
#include "utils.h"

// ============================================================================
// Host-side schedule construction
// ============================================================================
void build_diffusion_schedule(DiffusionSchedule* sched) {
    float alpha_bar_running = 1.0f;
    for (int t = 0; t < TIMESTEPS; t++) {
        // Linear beta schedule, exactly as in Ho et al. 2020.
        float frac = (TIMESTEPS > 1) ? ((float)t / (float)(TIMESTEPS - 1)) : 0.0f;
        float beta_t = BETA_START + (BETA_END - BETA_START) * frac;
        float alpha_t = 1.0f - beta_t;
        alpha_bar_running *= alpha_t;

        sched->beta[t]  = beta_t;
        sched->alpha[t] = alpha_t;
        sched->alpha_bar[t] = alpha_bar_running;
        sched->sqrt_alpha_bar[t] = sqrtf(alpha_bar_running);
        sched->sqrt_one_minus_alpha_bar[t] = sqrtf(1.0f - alpha_bar_running);
    }
}

// ============================================================================
// Sinusoidal time embedding table (host compute, one-time device upload)
// ============================================================================
float* build_time_embedding_table_device() {
    float* h_table = (float*)malloc((size_t)TIMESTEPS * TIME_EMB_DIM * sizeof(float));

    for (int t = 0; t < TIMESTEPS; t++) {
        for (int i = 0; i < TIME_EMB_DIM / 2; i++) {
            // Standard Transformer/DDPM sinusoidal encoding:
            //   freq_i = 10000 ^ (-2i / d)
            //   emb[2i]   = sin(t * freq_i)
            //   emb[2i+1] = cos(t * freq_i)
            float freq  = powf(10000.0f, -2.0f * (float)i / (float)TIME_EMB_DIM);
            float angle = (float)t * freq;
            h_table[t * TIME_EMB_DIM + 2 * i]     = sinf(angle);
            h_table[t * TIME_EMB_DIM + 2 * i + 1] = cosf(angle);
        }
    }

    float* d_table = device_alloc_from_host(h_table, (size_t)TIMESTEPS * TIME_EMB_DIM);
    free(h_table);
    return d_table;
}

// ============================================================================
// KERNEL: forward_diffusion_kernel
// One thread per element of the (flattened) image. Each thread:
//   1. seeds its own private curand generator,
//   2. draws one standard-normal sample epsilon,
//   3. computes x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*epsilon.
// ============================================================================
__global__ void forward_diffusion_kernel(const float* __restrict__ x0,
                                          float* __restrict__ xt,
                                          float* __restrict__ noise_out,
                                          float sqrt_ab, float sqrt_omab,
                                          unsigned long long seed, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    curandState state;
    // Same `seed` for every thread in this launch, but a distinct
    // "subsequence" (2nd arg) per thread. curand guarantees per-subsequence
    // streams are statistically independent, which is what actually makes
    // this "one RNG per pixel" instead of every pixel seeing the same value.
    curand_init(seed, (unsigned long long)i, 0, &state);
    float eps = curand_normal(&state);

    noise_out[i] = eps;
    xt[i] = sqrt_ab * x0[i] + sqrt_omab * eps;
}

void diffusion_forward_sample(const float* d_x0, float* d_xt, float* d_noise_out,
                               const DiffusionSchedule& sched, int t,
                               unsigned long long seed) {
    const int n = IMG_ELEMS;
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    forward_diffusion_kernel<<<blocks, threads>>>(
        d_x0, d_xt, d_noise_out,
        sched.sqrt_alpha_bar[t], sched.sqrt_one_minus_alpha_bar[t],
        seed, n);
    CUDA_CHECK_LAST();
}

// ============================================================================
// KERNEL: reverse_diffusion_step_kernel
// One thread per pixel. Applies the DDPM reverse-process mean update and
// (except at t==0) injects fresh noise scaled by sigma_t = sqrt(beta_t).
// ============================================================================
__global__ void reverse_diffusion_step_kernel(const float* __restrict__ xt,
                                               const float* __restrict__ eps_pred,
                                               float* __restrict__ x_prev,
                                               float inv_sqrt_alpha,
                                               float noise_coef,
                                               float sigma,
                                               bool add_noise,
                                               unsigned long long seed, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float mean = inv_sqrt_alpha * (xt[i] - noise_coef * eps_pred[i]);

    if (add_noise) {
        curandState state;
        curand_init(seed, (unsigned long long)i, 0, &state);
        float z = curand_normal(&state);
        x_prev[i] = mean + sigma * z;
    } else {
        x_prev[i] = mean;
    }
}

void diffusion_reverse_step(const float* d_xt, const float* d_eps_pred,
                             float* d_x_prev, const DiffusionSchedule& sched,
                             int t, unsigned long long seed) {
    const int n = IMG_ELEMS;
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    float inv_sqrt_alpha = 1.0f / sqrtf(sched.alpha[t]);
    float noise_coef = sched.beta[t] / sched.sqrt_one_minus_alpha_bar[t];
    float sigma = sqrtf(sched.beta[t]);
    bool add_noise = (t > 0);

    reverse_diffusion_step_kernel<<<blocks, threads>>>(
        d_xt, d_eps_pred, d_x_prev,
        inv_sqrt_alpha, noise_coef, sigma, add_noise, seed, n);
    CUDA_CHECK_LAST();
}

// ============================================================================
// KERNEL: prior_noise_kernel -- fills a buffer with iid N(0,1) samples.
// Used once to create x_T at the start of generation.
// ============================================================================
__global__ void prior_noise_kernel(float* __restrict__ x,
                                    unsigned long long seed, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandState state;
    curand_init(seed, (unsigned long long)i, 0, &state);
    x[i] = curand_normal(&state);
}

void diffusion_sample_prior(float* d_x, unsigned long long seed) {
    const int n = IMG_ELEMS;
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    prior_noise_kernel<<<blocks, threads>>>(d_x, seed, n);
    CUDA_CHECK_LAST();
}
```

### Line-by-Line Explanation

- **`DiffusionSchedule` lives on the host only.** This is a deliberate HPC choice: each kernel launch needs only *one* timestep's worth of scalars (`alpha[t]`, `beta[t]`, ...), so we look those up in host code and pass them **by value** as kernel arguments (they land in fast constant/parameter memory automatically) rather than uploading the whole 300-entry table to the device and having every thread perform a global-memory load `sched[t]` — that would be strictly slower for zero benefit.
- **`build_diffusion_schedule`**: a single sequential host loop computing `β_t` linearly between `BETA_START=1e-4` and `BETA_END=0.02`, then `α_t = 1-β_t`, then the running product `alpha_bar_running *= alpha_t` (this *is* $\bar\alpha_t=\prod_{s\le t}\alpha_s$ — computed incrementally instead of a fresh product each iteration, turning an $O(T^2)$ naive computation into $O(T)$). 300 scalar iterations costs nothing.
- **`build_time_embedding_table_device`**: implements the exact Transformer positional-encoding formula, $\text{freq}_i = 10000^{-2i/d}$, sin/cos interleaved. Because the network only ever sees $t\in\{0,\dots,T-1\}$ at both train and sample time (a closed, discrete system — no need to generalize to unseen continuous `t`), a plain lookup table is fully sufficient; there's no approximation happening here, it's the same table a "real" implementation would compute.
- **`forward_diffusion_kernel` — coalesced memory access.** Thread `i` reads `x0[i]` and writes `xt[i]`/`noise_out[i]` — for a warp (32 consecutive threads, `i..i+31`), that's 32 consecutive 4-byte floats = a single contiguous 128-byte region per array per warp. The GPU's memory controller services this as one (or two, depending on cache-line alignment) wide transaction instead of 32 separate ones — this is the textbook definition of a coalesced access, and it's why every elementwise kernel in this project is written "index = `blockIdx.x*blockDim.x+threadIdx.x`, threadIdx.x maps to the fastest-varying array dimension."
- **`curand_init(seed, i, 0, &state)`** — the **second** argument is curand's "subsequence" number. Two calls with the same `seed` but different subsequence indices produce statistically independent streams (this is what curand explicitly guarantees, unlike naively XOR-ing a global seed with the index yourself). Every pixel therefore gets an independent Gaussian sample, and every *training iteration* gets fresh noise because we increment `seed` (a host-side counter) between calls.
- **Performance tradeoff, stated explicitly**: `curand_init` does real work (several rounds of a hash/permutation to decorrelate subsequences) — re-running it from scratch for all ~3,072 threads on *every single kernel launch* is not the fastest possible design. A production training loop would instead allocate one persistent `curandState` array once and reuse it call after call. For this project's scope (3,072-element images, a few thousand iterations) it's microseconds of overhead and buys a simpler, fully self-contained kernel — the tradeoff the prompt explicitly asked for ("custom seeded RNG inside the kernel").
- **`reverse_diffusion_step_kernel`** implements $x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta\right)+\sqrt{\beta_t}\,z$ directly — `inv_sqrt_alpha`, `noise_coef`, and `sigma` are precomputed *once on the host* per step (not recomputed by every one of the 3,072 threads) and passed as scalar kernel arguments. `add_noise` is `false` only at `t==0`, matching Ho et al.'s Algorithm 2 exactly (the very last step is deterministic).

# 4. Neural Network Kernels (`network.h` / `network.cu`)

**Architecture** (a "simple CNN backbone," which the brief explicitly permits as an alternative to a full U-Net, chosen to keep 13 files mathematically airtight rather than stretch a skip-connected U-Net thin):

$$z_1=\text{conv}_1(x_t)+\text{tproj}_1(t)\ \ (\text{broadcast}),\quad h_1=\text{SiLU}(z_1)$$
$$z_2=\text{conv}_2(h_1)+\text{tproj}_2(t),\quad h_2=\text{SiLU}(z_2)$$
$$\text{out}=\text{conv}_3(h_2)\quad(\text{linear, predicted noise } \epsilon_\theta)$$

3×3/pad-1/stride-1 convs, channels 3→32→32→3, each hidden layer FiLM-conditioned on `t` via a small linear projection of the sinusoidal embedding, added as a per-channel bias before the activation.

```cpp
#ifndef NETWORK_H
#define NETWORK_H

#include "common.h"

// ----------------------------------------------------------------------------
// ConvLayer / LinearLayer: each bundles a parameter tensor with its gradient
// buffer (dW/db, populated by train.cu) and its Adam first/second moment
// buffers (mW/vW/mb/vb, also updated by train.cu). Keeping grad + optimizer
// state alongside the parameter itself means adam_step() can walk the
// network as a flat list of (param, grad, m, v, size) tuples -- see train.cu.
// ----------------------------------------------------------------------------
struct ConvLayer {
    int Cin, Cout;              // CONV_K x CONV_K spatial size is a global constant
    float *W, *b;                // parameters:  W is (Cout,Cin,K,K), b is (Cout)
    float *dW, *db;               // gradients, same shapes as W, b
    float *mW, *vW, *mb, *vb;     // Adam moment buffers, same shapes as W, b
};

struct LinearLayer {
    int in_dim, out_dim;
    float *W, *b;                 // W is (out_dim, in_dim), b is (out_dim)
    float *dW, *db;
    float *mW, *vW, *mb, *vb;
};

// ----------------------------------------------------------------------------
// Network: the full micro noise-predictor epsilon_theta(x_t, t).
//
//   z1 = conv1(x_t) + time1(temb)[:,None,None]   ; h1 = SiLU(z1)
//   z2 = conv2(h1)  + time2(temb)[:,None,None]   ; h2 = SiLU(z2)
//   out = conv3(h2)                                (predicted noise, linear)
//
// Besides the 3 conv + 2 linear layers, the struct also owns every
// intermediate activation buffer (z1,h1,z2,h2,tb1,tb2,out) so that
// network_backward() (train.cu) can read them back during backprop instead
// of recomputing the forward pass. This "activation cache" pattern is the
// direct CUDA analogue of what autograd does implicitly in PyTorch/TF.
// ----------------------------------------------------------------------------
struct Network {
    ConvLayer   conv1;   // IMG_CHANNELS -> HIDDEN_CH
    LinearLayer time1;   // TIME_EMB_DIM -> HIDDEN_CH
    ConvLayer   conv2;   // HIDDEN_CH    -> HIDDEN_CH
    LinearLayer time2;   // TIME_EMB_DIM -> HIDDEN_CH
    ConvLayer   conv3;   // HIDDEN_CH    -> IMG_CHANNELS

    // ---- forward activation cache (written by network_forward) ----
    const float* last_input;   // (IMG_CHANNELS,IMG_SIZE,IMG_SIZE) -- NOT owned
    const float* last_temb;    // (TIME_EMB_DIM)                   -- NOT owned
    float *z1, *h1;             // (HIDDEN_CH,IMG_SIZE,IMG_SIZE) each
    float *tb1;                  // (HIDDEN_CH) -- time1's output for this step
    float *z2, *h2;              // (HIDDEN_CH,IMG_SIZE,IMG_SIZE) each
    float *tb2;                   // (HIDDEN_CH)
    float *out;                   // (IMG_CHANNELS,IMG_SIZE,IMG_SIZE) predicted noise

    // ---- backward activation-gradient cache (written by network_backward) ----
    // NOTE: there is no separate "d_bias1 / d_bias2" buffer. Because tb_k
    // (the time projection) and b_k (the conv bias) enter their shared
    // pre-activation z_k purely additively, dL/db_k and dL/d(tb_k) are
    // IDENTICALLY the same tensor (proved by finite-difference check during
    // development). network_backward() computes it once directly into
    // conv{1,2}.db and reuses that same buffer as the upstream gradient for
    // the corresponding time{1,2} linear layer -- see train.cu.
    float *d_out;                 // dL/d(out)
    float *d_h2, *d_z2;           // dL/d(h2), dL/d(z2)
    float *d_h1, *d_z1;            // dL/d(h1), dL/d(z1)
};

// Allocates every device buffer above and randomly initializes all learnable
// parameters with He/Kaiming-scaled Gaussian noise (biases start at zero).
// Gradient and Adam-moment buffers start at zero.
void network_init(Network* net, unsigned int seed);

// Frees every device buffer owned by `net`.
void network_free(Network* net);

// Forward pass: net->out = epsilon_theta(d_x, d_temb). `d_x` (IMG_ELEMS
// floats) and `d_temb` (TIME_EMB_DIM floats) are device pointers that MUST
// remain valid until any subsequent network_backward() call, since backprop
// re-reads them (they are cached by pointer, not copied).
void network_forward(Network* net, const float* d_x, const float* d_temb);

#endif // NETWORK_H
```

```cpp
// network.cu
//
// FORWARD-PASS kernels and the host-side orchestration that strings them
// into epsilon_theta(x_t, t). Backward-pass kernels live in train.cu; the
// two files never call each other's __global__ functions directly, only
// through the plain host functions declared in network.h / train.h.

#include <cmath>
#include <cstdlib>
#include <random>

#include "common.h"
#include "network.h"
#include "utils.h"

// ============================================================================
// KERNEL: conv2d_forward_tiled_kernel
//
// Computes output[co,y,x] = bias[co] + sum_{ci,kh,kw} input[ci,y+kh-1,x+kw-1]
// * weight[co,ci,kh,kw]  (zero-padding outside [0,H)x[0,W)), for a 3x3,
// stride-1, "same"-padding convolution.
//
// THREAD/BLOCK MAPPING
//   block  = (TILE_DIM, TILE_DIM) = (16,16) threads -> one thread per output
//            pixel within a 16x16 output tile.
//   grid.x = ceil(W / TILE_DIM), grid.y = ceil(H / TILE_DIM)  -> tiles the
//            output plane.
//   grid.z = Cout                                              -> one block
//            "layer" per output channel, so blockIdx.z selects which output
//            channel this whole block is responsible for.
//
// SHARED-MEMORY TILING
//   For a fixed input channel ci, every one of the 256 threads in a block
//   needs a 3x3 neighborhood of `input[ci]` around its own output pixel.
//   Threads in the same block have OVERLAPPING neighborhoods (each interior
//   pixel is read by up to 9 different threads), so instead of letting every
//   thread re-fetch its neighborhood from global memory independently
//   (9x redundant global traffic), the block first cooperatively STAGES a
//   single (TILE_DIM+2)x(TILE_DIM+2)=18x18 "halo" tile of `input[ci]` into
//   __shared__ memory once, then every thread reads its 3x3 window out of
//   that fast on-chip tile. This is the classic tiled-convolution pattern:
//   trade redundant DRAM reads for redundant (but far cheaper) shared-memory
//   reads.
// ============================================================================
__global__ void conv2d_forward_tiled_kernel(const float* __restrict__ input,
                                             const float* __restrict__ weight,
                                             const float* __restrict__ bias,
                                             float* __restrict__ output,
                                             int Cin, int Cout) {
    const int H = IMG_SIZE, W = IMG_SIZE;

    // One shared tile, reused (overwritten) once per input channel.
    __shared__ float tile[HALO_DIM][HALO_DIM];

    const int co = blockIdx.z;                    // which output channel
    const int tx = threadIdx.x, ty = threadIdx.y;  // 0..15, 0..15
    const int out_x = blockIdx.x * TILE_DIM + tx;  // this thread's output column
    const int out_y = blockIdx.y * TILE_DIM + ty;  // this thread's output row

    float acc = 0.0f;

    for (int ci = 0; ci < Cin; ci++) {
        // ---- Cooperative halo load ----
        // HALO_DIM (18) > TILE_DIM (16), so a 16x16 block of threads must
        // load an 18x18 tile: each thread loads its "home" element plus,
        // for the threads nearest the tile edges, one extra element in x
        // and/or y (a `+= TILE_DIM` strided loop is the standard way to let
        // a smaller thread block cover a larger shared-memory footprint).
        for (int dy = ty; dy < HALO_DIM; dy += TILE_DIM) {
            int in_y = blockIdx.y * TILE_DIM + dy - CONV_PAD;
            for (int dx = tx; dx < HALO_DIM; dx += TILE_DIM) {
                int in_x = blockIdx.x * TILE_DIM + dx - CONV_PAD;
                float v = 0.0f;
                if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                    v = input[ci * H * W + in_y * W + in_x];
                }
                tile[dy][dx] = v;   // zero-padded halo cells stay 0.0f
            }
        }
        // Barrier: every thread in the block must finish writing the tile
        // before any thread starts reading it below -- otherwise a "fast"
        // thread could read stale/garbage shared memory from a slot a
        // "slow" thread hasn't written yet (a shared-memory race).
        __syncthreads();

        if (out_x < W && out_y < H) {
            #pragma unroll
            for (int kh = 0; kh < CONV_K; kh++) {
                #pragma unroll
                for (int kw = 0; kw < CONV_K; kw++) {
                    float wv = weight[((co * Cin + ci) * CONV_K + kh) * CONV_K + kw];
                    acc += wv * tile[ty + kh][tx + kw];
                }
            }
        }
        // Barrier: nobody may start overwriting `tile` for ci+1 until every
        // thread has finished reading ci's tile above.
        __syncthreads();
    }

    if (out_x < W && out_y < H) {
        output[co * H * W + out_y * W + out_x] = acc + bias[co];
    }
}

static void conv2d_forward(const float* d_input, const ConvLayer& layer, float* d_output) {
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((IMG_SIZE + TILE_DIM - 1) / TILE_DIM,
              (IMG_SIZE + TILE_DIM - 1) / TILE_DIM,
              layer.Cout);
    conv2d_forward_tiled_kernel<<<grid, block>>>(d_input, layer.W, layer.b, d_output,
                                                  layer.Cin, layer.Cout);
    CUDA_CHECK_LAST();
}

// ============================================================================
// KERNEL: linear_forward_kernel  --  y = W*x + b, a small dense layer used
// to project the sinusoidal time embedding down to a per-channel bias.
// One thread per output element; since out_dim <= HIDDEN_CH (32) this never
// needs more than a single warp, so no shared memory / reduction is needed.
// ============================================================================
__global__ void linear_forward_kernel(const float* __restrict__ x,
                                       const float* __restrict__ W,
                                       const float* __restrict__ b,
                                       float* __restrict__ y,
                                       int in_dim, int out_dim) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= out_dim) return;
    float s = b[o];
    for (int i = 0; i < in_dim; i++) {
        s += W[o * in_dim + i] * x[i];
    }
    y[o] = s;
}

static void linear_forward(const float* d_x, const LinearLayer& layer, float* d_y) {
    int threads = 32;
    int blocks = (layer.out_dim + threads - 1) / threads;
    linear_forward_kernel<<<blocks, threads>>>(d_x, layer.W, layer.b, d_y,
                                                layer.in_dim, layer.out_dim);
    CUDA_CHECK_LAST();
}

// ============================================================================
// KERNEL: add_time_bias_kernel -- broadcast-adds a per-channel scalar
// (the projected time embedding) onto every spatial position of that
// channel: z[c,y,x] += time_bias[c]. This is how the network is conditioned
// on the diffusion timestep t (FiLM-style additive conditioning).
// ============================================================================
__global__ void add_time_bias_kernel(float* __restrict__ z,
                                      const float* __restrict__ time_bias,
                                      int C, int HW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * HW;
    if (idx >= total) return;
    int c = idx / HW;              // which channel this element belongs to
    z[idx] += time_bias[c];
}

static void add_time_bias(float* d_z, const float* d_time_bias, int C) {
    int hw = IMG_SIZE * IMG_SIZE;
    int total = C * hw;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    add_time_bias_kernel<<<blocks, threads>>>(d_z, d_time_bias, C, hw);
    CUDA_CHECK_LAST();
}

// ============================================================================
// KERNEL: silu_forward_kernel -- elementwise SiLU/Swish: h = z * sigmoid(z).
// Fully memory-bound, embarrassingly parallel: one thread per element, no
// shared memory or synchronization needed at all.
// ============================================================================
__global__ void silu_forward_kernel(const float* __restrict__ z,
                                     float* __restrict__ h, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float zi = z[i];
    float sig = 1.0f / (1.0f + expf(-zi));
    h[i] = zi * sig;
}

static void silu_forward(const float* d_z, float* d_h, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    silu_forward_kernel<<<blocks, threads>>>(d_z, d_h, n);
    CUDA_CHECK_LAST();
}

// ============================================================================
// network_forward: strings the kernels above into the full 3-conv-layer
// noise predictor described in network.h.
// ============================================================================
void network_forward(Network* net, const float* d_x, const float* d_temb) {
    net->last_input = d_x;
    net->last_temb = d_temb;
    const int hidden_elems = HIDDEN_CH * IMG_SIZE * IMG_SIZE;

    // ---- Layer 1: conv1 -> + time1(temb) -> SiLU ----
    conv2d_forward(d_x, net->conv1, net->z1);
    linear_forward(d_temb, net->time1, net->tb1);
    add_time_bias(net->z1, net->tb1, HIDDEN_CH);
    silu_forward(net->z1, net->h1, hidden_elems);

    // ---- Layer 2: conv2 -> + time2(temb) -> SiLU ----
    conv2d_forward(net->h1, net->conv2, net->z2);
    linear_forward(d_temb, net->time2, net->tb2);
    add_time_bias(net->z2, net->tb2, HIDDEN_CH);
    silu_forward(net->z2, net->h2, hidden_elems);

    // ---- Layer 3: conv3, linear output = predicted noise ----
    conv2d_forward(net->h2, net->conv3, net->out);
}

// ============================================================================
// Parameter initialization + (de)allocation
// ============================================================================
static void init_conv_layer(ConvLayer* layer, int Cin, int Cout, std::mt19937& rng) {
    layer->Cin = Cin;
    layer->Cout = Cout;
    const int wsize = Cout * Cin * CONV_K * CONV_K;

    // He/Kaiming initialization: std = sqrt(2 / fan_in), fan_in = Cin*K*K.
    // Keeps activation variance roughly stable across layers at init time,
    // which matters here because there is no batch/layer norm in this
    // minimal network to correct a bad initial scale.
    float std = sqrtf(2.0f / (float)(Cin * CONV_K * CONV_K));
    std::normal_distribution<float> dist(0.0f, std);

    float* hW = (float*)malloc((size_t)wsize * sizeof(float));
    for (int i = 0; i < wsize; i++) hW[i] = dist(rng);
    layer->W = device_alloc_from_host(hW, wsize);
    free(hW);

    layer->b  = device_alloc_zero(Cout);   // biases start at 0
    layer->dW = device_alloc_zero(wsize);
    layer->db = device_alloc_zero(Cout);
    layer->mW = device_alloc_zero(wsize);
    layer->vW = device_alloc_zero(wsize);
    layer->mb = device_alloc_zero(Cout);
    layer->vb = device_alloc_zero(Cout);
}

static void init_linear_layer(LinearLayer* layer, int in_dim, int out_dim, std::mt19937& rng) {
    layer->in_dim = in_dim;
    layer->out_dim = out_dim;
    const int wsize = out_dim * in_dim;

    float std = sqrtf(2.0f / (float)in_dim);
    std::normal_distribution<float> dist(0.0f, std);

    float* hW = (float*)malloc((size_t)wsize * sizeof(float));
    for (int i = 0; i < wsize; i++) hW[i] = dist(rng);
    layer->W = device_alloc_from_host(hW, wsize);
    free(hW);

    layer->b  = device_alloc_zero(out_dim);
    layer->dW = device_alloc_zero(wsize);
    layer->db = device_alloc_zero(out_dim);
    layer->mW = device_alloc_zero(wsize);
    layer->vW = device_alloc_zero(wsize);
    layer->mb = device_alloc_zero(out_dim);
    layer->vb = device_alloc_zero(out_dim);
}

void network_init(Network* net, unsigned int seed) {
    std::mt19937 rng(seed);

    init_conv_layer(&net->conv1, IMG_CHANNELS, HIDDEN_CH, rng);
    init_linear_layer(&net->time1, TIME_EMB_DIM, HIDDEN_CH, rng);
    init_conv_layer(&net->conv2, HIDDEN_CH, HIDDEN_CH, rng);
    init_linear_layer(&net->time2, TIME_EMB_DIM, HIDDEN_CH, rng);
    init_conv_layer(&net->conv3, HIDDEN_CH, IMG_CHANNELS, rng);

    const int hidden_elems = HIDDEN_CH * IMG_SIZE * IMG_SIZE;

    net->z1  = device_alloc(hidden_elems);
    net->h1  = device_alloc(hidden_elems);
    net->tb1 = device_alloc(HIDDEN_CH);
    net->z2  = device_alloc(hidden_elems);
    net->h2  = device_alloc(hidden_elems);
    net->tb2 = device_alloc(HIDDEN_CH);
    net->out = device_alloc(IMG_ELEMS);

    net->d_out    = device_alloc(IMG_ELEMS);
    net->d_h2     = device_alloc(hidden_elems);
    net->d_z2     = device_alloc(hidden_elems);
    net->d_h1     = device_alloc(hidden_elems);
    net->d_z1     = device_alloc(hidden_elems);

    net->last_input = nullptr;
    net->last_temb  = nullptr;
}

static void free_conv_layer(ConvLayer* l) {
    CUDA_CHECK(cudaFree(l->W));  CUDA_CHECK(cudaFree(l->b));
    CUDA_CHECK(cudaFree(l->dW)); CUDA_CHECK(cudaFree(l->db));
    CUDA_CHECK(cudaFree(l->mW)); CUDA_CHECK(cudaFree(l->vW));
    CUDA_CHECK(cudaFree(l->mb)); CUDA_CHECK(cudaFree(l->vb));
}

static void free_linear_layer(LinearLayer* l) {
    CUDA_CHECK(cudaFree(l->W));  CUDA_CHECK(cudaFree(l->b));
    CUDA_CHECK(cudaFree(l->dW)); CUDA_CHECK(cudaFree(l->db));
    CUDA_CHECK(cudaFree(l->mW)); CUDA_CHECK(cudaFree(l->vW));
    CUDA_CHECK(cudaFree(l->mb)); CUDA_CHECK(cudaFree(l->vb));
}

void network_free(Network* net) {
    free_conv_layer(&net->conv1);
    free_linear_layer(&net->time1);
    free_conv_layer(&net->conv2);
    free_linear_layer(&net->time2);
    free_conv_layer(&net->conv3);

    CUDA_CHECK(cudaFree(net->z1));  CUDA_CHECK(cudaFree(net->h1));  CUDA_CHECK(cudaFree(net->tb1));
    CUDA_CHECK(cudaFree(net->z2));  CUDA_CHECK(cudaFree(net->h2));  CUDA_CHECK(cudaFree(net->tb2));
    CUDA_CHECK(cudaFree(net->out));

    CUDA_CHECK(cudaFree(net->d_out));
    CUDA_CHECK(cudaFree(net->d_h2)); CUDA_CHECK(cudaFree(net->d_z2));
    CUDA_CHECK(cudaFree(net->d_h1)); CUDA_CHECK(cudaFree(net->d_z1));
}
```

### Line-by-Line Explanation

**`conv2d_forward_tiled_kernel` (the centerpiece kernel):**
- **Grid/block mapping**: `block=(16,16)` → 256 threads, one per output pixel in a 16×16 tile. `grid=(2,2,Cout)` since `IMG_SIZE/TILE_DIM = 32/16 = 2` exactly (no partial tiles for this project's fixed 32×32 images, though the boundary checks below make the kernel correct for *any* size). `blockIdx.z` selects the output channel `co` — this is the "one block-layer per output channel" mapping, so each of the `Cout` output planes is computed by an independent set of blocks with no cross-channel synchronization needed.
- **`__shared__ float tile[HALO_DIM][HALO_DIM]`** (`HALO_DIM=18`) is **static shared memory**: fixed size, known at compile time, allocated once per block for the block's lifetime — contrast this with the *dynamic* shared memory used in `train.cu`'s reduction kernels (see section 5).
- **The halo-load loop** (`for dy = ty; dy < HALO_DIM; dy += TILE_DIM`) is the standard trick for letting a 16×16 thread block populate an 18×18 shared tile: most threads load exactly one element, but threads near the tile's right/bottom edge loop a second time to also grab the 2 extra halo columns/rows (`18 = 16 + 2`, so `ceil(18/16)=2` iterations cover it). `in_y`/`in_x` map shared-tile coordinates back to global image coordinates, offset by `-CONV_PAD` — reads outside `[0,H)×[0,W)` are simply written as `0.0f` (implementing zero-padding directly in shared memory rather than branching inside the inner conv loop).
- **Why two `__syncthreads()` calls per `ci` iteration, not one**: the first is a **read-after-write barrier** — every thread must finish *writing* its slice of `tile` before any thread starts *reading* neighboring slices written by other threads (without it, this is a classic shared-memory race — a fast thread could read a stale value a slower thread hasn't written yet). The second is a **write-after-read barrier** — before the loop moves to `ci+1` and starts *overwriting* `tile`, every thread must have finished *reading* the current tile's values in its accumulation step.
- **Reuse arithmetic**: for every input channel, a 16×16 block reads an 18×18=324-element tile from global memory once, then every one of the 256 threads reuses up to 9 of those cached values from shared memory. Without tiling, the same 3×3 neighborhoods would be re-fetched from global (DRAM) memory independently by every thread that needs them — up to 9× redundant DRAM traffic for interior pixels. Shared memory has roughly two orders of magnitude lower latency and higher bandwidth than global memory, so this trade (some redundant *shared*-memory reads in exchange for eliminating redundant *global*-memory reads) is a large net win.
- **Coalescing inside the halo load**: `threadIdx.x` (`tx`) is the fastest-varying index and maps to the innermost (contiguous) `in_x`/column dimension — so within a warp, 32 consecutive threads read 32 consecutive floats from `input[...]`, again a fully coalesced 128-byte transaction.
- **`#pragma unroll`** on the 3×3 accumulation loops is a hint to `ptxas` to fully unroll these fixed-trip-count loops into straight-line code, eliminating loop-branch overhead and letting the compiler schedule the 9 fused-multiply-adds without loop bookkeeping.

**`linear_forward_kernel` / `add_time_bias_kernel` / `silu_forward_kernel`:**
- The linear layer is intentionally the simplest possible kernel: `out_dim ≤ 32` means a single warp (or less) covers the whole output vector, so there's no benefit to shared memory or a reduction — each of the ≤32 threads just does a private serial dot-product over `in_dim=32` inputs.
- `add_time_bias_kernel` implements the **FiLM-style conditioning**: `c = idx / HW` recovers which channel a flattened index belongs to, then broadcasts that channel's single scalar bias across all `H*W` spatial positions — this is literally how the network learns "what timestep am I denoising."
- `silu_forward_kernel` computes $h = z\cdot\sigma(z)$ directly from the sigmoid definition; it's fully memory-bound (1 read, 1 write, a handful of FLOPs per element) with zero inter-thread dependency, the simplest possible kernel shape.

**Host orchestration (`network_forward`) and initialization:**
- `network_forward` is pure sequencing: 3× `conv2d_forward` calls interleaved with the two time-conditioning branches and two `silu_forward` calls — it caches `d_x` and `d_temb` by pointer (not by copy) into `net->last_input`/`net->last_temb` specifically so `network_backward` (section 5) can re-read the exact tensors used this forward pass without redundant transfers.
- **He/Kaiming initialization** (`std = sqrt(2/fan_in)`) is used instead of naive `N(0,1)` weights because this network has no batch/layer normalization to self-correct a bad initial activation scale — starting variance-controlled avoids the first few hundred training iterations being spent just fighting exploding/vanishing activations.
- Every buffer allocation funnels through `device_alloc`/`device_alloc_zero` (section 2), and `network_free` mirrors `network_init` exactly, freeing every one of the ~30 device buffers this struct owns — a common source of real-world CUDA memory leaks is exactly this kind of large "many small allocations" struct where one is accidentally missed in cleanup, so it's worth the tedium of writing it out explicitly and completely as done here.

# 5. Backpropagation & Optimizer (`train.h` / `train.cu`)

**The chain rule, spelled out**, since this is where hand-written backprop most commonly goes wrong. Let $L=\text{MSE}(\text{out},\epsilon)$. Working backward through exactly the forward sequence:

$$\frac{\partial L}{\partial \text{out}}\ \xrightarrow{\text{conv}_3^{-1}}\ \frac{\partial L}{\partial h_2}\ \xrightarrow{\text{SiLU}'}\ \frac{\partial L}{\partial z_2}\ \xrightarrow{\text{split}}\ \begin{cases}\dfrac{\partial L}{\partial b_2},\ \dfrac{\partial L}{\partial \text{tb}_2}\\[4pt]\xrightarrow{\text{conv}_2^{-1}}\ \dfrac{\partial L}{\partial h_1}\end{cases}\ \xrightarrow{\text{SiLU}'}\ \frac{\partial L}{\partial z_1}\ \xrightarrow{\text{split}}\ \begin{cases}\dfrac{\partial L}{\partial b_1},\ \dfrac{\partial L}{\partial \text{tb}_1}\\[4pt]\xrightarrow{\text{conv}_1 \text{ weight grad only}}\end{cases}$$

Two identities make the CUDA code shorter than the graph above might suggest, **both verified numerically with finite differences before writing a line of CUDA**:
1. Since $z_k = \text{conv}_{\text{raw}} + b_k + \text{tb}_k$ is a pure sum, $\partial L/\partial b_k = \partial L/\partial \text{tb}_k = \sum_{h,w}\partial L/\partial z_k$ — **the exact same tensor**. One reduction kernel computes both.
2. `x_t` (the network's input) is data, not a parameter — so `conv1`'s backward pass only needs the *weight* gradient, never an input gradient. This is why `network_backward` below never computes `conv2d_backward_input` for `conv1`.

```cpp
#ifndef TRAIN_H
#define TRAIN_H

#include "network.h"

// Computes mean-squared-error loss = mean((pred - target)^2) over n elements
// using a two-stage parallel reduction (block-level shared-memory tree
// reduction + atomicAdd across blocks). Returns the scalar loss on the host.
float compute_mse_loss(const float* d_pred, const float* d_target, int n);

// Full backward pass. Given d_target (the ground-truth noise epsilon used
// by the most recent diffusion_forward_sample call), computes dL/dout and
// backpropagates it through conv3 -> SiLU -> {conv2,time2} -> SiLU ->
// {conv1,time1}, filling every layer's dW/db (or dW/db for LinearLayer)
// gradient buffer. Requires the activation cache from a matching, immediately
// preceding network_forward() call (net->z1,h1,z2,h2,out,last_input,last_temb).
void network_backward(Network* net, const float* d_target);

// One Adam update step for every learnable parameter in `net`, consuming the
// gradients populated by the most recent network_backward(). `step` is the
// 1-indexed optimizer iteration count (used for Adam's bias correction).
void adam_step(Network* net, int step, float lr);

#endif // TRAIN_H
```

```cpp
// train.cu
//
// BACKWARD-PASS kernels (manual reverse-mode differentiation of every op in
// network.cu, applying the chain rule op-by-op) plus the Adam optimizer.
// Every kernel here consumes an upstream gradient ("dL/d<tensor>") and
// produces either a parameter gradient (dW, db) or a downstream activation
// gradient to keep propagating backward -- exactly mirroring, in reverse,
// the sequence of calls inside network_forward().

#include <cmath>

#include "common.h"
#include "network.h"
#include "train.h"
#include "utils.h"

// ============================================================================
// KERNEL: mse_loss_kernel -- computes sum((pred-target)^2) with a two-level
// parallel reduction:
//   Level 1 (per block): each thread squares one difference and the block
//   performs a shared-memory tree reduction down to a single partial sum.
//   Level 2 (across blocks): each block's lone survivor (tid==0) does one
//   atomicAdd into a single global accumulator. With only ~12 blocks (n=3072
//   elements, 256 threads/block) atomic contention here is negligible.
// The mean is taken on the host by dividing by n after the kernel returns.
// ============================================================================
__global__ void mse_loss_kernel(const float* __restrict__ pred,
                                 const float* __restrict__ target,
                                 float* __restrict__ loss_accum, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;
    if (i < n) {
        float diff = pred[i] - target[i];
        val = diff * diff;
    }
    sdata[tid] = val;
    __syncthreads();

    // Tree reduction: at each step half of the still-active threads add a
    // neighbor's value into their own slot; the "active half" shrinks by 2x
    // every iteration (blockDim.x -> .../2 -> .../4 -> ... -> 1). Every
    // iteration needs its own __syncthreads() because iteration k+1 reads
    // values that iteration k's *other* threads just wrote.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(loss_accum, sdata[0]);
}

float compute_mse_loss(const float* d_pred, const float* d_target, int n) {
    float* d_loss = device_alloc_zero(1);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    size_t shmem = threads * sizeof(float);

    mse_loss_kernel<<<blocks, threads, shmem>>>(d_pred, d_target, d_loss, n);
    CUDA_CHECK_LAST();

    float h_loss = 0.0f;
    copy_to_host(&h_loss, d_loss, 1);
    CUDA_CHECK(cudaFree(d_loss));
    return h_loss / (float)n;
}

// ============================================================================
// KERNEL: mse_grad_kernel -- dL/dpred[i] = (2/n) * (pred[i] - target[i]).
// This is the seed gradient that everything else below propagates backward.
// ============================================================================
__global__ void mse_grad_kernel(const float* __restrict__ pred,
                                 const float* __restrict__ target,
                                 float* __restrict__ dpred, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dpred[i] = (2.0f / (float)n) * (pred[i] - target[i]);
}

// ============================================================================
// KERNEL: conv2d_backward_weight_kernel
//
//   dW[co,ci,kh,kw] = sum_{y,x} dOut[co,y,x] * input_padded[ci, y+kh-CONV_PAD,
//                                                                x+kw-CONV_PAD]
//
// One thread per WEIGHT element (not per pixel!): thread `idx` decodes its
// own (co,ci,kh,kw) from a flat index and then walks the full HxW output
// plane accumulating a scalar. This is the transpose of the forward kernel's
// mapping (forward: one thread per output pixel, looped over weights;
// backward-weight: one thread per weight, looped over pixels) -- the natural
// shape for this gradient, since every (co,ci,kh,kw) weight touches every
// spatial position exactly once per forward pass.
// ============================================================================
__global__ void conv2d_backward_weight_kernel(const float* __restrict__ dout,
                                               const float* __restrict__ input,
                                               float* __restrict__ dW,
                                               int Cin, int Cout) {
    const int H = IMG_SIZE, W = IMG_SIZE;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Cout * Cin * CONV_K * CONV_K;
    if (idx >= total) return;

    // Decode idx -> (co, ci, kh, kw) for weight layout (Cout,Cin,K,K).
    int kw = idx % CONV_K;
    int t1 = idx / CONV_K;
    int kh = t1 % CONV_K;
    int t2 = t1 / CONV_K;
    int ci = t2 % Cin;
    int co = t2 / Cin;

    float sum = 0.0f;
    for (int y = 0; y < H; y++) {
        int in_y = y + kh - CONV_PAD;
        if (in_y < 0 || in_y >= H) continue;
        for (int x = 0; x < W; x++) {
            int in_x = x + kw - CONV_PAD;
            if (in_x < 0 || in_x >= W) continue;
            sum += dout[co * H * W + y * W + x] * input[ci * H * W + in_y * W + in_x];
        }
    }
    dW[idx] = sum;
}

static void conv2d_backward_weight(const float* d_dout, const float* d_input, ConvLayer& layer) {
    int total = layer.Cout * layer.Cin * CONV_K * CONV_K;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    conv2d_backward_weight_kernel<<<blocks, threads>>>(d_dout, d_input, layer.dW,
                                                         layer.Cin, layer.Cout);
    CUDA_CHECK_LAST();
}

// ============================================================================
// KERNEL: conv2d_backward_input_kernel
//
//   dIn[ci,y,x] = sum_{co,kh,kw} dOut[co, y-kh+CONV_PAD, x-kw+CONV_PAD] *
//                                W[co,ci,kh,kw]     (terms outside [0,H)x[0,W) dropped)
//
// This is mathematically a "full" cross-correlation of dOut with the SAME
// (unflipped) weight tensor, evaluated at shifted output coordinates -- the
// well-known fact that conv2d's input-gradient is itself a convolution with
// a 180-degree-rotated kernel. We implement it directly from the shifted-
// index definition rather than materializing a rotated kernel, since for a
// 3x3 filter the direct form is just as simple and avoids a separate flip
// kernel. One thread per INPUT element; loops over (co,kh,kw).
// ============================================================================
__global__ void conv2d_backward_input_kernel(const float* __restrict__ dout,
                                              const float* __restrict__ weight,
                                              float* __restrict__ dinput,
                                              int Cin, int Cout) {
    const int H = IMG_SIZE, W = IMG_SIZE;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Cin * H * W;
    if (idx >= total) return;

    int x = idx % W;
    int t1 = idx / W;
    int y = t1 % H;
    int ci = t1 / H;

    float sum = 0.0f;
    for (int co = 0; co < Cout; co++) {
        #pragma unroll
        for (int kh = 0; kh < CONV_K; kh++) {
            int out_y = y - kh + CONV_PAD;
            if (out_y < 0 || out_y >= H) continue;
            #pragma unroll
            for (int kw = 0; kw < CONV_K; kw++) {
                int out_x = x - kw + CONV_PAD;
                if (out_x < 0 || out_x >= W) continue;
                float wv = weight[((co * Cin + ci) * CONV_K + kh) * CONV_K + kw];
                sum += dout[co * H * W + out_y * W + out_x] * wv;
            }
        }
    }
    dinput[idx] = sum;
}

static void conv2d_backward_input(const float* d_dout, const ConvLayer& layer, float* d_dinput) {
    int total = layer.Cin * IMG_SIZE * IMG_SIZE;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    conv2d_backward_input_kernel<<<blocks, threads>>>(d_dout, layer.W, d_dinput,
                                                        layer.Cin, layer.Cout);
    CUDA_CHECK_LAST();
}

// ============================================================================
// KERNEL: channel_sum_kernel -- dBias[c] = sum_{h,w} dY[c,h,w].
//
// This single reduction kernel computes TWO different gradients in this
// network, because both are literally the same mathematical quantity:
//   - a conv layer's bias gradient (bias is added per-channel, broadcast
//     over space, so its gradient is exactly this spatial sum), and
//   - the upstream gradient flowing into a time-projection layer's output
//     tb_k (which is also added per-channel, broadcast over space).
// One block per channel; within a block, threads first do a grid-stride-like
// strided loop to cover all HW=1024 positions with only 256 threads (4
// elements/thread), then a shared-memory tree reduction combines the 256
// partial sums -- the same two-stage pattern as mse_loss_kernel above.
// ============================================================================
__global__ void channel_sum_kernel(const float* __restrict__ dY,
                                    float* __restrict__ dBias, int HW) {
    extern __shared__ float sdata[];
    int c = blockIdx.x;
    int tid = threadIdx.x;

    float sum = 0.0f;
    for (int i = tid; i < HW; i += blockDim.x) {
        sum += dY[c * HW + i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) dBias[c] = sdata[0];
}

static void channel_sum(const float* d_dY, float* d_dBias, int C) {
    int hw = IMG_SIZE * IMG_SIZE;
    int threads = 256;
    size_t shmem = threads * sizeof(float);
    channel_sum_kernel<<<C, threads, shmem>>>(d_dY, d_dBias, hw);
    CUDA_CHECK_LAST();
}

// ============================================================================
// KERNEL: silu_backward_kernel
//
//   SiLU(z) = z * sigmoid(z)
//   d/dz SiLU(z) = sigmoid(z) + z*sigmoid(z)*(1-sigmoid(z))
//                = sigmoid(z) * (1 + z*(1 - sigmoid(z)))
//   dL/dz = dL/dh * d/dz SiLU(z)          (chain rule)
// ============================================================================
__global__ void silu_backward_kernel(const float* __restrict__ z,
                                      const float* __restrict__ dh,
                                      float* __restrict__ dz, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float zi = z[i];
    float sig = 1.0f / (1.0f + expf(-zi));
    float dsilu_dz = sig * (1.0f + zi * (1.0f - sig));
    dz[i] = dh[i] * dsilu_dz;
}

static void silu_backward(const float* d_z, const float* d_dh, float* d_dz, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    silu_backward_kernel<<<blocks, threads>>>(d_z, d_dh, d_dz, n);
    CUDA_CHECK_LAST();
}

// ============================================================================
// KERNEL: linear_backward_weight_kernel -- dW[o,i] = dy[o] * x[i].
//
// One thread per (o,i) weight element. Because temb (the linear layer's
// input) is a FIXED sinusoidal table, not a learned tensor, we deliberately
// do not compute dL/dtemb here -- there is nothing upstream of it to update,
// so backprop simply stops at this layer's own parameters.
// db[o] = dy[o] is an exact identity (bias contributes additively with
// coefficient 1), so rather than launch a second kernel we just device-to-
// device memcpy dy into layer.db in the host wrapper below.
// ============================================================================
__global__ void linear_backward_weight_kernel(const float* __restrict__ dy,
                                               const float* __restrict__ x,
                                               float* __restrict__ dW,
                                               int in_dim, int out_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = out_dim * in_dim;
    if (idx >= total) return;
    int i = idx % in_dim;
    int o = idx / in_dim;
    dW[idx] = dy[o] * x[i];
}

static void linear_backward(const float* d_dy, const float* d_x, LinearLayer& layer) {
    int total = layer.out_dim * layer.in_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    linear_backward_weight_kernel<<<blocks, threads>>>(d_dy, d_x, layer.dW,
                                                         layer.in_dim, layer.out_dim);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(layer.db, d_dy, layer.out_dim * sizeof(float),
                           cudaMemcpyDeviceToDevice));
}

// ============================================================================
// network_backward: runs the chain rule in reverse over exactly the sequence
// of ops network_forward() applied, using the activation cache network_forward
// left behind in `net`.
// ============================================================================
void network_backward(Network* net, const float* d_target) {
    const int hidden_elems = HIDDEN_CH * IMG_SIZE * IMG_SIZE;

    // ---- seed gradient: dL/d(out) from the MSE loss ----
    {
        int threads = 256;
        int blocks = (IMG_ELEMS + threads - 1) / threads;
        mse_grad_kernel<<<blocks, threads>>>(net->out, d_target, net->d_out, IMG_ELEMS);
        CUDA_CHECK_LAST();
    }

    // ---- conv3 backward: out = conv3(h2) + b3 (no time-bias, no activation) ----
    conv2d_backward_weight(net->d_out, net->h2, net->conv3);      // -> conv3.dW
    channel_sum(net->d_out, net->conv3.db, IMG_CHANNELS);          // -> conv3.db
    conv2d_backward_input(net->d_out, net->conv3, net->d_h2);      // -> dL/dh2

    // ---- SiLU backward, layer 2: h2 = SiLU(z2) ----
    silu_backward(net->z2, net->d_h2, net->d_z2, hidden_elems);    // -> dL/dz2

    // ---- shared bias gradient for {conv2.b, time2 output}; z2 = conv2_raw + b2 + tb2 ----
    channel_sum(net->d_z2, net->conv2.db, HIDDEN_CH);               // -> conv2.db AND dL/dtb2
    linear_backward(net->conv2.db, net->last_temb, net->time2);     // -> time2.dW, time2.db

    // ---- conv2 backward (weight + propagate into h1) ----
    conv2d_backward_weight(net->d_z2, net->h1, net->conv2);         // -> conv2.dW
    conv2d_backward_input(net->d_z2, net->conv2, net->d_h1);        // -> dL/dh1

    // ---- SiLU backward, layer 1: h1 = SiLU(z1) ----
    silu_backward(net->z1, net->d_h1, net->d_z1, hidden_elems);     // -> dL/dz1

    // ---- shared bias gradient for {conv1.b, time1 output}; z1 = conv1_raw + b1 + tb1 ----
    channel_sum(net->d_z1, net->conv1.db, HIDDEN_CH);                // -> conv1.db AND dL/dtb1
    linear_backward(net->conv1.db, net->last_temb, net->time1);      // -> time1.dW, time1.db

    // ---- conv1 backward: weight only. net->last_input (x_t) is DATA, not a
    // learned parameter, so we deliberately never call conv2d_backward_input
    // here -- there is nothing upstream to keep propagating into. ----
    conv2d_backward_weight(net->d_z1, net->last_input, net->conv1); // -> conv1.dW
}

// ============================================================================
// KERNEL: adam_update_kernel -- standard Adam (Kingma & Ba, 2015), applied
// elementwise to a flat parameter array. One thread per parameter element.
//   m <- beta1*m + (1-beta1)*g          (biased 1st-moment / momentum)
//   v <- beta2*v + (1-beta2)*g^2        (biased 2nd-moment / RMS scale)
//   m_hat <- m / (1 - beta1^step)       (bias-correct, since m,v start at 0)
//   v_hat <- v / (1 - beta2^step)
//   param <- param - lr * m_hat / (sqrt(v_hat) + eps)
// beta1_t = beta1^step and beta2_t = beta2^step are computed ONCE on the
// host and passed in as scalars, so we don't waste a powf() call per thread.
// ============================================================================
__global__ void adam_update_kernel(float* __restrict__ param,
                                    const float* __restrict__ grad,
                                    float* __restrict__ m,
                                    float* __restrict__ v,
                                    int n, float lr,
                                    float beta1, float beta2, float eps,
                                    float beta1_t, float beta2_t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float g = grad[i];
    float m_new = beta1 * m[i] + (1.0f - beta1) * g;
    float v_new = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = m_new;
    v[i] = v_new;

    float m_hat = m_new / (1.0f - beta1_t);
    float v_hat = v_new / (1.0f - beta2_t);

    param[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

static void adam_update(float* param, const float* grad, float* m, float* v, int n,
                         float lr, float beta1_t, float beta2_t) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    adam_update_kernel<<<blocks, threads>>>(param, grad, m, v, n, lr,
                                             ADAM_BETA1, ADAM_BETA2, ADAM_EPS,
                                             beta1_t, beta2_t);
    CUDA_CHECK_LAST();
}

void adam_step(Network* net, int step, float lr) {
    // Bias-correction terms depend only on the step count, so compute them
    // once here rather than once per parameter tensor.
    float beta1_t = powf(ADAM_BETA1, (float)step);
    float beta2_t = powf(ADAM_BETA2, (float)step);

    // conv1
    {
        int wsize = net->conv1.Cout * net->conv1.Cin * CONV_K * CONV_K;
        adam_update(net->conv1.W, net->conv1.dW, net->conv1.mW, net->conv1.vW, wsize, lr, beta1_t, beta2_t);
        adam_update(net->conv1.b, net->conv1.db, net->conv1.mb, net->conv1.vb, net->conv1.Cout, lr, beta1_t, beta2_t);
    }
    // time1
    {
        int wsize = net->time1.out_dim * net->time1.in_dim;
        adam_update(net->time1.W, net->time1.dW, net->time1.mW, net->time1.vW, wsize, lr, beta1_t, beta2_t);
        adam_update(net->time1.b, net->time1.db, net->time1.mb, net->time1.vb, net->time1.out_dim, lr, beta1_t, beta2_t);
    }
    // conv2
    {
        int wsize = net->conv2.Cout * net->conv2.Cin * CONV_K * CONV_K;
        adam_update(net->conv2.W, net->conv2.dW, net->conv2.mW, net->conv2.vW, wsize, lr, beta1_t, beta2_t);
        adam_update(net->conv2.b, net->conv2.db, net->conv2.mb, net->conv2.vb, net->conv2.Cout, lr, beta1_t, beta2_t);
    }
    // time2
    {
        int wsize = net->time2.out_dim * net->time2.in_dim;
        adam_update(net->time2.W, net->time2.dW, net->time2.mW, net->time2.vW, wsize, lr, beta1_t, beta2_t);
        adam_update(net->time2.b, net->time2.db, net->time2.mb, net->time2.vb, net->time2.out_dim, lr, beta1_t, beta2_t);
    }
    // conv3
    {
        int wsize = net->conv3.Cout * net->conv3.Cin * CONV_K * CONV_K;
        adam_update(net->conv3.W, net->conv3.dW, net->conv3.mW, net->conv3.vW, wsize, lr, beta1_t, beta2_t);
        adam_update(net->conv3.b, net->conv3.db, net->conv3.mb, net->conv3.vb, net->conv3.Cout, lr, beta1_t, beta2_t);
    }
}
```

### Line-by-Line Explanation

- **`extern __shared__ float sdata[];`** (in both `mse_loss_kernel` and `channel_sum_kernel`) is **dynamic** shared memory — unlike `network.cu`'s fixed-size `tile[18][18]`, this array's size isn't known until the *launch site* supplies it as the third `<<<blocks, threads, shmem_bytes>>>` argument (`shmem = threads * sizeof(float)`), which is exactly what `compute_mse_loss` and `channel_sum` do.
- **The tree-reduction loop** `for (s = blockDim.x/2; s>0; s>>=1)` is the canonical parallel-reduction pattern: iteration 1 has 128 active threads each adding their "mirror" partner 128 slots away; iteration 2 has 64 threads adding partners 64 slots away; and so on down to 1 thread holding the full block sum in `sdata[0]`. Each iteration needs its own `__syncthreads()` because it consumes values other threads wrote in the *previous* iteration.
- **`atomicAdd(loss_accum, sdata[0])`** combines the ~12 per-block partial sums into one global total. With so few blocks, atomic contention is negligible — this is the textbook "grid-level reduction via atomics" finish for a reduction that's too large for one block.
- **`conv2d_backward_weight_kernel`'s index decode** (`kw = idx % K; ... co = t2 / Cin`) is the exact inverse of the flattening formula `((co*Cin+ci)*K+kh)*K+kw` used everywhere weights are stored — get this decode wrong (e.g. swap `ci`/`co`) and gradients silently update the wrong weights, which is precisely the class of bug the NumPy finite-difference check (see the verification section) is designed to catch. **One thread per weight, looped over all `H*W` output pixels** is the natural shape here because every weight element participates in *every* spatial position's forward computation exactly once — the transpose of the forward kernel's "one thread per pixel, loop over weights."
- **`conv2d_backward_input_kernel`** implements $\partial L/\partial\text{in}[ci,y,x]=\sum_{co,kh,kw}\partial L/\partial\text{out}[co,y-kh+\text{pad},x-kw+\text{pad}]\cdot W[co,ci,kh,kw]$ — the well-known fact that a convolution's input-gradient is itself a convolution of the output-gradient with the *same, unflipped* kernel, evaluated at shifted coordinates. The `if (out_y<0||out_y>=H) continue;` guards implement the implicit zero-padding of $\partial L/\partial\text{out}$ outside its valid range.
- **The bias-sharing trick**: `channel_sum(net->d_z2, net->conv2.db, HIDDEN_CH)` writes the spatial-sum gradient **directly into `conv2.db`** (no intermediate buffer), and the *very next line* feeds that same `net->conv2.db` pointer into `linear_backward` as the upstream gradient for `time2`. This isn't a shortcut taken for convenience — it's mathematically forced: since `z2 = conv2_raw + b2 + tb2` is a sum, $\partial z_2/\partial b_2 = \partial z_2/\partial \text{tb}_2 = 1$ identically, so both downstream gradients truly are the same tensor. (This exact identity was asserted and checked in the NumPy verification, `assert np.allclose(db2_direct, dbias2)`.)
- **`network_backward`'s final line** deliberately calls `conv2d_backward_weight` for `conv1` but *never* `conv2d_backward_input` — `net->last_input` is the noisy image `x_t`, which is data sampled fresh every training iteration, not a parameter with anything upstream of it to update. Computing its gradient would be wasted work.
- **`adam_update_kernel`**: textbook Adam — biased first/second moment updates, then bias-correction (`m_hat`, `v_hat`) to counteract the fact that `m`/`v` start at exactly zero and would otherwise be biased toward zero in early steps, then the parameter update itself. `beta1_t = β1^step` and `beta2_t = β2^step` are computed **once on the host per `adam_step` call**, not per-thread — a `powf` call is meaningfully expensive compared to the rest of this kernel's per-thread work (a handful of FMAs), so hoisting it out of the ~13,000-times-replicated thread body to a single host-side computation is a real, worthwhile optimization.
- **`adam_step`** is written as five explicit `{...}` blocks (one per layer) rather than a loop over an array of layer structs — `ConvLayer` and `LinearLayer` are *different C++ types* with different member layouts, so a truly generic loop would need either a common base class or void-pointer type erasure; five short, explicit, obviously-correct blocks were judged clearer for a project of this size than either of those.

# 6. Reverse Diffusion / Generation (`generate.h` / `generate.cu`)

```cpp
#ifndef GENERATE_H
#define GENERATE_H

#include "network.h"
#include "diffusion.h"

// Runs the full reverse-diffusion sampling loop and writes the result to disk:
//   1. x_T ~ N(0, I)                                    (diffusion_sample_prior)
//   2. for t = T-1 .. 0:
//        eps_pred = epsilon_theta(x_t, t)                (network_forward)
//        x_{t-1}  = reverse_step(x_t, eps_pred, t)        (diffusion_reverse_step)
//   3. save x_0 to `output_path` as a PNG                 (save_image_chw)
//
// `net` must already be trained. `sched` is the precomputed noise schedule.
// `d_temb_table` is the device sinusoidal time-embedding table returned by
// build_time_embedding_table_device(). `seed` seeds every curand draw made
// during sampling (the initial prior noise plus each step's injected noise).
void generate_image(Network* net, const DiffusionSchedule& sched,
                     const float* d_temb_table, const char* output_path,
                     unsigned long long seed);

#endif // GENERATE_H
```

```cpp
// generate.cu
//
// Reverse diffusion (sampling / generation). No new __global__ kernels are
// defined here -- generation is pure orchestration, reusing network_forward
// (network.cu) and diffusion_reverse_step / diffusion_sample_prior
// (diffusion.cu) inside a host-side loop over t = T-1 .. 0.

#include <cstdio>
#include <cstdlib>
#include <utility>

#include "common.h"
#include "generate.h"
#include "utils.h"

void generate_image(Network* net, const DiffusionSchedule& sched,
                     const float* d_temb_table, const char* output_path,
                     unsigned long long seed) {
    float* d_x      = device_alloc(IMG_ELEMS);   // holds x_t for the current step
    float* d_x_prev = device_alloc(IMG_ELEMS);   // scratch target for x_{t-1}

    // ---- 1. x_T ~ N(0, I): the generation chain always starts from pure,
    // structureless Gaussian noise -- by construction (see the schedule:
    // alpha_bar_{T-1} is tiny), q(x_T | x_0) is indistinguishable from
    // N(0,I) for essentially any x_0, so sampling starts exactly there. ----
    diffusion_sample_prior(d_x, seed);

    printf("[generate] reverse diffusion: T=%d steps, image %dx%dx%d\n",
           TIMESTEPS, IMG_CHANNELS, IMG_SIZE, IMG_SIZE);

    // ---- 2. iteratively denoise from t = T-1 down to t = 0 ----
    for (int t = TIMESTEPS - 1; t >= 0; t--) {
        const float* d_temb_t = d_temb_table + (size_t)t * TIME_EMB_DIM;

        // Ask the network what noise it thinks is present in x_t.
        network_forward(net, d_x, d_temb_t);   // net->out = eps_theta(x_t, t)

        // Remove (an estimate of) that noise, re-inject a smaller amount of
        // fresh stochastic noise (unless this is the very last step, t==0).
        diffusion_reverse_step(d_x, net->out, d_x_prev, sched, t,
                                seed + (unsigned long long)(TIMESTEPS - t));

        // x_prev just became the new "current" sample; swap pointers rather
        // than copying IMG_ELEMS floats device-to-device every step.
        std::swap(d_x, d_x_prev);

        if (t % 50 == 0 || t == TIMESTEPS - 1) {
            printf("  t = %4d / %d\n", t, TIMESTEPS);
        }
    }

    // ---- 3. d_x now holds x_0: copy to host, denormalize, write PNG ----
    float* h_x0 = (float*)malloc((size_t)IMG_ELEMS * sizeof(float));
    copy_to_host(h_x0, d_x, IMG_ELEMS);

    if (save_image_chw(output_path, h_x0)) {
        printf("[generate] wrote generated image to '%s'\n", output_path);
    }

    free(h_x0);
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_x_prev));
}
```

### Line-by-Line Explanation

- No new kernels here at all — generation is **pure host-side orchestration** reusing `network_forward` (section 4) and `diffusion_reverse_step`/`diffusion_sample_prior` (section 3). This is one of the appealing properties of the noise-prediction parameterization: the *exact same* network, called with the *exact same* function signature, does double duty for both training (predict noise to compute a loss) and sampling (predict noise to subtract it) — nothing about `network_forward` needs to know or care which mode it's being used in.
- **`std::swap(d_x, d_x_prev)`** avoids a 3,072-float device-to-device `cudaMemcpy` on every one of the 300 iterations — instead of copying data, we just swap which host-side pointer variable refers to which of the two pre-allocated device buffers. This is a small but real optimization: 300 avoided memcpys instead of 300 performed ones, at zero extra code complexity.
- **Why `x_T ~ N(0,I)` is a valid starting point, not an approximation**: the schedule is constructed so `alpha_bar[T-1]` is extremely close to 0 (`β` ramps up to 0.02 over 300 steps), meaning $q(x_T\mid x_0)\approx\mathcal N(0,I)$ regardless of what `x_0` was — the forward process has, by step `T`, destroyed essentially all information about the original image. This is *why* sampling can start from literal random noise and still be asked to hallucinate a plausible `x_0`.
- **Seed scheme**: `seed + (TIMESTEPS - t)` guarantees every one of the 300 reverse steps draws a *different* noise realization (reusing the same noise across steps would correlate errors across the whole chain and visibly bias the output).

# 7. Orchestration (`main.cu`)

```cpp
// main.cu
//
// Orchestration. Ties together utils.cu (image I/O), diffusion.cu (noise
// schedule + forward/reverse kernels), network.cu (forward pass),
// train.cu (backward pass + Adam), and generate.cu (sampling loop) into a
// single end-to-end "load image -> train -> generate" pipeline.

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "common.h"
#include "utils.h"
#include "diffusion.h"
#include "network.h"
#include "train.h"
#include "generate.h"

struct Args {
    std::vector<std::string> image_paths;
    int iters = TRAIN_ITERS_DEFAULT;
    std::string out_path = "output/generated.png";
    unsigned int seed = 1234u;
};

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s <image1> [image2 ...] [--iters N] [--out path.png] [--seed N]\n\n"
        "  Trains a minimal from-scratch CUDA DDPM to overfit the given\n"
        "  image(s), then generates a new sample via reverse diffusion.\n\n"
        "  --iters N     number of training iterations (default %d)\n"
        "  --out  path   where to write the generated PNG (default %s)\n"
        "  --seed N      RNG seed for init/training/sampling (default 1234)\n",
        prog, TRAIN_ITERS_DEFAULT, "output/generated.png");
}

static Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--iters" && i + 1 < argc) {
            args.iters = std::atoi(argv[++i]);
        } else if (a == "--out" && i + 1 < argc) {
            args.out_path = argv[++i];
        } else if (a == "--seed" && i + 1 < argc) {
            args.seed = (unsigned int)std::atoi(argv[++i]);
        } else if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            args.image_paths.push_back(a);
        }
    }
    return args;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    if (args.image_paths.empty()) {
        fprintf(stderr, "Error: no input images given.\n\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    // ------------------------------------------------------------------
    // 1. Load the (tiny) dataset. The project is scoped to overfit a
    //    handful of images, so we simply keep every image resident on the
    //    device for the whole run and pick one at random each iteration.
    // ------------------------------------------------------------------
    const int num_images = (int)args.image_paths.size();
    std::vector<float*> d_dataset(num_images);
    for (int i = 0; i < num_images; i++) {
        float* h_img = load_image_chw(args.image_paths[i].c_str());
        if (!h_img) {
            fprintf(stderr, "Error: could not load '%s'\n", args.image_paths[i].c_str());
            return EXIT_FAILURE;
        }
        d_dataset[i] = device_alloc_from_host(h_img, IMG_ELEMS);
        free(h_img);
        printf("[main] loaded '%s' -> resized to %dx%dx%d, normalized to [-1,1]\n",
               args.image_paths[i].c_str(), IMG_CHANNELS, IMG_SIZE, IMG_SIZE);
    }

    // ------------------------------------------------------------------
    // 2. Diffusion schedule (host) + sinusoidal time-embedding table (device).
    // ------------------------------------------------------------------
    DiffusionSchedule sched;
    build_diffusion_schedule(&sched);
    float* d_temb_table = build_time_embedding_table_device();
    printf("[main] diffusion schedule ready: T=%d, beta in [%.1e, %.1e]\n",
           TIMESTEPS, BETA_START, BETA_END);

    // ------------------------------------------------------------------
    // 3. Network (parameters, gradients, Adam moments, activation cache).
    // ------------------------------------------------------------------
    Network net;
    network_init(&net, args.seed);
    printf("[main] network initialized: conv(%d->%d)-conv(%d->%d)-conv(%d->%d), "
           "time-emb dim=%d\n",
           IMG_CHANNELS, HIDDEN_CH, HIDDEN_CH, HIDDEN_CH, HIDDEN_CH, IMG_CHANNELS,
           TIME_EMB_DIM);

    // ------------------------------------------------------------------
    // 4. Training loop: SGD (batch size 1) over (random image, random t).
    //    Each iteration corrupts one image to a random timestep, asks the
    //    network to predict the noise that was added, and updates the
    //    network's weights to reduce the MSE between predicted and true
    //    noise -- this IS the entire DDPM training objective (Ho et al.
    //    2020, Eq. 14: the "simple" loss L_simple).
    // ------------------------------------------------------------------
    float* d_xt    = device_alloc(IMG_ELEMS);  // x_t for the current step
    float* d_noise = device_alloc(IMG_ELEMS);  // ground-truth epsilon for this step

    std::mt19937 rng(args.seed + 1);
    std::uniform_int_distribution<int> pick_image(0, num_images - 1);
    std::uniform_int_distribution<int> pick_t(0, TIMESTEPS - 1);

    unsigned long long noise_seed_counter = 1;
    float last_logged_loss = 0.0f;

    printf("[main] training for %d iterations...\n", args.iters);
    for (int iter = 1; iter <= args.iters; iter++) {
        int img_idx = pick_image(rng);
        int t       = pick_t(rng);

        // q(x_t | x0): corrupt the chosen clean image to timestep t.
        diffusion_forward_sample(d_dataset[img_idx], d_xt, d_noise, sched, t,
                                  noise_seed_counter++);

        // epsilon_theta(x_t, t) -> loss -> gradients -> Adam update.
        const float* d_temb_t = d_temb_table + (size_t)t * TIME_EMB_DIM;
        network_forward(&net, d_xt, d_temb_t);
        network_backward(&net, d_noise);
        adam_step(&net, iter, LEARNING_RATE);

        if (iter == 1 || iter % LOG_EVERY == 0) {
            last_logged_loss = compute_mse_loss(net.out, d_noise, IMG_ELEMS);
            printf("  iter %6d / %d   mse_loss = %.6f   (t=%3d, img=%d)\n",
                   iter, args.iters, last_logged_loss, t, img_idx);
        }
    }
    printf("[main] training complete. last logged loss = %.6f\n", last_logged_loss);

    // ------------------------------------------------------------------
    // 5. Generation: pure noise -> T reverse steps -> PNG on disk.
    // ------------------------------------------------------------------
    generate_image(&net, sched, d_temb_table, args.out_path.c_str(),
                    /*seed=*/ noise_seed_counter + 999);

    // ------------------------------------------------------------------
    // 6. Cleanup.
    // ------------------------------------------------------------------
    for (int i = 0; i < num_images; i++) CUDA_CHECK(cudaFree(d_dataset[i]));
    CUDA_CHECK(cudaFree(d_xt));
    CUDA_CHECK(cudaFree(d_noise));
    CUDA_CHECK(cudaFree(d_temb_table));
    network_free(&net);

    printf("[main] done.\n");
    return EXIT_SUCCESS;
}
```

### Line-by-Line Explanation

- **CLI parsing** accepts any number of positional image paths (dataset) plus `--iters`/`--out`/`--seed` flags — anything not matching a flag is pushed onto `image_paths`, so `./ddpm data/a.png data/b.png --iters 8000` works for a "few images" run, and `./ddpm data/cat.png` alone gives the single-image overfitting case the brief calls out explicitly.
- **Step 1 (dataset load)**: every image is decoded, resized, normalized (`load_image_chw`), uploaded to the device, and the *host* copy is immediately freed (`free(h_img)`) — only the device copies persist for the run, since that's all the training loop touches.
- **Step 4 is where the entire DDPM training algorithm lives**, and it's deliberately just five function calls: sample a random `(image, t)` pair → corrupt the image (`diffusion_forward_sample`) → predict the noise (`network_forward`) → backprop against the *true* noise (`network_backward`) → update weights (`adam_step`). This is **stochastic gradient descent with batch size 1**: each iteration is a single (image, timestep) sample rather than a mini-batch. It's a deliberate scope decision — the brief's own guidance to "prioritize a minimal architecture...so it fits" extends naturally to this: adding a batch dimension would mean every kernel in sections 4–5 gains a batch index and every bias-gradient reduction becomes a sum over batch *and* space, non-trivially more code for a project whose explicit goal is *overfitting a handful of images*, where batch-size-1 SGD converges just fine.
- **Logging** only calls `compute_mse_loss` every `LOG_EVERY=250` iterations (plus iteration 1), not every iteration — the loss kernel's parallel reduction is cheap, but there's no reason to synchronize/readback to the host every single step when only human-readable progress is needed.
- **Step 6 cleanup** frees every device allocation this file made directly (`d_dataset[i]`, `d_xt`, `d_noise`, `d_temb_table`) and delegates the network's ~30 internal buffers to `network_free` — mirroring exactly what section 4 allocates, so nothing leaks across a run.

# Building and Running

```bash
# 1. unzip the project (below), then:
cd ddpm_cuda
make ARCH=sm_86        # match your GPU; see the Makefile header for a table

# 2. put a real image in data/, then:
./bin/ddpm data/your_photo.png --iters 5000 --out output/generated.png

# or, for a few images at once:
./bin/ddpm data/img1.png data/img2.png data/img3.png --iters 8000

# convenience wrapper (same thing):
make run IMG=data/your_photo.png ITERS=5000
```

Expected console output looks like:
```
[main] loaded 'data/your_photo.png' -> resized to 3x32x32, normalized to [-1,1]
[main] diffusion schedule ready: T=300, beta in [1.0e-04, 2.0e-02]
[main] network initialized: conv(3->32)-conv(32->32)-conv(32->3), time-emb dim=32
[main] training for 5000 iterations...
  iter      1 / 5000   mse_loss = 1.014221   (t=149, img=0)
  iter    250 / 5000   mse_loss = 0.612903   (t= 87, img=0)
  ...                                           <- should trend toward ~0 as it overfits
  iter   5000 / 5000   mse_loss = 0.041822   (t= 12, img=0)
[main] training complete. last logged loss = 0.041822
[generate] reverse diffusion: T=300 steps, image 3x32x32
  t =  299 / 300
  ...
  t =    0 / 300
[generate] wrote generated image to 'output/generated.png'
[main] done.
```

# How This Was Verified (no GPU available in my environment)

I don't have `nvcc`/a GPU in this sandbox, so "trust me" isn't good enough for a claim of "production-ready, compilable." What I actually did:

1. **Finite-difference gradient checking (NumPy)** — implemented the exact forward/backward formulas used in `train.cu` (conv2d, linear, SiLU, MSE, the bias-sharing identity) and checked every one of the 10 parameter gradients against central-difference numerical gradients. All matched to ~1e-7 relative error.
2. **Tiled-kernel indexing emulation** — reproduced `conv2d_forward_tiled_kernel`'s exact block/halo-load/sync control flow in Python (including a non-multiple-of-`TILE_DIM` size to exercise boundary tiles) and confirmed it's numerically identical to a direct reference convolution.
3. **Closed-form sampling check** — verified the reverse-step coefficients (`inv_sqrt_alpha`, `noise_coef`, `sigma`) against the DDPM posterior mean $\tilde\mu(x_t,x_0)$ derived independently via the $x_0$-parameterization; both formulas agreed to float64 precision.
4. **Full structural compile + link + run** — wrote a CUDA-runtime "shim" (fake `cudaMalloc`/`cudaMemcpy`/`curand`/`dim3`, `<<<...>>>` stripped) and compiled all six real `.cu` files against it with `g++ -std=c++17 -Wall -Wextra`, linked them into one binary, and ran the actual end-to-end pipeline (load a synthetic PNG → train loop → 300-step generation loop → save PNG) — it completed with no crashes and produced a structurally valid PNG. This catches exactly the class of bug that "compiles per-file but fails to link" or "segfaults on a real run" represents, which finite-difference checks alone can't.

What this *doesn't* prove: actual numerical convergence quality on a real GPU, or that `nvcc` accepts 100% of the CUDA-specific syntax (`__shared__`, `<<<...>>>`, `curand_kernel.h`) with zero warnings — that step needs real hardware, which I don't have here. The syntax used throughout (dynamic vs. static shared memory, `__restrict__`, the tiling pattern, device-API curand) follows standard, well-established CUDA idioms.

# Design Decisions & Extending This Further

- **Why a 3-layer CNN instead of a full U-Net**: the brief explicitly permits "a micro-UNet **or** simple CNN backbone." A real U-Net's down/up-sampling path and concatenated skip connections would roughly double the backward-pass code (concatenation backprop splits a gradient by channel-slice; strided down/up-sampling needs its own backward kernels) for a project whose stated goal is a complete, correct, from-scratch reference — I chose depth-of-correctness over breadth-of-architecture.
- **Batch size 1**: see the `main.cu` explanation above — extending every kernel to a batch dimension is mechanical (add a `B` loop/index everywhere) but roughly doubles the line count for something orthogonal to the DDPM math itself.
- **Straightforward extensions**, roughly in order of effort: (1) swap the linear beta schedule for cosine (Nichol & Dhariwal 2021) — a one-function change in `diffusion.cu`; (2) add a real down/up-sampling U-Net with concatenated skips; (3) replace per-launch `curand_init` with a persistent `curandState` buffer for a measurable speedup at scale; (4) batch training with a 4th tensor dimension; (5) mixed precision (`__half`) for the conv kernels.

I've packaged the full project (including the genuine `stb_image.h`/`stb_image_write.h`) below — unzip and `make` on a CUDA machine.

> Presented file
>
> - Presenting file(s)...
> - **Done**

> File: [/mnt/user-data/outputs/ddpm_cuda.zip](https://claude.ai/api/organizations/7e71ebc7-9770-4a75-a0fe-fd1afa7586d2/conversations/9fe9937b-4328-4cd5-9ca6-353781998a2b/wiggle/download-file?path=%2Fmnt%2Fuser-data%2Foutputs%2Fddpm_cuda.zip)

## User:

> 8/21/2026 16:04:55

give some study material for this. with links

## Assistant:

> 8/21/2026 16:07:59

> Architected curated resource guide spanning diffusion theory, CUDA fundamentals, optimization techniques.
>
> - Denoising Diffusion Probabilistic Models Ho et al 2020 arxiv
> - Lilian Weng "What are Diffusion Models" blog lilianweng.github.io
> - HuggingFace "The Annotated Diffusion Model" blog
> - Mark Harris NVIDIA "Optimizing Parallel Reduction in CUDA" pdf
> - NVIDIA CUDA C++ Programming Guide docs.nvidia.com shared memory
> - CUDA cuRAND device API documentation curand_init curand_normal
> - Adam optimizer paper Kingma Ba arxiv 1412.6980
> - Simon Boehm "How to Optimize a CUDA Matmul Kernel" blog siboehm
> - CS231n Stanford convolutional neural networks course notes cs231n.github.io
> - Andrej Karpathy "Neural Networks: Zero to Hero" YouTube playlist micrograd
> - Nichol Dhariwal "Improved Denoising Diffusion Probabilistic Models" arxiv 2102.09672
> - Architected curated resource guide spanning diffusion theory, CUDA fundamentals, optimization techniques.
> - **Done**

Here's a curated set of resources, organized to match the project — theory first, then the CUDA/HPC substrate, then the two specific techniques the kernels lean on, then the calculus underneath `train.cu`.

## 1. The DDPM math itself

Start here to understand *what* `diffusion.cu` and the training loop in `main.cu` are actually computing.

> Used tool
>
> - Papers and explainers behind the DDPM math implemented in diffusion.cu
> - **Done**

## 2. CUDA & GPU programming fundamentals

The platform everything is built on. If any of the grid/block/`__shared__`/`__syncthreads()` reasoning in the explanations felt unfamiliar, these are the primary sources.

> Used tool
>
> - Official NVIDIA references for the CUDA language, memory model, and curand API
> - **Done**

## 3. The two signature HPC patterns in the code

`conv2d_forward_tiled_kernel` and the `mse_loss_kernel`/`channel_sum_kernel` reductions are the two most technique-dense kernels in the project. These are the classic write-ups those patterns come from — worth reading end to end, since both build up from a naive version to the one actually used.

> Used tool
>
> - Worklogs behind the tiling and reduction techniques used in network.cu and train.cu
> - **Done**

## 4. The math underneath `train.cu`: backprop, conv gradients, Adam

If the manual chain-rule / conv-backward derivation was the hardest part to follow, this is the fastest path to being able to derive it yourself instead of taking it on faith.

> Used tool
>
> - Papers and courses behind the manual backprop and Adam optimizer in train.cu
> - **Done**

**If you only have time for three:** Lilian Weng's post for the diffusion math, the CUDA Programming Guide's "Thread Hierarchy" + "Shared Memory" sections for the HPC mechanics, and CS231n's conv-net notes for the backprop derivation. Together those three cover what's actually needed to read every kernel in the project without taking anything on faith. Karpathy's `micrograd` (first video in the playlist above, code at `github.com/karpathy/nn-zero-to-hero`) is the single best complement to `train.cu` specifically — it's the same manual-backprop idea, just for scalars instead of tensors, so it's a much easier place to build the intuition before looking at the CUDA version.

---

**Sources:**

- [Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/pdf/2107.03006)
- [[2107.03006] Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006)
- [A Comprehensive Review of Diffusion Models in Smart Agriculture: Progress, Applications, and Challenges](https://arxiv.org/pdf/2507.18376)
- [Binary Diffusion Probabilistic Model](https://arxiv.org/pdf/2501.13915)
- [Differentially Private Diffusion Models Generate Useful Synthetic Images](https://arxiv.org/pdf/2302.13861)
- [GitHub - hojonathanho/diffusion: Denoising Diffusion Probabilistic Models · GitHub](https://github.com/hojonathanho/diffusion)
- [Denoising Diffusion Probabilistic Models in Six Simple Steps](https://arxiv.org/pdf/2402.04384)
- [On Calibrating Diffusion Probabilistic Models](https://arxiv.org/pdf/2302.10688)
- [Evolvable Conditional Diffusion](https://arxiv.org/pdf/2506.13834)
- [Denoising Diffusion Probabilistic Models for Styled Walking Synthesis](https://arxiv.org/pdf/2209.14828)
- [Fast and Scalable Analytical Diffusion](https://arxiv.org/html/2602.16498)
- [Improved Denoising Diffusion Probabilistic Models](https://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf)
- [[2006.11239] Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Denoising diffusion probabilistic models are optimally adaptive to unknown low dimensionality](https://arxiv.org/html/2410.18784v3)
- [Denoising Diffusion Probabilistic Models](https://hojonathanho.github.io/diffusion/)
- [Denoising Diffusion Probabilistic Models | alphaXiv](https://www.alphaxiv.org/overview/2006.11239)
- [Denoising diffusion probabilistic models | Proceedings of the 34th International Conference on Neural Information Processing Systems](https://dl.acm.org/doi/abs/10.5555/3495724.3496298)
- [Denoising Diffusion Probabilistic Models | by Eleventh Hour Enthusiast | Medium](https://medium.com/@EleventhHourEnthusiast/denoising-diffusion-probabilistic-models-63b4fd3a3b67)
- [Denoising Diffusion Probabilistic Models Jonathan Ho UC Berkeley](https://arxiv.org/pdf/2006.11239)
- [[PDF] Denoising Diffusion Probabilistic Models | Semantic Scholar](https://www.semanticscholar.org/paper/Denoising-Diffusion-Probabilistic-Models-Ho-Jain/5c126ae3421f05768d8edd97ecd44b1364e2c99a)
- [What are Diffusion Models? | Lil'Log](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Diffusion Model](https://donaldlab.cs.duke.edu/Teaching/Seminar22/Recitations/Diffusion%20Model.pdf)
- [Diffusion Models for Video Generation | Lil'Log](https://lilianweng.github.io/posts/2024-04-12-diffusion-video/)
- [Fully Annotated Guide to "What are Diffusion Models?" | The Kiseki Log](https://ki-seki.github.io/posts/250902-diffusion-annotated/)
- [Awesome-Diffusion-Models | A collection of resources and papers on Diffusion Models](https://diff-usion.github.io/Awesome-Diffusion-Models/)
- [A Visual Guide to How Diffusion Models Work | Towards Data Science](https://towardsdatascience.com/a-visual-guide-to-how-diffusion-models-work/)
- [Posts Archive Search Tags FAQ emojisearch.app What are Diffusion Models?](https://www2.cs.uh.edu/~ceick/ai/DF_Reading.pdf)
- [Posts | Lil'Log](https://lilianweng.github.io/posts/)
- [Lil'Log](https://lilianweng.github.io/)
- [A Gentle Introduction and Tutorial on Deep Generative Models in Transportation Research](https://arxiv.org/pdf/2410.07066)
- [Video-Generation | Lil'Log](https://lilianweng.github.io/tags/video-generation/)
- [High-resolution image generation using diffusion models](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12561851)
- [Non-destructive generative image editing](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12462456)
- [Generative-Model | Lil'Log](https://lilianweng.github.io/tags/generative-model/)
- [Diffusion model](https://en.wikipedia.org/wiki/Diffusion_model)
- [arxiv.org](https://arxiv.org/pdf/2209.00796v7)
- [DEV Community](https://dev.to/gilles_hamelink_ea9ff7d93/unlocking-the-power-of-diffusion-models-a-deep-dive-into-language-efficiency-4e09)
- [arxiv.org](https://arxiv.org/pdf/2209.00796v8)
- [0% found this document useful (0 votes)](https://www.scribd.com/document/824579133/Diffusion-Models-in-Deep-Learning)
- [Hacker Newsnew | past | comments | ask | show | jobs | submit | ycy's commentslogin](https://news.ycombinator.com/threads?id=ycy)
- [Annotated-Diffusion-Model/README.md at main · jackaduma/Annotated-Diffusion-Model](https://github.com/jackaduma/Annotated-Diffusion-Model/blob/main/README.md)
- [GitHub - jackaduma/Annotated-Diffusion-Model: The Annotated Diffusion Model · GitHub](https://github.com/jackaduma/Annotated-Diffusion-Model)
- [HuggingFace ブログ : 注釈付き拡散モデル – ClassCat® Blog](https://torch.classcat.com/2022/11/06/huggingface-blog-annotated-diffusion/)
- [blog/stable_diffusion.md at main · huggingface/blog](https://github.com/huggingface/blog/blob/main/stable_diffusion.md)
- [huggingface-blog/stable_diffusion.md at main · mbrukman/huggingface-blog](https://github.com/mbrukman/huggingface-blog/blob/main/stable_diffusion.md)
- [Stable Diffusion with 🧨 Diffusers](https://huggingface.co/blog/stable_diffusion)
- [Stable Diffusion with Hugging Face Inference Endpoints](https://www.philschmid.de/stable-diffusion-inference-endpoints)
- [Unit 1: An Introduction to Diffusion Models · Hugging Face](https://huggingface.co/learn/diffusion-course/unit1/1)
- [tglcourse - An Introduction to Diffusion Models](https://johnowhitaker.github.io/tglcourse/dm1.html)
- [Japanese Stable Diffusion](https://huggingface.co/blog/japanese-stable-diffusion)
- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)
- [Evaluating a Synthetic Image Dataset Generated with Stable Diffusion](https://arxiv.org/pdf/2211.01777)
- [Are Diffusion Models Vision-And-Language Reasoners?](https://arxiv.org/pdf/2305.16397)
- [The Annotated Diffusion Model and Study Notes | Changjiang Cai | Researcher on Computer Vision](https://www.changjiangcai.com/studynotes/2024-09-15-annotated-diffusion/)
- [The annotated diffusion models](https://www.kaggle.com/code/aisuko/the-annotated-diffusion-models)
- [discuss.huggingface.co](https://discuss.huggingface.co/t/dreambooth-tutorial/57498)
- [Hugging Face](https://huggingface.co/datasets/huggingface/documentation-images/tree/2e5d3ae6cb3db7c3b4cac7768f7ff7accea69071/blog)
- [Hugging Face's logo Hugging Face](https://huggingface.co/datasets/huggingface/documentation-images/tree/b5c3c91818547806de18f5ec457de255b71a84e8/blog/diffusion-models-event)
- [Hugging Face](https://huggingface.co/datasets/huggingface/documentation-images/commits/refs%2Fpr%2F98/blog?p=2)
- [I recently reread "Optimizing Parallel Reduction in CUDA" by Mark Harris and was wondering if it was still considered the best solution. As far as I can tell the solution is written for CC ≤1.3 and today CC=12.0](https://gist.github.com/troelsy/fff6aac2226e080dcebf05531a11d44e)
- [GitHub - mahmoudmaftah/MaxReduction-Cuda: Six-stage CUDA parallel reduction optimization: from basic global memory to warp shuffles and bank conflict avoidance · GitHub](https://github.com/mahmoudmaftah/MaxReduction-Cuda)
- [Optimizing Parallel Reduction in CUDA Mark Harris NVIDIA Developer Technology](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
- [Harris, M. (2007) Optimizing Parallel Reduction in CUDA. Nvidia Developer Technology, 2, 70. - References - Scientific Research Publishing](https://www.scirp.org/reference/referencespapers?referenceid=3741198)
- [Optimizing Parallel Reduction in CUDA Optimizing Parallel Reduction in CUDA](https://people.maths.ox.ac.uk/gilesm/cuda/prac4/reduction.pdf)
- [Optimizing Parallel Reduction in CUDA Mark Harris NVIDIA Developer Technology](https://vuduc.org/teaching/cse6230-hpcta-fa12/slides/cse6230-fa12--05b-reduction-notes.pdf)
- [Harris, M. (2008) Optimizing Parallel Reduction in CUDA. NVIDIA. - References - Scientific Research Publishing](https://www.scirp.org/reference/ReferencesPapers?ReferenceID=1546858)
- [CS 380 - GPU and GPGPU Programming Lecture 21: CUDA Memory, Pt. 2 GPU Reduction](https://vccvisualization.org/teaching/CS380/CS380_fall2020_lecture_21.pdf)
- [CS 380 - GPU and GPGPU Programming Lecture 21: GPU Parallel Reduction](https://vccvisualization.org/teaching/CS380/CS380_fall2024_lecture_21.pdf)
- [Equivalence Checking of ML GPU Kernels](https://arxiv.org/pdf/2511.12638)
- [Optimizing Parallel Reduction in CUDA : NOTES | PDF](https://www.slideshare.net/slideshow/optimizing-parallel-reduction-in-cuda-notes/249770220)
- [Algorithmic Strategies for Optimizing the Parallel Reduction Primitive in CUDA](https://www.csd.uwo.ca/~mmorenom/CS433-CS9624/Resources/Algorithmic_strategies_for_optimizing_the_parallel_reduction_primitive_in_CUDA.pdf)
- [Optimizing Parallel Reduction in CUDA. Mark Harris](https://www.pinterest.com/pin/244179611033230009/)
- [Optimizing parallel reduction in CUDA Follows the presentation](https://slidetodoc.com/optimizing-parallel-reduction-in-cuda-follows-the-presentation/)
- [pdfs/NVIDIA - Optimizing Parallel Reduction in CUDA - Slides.pdf at master · tpn/pdfs](https://github.com/tpn/pdfs/blob/master/NVIDIA%20-%20Optimizing%20Parallel%20Reduction%20in%20CUDA%20-%20Slides.pdf)
- [Optimizing Parallel Reduction in CUDA Mark Harris NVIDIA](https://slidetodoc.com/optimizing-parallel-reduction-in-cuda-mark-harris-nvidia/)
- [A Study of Single and Multi-device Synchronization Methods in Nvidia   GPUs](https://arxiv.org/pdf/2004.05371)
- [Patterns and Rewrite Rules for Systematic Code Generation (From   High-Level Functional Patterns to High-Performance OpenCL Code)](https://arxiv.org/pdf/1502.02389)
- [GPU Load Balancing](https://arxiv.org/pdf/2212.08964)
- [Optimizing Parallel Reduction in NVIDIA's CUDA - YouTube](https://www.youtube.com/watch?v=5YDtuaFGe1c)
- [CUDA C++ Programming Guide Release 13.3 NVIDIA Corporation Jun 25, 2026](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf)
- [CUDA C++ Programming Guide Release 13.0 NVIDIA Corporation Nov 02, 2025](https://docs.nvidia.com/cuda/archive/13.0.2/pdf/CUDA_C_Programming_Guide.pdf)
- [CUDA C++ Programming Guide — CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/archive/12.3.1/cuda-c-programming-guide/index.html)
- [CUDA C++ Programming Guide (Legacy) — CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Programming Guide Release 12.2 NVIDIA Jun 22, 2023](https://docs.nvidia.com/cuda/archive/12.2.0/pdf/CUDA_C_Programming_Guide.pdf)
- [PG-02829-001_v11.6 | February 2022 CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/archive//11.6.0/pdf/CUDA_C_Programming_Guide.pdf)
- [PG-02829-001_v12.0 | December 2022 CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/archive/12.0.0/pdf/CUDA_C_Programming_Guide.pdf)
- [CUDA C++ PROGRAMMING GUIDE PG-02829-001_v11.0 | August 2020 Design Guide](https://docs.nvidia.com/cuda/archive//11.0/pdf/CUDA_C_Programming_Guide.pdf)
- [Contents — CUDA C++ Best Practices Guide 12.8 documentation](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-best-practices-guide/contents.html)
- [PG-02829-001_v11.1 | September 2020 CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/archive/11.1.0/pdf/CUDA_C_Programming_Guide.pdf)
- [Contents — CUDA C++ Best Practices Guide 13.3 documentation](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/contents.html)
- [CUDA Best Practices Guide — CUDA C++ Best Practices Guide 13.3 documentation](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [Programming Guide :: CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/archive/10.2/cuda-c-programming-guide/index.html)
- [2.1. Intro to CUDA C++ — CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)
- [CUDA Programming Guide — CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)
- [5.7. CUDA C++ Memory Model — CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html)
- [docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/search.html)
- [docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.3.0/cuda-c-programming-guide/search.html)
- [docs.nvidia.com](https://docs.nvidia.com/cuda/archive/12.2.2/cuda-c-programming-guide/search.html)
- [cuRAND :: CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/curand/device-api-overview.html)
- [CUDA CURAND Library PG-05328-032_V01 August, 2010](https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/CURAND_Library.pdf)
- [PG-05328-050 _v12.3 | October 2023 cuRAND Library Programming Guide](https://docs.nvidia.com/cuda/archive/12.3.1/pdf/CURAND_Library.pdf)
- [CUDA Toolkit 4.2 CURAND Guide PG-05328-041_v01 | March 2012](https://developer.download.nvidia.com/compute/DevZone/docs/html/CUDALibraries/doc/CURAND_Library.pdf)
- [CURAND LIBRARY PG-05328-050 _v5.5 | July 2013 Programming Guide](https://cseweb.ucsd.edu/classes/wi15/cse262-a/static/cuda-5.5-doc/pdf/CURAND_Library.pdf)
- [CURAND LIBRARY PG-05328-050 _v5.5 | May 2013 Programming Guide](https://cs.colby.edu/courses/S14/cs336/online_materials/CURAND_Library.pdf)
- [CUDA Libraries | CUDA Succinctly | Syncfusion®](https://www.syncfusion.com/succinctly-free-ebooks/cuda/cuda-libraries)
- [professional-cuda-c-programming/examples/chapter08/rand-kernel.cu at master · deeperlearning/professional-cuda-c-programming](https://github.com/deeperlearning/professional-cuda-c-programming/blob/master/examples/chapter08/rand-kernel.cu)
- [CUDA Toolkit 5.0 CURAND Guide PG-05328-050_v02 | September 2012](https://www.math.univ-paris13.fr/~cuvelier/docs/Informatique/CUDA/docs/5.0/CURAND_Library.pdf)
- [CUDA Toolkit 4.1 CURAND Guide PG-05328-041_v01 | January 2012](http://wwwteor.mi.infn.it/~vicini/CUDA/doc/CURAND_guida.pdf)
- [PG-05328-050 _vRelease Version | January 2024 cuRAND Library](https://docs.nvidia.com/cuda/pdf/CURAND_Library.pdf)
- [CURAND LIBRARY PG-05328-050 _vRelease Version | July 2017 Programming Guide](https://ynitta.com/lec/cuda/doc_v9_0/pdf/CURAND_Library.pdf)
- [cuRAND Usage: CUDA Feature Testing](https://codingbyexample.com/2020/09/15/curand/)
- [The CURAND library Will Landau Host interface Device interface](https://wlandau.github.io/gpu/lectures/curand/curand.pdf)
- [cuRAND Device APIs (nvmath.device.random)#](https://docs.nvidia.com/cuda/archive/13.1.0/nvmath-python/0.2.1/device-apis/curand.html)
- [CURAND Library](https://docs.nvidia.com/cuda/archive/12.5.1/pdf/CURAND_Library.pdf)
- [Random Number Runtime Library APIs](https://docs.nvidia.com/hpc-sdk/archive/25.1/compilers/fortran-cuda-interfaces/cfrand-runtime.html)
- [cuRAND (nvmath.bindings.curand)#](https://docs.nvidia.com/cuda/archive/13.1.0/nvmath-python/0.2.1/bindings/curand.html)
- [3. Device API Overview](https://cs400-web.cs.wisc.edu/nchou/s/cuda-9.1/doc/html/curand/device-api-overview.html)
- [3. Device API Overview](https://docs.nvidia.com/cuda/archive/12.0.1/curand/device-api-overview.html)
- [Paper Summary: Adam: A Method for Stochastic Optimization](https://karan3-zoh.medium.com/paper-summary-adam-a-method-for-stochastic-optimization-4b0a030a019f)
- [Kingma, D.P. and Ba, J. (2017) Adam A Method for Stochastic Optimization. arXiv 1412.6980. - References - Scientific Research Publishing](https://www.scirp.org/reference/referencespapers?referenceid=3756897)
- [ADAM: The Smart Optimizer That Changed Deep Learning. | by Vivek Tiwari | Medium](https://medium.com/@vivek_tiwari_vt/adam-the-smart-optimizer-that-changed-deep-learning-077070575899)
- [Kingma, D.P. and Ba, J. (2014) Adam A Method for Stochastic Optimization. arXiv 1412.6980. - References - Scientific Research Publishing](https://www.scirp.org/reference/referencespapers?referenceid=3680973)
- [[PDF] Adam: A Method for Stochastic Optimization by Diederik P. Kingma, Jimmy Ba · 10.48550/arxiv.1412.6980 · OA.mg](https://oa.mg/work/10.48550/arxiv.1412.6980)
- [[1412.6980v8] Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980v8?hl%3Aja=)
- [On a Conjecture Regarding the Adam Optimizer](https://arxiv.org/pdf/2111.08162)
- [Adam: A method for stochastic optimization | BibSonomy](https://www.bibsonomy.org/bibtex/57d2ac873f398f21bb94790081e80394)
- [Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/pdf/1907.08610)
- [Machine Learning](https://arxiv.org/pdf/2512.11133)
- [SADAM: Stochastic Adam, A Stochastic Operator for First- ...](https://arxiv.org/pdf/2205.10247)
- [Adam: A Method for Stochastic Optimization](https://bibbase.org/network/publication/kingma-ba-adamamethodforstochasticoptimization-2015)
- [Hindsight-Guided Momentum (HGM) Optimizer: An Approach to Adaptive Learning Rate](https://arxiv.org/pdf/2506.22479)
- [Dive into Deep Learning](https://arxiv.org/pdf/2106.11342)
- [An Automatic Learning Rate Schedule Algorithm for Achieving Faster   Convergence and Steeper Descent](https://arxiv.org/pdf/2310.11291)
- [Reevaluating Theoretical Analysis Methods for Optimization in Deep Learning](https://arxiv.org/pdf/2407.01825)
- [Nesterov acceleration despite very noisy gradients](https://arxiv.org/pdf/2302.05515)
- [Adam: A Method for Stochastic Optimization : Diederik P. Kingma : Free Download, Borrow, and Streaming : Internet Archive](https://archive.org/details/arxiv-1412.6980)
- [Under review as a conference paper at ICLR 2015](https://arxiv.org/pdf/1412.6980v1)
- [Published as a conference paper at ICLR 2015](https://arxiv.org/pdf/1412.6980)
- [Simon Boehm (@Si_Boehm) / X](https://x.com/si_boehm?lang=en)
- [Simon Boehm on X: "I wrote the most naive CUDA matrix multiply and iteratively optimised it to ~80% of cuBLAS performance: https://t.co/zHBigTKhCB" / X](https://x.com/Si_Boehm/status/1610335205767933952?lang=en)
- [Accelerating GPGPU Simulation by Strategically Parallelizing the Compute Bottleneck](https://drops.dagstuhl.de/entities/document/10.4230/OASIcs.PARMA-DITAM.2026.6)
- [Simon Boehm github.com/siboehm Technical Blog: siboehm.com Open Source Projects](https://siboehm.com/assets/resume.pdf)
- [Modern GPU Matmul Optimization](https://www.cloudrift.ai/blog/gpu-matmul-optimization)
- [GPU Kernel Scientist: An LLM-Driven Framework for Iterative Kernel Optimization](https://arxiv.org/html/2506.20807v2)
- [Good blogs about Matmul Kernel. “How to Optimize a CUDA Matmul Kernel… | by Hao Zhuang, AI and Compute | Medium](https://medium.com/@zhuangh/good-blogs-about-matmul-kernel-5e0de6413b57)
- [Let’s Do Some Basic GPU Optimizations | by Matthew Gunton | Medium](https://medium.com/@mgunton7/lets-do-some-basic-gpu-optimizations-e2f861e5ad04)
- [How to optimize a CUDA matmul kernel for cuBLAS-like performance (2022) | Hacker News](https://news.ycombinator.com/item?id=41079083)
- [CUDA Matrix Multiplication Optimization: From Naive to Near-cuBLAS | Abhik Sarkar](https://www.abhik.ai/articles/cuda-matrix-multiplication-optimization)
- [Advanced Matrix Multiplication Optimization on NVIDIA GPUs](https://salykova.github.io/sgemm-gpu)
- [GitHub - hova88/CUDA-MatMul-Practice · GitHub](https://github.com/hova88/CUDA-MatMul-Practice)
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
- [Surfacing a 60% performance bug in cuBLAS](https://www.cloudrift.ai/blog/beating-cublas-on-rtx-5090)
- [How to Optimize a CUDA Matmul Kernel for CuBLAS-Like Performance: A Worklog | Hacker News](https://news.ycombinator.com/item?id=34256392)
- [Optimizing Matrix Multiplication Algorithms and MatMul/GEMM Kernels](https://www.aussieai.com/research/matmul)
- [CUDA-L2: Surpassing cuBLAS Performance for Matrix Multiplication through Reinforcement Learning](https://arxiv.org/pdf/2512.02551)
- [Making LLMs Optimize Multi-Scenario CUDA Kernels Like Experts](https://arxiv.org/pdf/2603.07169)
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like ...](https://app.daily.dev/posts/how-to-optimize-a-cuda-matmul-kernel-for-cublas-like-performance-a-worklog-aatojbu6x)
- [GitHub - Shawn-wave/CS231n: All lecture notes and assignments for CS231n: Convolutional Neural Networks for Visual Recognition class by Stanford](https://github.com/Shawn-wave/CS231n)
- [GitHub - cs231n/cs231n.github.io: Public facing notes page · GitHub](https://github.com/cs231n/cs231n.github.io)
- [GitHub - huohuotm/CS231: Stanford course CS231n: Convolutional Neural Networks for Visual Recognition. · GitHub](https://github.com/huohuotm/CS231)
- [GitHub - morteza-im/cs231n: Note and Assignments for CS231n: Convolutional Neural Networks for Visual Recognition · GitHub](https://github.com/morteza-im/cs231n)
- [CS231n Convolutional Neural Networks for Visual Recognition](https://iamavenger.github.io/cs231n/)
- [GitHub - lalithnag/cs231n: Working repository of cs231n course on CNN & Deep Learning offered by Stanford · GitHub](https://github.com/lalithnag/cs231n)
- [GitHub - madalinabuzau/cs231n-convolutional-neural-networks-solutions: Assignment solutions for the CS231n course taught by Stanford on visual recognition. Spring 2017 solutions are for both deep learning frameworks: TensorFlow and PyTorch. · GitHub](https://github.com/madalinabuzau/cs231n-convolutional-neural-networks-solutions)
- [GitHub - jariasf/CS231n: My assignment solutions for CS231n - Convolutional Neural Networks for Visual Recognition · GitHub](https://github.com/jariasf/CS231n)
- [GitHub - cmh325/CS231n-stanford-course-material: Working through CS231n: Convolutional Neural Networks for Visual Recognition · GitHub](https://github.com/cmh325/CS231n-stanford-course-material)
- [GitHub - miguelfrde/stanford-cs231n: Coursework for Stanford CS231n Convolutional Neural Networks for Visual Recognition](https://github.com/miguelfrde/stanford-cs231n)
- [GitHub - MahanFathi/CS231: Complete Assignments for CS231n: Convolutional Neural Networks for Visual Recognition · GitHub](https://github.com/MahanFathi/CS231)
- [GitHub - Yorko/stanford_cs231n_2019: Solutions and comments to assignments for 2019 Stanford's course on convolutional neural networks](https://github.com/Yorko/stanford_cs231n_2019)
- [GitHub - Andrew-Ng-s-number-one-fan/CS231n-Convolutional-Neural-Networks-for-Visual-Recognition: A course offered by Stanford University · GitHub](https://github.com/Andrew-Ng-s-number-one-fan/CS231n-Convolutional-Neural-Networks-for-Visual-Recognition)
- [Convolutional Neural Networks (CNNs / ConvNets)](https://cs231n.github.io/convolutional-networks/)
- [GitHub - maxis42/Convolutional-Neural-Networks-Stanford-CS231n: Stanford CS class CS231n: Convolutional Neural Networks for Visual Recognition · GitHub](https://github.com/maxis42/Convolutional-Neural-Networks-Stanford-CS231n)
- [CS231n Final Lecture Notes on Convolutional Neural Networks and Image Classification - Studocu](https://www.studocu.com/en-us/document/stanford-university/convolutional-neural-networks-for-visual-recognition/1-cs231n-convolutional-neural-networks-for-visual-recognition/93095363)
- [GitHub - amanchadha/stanford-cs231n-assignments-2020: This repository contains my solutions to the assignments for Stanford's CS231n "Convolutional Neural Networks for Visual Recognition" (Spring 2020). · GitHub](https://github.com/amanchadha/stanford-cs231n-assignments-2020)
- [CS231n Deep Learning for Computer Vision](https://cs231n.github.io/)
- [GitHub - YYZhang2025/Stanford-CS231N: My solution and notes for the Stanford CS231N Deep Learning for Computer Vision · GitHub](https://github.com/YYZhang2025/Stanford-CS231N)
- [github.com](https://github.com/israfelsr/cs231n)
- [Implementation of Karpathy's Neural Networks: Zero to Hero Lecture Series](https://chizkidd.github.io/2024/07/15/karpathy-nn-zero-to-hero/)
- [Andrey Karpathy. Neural Networks: Zero to Hero](https://averkij.github.io/karcaps/)
- [Andrej Karpathy's Neural Networks: Zero to Hero — 1) Intro to Neural Networks and Backpropagation - DEV Community](https://dev.to/jun07/andrej-karpathys-neural-networks-zero-to-hero-1-intro-to-neural-networks-and-backpropagation-4j5f)
- [How Open Ai’s Andrej Karpathy Made One of the Best Tutorials in Deep Learning | by Usama Ahmed | Medium](https://medium.com/@0ssamaak0/how-open-ais-andrej-karpathy-made-one-of-the-best-tutorials-in-deep-learning-e6b6445a2d05)
- [Karpathy’s neural networks series | by Stephen Jonany | Medium](https://medium.com/@sjonany/karpathys-micrograd-walkthrough-535718235150)
- [GitHub - chizkidd/Karpathy-Neural-Networks-Zero-to-Hero: Detailed python notes & code for lectures and exercises of Andrej Karpathy's course "Neural Networks: Zero to Hero." The course is focused on building neural networks from scratch. · GitHub](https://github.com/chizkidd/Karpathy-Neural-Networks-Zero-to-Hero)
- [Andrej Karpathy's micrograd - Neural networks: Zero to Hero - YouTube](https://www.youtube.com/watch?v=44_j7ufypfw)
- [Andrej Karpathy Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLXYLzZ3XzIbi4lL43O6fIU_ojuZwBO6vi)
- [Neural Networks: Zero To Hero](https://karpathy.ai/zero-to-hero.html)
- [neolateral - Notes from Andrej Karpathy's building micrograd video.](https://neolateral.in/notes-from-andrej-karpathys-building-micrograd-video)
- [GitHub - karpathy/nn-zero-to-hero: Neural Networks: Zero to Hero · GitHub](https://github.com/karpathy/nn-zero-to-hero)
- [Neural Networks: Zero to Hero](https://music.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- [Neural Networks: Zero to Hero - csdiy.wiki](https://csdiy.wiki/en/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD/Neural%20Networks%EF%BC%9AZero%20to%20Hero/)
- [My lessons with Andrej - Bharat Bheesetti](https://bharatbheesetti.com/posts/zero_to_hero)
- [Learn Neural Networks with Andrej Karpathy - Brian Sigafoos](https://briansigafoos.com/neural-networks-karpathy/)
- [Neural Networks: Zero to Hero | Hacker News](https://news.ycombinator.com/item?id=35459031)
- [Building Neural Networks from Scratch: A Study Guide for Karpathy's 'Neural Networks: Zero to Hero' · Zing Forum](https://www.zingnex.cn/en/forum/thread/karpathyneural-networks-zero-to-hero)
- [Neural Networks: Zero to Hero - YouTube](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- [NeuralNetworks Zero To Hero](https://github.com/anubhavmaity/NeuralNetworks-Zero-To-Hero)
- [Neural Networks : Zero to Hero - 신경망 개발을 밑바닥 부터 코딩하며 배우기](https://news.hada.io/topic?id=8322)
- [[2102.09672] Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)
- [Denoising Diffusion Probabilistic Model: practical experiments for landscape generation from segmentation masks using EDEN — Multimodal Synthetic Dataset of Enclosed Garden Scenes | by Olga Mindlina | Medium](https://medium.com/@olga.mindlina/denoising-diffusion-probabilistic-model-practical-experiments-for-landscape-generation-from-688ae87f3b73)
- [Representation Learning with Diffusion Models](https://arxiv.org/pdf/2210.11058)
- [Improved Denoising Diffusion Probabilistic Models | alphaXiv](https://www.alphaxiv.org/resources/2102.09672v1)
- [A Comprehensive Survey on Diffusion Models and Their Applications](https://arxiv.org/pdf/2408.10207)
- [Understanding Galaxy Morphology Evolution Through Cosmic Time via Redshift Conditioned Diffusion Models](https://arxiv.org/pdf/2411.18440)
- [Image Super-Resolution via Iterative Refinement](https://arxiv.org/pdf/2104.07636)
- [Improved Denoising Diffusion Probabilistic Models](https://www.researchgate.net/publication/349492049_Improved_Denoising_Diffusion_Probabilistic_Models)
- [DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models](https://arxiv.org/pdf/2211.01095)
- [Improved Denoising Diffusion Probabilistic Models - researchr publication](https://researchr.org/publication/NicholD21)
- [The Diffusion Duality, Chapter II: $\Psi$-Samplers](https://arxiv.org/pdf/2602.21185)
- [Improving Denoising Diffusion Models via Simultaneous Estimation of Image and Noise](https://arxiv.org/pdf/2310.17167)
- [Accelerating Video Diffusion Models via Distribution Matching](https://arxiv.org/pdf/2412.05899)
- [Evolutionary Caching to Accelerate Your Off-the-Shelf Diffusion Model](https://arxiv.org/pdf/2506.15682)
- [FREPix: Frequency-Heterogeneous Flow Matching for Pixel-Space Image Generation](https://arxiv.org/pdf/2605.06421)
- [Improved Denoising Diffusion Probabilistic Models · Sophon](https://sophon.at/papers/pwc-5063)
- [[PDF] Improved Denoising Diffusion Probabilistic Models | Semantic Scholar](https://www.semanticscholar.org/paper/Improved-Denoising-Diffusion-Probabilistic-Models-Nichol-Dhariwal/de18baa4964804cf471d85a5a090498242d2e79f)
- [Diffusion models beat GANs on image synthesis | Proceedings of the 35th International Conference on Neural Information Processing Systems](https://dl.acm.org/doi/10.5555/3540261.3540933)
- [Alex Nichol, Prafulla Dhariwal · Improved Denoising Diffusion Probabilistic Models](https://slideslive.com/38959180/improved-denoising-diffusion-probabilistic-models)

Powered by Claude Exporter (https://www.ai-chat-exporter.net)
