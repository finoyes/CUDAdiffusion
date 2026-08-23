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