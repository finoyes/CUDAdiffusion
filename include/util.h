#ifdef UTILS_H
#define UTILS_H

#include <cstddef>

//both functions of the image I/O works in "CHW, float, [-1,1]" layout : index(c,y,x) = c*IMG_SIZE^2 + y*IMG_SIZE + x (every kernel in network and diffusion assumes)
//utils connects the stb_image and stb_write_image to the memory allocator???

//loads image file of stb support image extension 
//forces to 3 channels, nearest-neighbor resizes to IMG_SIZE^2
//and returns a freshly malloc's host float array of length IMG_ELEMS holding]
//CHW data normalizes to [-1,1]. caller owns the returned pointer (freee()).
//Returns nullptr on failure
float* load_img_chw(const char* path);

//writes a CHW float array (values expected in roughly [-1,1], clamped
//defensly) of lenth IMG_ELEMS out to 'path' as an 8-bit PNG.
bool save_img_chw(const char* path, const float* chw);

//HOST to device transfer helpers, this is connected to the image I/O operation
//this is this explicit wrapers around cudaMalloc/cudamemcpy/cudaFree so evrery allocation site in the system goes thruogh one error check. (CUDA_CHECK in commom.h)


//Allocates 'n_floats' floats of device global memory.
float* device_alloc(size_t n_floats);

//Allocates 'n_float' floats of device memory and zero-initilizes the
//(used for gradients and Adam moment buffers, which must start at 0)
float* device_alloc_zero(size_t n_floats);

//copies 'n_floats' floats from host array to the existing device array
void copy_device(float* d_dst, const float* h_src, size_t n_floats);

//copies 'n_floats' floats from an existing device arrat into a host array
void copy_host(float* h_dst, const float* d_src, size_t n_floats);

//alloc device buffer and immediately upload 'h_src' into it.
float* device_alloc_from_host(const float* h_src, size_t n_floats);

#endif
