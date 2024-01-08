#ifndef CONV_KERNEL_H
#define CONV_KERNEL_H

void startTimer();
float stopTimer();

__global__ void im2col_kernel(float* restrict image, float* restrict data_col, int height_in, int width_in, int channel_in, int height_out, int width_out, int height_kernel, int width_kernel, int pad_h, int pad_w, int stride);
void matrix_multiply_gpu(const float* A, float* C, int m, int n, int k);
void copyWeightsToConstant(float* host_weights, size_t num_weights);

#endif
