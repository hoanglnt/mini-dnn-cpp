#ifndef CONV_KERNEL_H
#define CONV_KERNEL_H

void startTimer();
float stopTimer();

void copyWeightsToConstant(float* host_weights, size_t num_weights);

void im2col_gpu1(const float* image, float* data_col, int height_in, int width_in, int channel_in, int height_out, int width_out, int height_kernel, int width_kernel, int pad_h, int pad_w, int stride);
void im2col_gpu2(const float* image, float* data_col, int height_in, int width_in, int channel_in, int height_out, int width_out, int height_kernel, int width_kernel, int pad_h, int pad_w, int stride);
void matrix_multiply_gpu1(const float* A, const float* B, float* C, int m, int n, int k);
void matrix_multiply_gpu2(const float* A, float* C, int m, int n, int k);

#endif
