#ifndef CONV_KERNEL_H
#define CONV_KERNEL_H

void startTimer();
float stopTimer();

void im2col_gpu(const float* image, float* data_col, int height_in, int width_in, int channel_in, int height_out, int width_out, int height_kernel, int width_kernel, int pad_h, int pad_w, int stride);

#endif
