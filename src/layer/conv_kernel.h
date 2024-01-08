#ifndef CONV_KERNEL_H
#define CONV_KERNEL_H

void startTimer();
float stopTimer();

void unrollGPUWrapper(int C, int H, int W, int K, float* image, float* data_col);

#endif
