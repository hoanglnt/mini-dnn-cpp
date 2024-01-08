#include <iostream>
#include "conv_kernel.h"
__constant__ float const_weights[2560];

#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
		fprintf(stderr, "code: %d, reason: %s\n", error,\
				cudaGetErrorString(error));\
		exit(EXIT_FAILURE);\
	}\
}

struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start, 0);
		cudaEventSynchronize(start);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	float Elapsed()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};

static GpuTimer timer;
void startTimer()
{
    timer.Start();
}

float stopTimer()
{
    timer.Stop();

	return timer.Elapsed();
}   

__global__ void im2col_kernel(float* image, float* data_col, int height_in, int width_in, int channel_in, int height_out, int width_out, int height_kernel, int width_kernel, int pad_h, int pad_w, int stride) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= height_out * width_out) return; // One thread per output element

    int h_out = index / width_out; // Determine which output row and column this thread should handle
    int w_out = index % width_out;

    for (int c = 0; c < channel_in; c++) { // For each channel
        for (int i = 0; i < height_kernel; i++) { // For each row in the kernel
            for (int j = 0; j < width_kernel; j++) { // For each column in the kernel

                int im_row = h_out * stride - pad_h + i; // Calculate corresponding input row
                int im_col = w_out * stride - pad_w + j; // Calculate corresponding input column

                float val = 0; // Default to zero for padding
                if (im_row >= 0 && im_row < height_in && im_col >= 0 && im_col < width_in) {
                    val = image[im_row * width_in + im_col + c * height_in * width_in]; // Adjust for channel
                }

                // Calculate index in data_col
                int data_col_idx = (c * height_kernel * width_kernel + i * width_kernel + j) * height_out * width_out + index;
                data_col[data_col_idx] = val;
            }
        }
    }
}


__global__ void matrix_multiplication_kernel(float* A, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        float value = 0;
        for (int h = 0; h < n; h++) {
            value += A[row * n + h] * const_weights[h * k + col];  // Using weights from constant memory
        }
        C[row * k + col] = value;
    }
}

void copyWeightsToConstant(float* host_weights, size_t num_weights) {
    cudaMemcpyToSymbol(const_weights, host_weights, sizeof(float) * num_weights);
}

void im2col_gpu(const float* image, float* data_col, int height_in, int width_in, int channel_in, int height_out, int width_out, int height_kernel, int width_kernel, int pad_h, int pad_w, int stride) {
    // Allocate memory on device
    float *d_image, *d_data_col;
    size_t image_size = sizeof(float) * height_in * width_in * channel_in;
    size_t data_col_size = sizeof(float) * height_out * width_out * height_kernel * width_kernel * channel_in;
    cudaMalloc(&d_image, image_size);
    cudaMalloc(&d_data_col, data_col_size);

    // Copy image to device
    cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice);

    // Calculate grid and block sizes
    int threads = 256; // This can be tuned for your specific GPU
    int blocks = (height_out * width_out + threads - 1) / threads;

    // Launch kernel    
    im2col_kernel<<<blocks, threads>>>(d_image, d_data_col, height_in, width_in, channel_in, height_out, width_out, height_kernel, width_kernel, pad_h, pad_w, stride);

    // Copy result back to host
    cudaMemcpy(data_col, d_data_col, data_col_size, cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_image);
    cudaFree(d_data_col);
}

void matrix_multiply_gpu(const float* A, float* C, int m, int n, int k) {
    // Allocate device memory for A and C only
    float *d_A, *d_C;
    cudaMalloc(&d_A, sizeof(float) * m * n);
    cudaMalloc(&d_C, sizeof(float) * m * k);

    // Copy host memory to device for A
    cudaMemcpy(d_A, A, sizeof(float) * m * n, cudaMemcpyHostToDevice);=

    // Kernel launch parameters
    dim3 threadsPerBlock(16, 16); // This can be tuned
    dim3 numBlocks((k + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the modified matrix multiplication kernel
    matrix_multiplication_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_C, m, n, k);

    // Copy result back to host
    cudaMemcpy(C, d_C, sizeof(float) * m * k, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_C);
}


