#include <iostream>
#include "conv_kernel.h"

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

__global__ void im2col_kernel(const float* image, float* data_col, int height_in, int width_in, int channel_in, int height_out, int width_out, int height_kernel, int width_kernel, int pad_h, int pad_w, int stride) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index
    if (index >= height_out * width_out * channel_in) return; // Check if within bounds

    int w_out = index % width_out;
    int h_out = (index / width_out) % height_out;
    int c_in = index / (width_out * height_out);

    int h_offset = h_out * stride - pad_h;
    int w_offset = w_out * stride - pad_w;

    for (int i = 0; i < height_kernel; ++i) {
        for (int j = 0; j < width_kernel; ++j) {
            int im_row = h_offset + i;
            int im_col = w_offset + j;
            int col_index = (i * width_kernel + j) * channel_in + c_in;
            if (im_row >= 0 && im_row < height_in && im_col >= 0 && im_col < width_in) {
                data_col[index + col_index * height_out * width_out] = image[im_row * width_in + im_col + c_in * height_in * width_in];
            } else {
                data_col[index + col_index * height_out * width_out] = 0;
            }
        }
    }
}

void im2col_gpu(const Matrix& bottom, Matrix& data_col) {
    // Extract dimensions and parameters from the bottom and data_col matrices
    int height_in = ...; // Set the actual value
    int width_in = ...; // Set the actual value
    int channel_in = ...; // Set the actual value
    int height_out = ...; // Calculate or set the actual value
    int width_out = ...; // Calculate or set the actual value
    int height_kernel = ...; // Set the actual value
    int width_kernel = ...; // Set the actual value
    int pad_h = ...; // Set the actual value
    int pad_w = ...; // Set the actual value
    int stride = ...; // Set the actual value

    // Allocate memory on the GPU
    float* d_image;
    float* d_data_col;
    cudaMalloc(&d_image, sizeof(float) * bottom.size()); // bottom.size() should be total elements in bottom
    cudaMalloc(&d_data_col, sizeof(float) * data_col.size()); // data_col.size() should be total elements in data_col

    // Copy data from CPU to GPU
    cudaMemcpy(d_image, bottom.data(), sizeof(float) * bottom.size(), cudaMemcpyHostToDevice);

    // Calculate grid and block sizes
    int numThreads = 256; // This can be tuned
    int numBlocks = (height_out * width_out * channel_in + numThreads - 1) / numThreads;

    // Launch kernel
    im2col_kernel<<<numBlocks, numThreads>>>(d_image, d_data_col, height_in, width_in, channel_in, height_out, width_out, height_kernel, width_kernel, pad_h, pad_w, stride);

    // Copy result back to CPU
    cudaMemcpy(data_col.data(), d_data_col, sizeof(float) * data_col.size(), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_image);
    cudaFree(d_data_col);
}