#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>

int main() {
	cudnnHandle_t handle;
	cudnnCreate(&handle);

	cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
	cudnnTensorFormat_t tensor_format = CUDNN_TENSOR_NCHW;
	int n = 1, c = 1, h = 1, w = 10;

	int NUM_ELEMENTS = n*c*h*w;
	cudnnTensorDescriptor_t x_desc;
	cudnnCreateTensorDescriptor(&x_desc);
	cudnnSetTensor4dDescriptor(x_desc, tensor_format, dtype, n, c, h, w);

	float *x;
	cudaMallocManaged(&x, NUM_ELEMENTS * sizeof(float));
	for (int i=0; i<NUM_ELEMENTS; ++i) {
		x[i] = i * 1.05f;
	}
	std::cout << "Original data: ";
	for (int i=0; i<NUM_ELEMENTS; ++i) {
		std::cout << x[i] << " ";
	}
	std::cout << '\n';

	float alpha[1] = {1.0f};
	float beta[1] = {0.0f};

	cudnnActivationDescriptor_t sigmoid_activation;
	cudnnActivationMode_t mode = CUDNN_ACTIVATION_SIGMOID;
	cudnnNanPropagation_t prop = CUDNN_NOT_PROPAGATE_NAN;
	cudnnCreateActivationDescriptor(&sigmoid_activation);
	cudnnSetActivationDescriptor(sigmoid_activation, mode, prop, 0.0f);

	cudnnActivationForward(
		handle,
		sigmoid_activation,
		alpha,
		x_desc,
		x,
		beta,
		x_desc,
		x
	);

	cudnnDestroy(handle);
	std::cout << "\nDestroyed cuDNN handle.\n";
	std::cout << "New array: ";
	for (int i=0; i<NUM_ELEMENTS; ++i) {
		std::cout << x[i] << " ";
	}
	std::cout << '\n';
	cudaFree(x);

	return 0;
}
