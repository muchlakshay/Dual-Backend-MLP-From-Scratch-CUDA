#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

//--------------- FORWARD DECLERATION FOR ALL THE KERNELS ---------------

namespace Kernels {
	__global__ void print_data(float* data, int rows, int cols);
	__global__ void scalarOperations(float* mat_a, float* mat_b, float scalar, int row, int cols, int operationID);
	__global__ void matrixOperations(float* matrix_a, float* matrix_b, float* matrix_c, int rows, int cols, int operationID);
	__global__ void transpose(float* matrix, float* matrix_t, int rows, int cols);
	__global__ void shuffle(float* original, float* shuffled, int* indices, int rows, int cols);
	__global__ void constant(float* mat, float constant, int rows, int cols);
	__global__ void rowBroadcast(float* vec, float* mat, float* store_in, int rows, int cols);
	__global__ void colBroadcast(float* vec, float* mat, float* store_in, int rows, int cols);
	__global__ void matmul(const float* A, const float* B, float* C, int M, int N, int K);
	__global__ void relu(float* preActivations, float* activations, int rows, int cols);
	__global__ void sigmoid(float* preActivations, float* activations, int rows, int cols);
	__global__ void leaky_relu(float* preActivations, float* activations, int rows, int cols);
	__global__ void tanh(float* preActivations, float* activations, int rows, int cols);
	__global__ void calculate_exps(float* activations, float* store_in, int rows, int cols);
	__global__ void calculate_max_and_sum_exp(float* preActivations, float* store_exp_sum, float* store_max, int rows, int cols);
	__global__ void softmax(float* preActivations, float* activations, float* exp_sum, float* max, int rows, int cols);
	__global__ void elementWiseProduct(float* matrix_A, float* matrix_B, float* store_in, int rows, int cols);
	__global__ void activationPrime(float* preActivation, float* store_in, int activation_id, int rows, int cols);
	__global__ void rowWiseSum(float* matrix, float* store_in, int rows, int cols);

	__global__ void computeWeightedSum(float* input, float* weights_transposed,
		float* preActivations, float* biases, int M, int N, int K);
	__global__ void compute_dC_da(float* activations, float* Y_train, float* errors,
		int lossFn_id, int rows, int cols);

	__global__ void cross_entropy_loss(float* activations, float* Y_train, float* loss, int rows, int cols);
	__global__ void binary_cross_entropy(float* activations, float* Y_train, float* loss, int rows, int cols);
	__global__ void MSE(float* activations, float* Y_train, float* loss, int rows, int cols);
}
