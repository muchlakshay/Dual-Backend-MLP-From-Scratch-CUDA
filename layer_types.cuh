#pragma once

#include "matrix.cuh"
#include <Eigen/Dense>
#include "typedefs.cuh"

//--------------- LAYER TYPES---------------

//layer struct for CPU backend
struct DenseLayer {
	EigenMatrix weights;
	EigenVector biases;
	EigenMatrix preActivations;
	EigenMatrix activations;
	EigenMatrix weightsGradients;
	EigenMatrix biasesGradients;
	EigenMatrix errors;
	EigenMatrix prev_weights_grad;
	EigenVector prev_biases_grad;
	std::string activationFunction;
	int neurons{};
	int in;
};

//layer struct for GPU backend
struct DenseLayerGPU {
	Matrix weights;
	Matrix weights_transposed;
	Matrix biases;
	Matrix preActivations;
	Matrix activations;
	Matrix weights_gradients;
	Matrix biases_gradients;
	Matrix errors;
	Matrix errors_transposed;
	Matrix activation_primes;
	Matrix scaled_weights_gradients;
	Matrix scaled_biases_gradients;
	Matrix scaled_prev_weights_grad;
	Matrix scaled_prev_biases_grad;
	Matrix prev_weights_grad;
	Matrix prev_biases_grad;

	std::string activationFunction;
	int neurons;
	int in;
};