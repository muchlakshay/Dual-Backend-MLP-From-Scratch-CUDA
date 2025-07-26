#pragma once

#include <iostream>
#include <vector>
#include "matrix.cuh"
#include "utility.cuh"
#include "layer_types.cuh"
#include "typedefs.cuh"
#include <cuda_runtime.h>
#include "KERNELS.cuh"

//Namespace For Network's Data
namespace nnData {
	Matrix X_train;
	Matrix Y_train;
	Matrix shuffled_X_train;
	Matrix shuffled_Y_train;
	Matrix temp_for_shuffle_1;
	Matrix temp_for_shuffle_2;
	std::vector<DenseLayerGPU> layersGPU;
	float* loss;
}

//Allocate Managed Memory For Loss
void allocateLossMem() {
	cudaMallocManaged(&nnData::loss, sizeof(float));
	memset(nnData::loss, 0.0f, sizeof(float)); //Set Loss To 0
}

//Loads The Training Data Into GPU
void loadDataIntoGPU(const EigenMatrix& X_train, const EigenMatrix& Y_train, bool enableShuffling) {

	//Dims Of Data
	int X_train_rows = X_train.rows();
	int X_train_cols = X_train.cols();
	int Y_train_rows = Y_train.rows();
	int Y_train_cols = Y_train.cols();

	//Flattened Data Stored In An std::vector
	auto X_train_flatten{ utility::flatten(X_train) };
	auto Y_train_flatten{ utility::flatten(Y_train) };

	//Assign The X_train And Y_train
	nnData::X_train = Matrix{ X_train_flatten.data(), X_train_rows, X_train_cols };
	nnData::Y_train = Matrix{ Y_train_flatten.data(), Y_train_rows, Y_train_cols };

	//Pre Allocation Of Temp Matricies For Shuffling
	if (enableShuffling) {
		nnData::shuffled_X_train.reserve(X_train_rows, X_train_cols);
		nnData::shuffled_Y_train.reserve(Y_train_rows, Y_train_cols);
	}
}

//Load All The Host Side Layers Into The GPU And Pre-Allocate Memory For Them
void loadLayersDataToGPU(std::vector<DenseLayer> host_layers, int batch_size) {

	//Iterate Through Host Layers
	for (const auto& layer : host_layers) {
		DenseLayerGPU layer_gpu;

		auto host_weights_flattened{ utility::flatten(layer.weights) }; //std::vector containing flattened weights
		auto host_biases_flattened{ utility::flatten(layer.biases) };   //std::vector containing biases

		//set layer Neurons And Input Size
		layer_gpu.neurons = layer.neurons;		
		layer_gpu.in = layer.in;

		//Pre-Allocate Memory For All The Data Members Of DenseLayerGPU Struct
		layer_gpu.weights = Matrix{ host_weights_flattened.data(), layer.neurons, layer.in };
		layer_gpu.weights_transposed.reserve(layer.in, layer.neurons);
		layer_gpu.biases.reserve(1, layer.neurons);
		layer_gpu.biases.setConstant(0.0f);
		layer_gpu.preActivations.reserve(batch_size, layer.neurons);
		layer_gpu.activation_primes.reserve(batch_size, layer.neurons);
		layer_gpu.activations.reserve(batch_size, layer.neurons);
		layer_gpu.weights_gradients.reserve(layer.neurons, layer.in);
		layer_gpu.biases_gradients.reserve(1, layer.neurons);
		layer_gpu.activationFunction = layer.activationFunction;
		layer_gpu.errors.reserve(batch_size, layer.neurons);
		layer_gpu.errors_transposed.reserve(layer.neurons, batch_size);
		layer_gpu.scaled_biases_gradients.reserve(1, layer.neurons);
		layer_gpu.scaled_weights_gradients.reserve(layer.neurons, layer.in);
		layer_gpu.prev_weights_grad.reserve(layer.neurons, layer.in);
		layer_gpu.prev_weights_grad.setConstant(0.0f);
		layer_gpu.prev_biases_grad.reserve(1, layer.neurons);
		layer_gpu.prev_biases_grad.setConstant(0.0f);
		layer_gpu.scaled_prev_weights_grad.reserve(layer.neurons, layer.in);
		layer_gpu.scaled_prev_biases_grad.reserve(1, layer.neurons);

		//Push Back The Layer Struct Into Vector Containing All The Layer Structs
		nnData::layersGPU.push_back(std::move(layer_gpu));
	}

}

//Shuffle Data On GPU
void shuffleDataGPU() {
	int* indices{ make_device_indices(nnData::X_train.getRows()) };		//Generate Shuffled Device Indices 
	nnData::X_train.shuffle(nnData::shuffled_X_train, indices, false);	//Shuffle X_train Using Those Indicies
	nnData::Y_train.shuffle(nnData::shuffled_Y_train, indices);			//Shuffle Y_train using Those Indicies

	nnData::temp_for_shuffle_1 = std::move(nnData::X_train);			//Move X_train Into A Temp
	nnData::temp_for_shuffle_2 = std::move(nnData::Y_train);			//Move Y_train Into A Temp

	nnData::X_train = std::move(nnData::shuffled_X_train);				//Move Shuffled Data Into X_train
	nnData::Y_train = std::move(nnData::shuffled_Y_train);				//Move Shuffled Data Into Y_train

	nnData::shuffled_X_train = std::move(nnData::temp_for_shuffle_1);	//Move Temp To Shuffled_X_train To Store Next Shuffled Data
	nnData::shuffled_Y_train = std::move(nnData::temp_for_shuffle_2);	//Move Temp To Shuffled_Y_train To Store Next Shuffled Data

}

//Feed Forward Thorugh The Network
void feedForwardGPU(Matrix batch, int batch_size) {
	
	for (int i{}; i < nnData::layersGPU.size(); ++i) {
		auto& layer{ nnData::layersGPU[i] };

		//Set Matricies Dims According To Batch Size -> Last Batch's Size May Be Diffrent From Specified Batch Size
		//This Will Cause Problems
		int layer_cols_PA{ layer.preActivations.getCols() };
		layer.preActivations.setDims(batch_size, layer_cols_PA);
		layer.activations.setDims(batch_size, layer_cols_PA);
		layer.errors.setDims(batch_size, layer_cols_PA);
		layer.errors_transposed.setDims(layer_cols_PA, batch_size);
		layer.activation_primes.setDims(batch_size, layer_cols_PA);

		//Launch Kernel To Calculate Weighted Sum
		dim3 BLOCK_SIZE(16, 16);
		dim3 GRID_SIZE(
			(layer.neurons + BLOCK_SIZE.x - 1) / BLOCK_SIZE.x,
			(batch_size + BLOCK_SIZE.y - 1) / BLOCK_SIZE.y
		);

		//Transpose Weights
		layer.weights.transpose(layer.weights_transposed);

		Kernels::computeWeightedSum << <GRID_SIZE, BLOCK_SIZE >> > (
			i == 0 ? batch.data() : nnData::layersGPU[i - 1].activations.data(),
			layer.weights_transposed.data(),
			layer.preActivations.data(),
			layer.biases.data(),
			batch_size,
			layer.neurons,
			layer.in
			);

		cudaDeviceSynchronize();
		CUDA_CHECK(cudaGetLastError());

		//Apply Activations
		if (layer.activationFunction == "sigmoid") layer.preActivations.sigmoid(layer.activations);
		else if (layer.activationFunction == "relu") layer.preActivations.relu(layer.activations);
		else if (layer.activationFunction == "leaky_relu") layer.preActivations.leaky_relu(layer.activations);
		else if (layer.activationFunction == "tanh") layer.preActivations.tanh(layer.activations);
		else if (layer.activationFunction == "softmax") layer.preActivations.softmax(layer.activations);
		else if (layer.activationFunction == "linear") layer.activations.clone(layer.preActivations);

	}

}

//Backpropagate Through The Network And Calculate Gradients
void backprogatateGPU(Matrix X_train, Matrix Y_train, int batch_size, const std::string& loss_function) {

	//Output Layer Error Calculations
	DenseLayerGPU& layer{ nnData::layersGPU.back() };

	//Directly Calculate Errors By Doing (Aactivations-Y_train) If Activation and Loss Functions are Softmax/CE or Sigmoid/BCE
	if (layer.activationFunction == "softmax" && loss_function == "cross_entropy") layer.activations.subtractInto(Y_train, layer.errors);
	else if (layer.activationFunction == "sigmoid" && loss_function == "binary_cross_entropy") layer.activations.subtractInto(Y_train, layer.errors);
	//Do [ dC/da ⊙ da/dz] else 
	else {
		compute_dC_da_GPU(layer.activations, Y_train, layer.errors, loss_function);
		layer.activations.activationPrimes(layer.activation_primes, layer.activationFunction);
		layer.errors.elementWiseProduct(layer.activation_primes, layer.errors);
	}

	//Hidden Layer Error Calculations

	if (nnData::layersGPU.size() >= 2) {	//In Case There Is Only One Layer
		for (int i = nnData::layersGPU.size() - 2; i >= 0; --i) {
			DenseLayerGPU& current_layer{ nnData::layersGPU[i] };
			DenseLayerGPU& next_layer{ nnData::layersGPU[i + 1] };

			//Perform [ err(l+1) * weights(l+1) ⊙ da(l)/dz(l) ]
			next_layer.errors.MatMulInto(next_layer.weights, current_layer.errors);
			current_layer.preActivations.activationPrimes(current_layer.activation_primes, current_layer.activationFunction);
			current_layer.errors.elementWiseProduct(current_layer.activation_primes, current_layer.errors);
		}
	}

	//Gradients Calculations

	for (int i{}; i < nnData::layersGPU.size(); ++i) {
		DenseLayerGPU& layer{ nnData::layersGPU[i] };
		layer.errors.transpose(layer.errors_transposed);

		//Weigths Gradients Calculation
		layer.errors_transposed.MatMulInto(i == 0 ? X_train : nnData::layersGPU[i - 1].activations, layer.weights_gradients);
		layer.weights_gradients.scalarOperation(batch_size, '/', layer.weights_gradients); //Average Over Batch
		//Biases Gradients Calculation
		layer.errors.rowWiseSum(layer.biases_gradients);
		layer.biases_gradients.scalarOperation(batch_size, '/', layer.biases_gradients);   //Average Over Batch
	}
}

//Update The Parameters
void updateWeightsGPU(float learning_rate, float momentum_coef) {
	for (auto& layer : nnData::layersGPU) {   // Iterate Through Each Layer
		//Calcuate Velocity By Doing [V = M_coef * prev_V + grads] [Acts Like Momentum If Momentum Coefficient != 0, SGD Otherwise]

		layer.prev_weights_grad.scalarOperation(momentum_coef, '*', layer.scaled_prev_weights_grad);
		layer.scaled_prev_weights_grad.addInto(layer.weights_gradients, layer.prev_weights_grad);
		layer.prev_weights_grad.scalarOperation(learning_rate, '*', layer.scaled_weights_gradients);

		layer.prev_biases_grad.scalarOperation(momentum_coef, '*', layer.scaled_prev_biases_grad);
		layer.scaled_prev_biases_grad.addInto(layer.biases_gradients, layer.prev_biases_grad);
		layer.prev_biases_grad.scalarOperation(learning_rate, '*', layer.scaled_biases_gradients);

		//Weights Update [W = W - V]
		layer.weights.subtractInto(layer.scaled_weights_gradients, layer.weights);
		//Biases Update  [b = b - v]
		layer.biases.subtractInto(layer.scaled_biases_gradients, layer.biases);
	}
}
 //Load The Trained Parameters To Host
void loadParametersToHost(std::vector<DenseLayer>& host_layers) {

	// Ensure both host and GPU layer vectors are the same size
	assert(host_layers.size() == nnData::layersGPU.size());

	for (int i{}; i < host_layers.size(); ++i) {
		DenseLayer& layer_cpu{ host_layers[i] };
		DenseLayerGPU& layer_gpu{ nnData::layersGPU[i] };

		// Allocate host-side vectors to store weights and biases copied from GPU
		std::vector<ElementType> weights_host_vec(layer_cpu.in * layer_cpu.neurons);
		std::vector<ElementType> biases_host_vec(layer_cpu.neurons);

		// Copy weights from GPU to host memory
		std::size_t weights_size{ sizeof(ElementType) * weights_host_vec.size() };
		auto cpy1 = cudaMemcpy(weights_host_vec.data(), layer_gpu.weights.data(), weights_size, cudaMemcpyDeviceToHost);
		CUDA_CHECK(cpy1);

		// Copy biases from GPU to host memory
		std::size_t biases_size{ sizeof(ElementType) * biases_host_vec.size() };
		auto cpy2 = cudaMemcpy(biases_host_vec.data(), layer_gpu.biases.data(), biases_size, cudaMemcpyDeviceToHost);
		CUDA_CHECK(cpy2);

		// Transfer the flattened GPU weights into the structured Eigen matrix (row-major order)
		for (int row{}; row < layer_cpu.neurons; ++row) {
			for (int col{}; col < layer_cpu.in; ++col) {
				layer_cpu.weights(row, col) = weights_host_vec[row * layer_cpu.in + col];
			}
			// Copy bias for each neuron
			layer_cpu.biases(row) = biases_host_vec[row];
		}
	}
}

// Calculates loss on GPU based on the specified loss function.
// Supports Cross Entropy, Binary Cross Entropy, and Mean Squared Error (MSE)
void calculateLossGPU(Matrix activations, Matrix Y_train, const std::string& loss_function) {

	// Dispatch the appropriate GPU loss kernel based on the selected loss function
	if (loss_function == "cross_entropy") cross_entropy_GPU(activations, Y_train, nnData::loss);
	else if (loss_function == "binary_cross_entropy") binary_cross_entropy_GPU(activations, Y_train, nnData::loss);
	else if (loss_function == "MSE") MSE_GPU(activations, Y_train, nnData::loss);

	// Average the accumulated loss over the batch size (number of rows)
	*nnData::loss /= activations.getRows();
}
