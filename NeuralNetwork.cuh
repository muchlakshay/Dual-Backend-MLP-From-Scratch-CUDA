#pragma once

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <random>
#include "matrix.cuh"
#include "utility.cuh"
#include <cmath>
#include "gpuBackend.cuh"
#include "layer_types.cuh"
#include "typedefs.cuh"
#include <fstream>
#include <iomanip>
#include <utility>

//------------------- ACTIVATION STUFF -------------------

namespace ActivationsLambdas {

	//Activation Function Lambdas
	auto sigmoid{ [](ElementType z) -> ElementType {return 1.0 / (1.0 + std::expf(-z)); } };
	auto relu{ [](ElementType z) -> ElementType {return std::fmaxf(0.0, z); } };
	auto leakyRelu{ [](ElementType z) -> ElementType {return z <= 0.0 ? 0.1 * z : z; } };
	auto tanh{ [](ElementType z) -> ElementType {return std::tanhf(z); } };

	//Activation Function Derivatives Lambdas
	auto sigmoid_prime{ [](ElementType z) -> ElementType { return sigmoid(z) * (1.0 - sigmoid(z)); } };
	auto relu_prime{ [](ElementType z) -> ElementType {return z > 0.0 ? 1.0 : 0.0; } };
	auto leaky_relu_prime{ [](ElementType z) -> ElementType {return z > 0.0 ? 1.0 : 0.1; } };
	auto tanh_prime{ [](ElementType z) -> ElementType {return 1.0 - std::tanhf(z) * std::tanhf(z); } };

	//Function To Compute da/dz During Backpropagation
	static EigenMatrix compute_da_dz(const std::string& activation_fn,
		const EigenMatrix& preActivations) {
		if (activation_fn == "sigmoid") return preActivations.unaryExpr(sigmoid_prime);
		else if (activation_fn == "relu") return preActivations.unaryExpr(relu_prime);
		else if (activation_fn == "leaky_relu") return preActivations.unaryExpr(leaky_relu_prime);
		else if (activation_fn == "tanh") return preActivations.unaryExpr(tanh_prime);
		else if (activation_fn == "linear") return EigenMatrix::Constant(preActivations.rows(), preActivations.cols(), 1.0f);
	}

};

//------------------- NN CLASS FORWARD DECLERATION -------------------

class NeuralNetwork {
public:

	//Wight Initializer Types
	enum class Initializer {
		He_Normal,
		He_Uniform,
		Xavier_Normal,
		Xavier_Uniform,
	};

	//Training Devices
	enum class TrainingDevice {
		CPU, GPU
	};

private:
	//Data Members
	std::vector<DenseLayer> layers;
	ElementType learning_rate{};
	std::string loss_function{};
	int input_size{};
	int batch_size{};
	bool verbose;
	ElementType loss{};
	ElementType momentum_coef{};

	//Private Member Functions
	void initialize(int fan_in, int fan_out, const Initializer& initializer, DenseLayer& layer);
	void feedForwardCPU(const EigenMatrix& batch);
	void backpropagateCPU(const EigenMatrix& X_train, const EigenMatrix& Y_train);
	void updateWeightsCPU();
	EigenMatrix compute_dC_da(EigenMatrix& activations, const EigenMatrix& Y_train, const std::string& loss_function);
	float binaryCrossEntropyLoss(const EigenMatrix& activations, const EigenMatrix& Y_train);
	void calculateLoss(const EigenMatrix& output, const EigenMatrix Y_train, int batch_size);
	float cross_entropy_loss(const EigenMatrix& Y_train, const EigenMatrix& activations);
	void softmax(const EigenMatrix& preActivation, EigenMatrix& activations);
	std::pair<int, std::string> check_cuda_support();
public:

	//Public Member Functions
	void learn(EigenMatrix& X_train, EigenMatrix& Y_train, int epochs, const TrainingDevice& device, bool enableShuffling);
	void input(int size);
	void extend(int neurons, const std::string& activation_function,
		const Initializer& initializer);
	void assemble(const std::string& Loss_function,
		ElementType Learning_rate, int Batch_size, ElementType Momentum_coef = 0.0f, bool Verbose = true);
	void info() const;
	void exportModel(const std::string& filename);
	void importModel(const std::string& filename);
	EigenMatrix predict(const EigenMatrix& to_predict);

	//Getter Functions
	const auto getWeights() const;
	const auto getBiases() const;
	ElementType getLearningRate() const { return learning_rate; }
	int getBatchSize() const { return batch_size; }
	int getInputSize() const { return input_size; }
};


//-------------------MEMBER FUNCTIONS DEFINITIONS-------------------

//initialize() Function Initializes The Weights Using Passed Weight Initialization Type
void NeuralNetwork::initialize(int fan_in, int fan_out, const Initializer& initializer, DenseLayer& layer) {

	//PRNG Instantiation
	std::random_device device;
	std::mt19937 mt{ device() };

	//Lambda To Generate Random Parameters
	auto init{
		[&](auto& dist, auto& weights) {
			for (int row{}; row < weights.rows(); ++row) {
				for (int col{}; col < weights.cols(); ++col) weights(row, col) = dist(mt);
			}
		}
	};

	//Temp Matrix To Store Generated Weights
	EigenMatrix weights(fan_out, fan_in);

	//Xavier Normal
	if (initializer == Initializer::Xavier_Normal) {
		std::normal_distribution<ElementType> distribution(0, std::sqrt(1.0 / fan_in));
		init(distribution, weights);
	}

	//Xavier Uniform
	else if (initializer == Initializer::Xavier_Uniform) {
		float limit = std::sqrt(6.0 / (fan_in + fan_out));
		std::uniform_real_distribution<ElementType> distribution(-limit, limit);
		init(distribution, weights);
	}

	//He Normal
	else if (initializer == Initializer::He_Normal) {
		std::normal_distribution<ElementType> distribution(0, std::sqrt(2.0 / fan_in));
		init(distribution, weights);
	}

	//He Uniform
	else if (initializer == Initializer::He_Uniform) {
		float limit = std::sqrt(6.0 / fan_in);
		std::uniform_real_distribution<ElementType> distribution(-limit, limit);
		init(distribution, weights);
	}
	else {
		throw std::invalid_argument("Unknown initializer type.");
	}

	//Assigning Data Members Of Layer
	layer.weights = weights;
	layer.weightsGradients = EigenMatrix(weights.rows(), weights.cols());
	layer.biases = EigenVector::Zero(fan_out);
	layer.biasesGradients = EigenVector(fan_out);
}

//feedForwardCPU() Function Will Do The Forward Pass Of Data In The Network
void NeuralNetwork::feedForwardCPU(const EigenMatrix& batch) {

	//Lambda For Applying Activations On Pre Activation Matrix
	auto computeActivation{
	[&](const auto& preActivations, auto& activations, const auto& activationFn) {
		if (activationFn == "sigmoid")
			activations = preActivations.unaryExpr(ActivationsLambdas::sigmoid);
		else if (activationFn == "relu")
			activations = preActivations.unaryExpr(ActivationsLambdas::relu);
		else if (activationFn == "leaky_relu")
			activations = preActivations.unaryExpr(ActivationsLambdas::leakyRelu);
		else if (activationFn == "softmax")
			softmax(preActivations, activations);
		else if (activationFn == "tanh")
			activations = preActivations.unaryExpr(ActivationsLambdas::tanh);
		else if (activationFn == "linear")
			activations = preActivations;
		}
	};

	//FeedForward Loop - Pass The Data From Each Layer
	for (int i{}; i < layers.size(); ++i) {
		//Layer Reference
		auto& layer{ layers[i] };

		//Matrix Multiplication of Previous Layer Activation or Batch and Current Layer Weights
		layer.preActivations = (i == 0 ? batch : layers[i - 1].activations) * layer.weights.transpose();

		//Add Biases
		layer.preActivations.rowwise() += layer.biases.transpose();

		//Apply Activations
		computeActivation(layer.preActivations, layer.activations, layer.activationFunction);
	}
}

//backpropateCPU() Will Backpropagate Through The Network And Calculate The Gradietns
void NeuralNetwork::backpropagateCPU(const EigenMatrix& X_train, const EigenMatrix& Y_train) {

	//Output Layer Error Calculation
	DenseLayer& opt_layer{ layers.back() };

	//Directly Calculate Errors By Doing (Aactivations-Y_train) If Activation and Loss Functions are Softmax/CE or Sigmoid/BCE
	if (opt_layer.activationFunction == "softmax" && loss_function == "cross_entropy") opt_layer.errors = opt_layer.activations - Y_train;
	else if (opt_layer.activationFunction == "sigmoid" && loss_function == "binary_cross_entropy") opt_layer.errors = opt_layer.activations - Y_train;
	//Do [ dC/da ⊙ da/dz] else 
	else {
		std::string activationFn{ opt_layer.activationFunction };
		compute_dC_da(opt_layer.activations, Y_train, loss_function).cwiseProduct(
			ActivationsLambdas::compute_da_dz(activationFn, opt_layer.preActivations));
	}

	//Hidden Layer Error Calculation
	if (layers.size() >= 2) {		//In Case There Is Only One Layer
		for (int i = layers.size() - 2; i >= 0; --i) {
			DenseLayer& current_layer{ layers[i] };
			DenseLayer& next_layer{ layers[i + 1] };

			//Perform [ err(l+1) * weights(l+1) ⊙ da(l)/dz(l) ]
			current_layer.errors = (next_layer.errors * next_layer.weights).cwiseProduct(
				ActivationsLambdas::compute_da_dz(current_layer.activationFunction, current_layer.preActivations));
		}
	}

	//Weights and Biases Gradients Calculation
	for (int i{}; i < layers.size(); ++i) {
		DenseLayer& current_layer{ layers[i] };

		current_layer.weightsGradients = current_layer.errors.transpose() * (i == 0 ? X_train : layers[i - 1].activations);
		current_layer.weightsGradients /= Y_train.rows(); //Average Over Batch
		current_layer.biasesGradients = current_layer.errors.colwise().mean();

	}
}

//updateWeightsCPU() Will Update The Network Parameters
void NeuralNetwork::updateWeightsCPU() {
	for (auto& layer : layers) { // Iterate Through Each Layer
		//Calcuate Velocity By Doing [V = M_coef * prev_V + grads] [Acts Like Momentum If Momentum Coefficient != 0, SGD Otherwise]
		layer.prev_weights_grad = momentum_coef * layer.prev_weights_grad + layer.weightsGradients; 
		layer.prev_biases_grad = momentum_coef * layer.prev_biases_grad + layer.biasesGradients;

		//Update Weights [W = W - V]
		layer.weights -= learning_rate * layer.prev_weights_grad;
		layer.biases -= learning_rate * layer.prev_biases_grad;
	}
}

//compute_Dc_da() Will Compute [Dc/Da] During Backpropagation
EigenMatrix NeuralNetwork::compute_dC_da(EigenMatrix& activations, const EigenMatrix& Y_train, const std::string& loss_function) {
	if (loss_function == "MSE") return activations - Y_train; //MSE Derivative
	else if (loss_function == "cross_entropy_loss") {
		float eps{ 1e-7f };//To Prevent The Denominator From Becoming 0
		return -((Y_train.array()) / (activations.array() + eps)); //CE Derivative
	}
	else if (loss_function == "binary_cross_entropy") {
		float eps{ 1e-7f }; //To Prevent The Denominator From Becoming 0
		return (activations - Y_train).array() /
			((activations.array() + eps) * ((1.0f - activations.array()) + eps)); //BCE Derivative
	}
	else throw "Invalid Loss Function";
}

//Separate Function For Binary Cross Entropy Loss
float NeuralNetwork::binaryCrossEntropyLoss(const EigenMatrix& activations, const EigenMatrix& Y_train) {
	assert(activations.rows() == Y_train.rows() && activations.cols() == Y_train.cols());

	const float eps{ 1e-7f }; //To Prevent The Activation From Becoming 0
	float totalLoss{ 0.0f };

	for (int i = 0; i < activations.rows(); ++i) { //Iteate Thorught Every Row
		for (int j = 0; j < activations.cols(); ++j) { 
			float a = std::max(eps, std::min(activations(i, j), 1.0f - eps)); //Clamp The Activation If Its 0
			float y = Y_train(i, j);
			//Calculate BCE [ -(y*log(a)) + (1-y) * log(1-a) ] And Add It To totalLoss
			totalLoss += -(y * std::log(a) + (1.0f - y) * std::log(1.0f - a));
		}
	}

	return totalLoss / activations.rows(); //Average Over Batch
}
//Calculate Loss After ForwardPass
void NeuralNetwork::calculateLoss(const EigenMatrix& output, const EigenMatrix Y_train, int batch_size) {
	if (loss_function == "MSE") loss += (Y_train - output).array().square().sum() / (2.0 * batch_size);
	else if (loss_function == "cross_entropy") loss += cross_entropy_loss(Y_train, output);
	else if (loss_function == "binary_cross_entropy") loss += binaryCrossEntropyLoss(output, Y_train);
}

//Separate Function For Cross Entropy Loss
float NeuralNetwork::cross_entropy_loss(const EigenMatrix& Y_train, const EigenMatrix& activations) {
	const float epsilon{ 1e-15f }; //To Prevent The Activation From Becoming 0
	EigenMatrix activation_clamped = activations.array().min(1.0 - epsilon).max(epsilon); ////Clamp The Activation If Its 0
	float loss{ 0.0f };
	for (int i = 0; i < Y_train.rows(); ++i) {
		//Calculate CE
		loss -= (Y_train.row(i).array() * activation_clamped.row(i).array().log()).sum();
	}
	return loss;
}

//Sepatate Function For Softmax Activation
void NeuralNetwork::softmax(const EigenMatrix& preActivation, EigenMatrix& activations) {
	activations.resize(preActivation.rows(), preActivation.cols());

	for (int i = 0; i < preActivation.rows(); ++i) {
		float max{ preActivation.row(i).maxCoeff() }; //Max Logit

		EigenVector exp_row = (preActivation.row(i).array() - max).exp(); //Logits Exps
		float sum{ exp_row.sum() }; //Logits Exps Sum

		if (sum == 0.0) sum = 1e-7f; //Prevent Sum From Becoming 0

		activations.row(i) = exp_row / sum; 
	}
}

//Check For CUDA Support On GPU [Returns Success Code(1) or Error Code (0) And Associated Additional Info String]
std::pair<int, std::string> NeuralNetwork::check_cuda_support() {
	int device_count{};
	auto error{ cudaGetDeviceCount(&device_count) };
	//If Some Error Occured
	if (error != cudaSuccess) {
		return { 0, "CUDA Not Supported Or Driver Issues. Switch To CPU Training" };
	}
	//If No CUDA Supported Device Found
	if (device_count == 0) return { 0, "No CUDA-Capable GPU Fund. Switch To CPU Training" };
	else return { 1, "CUDA Supported Device Found" }; // If CUDA Supported Device Found
}

//Function To Start Training
void NeuralNetwork::learn(EigenMatrix& X_train, EigenMatrix& Y_train, int epochs, const TrainingDevice& device, bool enableShuffling) {
	assert(X_train.cols() == input_size && Y_train.cols() == layers.back().neurons);

	//Dims Of Data
	int train_data_rows{ static_cast<int>(X_train.rows()) };
	int train_data_cols{ static_cast<int>(X_train.cols()) };
	int Y_train_cols{ static_cast<int>(Y_train.cols()) };

	//If Selected Training Device Is CPU
	if (device == TrainingDevice::CPU) {

		std::cout << "Learning...\n";

		for (int epoch{ 1 }; epoch <= epochs; ++epoch) { // Epoch Loop
			if (enableShuffling) utility::shuffleData(X_train, Y_train); //Shuffle Data If enableShuffling Is True

			for (int start{}; start < train_data_rows; start += batch_size) { //Loop To Train Over Batches

				//Last Batch Size May Not Be Equal To Specified Batch Size
				int current_batch_size{ std::min(train_data_rows - start, batch_size) };
				//Batches
				EigenMatrix X_train_block{ X_train.block(start, 0, current_batch_size, train_data_cols) };
				EigenMatrix Y_train_block{ Y_train.block(start, 0, current_batch_size, Y_train_cols) };

				feedForwardCPU(X_train_block); //Feed Forward
				backpropagateCPU(X_train_block, Y_train_block); //Backpropagate
				updateWeightsCPU(); //Update Weights
				if (verbose) calculateLoss(layers.back().activations, Y_train_block, current_batch_size); //Calculate Loss If Verbose
			}
			//Print Epoch Number And Loss If Verbose
			if (verbose) {
				std::cout << "Epoch: " << epoch << " | " << "Loss: " << loss << "\n";
				loss = 0.0;
			}
		}
	}

	//If Selected Training Device Is GPU
	else if (device == TrainingDevice::GPU) {

		//Check For CUDA Support
		std::pair<int, std::string> cuda_availability{ check_cuda_support() };
		if (cuda_availability.first == 0) {
			std::cout << cuda_availability.second;
			return;
		}

		loadDataIntoGPU(X_train, Y_train, enableShuffling); //Load X_train And Y_train Into GPU
		loadLayersDataToGPU(layers, batch_size); //Load Weigths and Biases To GPU
		if (verbose) allocateLossMem(); //Allocate Memory For Loss, If Verbose

		std::cout << "Learning...\n";

		for (int epoch{ 1 }; epoch <= epochs; ++epoch) { //Epoch Loop
			if (enableShuffling) shuffleDataGPU(); //Shuffle Data On GPU is enableShuffling is Set To True

			for (int start{}; start < train_data_rows; start += batch_size) { //Loop To Train Over Batches

				//Last Batch Size May Not Be Equal To Specified Batch Size
				int current_batch_size{ std::min(train_data_rows - start, batch_size) };

				feedForwardGPU(nnData::X_train.rowBlock(start, current_batch_size, true), current_batch_size); //Forward Pass

				backprogatateGPU(nnData::X_train.rowBlock(start, current_batch_size, true),
					nnData::Y_train.rowBlock(start, current_batch_size, true), current_batch_size, loss_function); //Backpropagate

				updateWeightsGPU(learning_rate, momentum_coef); //Update Parameters

				if (verbose) calculateLossGPU(nnData::layersGPU.back().activations,
					nnData::Y_train.rowBlock(start, current_batch_size, true), loss_function); //Calculate Loss If Verbose

			}
			//Print Epoch And Loss, If Verbose
			if (verbose) {
				std::cout << "Epoch: " << epoch << " | " << "Loss: " << *nnData::loss << "\n";
				memset(nnData::loss, 0.0f, sizeof(float)); //Set Loss Again To 0
			}
		}
		loadParametersToHost(layers); //Load Trained Parameters To Host For Perdictions/Inference
		if (verbose) cudaFree(nnData::loss); //Free The Memory Allocated For Loss
		nnData::layersGPU.clear(); //Destroy All The Layers Structs To Free Memory
	}
}

//Set Neurons In Input Layer
void NeuralNetwork::input(int size) {
	assert(input_size == 0 && size > 0);
	input_size = size;
}

//Addon Layers In The Network
void NeuralNetwork::extend(int neurons, const std::string& activation_function,
	const Initializer& initializer) {

	assert(input_size != 0 && neurons > 0);

	DenseLayer layer;
	layer.neurons = neurons; //Neruons In The Layer
	layer.activationFunction = activation_function; //Activation Function Used In Layer
	layer.in = layers.size() == 0 ? input_size : layers.back().neurons; //Previous Layer Neurons
	initialize(layer.in, neurons, initializer, layer);//Initialize The Weights
	layer.prev_biases_grad = EigenMatrix::Constant(layer.neurons, layer.in, 0.0); //Velocity Matrix For Weights
	layer.prev_biases_grad = EigenVector::Zero(layer.neurons); //Velocity Vector For The Biases

	layers.push_back(layer); //Push Back The Layer Struct Into Vector That Contains All The Layers
}

//Sets The Loss Function, Learning Rate, Batch Size, Momentum Coefficient And Verbose Of Network
void NeuralNetwork::assemble(const std::string& Loss_function,
	ElementType Learning_rate, int Batch_size, ElementType Momentum_coef, bool Verbose) {

	loss_function = Loss_function;
	learning_rate = Learning_rate;
	batch_size = Batch_size;
	verbose = Verbose;
	momentum_coef = Momentum_coef;
}

//Getter For Weights Of The Network
const auto NeuralNetwork::getWeights() const {
	//Store All the Weights In An Vector And Return It
	std::vector<EigenMatrix> weights(layers.size());
	for (int i{}; i < weights.size(); ++i) {
		weights[i] = layers[i].weights;
	}
	return weights;
}

//Getter For Biases Of The Network
const auto NeuralNetwork::getBiases() const {
	//Store All the Biases In An Vector And Return It
	std::vector<EigenVector> biases(layers.size());
	for (int i{}; i < biases.size(); ++i) {
		biases[i] = layers[i].biases;
	}
	return biases;
}

//Print Some Useful Information About The Network
void NeuralNetwork::info() const {

	std::size_t total_weights{};
	std::size_t total_biases{};

	std::cout << "Layer 1: (Input Layer) \n\n"
		<< "Neurons: " << input_size << "\n\n";
	for (int i{}; i < layers.size(); i++) {
		int layerIdx{ i + 2 };
		auto& layer{ layers[i] };
		std::cout << "Layer: " << layerIdx << "\n\n"
			<< "Neurons: " << layer.neurons << "\n"
			<< "Activation: " << layer.activationFunction << "\n"
			<< "Weights: " << layer.weights.rows() << " x " << layer.weights.cols()
			<< " (" << layer.weights.rows() * layer.weights.cols() << ")" << "\n"
			<< "Biases : " << layer.biases.size() << "\n\n";

		total_weights += layer.weights.rows() * layer.weights.cols();
		total_biases += layer.biases.size();
	}
	std::cout << "Total Weights: " << total_weights << "\n"
		<< "Total Biases: " << total_biases << "\n"
		<< "Total Parameters: " << total_weights + total_biases << "\n";
}

//Export The Model Into An File 
void NeuralNetwork::exportModel(const std::string& filename) {
	std::ofstream file; 
	file.open(filename); //Open The File
	if (!file.is_open()) { //Checks If Any Error Occured
		std::cerr << "Some Error Occured. Couldn't Export The Model";
		return;
	}

	//Store Meta Data Of The Network
	file << std::fixed << std::setprecision(8);
	file << input_size << "\n";
	file << learning_rate << "\n";
	file << verbose << "\n";
	file << loss_function << "\n";
	file << batch_size << "\n";

	//Iterate Through Every Layer Of Network
	for (const auto& layer : layers) {
		//Frist Store Layer Weights Dims [Neurons x Input Size]
		file << layer.weights.rows() << " " << layer.weights.cols() << "\n";

		//Store All the Layer Wegiths In Flattened Way
		for (int row{}; row < layer.weights.rows(); ++row) {
			for (int col{}; col < layer.weights.cols(); ++col) {
				file << layer.weights(row, col) << " ";
			}
		}
		file << "\n";
		//Store All The Biases
		for (int neuron{}; neuron < layer.weights.rows(); ++neuron) {
			file << layer.biases(neuron) << " ";
		}
		file << "\n";
		file << layer.activationFunction << "\n"; //Store Activation Function In The End
	}
	file.close(); //Close The File
}

//Reconstruct The Network From A Exported Model File
void NeuralNetwork::importModel(const std::string& filename) {
	std::ifstream file;
	file.open(filename); //Open File
	if (!file.is_open()) { //Checks If Any Error Occured
		std::cerr << "Some Error Occured. Couldn't Import The Model";
		return;
	}

	//First Read All The Network Meta Data
	file >> input_size;
	file >> learning_rate;
	file >> verbose;
	file >> loss_function;
	file >> batch_size;

	int rows, cols; //[Neurons x input_size] of Weights
	while (file >> rows >> cols) { //Read The Dims Of Weights
		//Push An Default Initialized DenseLayer Struct Into The Layers Vector
		layers.push_back(DenseLayer{});
		DenseLayer& layer{ layers.back() };

		//Instantiate Temp Weights, Biases Matrix and Vector 
		EigenMatrix weights(rows, cols);
		EigenVector biases(rows);

		//Reads The Weights And Store Them in "Weights" Matrix
		for (int row{}; row < rows; ++row) {
			for (int col{}; col < cols; ++col) {
				file >> weights(row, col);
			}
		}
		//Reads The Biases And Store Them In "biases" Vector
		for (int neuron{}; neuron < rows; ++neuron) {
			file >> biases(neuron);
		}

		file >> layer.activationFunction; //Read Activation Function
		//Set The Layer Weights And Biases To The Temps
		layer.weights = weights;
		layer.biases = biases;
		layer.in = cols;
		layer.neurons = rows;

	}
	file.close(); //Close The File
}

//Make Predictions
EigenMatrix NeuralNetwork::predict(const EigenMatrix& to_predict) {
	//Perform Forward Pass And Returns The Activations Of Last Layer
	feedForwardCPU(to_predict);
	return layers.back().activations;
};

//Function To Calculate Accuracy
float calculateAccuracy(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& Y_true) {
	// Ensure both matrices have the same shape (i.e., same number of samples and classes)
	assert(predictions.rows() == Y_true.rows() && predictions.cols() == Y_true.cols());

	int correct = 0;     // Counter for correctly predicted samples
	int total = predictions.rows(); // Total number of samples

	for (int i = 0; i < total; ++i) {
		int predicted_class = -1;   // Index of class with highest predicted probability
		int actual_class = -1;		// Index of true class (from one-hot label)
		float max_pred = -1.0f;		// Highest predicted probability
		float max_actual = -1.0f;	// Highest value in actual one-hot label (should be 1.0 ideally)

		// Find predicted and actual class index by finding max in each row
		for (int j = 0; j < predictions.cols(); ++j) {
			if (predictions(i, j) > max_pred) {
				max_pred = predictions(i, j);
				predicted_class = j;
			}
			if (Y_true(i, j) > max_actual) {
				max_actual = Y_true(i, j);
				actual_class = j;
			}
		}

		// Count it as correct if predicted class matches the true class
		if (predicted_class == actual_class) {
			++correct;
		}
	}

	// Return accuracy as a percentage
	return static_cast<float>(correct) / total * 100;
}