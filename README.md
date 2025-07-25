# Dual-Backend-MLP-From-Scratch-CUDA
A fully from-scratch Multi-Layer Perceptron built in CUDA C++ with support for both GPU and CPU training.
It features a clean, modular API for defining network architectures, loss functions, and activation functions without relying on external machine learning libraries. Whether you're experimenting on a CPU or training faster on a GPU, this dual-backend system enables you to easily switch between the two, making it ideal for both educational purposes and custom, low-level deep learning research

Built by a teenager with a deep passion for AI systems and systems-level programming

## FEATURES
- Dual Backend: Train your models on either CPU or GPU (CUDA) with a simple switch
- Modular & Clean API: Easy to define and train models without any external dependencies
- Loss Functions: Mean Squared Error (MSE), Cross Entropy (CE), and Binary Cross Entropy (BCE)
- Activation Functions: Sigmoid, Relu, Leaky Relu, Tanh, and Linear
- Optimizers: vanilla Stochastic Gradient Descent (SGD), Mini-Batch SGD, and Momentum
- Weight Initialization Techniques: Xavier Normal, Xavier Uniform, He Normal, and He Uniform 
- Model Saving And Loading Mechanism
- Fully Customizable: Choose batch size, learning rate, architecture, backend, and more

## INSTALLATION
### Requirements
- [Eigen](https://eigen.tuxfamily.org/) Library
- CUDA Toolkit 7.5 or later
- A CUDA-capable NVIDIA GPU

### COMPILE AND RUN

```bash
git clone https://github.com/muchlakshay/Dual-Backend-MLP-From-Scratch-CUDA.git
cd Dual-Backend-MLP-From-Scratch-CUDA
```
- Open the cloned repo and compile & run the project

If you're using Windows (e.g., with Git Bash or PowerShell), make sure nvcc is in your system ```PATH```

```bash
nvcc main.cu kernles.cu matrix.cu -o main.exe
./main.exe
```

## USAGE

### Loading Data (From CSV Files)

To load data from a CSV file you can use ```loadcsv.cuh``` header file. First import the header file and then you can use the ```load_csv_eigen()``` function to load the data.

```load_csv_eigen(const std::string& filename, const std::string& target_column, float training_ratio = 0.8f)```

```cpp
#include loadcsv.cuh

EigenDataT data {load_csv_eigen("data.csv", "target_column", 0.7)};

std::cout << "Training Features\n" << data.X_train << "\n";
std::cout << "Training Labels\n"   << data.Y_train << "\n";
std::cout << "Testing Features\n"  << data.X_test  << "\n";
std::cout << "Testing Labels\n"    << data.Y_test  << "\n";

```

This funtion returns a Struct that contains Training Features (```X_train```), Training Labels (```Y_train```), Testing Features (```X_test```) and Testing Labels (```Y_test```)

You can also normalize the data using ```normalizeMatrix()``` function. It takes a reference to a ```EigenMatrix``` and then does in-place normalization

``` normalizeMatrix(EigenMatrix& matrix) ```

```cpp
normalizeMatrix(data.X_train);

std::cout << Normalized Training Features\n << data.X_train << "\n";
```
To one-hot-encode the labels you can use ```toOneHot()``` function. It takes Labels and number of classes as parameters and returns an ```EigenMatrix``` containing the one-hot-encoded labels (for multiclass calssification)

``` EigenMatrix toOneHot(EigenVector& labels, int num_labels) ```

```cpp
EigenMatrix Y_train_ohe { toOneHot(data.Y_train) };
EigenMatrix Y_test_ohe  { toOneHot(data.Y_test)  } 

std::cout<< "One Hot Encoded Training Labels\n" << Y_train_ohe << "\n";
std::cout<< "One Hot Encoded Testing Labels\n"  << Y_test_ohe  << "\n";

```

### Model Building 

To build a model architecture, first include the ``` NeuralNetwork.cuh``` header file and initialize a ```NeuralNetwork``` class object 

```cpp
#include "NeuralNetwork.cuh"

NeuralNetwork nn;
```

Now, first you have to define the size of input layer (number of columns in training features), you can do this using ```input()``` member function

``` void NeuralNetwork::input(int size) ```

```cpp
input(data.X_train.cols());
```
Then to add hidden layers or output layer use ```extend()``` member funtion

``` void NeuralNetwork::extend(int neurons, const std::string& activation_function, const Initializer& initializer) ```

- Supported activation function - "sigmoid", "relu", "tanh", "softmax" and "linear"
- Supported weight initializers - ```He_Uniform```, ```He_Normal```, ```Xavier_Uniform```, ```Xavier_Normal```

```cpp
//example
nn.extend(16, "relu", NeuralNetwork::Initializer::Xavier_Uniform);
```
To configure learning rate, optimizer, loss function, batch_size and verbose use ```assemble()``` member function

```void NeuralNetwork::assemble(const std::string& Loss_function, ElementType Learning_rate, int Batch_size, ElementType Momentum_coef=0.0f, bool Verbose=true)```

- Supported loss functions - "MSE", "cross_entropy", and "binary_cross_entropy"
- Supported optimizers - SGD [Default] (Keep ```Momentum_coef = 0.0f```) , Momentum (set ```Momentum_coef > 0.0```)

```cpp
nn.assemble("cross_entropy", 0.01f, 128, 0.95, true)
```

To start the training use ```learn()``` member function and for to predictions on data use ``` predict() ``` member function 

``` void NeuralNetwork::learn(EigenMatrix& X_train, EigenMatrix& Y_train, int epochs, const TrainingDevice& device, bool enableShuffling) ```

``` EigenMatrix NeuralNetwork::predict(const EigenMatrix& to_predict) ```

- Training Devices - ```CPU``` and ```GPU```

```cpp
//for CPU training
nn.learn(data.X_train, Y_train_ohe, 100, NeuralNetwork::TrainingDevice::CPU, false);

//for GPU training
 nn.learn(data.X_train, Y_train_ohe, 100, NeuralNetwork::TrainingDevice::GPU, false);

//doing predictions
auto predictions { nn.predict(data.X_test) };

```

Use ```info()``` member function to print information about the Network's Architecture

``` cpp
NeuralNetwork nn;
nn.input(10);
nn.extend(16, "relu", NeuralNetwork::Initializer::Xavier_Uniform);
nn.extend(5,  "softmax", NeuralNetwork::Initializer::Xavier_Uniform);

nn.info();

```
Output:-
```bash
Layer 1: (Input Layer)

Neurons: 10

Layer: 2

Neurons: 16
Activation: relu
Weights: 16 x 10 (160)
Biases : 16

Layer: 3

Neurons: 5
Activation: softmax
Weights: 5 x 16 (80)
Biases : 5

Total Weights: 240
Total Biases: 21
Total Parameters: 261

```

For saving the model after training use ``` exportModel() ``` member function and to import a exported model, use ``` importModel() ``` member function 

``` void NeuralNetwork::exportModel(const std::string& filename) ```
``` void NeuralNetwork::importModel(const std::string& filename) ```

```cpp
//exporting a model
nn.exportModel("model.txt");

//importing a model
NeuralNetwork nn2;
nn2.importModel("model.ext")
```
After importing the model, you can do predictions or further training, or fine-tuning on another dataset (transfer learning)

#### Final Example Pipe-Line

```cpp

#include "NeuralNetwork.cuh"
#include "loadcsv.cuh"

int main() {

  //Data Loading
  EigenDataT data { EigenDataT data {load_csv_eigen("data.csv", "target_column", 0.8)}; };

  //Normalizing Data
  normalizeMatrix(data.X_train);
  normalizeMatrix(data.X_test);

  //Model Building
  NeuralNetwork nn;
  nn.input(data.X_train.cols());
  nn.extend(16, "leaky_relu", NeuralNetwork::Initializer::He_Normal);
  nn.extend(4,  "leaky_relu", NeuralNetwork::Initializer::He_Normal);
  nn.extend(1,  "sigmoid",    NeuralNetwork::Initializer::He_Normal);
  nn.assemble("binary_cross_entropy", 0.001f, 64, 0.9, true);

  nn.learn(data.X_train, data.Y_train, 100, NeuralNetwork::TrainingDevice::GPU, true);

  //predictions
  auto predictions { nn.predict(data.X_test) };

 //exporting trained model
 nn.exportModel("model.txt");

  return 0;
}

```

### Example Training On MNIST Dataset

For loading MNIST dataset ```load_mnist.cuh```header file is used

```cpp
#include "NeuralNetwork.cuh"
#include "loadcsv.cuh"
#include "load_mnist.cuh"

int main() {

        //struct to load MNIST
	MNISTData mnist;

        //importing MNIST
	try {
		mnist = load_mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte", "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
	}
	catch (const std::exception& e) {
		std::cout << e.what();
	}

        //Building Model
	NeuralNetwork nn;
	nn.input(mnist.X_train.cols());
	nn.extend(16, "tanh", NeuralNetwork::Initializer::Xavier_Uniform);
	nn.extend(16, "tanh", NeuralNetwork::Initializer::Xavier_Uniform);
	nn.extend(10, "softmax", NeuralNetwork::Initializer::Xavier_Uniform);
	nn.assemble( "cross_entropy", 0.01f, 1000, 0.95f);

        //Starting Training 
	nn.learn(mnist.X_train, mnist.Y_train, 20, NeuralNetwork::TrainingDevice::GPU, false);
	nn.exportModel("model.txt");

        //Recognizing test digits
	auto predicted{ nn.predict(mnist.X_test) };

        //printing predictions
	for (int i{}; i < predicted.rows(); ++i) {
		std::cout << "Actual: " << mnist.Y_test.row(i) << " Predicted: " << predicted.row(i) << "\n";
	}

        //Calculation accuracy
	std::cout << "Accuracy: " << calculateAccuracy(predicted, mnist.Y_test) << "%";

       //exporting trained model (for future predictions or fine-tuning parameters more)
       nn.exportModel("model_mnist.txt");

	return 0;
}

```

Image of a few predictions and accuracy

<img width="1495" height="154" alt="image" src="https://github.com/user-attachments/assets/a188180e-874c-41c8-b23d-7194927ff0d1" />

## LIMITATION / KNOWN ISSUES 

- Unoptimized CUDA Kernels : The current GPU implementation prioritizes clarity, modularity, and reusability over low-level performance optimizations. It does not use Shared memory, Tiling strategies and Fused kernels but these can be added later for performance-critical use cases.
- Lack of Training Metrics : The model currently reports only the loss after each epoch. It does not track or display Accuracy (Of Each Epoch), Validation performance, Training time or speed metrics
- Only few activation and loss functions are available.
  
