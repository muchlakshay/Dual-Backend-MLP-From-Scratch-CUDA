# Dual-Backend-MLP-From-Scratch-CUDA-C-
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
- Visual Studio 2022 (with Desktop development with C++ and CUDA workload)
- CUDA Toolkit 7.5 or later
- A CUDA-capable NVIDIA GPU

### Build (Visual Studio)

```bash
git clone https://github.com/muchlakshay/Dual-Backend-MLP-From-Scratch-CUDA.git
cd Dual-Backend-MLP-From-Scratch-CUDA
```
- Open .sln file in Visual Studio
- CUDA files [.cu] should be marked as CUDA C/C++ in file properties
- Build the solution (Release or Debug)

## Build (CMake)

- I'll add it later (Currently, idk how to use it)

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

To start the training use ```learn()``` member function

``` void NeuralNetwork::learn(EigenMatrix& X_train, EigenMatrix& Y_train, int epochs, const TrainingDevice& device, bool enableShuffling) ```

- Training Devices - ```CPU``` and ```GPU```

```cpp
//for CPU training
nn.learn(data.X_train, Y_train_ohe, 100, NeuralNetwork::TrainingDevice::CPU, false);

//for GPU training
 nn.learn(data.X_train, Y_train_ohe, 100, NeuralNetwork::TrainingDevice::GPU, false);
```

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
  nn.extend(1, "sigmoid", NeuralNetwork::Initializer::He_Normal);
  nn.assemble("binary_cross_entropy", 0.001f, 64, 0.9, true);

  nn.learn(data.X_train, data.Y_train, 100, NeuralNetwork::TrainingDevice::GPU, true);

  //predictions
  auto prediction { nn.predict(data.X_test) };

  return 0;
}

```

## ARCHITECTURE OVERVIEW

## LIMITATION / KNOWN ISSUES 
