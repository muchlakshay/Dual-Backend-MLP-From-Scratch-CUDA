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
#include loadcsv.h

EigenDataT data {load_csv_eigen("data.csv", "target_column", 0.7)};

std::cout<< "Training Features\n" << data.X_train << "\n";


```
This funtion returns a Struct that contains Training Features (```X_train```) 

## ARCHITECTURE OVERVIEW

## LIMITATION / KNOWN ISSUES 
