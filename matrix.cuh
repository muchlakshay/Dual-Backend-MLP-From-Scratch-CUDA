#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <initializer_list>

//macro to check cuda errors
#define CUDA_CHECK(err) {cudaCheck(err, __FILE__, __LINE__);}

//Matrix Class 
class Matrix {

	//Data members
	float* m_data_ptr{ nullptr };
	int m_rows{};
	int m_cols{};
	int m_elements{};
	bool m_destroy{true};

	//private member functions -> wrappers over cudaMemcpy & cudaMalloc
	float* allocate(int size);
	void copy(const float* dst, const float* src, int size, cudaMemcpyKind kind);

public:
	//constructors
	Matrix() = default;
	Matrix(const float* data_ptr, int rows, int cols, bool destroy=true);
	Matrix(std::initializer_list<std::initializer_list<float>> list);
	Matrix(Matrix&& matrix) noexcept; //move constructor
	Matrix(Matrix& matrix);
	Matrix(int rows, int cols);

	//destructor (cudaFree the pointer if m_destroy flag is true)
	~Matrix() { if (m_destroy) cudaFree(m_data_ptr); }

	//copy and move assigment operators
	void operator=(Matrix&) = delete;
	void operator=(Matrix&& matrix) noexcept;
	void reserve(int rows, int cols);

	//public memeber functions
	void clone(Matrix& matrix);
	void moveFrom(float* device_ptr, int cols, int rows = 1);

	void shuffle(Matrix& matrix, int* device_indices, bool free_indices_after = true);
	void MatMulInto(Matrix& mat_2, Matrix& store_in);
	void setConstant(float constant);
	void elementWiseProduct(Matrix& matrix, Matrix& store_in);
	void activationPrimes(Matrix& store_in, const std::string& activation_fn);
	void transpose(Matrix& matrix) const;
	Matrix rowBlock(int from, int rows, bool isView=false);
	void rowWiseSum(Matrix& store_in);
	void scalarOperation(float scalar, const char& operation, Matrix& store_in);
	void addInto(Matrix& matrix, Matrix& store_in);
	void subtractInto(Matrix& matrix, Matrix& store_in);
	void rowBroadcast(Matrix& vector, Matrix& store_in);
	void colBroadcast(Matrix& vector, Matrix& store_in);
	void relu(Matrix& store);
	void sigmoid(Matrix& store);
	void leaky_relu(Matrix& store);
	void tanh(Matrix& store);
	void softmax(Matrix& store);
	friend void print(const Matrix& matrix); 
	float* const data() { return m_data_ptr; }
	int getRows() const { return m_rows; }
	int getCols() const { return m_cols; }
	void setDims(int rows, int cols);
	int size() const { return m_rows * m_cols; }

};

//non-member functions
int* make_device_indices(unsigned int seed, int rows);
int* make_device_indices(int rows);
void compute_dC_da_GPU(Matrix activations, Matrix Y_train, Matrix errors, const std::string& loss_function);
void cross_entropy_GPU(Matrix activations, Matrix Y_train, float* d_loss_ptr);
void binary_cross_entropy_GPU(Matrix activations, Matrix Y_train, float* d_loss_ptr);
void MSE_GPU(Matrix activations, Matrix Y_train, float* d_loss_ptr);

//function to check cuda errors
inline void cudaCheck(cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		std::cout << "CUDA Error In File: " << file
			<< "\nAt Line: " << line
			<< "\nError: " << cudaGetErrorString(err);
		exit(err);
	}
}
