#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <initializer_list>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <vector>
#include <numeric>
#include <random>
#include "KERNELS.cuh"
#include "matrix.cuh"

//A Wrapper Around cudaMalloc For Easy And Silent Errors Free Memory Allocations
float* Matrix::allocate(int size) {
	float* d_ptr{ nullptr };
	auto alloc{ cudaMalloc(&d_ptr, size * sizeof(float)) };
	CUDA_CHECK(alloc);
	return d_ptr;
}

//A Wrapper Around CudaMemcpy For Easy And Silent Errors Free Usage
void Matrix::copy(const float* dst, const float* src, int size, cudaMemcpyKind kind) {
	auto cpy{ cudaMemcpy((void**)dst, (void**)src, size * sizeof(float), kind) };
	CUDA_CHECK(cpy);
}

//Contructor To Construct The Object From A Float Pointer 
Matrix::Matrix(const float* data_ptr, int rows, int cols, bool destroy)
	: m_rows{ rows }, m_cols{ cols }, m_elements{ rows * cols }, m_destroy{destroy} {  //member initializer ist

	assert(m_cols != 0 && m_rows != 0 && data_ptr != nullptr); //Checks If Pointer isnt nullptr or dims arent 0
	m_data_ptr = allocate(m_elements);   //allocate memory to the m_data_ptr 
	cudaPointerAttributes attr;
	//fetch info about the pointer
	if (cudaPointerGetAttributes(&attr, data_ptr) == cudaSuccess) {
		if (attr.hostPointer != nullptr) copy(m_data_ptr, data_ptr, m_elements, cudaMemcpyHostToDevice); //If pointer is a host pointer
		else if (attr.devicePointer != nullptr) copy(m_data_ptr, data_ptr, m_elements, cudaMemcpyDeviceToDevice); //if pointer is a device pointer
	}
}

// Constructor to initialize Matrix from a nested initializer list (like {{1,2,3},{4,5,6}})
// It flattens the 2D structure and copies the data to GPU memory
Matrix::Matrix(std::initializer_list<std::initializer_list<float>> list) {

	bool isFirst{ true };
	m_rows = list.size(); // Number of rows in the matrix

	std::vector<float> flattened; // Temporary CPU-side buffer to flatten data

	for (const auto& val : list) {
		if (isFirst) {
			m_cols = val.size();				// Infer number of columns from first row
			m_elements = m_cols * m_rows;		// Total number of elements
			flattened.reserve(m_elements);		// Reserve memory for performance
		}
		// Ensure all rows have same number of columns
		assert(val.size() == m_cols);
		// Append row elements to flat vector
		flattened.insert(flattened.end(), val.begin(), val.end());
		isFirst = false;
	}
	// Allocate memory on GPU for flattened matrix
	m_data_ptr = allocate(m_elements);
	// Copy data from CPU to GPU
	copy(m_data_ptr, flattened.data(), m_elements, cudaMemcpyHostToDevice);
}

//Construct just to reserve memory 
Matrix::Matrix(int rows, int cols){
	reserve(rows, cols);
}

//Move Constructor
//Transfers ownership of resources from the temporary 'matrix' to 'this'
//Prevents deep copy and makes move operations fast and safe
Matrix::Matrix(Matrix&& matrix) noexcept 
	: m_data_ptr{ matrix.m_data_ptr },		// Take over GPU memory pointer
	m_rows{ matrix.m_rows },				// Take over row count
	m_cols{ matrix.m_cols },				// Take over col count
	m_elements{ matrix.m_elements },		// Take over total element count
	m_destroy{matrix.m_destroy} {			// Copy destruction flag (if applicable)

	//Invalidate the source matrix so it doesn't free memory we now own
	matrix.m_data_ptr = nullptr;
	matrix.m_rows = matrix.m_cols = matrix.m_elements = 0;
	
}

// Copy Constructor
// Delegates to another constructor that accepts raw data pointer and dimensions
Matrix::Matrix(Matrix& matrix) : Matrix{ matrix.m_data_ptr, matrix.m_cols, matrix.m_rows, matrix.m_destroy } {}

// Reserves GPU memory for the matrix with given dimensions
// Frees and reallocates only if the size changes
void Matrix::reserve(int rows, int cols) {
	int elements{ rows * cols };

	// Reallocate only if new size differs from current size
	if (m_elements != elements) {
		cudaFree(m_data_ptr);				// Free existing GPU memory
		m_data_ptr = allocate(elements);	// Allocate new memory on GPU
	}
	// Update metadata
	m_rows = rows; m_cols = cols; m_elements = elements;
}

// Move assignment operator
void Matrix::operator=(Matrix&& matrix) noexcept {
	// Handle self-assignment
	if (this == &matrix) return;
	// Free existing GPU memory, if allocated
	if (m_data_ptr) cudaFree(m_data_ptr);

	// Transfer ownership of data and metadata from the source matrix
	m_data_ptr = matrix.m_data_ptr;
	m_cols = matrix.m_cols;
	m_rows = matrix.m_rows;
	m_elements = matrix.m_elements;
	m_destroy = matrix.m_destroy;

	// Nullify the source matrix to avoid double free or accidental use
	matrix.m_data_ptr = nullptr;
	matrix.m_rows = matrix.m_cols = matrix.m_elements = 0; // Prevent destruction on moved-from object (optional safety)
}

// Clones the contents and structure of another Matrix into this one
void Matrix::clone(Matrix& matrix) {
	// If element count differs, free old memory and allocate new space
	if (matrix.m_elements != m_elements) {
		cudaFree(m_data_ptr);							// Free current memory
		m_data_ptr = allocate(matrix.m_elements);		// Allocate new memory matching the source matrix
	}	

	// Copy metadata
	m_elements = matrix.m_elements;
	m_rows = matrix.m_rows;
	m_cols = matrix.m_cols;
	m_destroy = matrix.m_destroy;

	// Copy data from source matrix on device to this matrix on device (device-to-device copy)
	copy(m_data_ptr, matrix.m_data_ptr, matrix.m_elements, cudaMemcpyDeviceToDevice);
}

// Extracts a block of consecutive rows from the matrix.
// If `isView` is true, it returns a non-owning view (no memory copied)
// If `isView` is false, it returns a deep copy of the rows (memory is allocated and data is copied)
Matrix Matrix::rowBlock(int from, int rows, bool isView){
	Matrix mat{};  // Resultant sub-matrix

	if (isView) {
		// Return a view: just adjust pointer and metadata, no allocation or copying
		mat.m_data_ptr = m_data_ptr+(m_cols*from);  // Point to the correct offset in original data
		mat.m_rows = rows;			
		mat.m_cols = m_cols;
		mat.m_elements = rows * m_cols;
		mat.m_destroy = false;  // Don't free memory, since it’s just a view
		return mat;
	}
	else
	{
		// Return a copy: allocate new memory and copy the data
		mat.m_rows = rows;
		mat.m_cols = m_cols;
		mat.m_elements = rows * m_cols;
		mat.m_data_ptr = allocate(mat.m_elements);  // Allocate new GPU memory
		copy(mat.m_data_ptr, m_data_ptr + (m_cols * from), sizeof(float)*mat.m_elements, cudaMemcpyDeviceToDevice);
		return mat;
	}
}

// Moves data ownership from an existing device pointer into this Matrix object.
// Assumes the pointer is already allocated on the device and valid.
// This function does not copy data, it just takes ownership (performs a move)
void Matrix::moveFrom(float* device_ptr, int rows, int cols) {

	cudaPointerAttributes attr;
	cudaPointerGetAttributes(&attr, device_ptr); // Get pointer attributes to ensure it's a device pointer
	assert(attr.devicePointer != nullptr);		 //must be a valid device pointer

	int ptr_size{ rows * cols};
	cudaFree(m_data_ptr);				// Free current device memory

	// Update metadata
	m_rows = rows; m_cols = cols; m_elements = ptr_size;
	// Take ownership of the incoming device memory
	m_data_ptr = device_ptr;
	// Invalidate the source device pointer
	device_ptr = nullptr;
}

// Shuffles the current matrix's rows using the provided device-side indices,
// and stores the result in another pre-allocated matrix.
// Parameters:
// - matrix: the output Matrix object where shuffled data will be stored
// - device_indices: device pointer holding shuffled row indices
// - free_indices_after: whether to free device_indices after use
void Matrix::shuffle(Matrix& matrix, int* device_indices, bool free_indices_after) {
	assert(this!=&matrix); // Ensure source and destination are not the same matrix

	// Define CUDA block and grid dimensions for 2D execution
	dim3 BLOCK_SIZE(16, 16);
	dim3 GRID_SIZE(
		(ceil(m_cols / static_cast<float>(BLOCK_SIZE.x))),
		(ceil(m_rows / static_cast<float>(BLOCK_SIZE.y)))
	);

	// Launch the shuffle kernel on the device
	Kernels::shuffle<<<GRID_SIZE, BLOCK_SIZE>>>(m_data_ptr, matrix.m_data_ptr, device_indices, m_rows, m_cols);
	// Wait for GPU to finish and check for any kernel errors
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());
	// Optionally free the device-side indices after shuffling
	if (free_indices_after) cudaFree(device_indices);
}

// Transposes the current matrix and stores the result in a given destination matrix
// The destination matrix must already be allocated with the correct number of elements
void Matrix::transpose(Matrix& matrix) const {
	assert(m_elements == matrix.m_elements && this != &matrix); // Ensure same size and not transposing into self

	// Update the shape of the destination matrix: rows become columns and vice versa
	matrix.m_cols = m_rows;
	matrix.m_rows = m_cols;

	// Define CUDA block and grid dimensions for 2D execution
	dim3 BLOCK_SIZE(16, 16);
	dim3 GRID_SIZE(
		(ceil(m_cols / static_cast<float>(BLOCK_SIZE.x))),
		(ceil(m_rows / static_cast<float>(BLOCK_SIZE.y)))
	);

	// Launch the CUDA transpose kernel
	Kernels::transpose << <GRID_SIZE, BLOCK_SIZE >> > (m_data_ptr,
		matrix.m_data_ptr, m_rows, m_cols);
	// Wait for GPU to finish and check for any kernel errors
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());
}

//Prints the matrix to console
void print(const Matrix& matrix) {
	//Launch the kernel to print the matrix
	Kernels::print_data << <1, 1 >> > (matrix.m_data_ptr, matrix.m_rows, matrix.m_cols);
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());
}

// Adds current matrix to another matrix and stores result in store_in
void Matrix::addInto(Matrix& matrix, Matrix& store_in){
	assert(matrix.m_cols == m_cols && matrix.m_rows == m_rows
		&& store_in.m_cols == m_cols && store_in.m_rows == m_rows); // Ensure dimensions match
	
	// Define CUDA block and grid dimensions
	dim3 BLOCK_SIZE(16, 16);
	dim3 GRID_SIZE(
		(ceil(m_cols / static_cast<float>(BLOCK_SIZE.x))),
		(ceil(m_rows / static_cast<float>(BLOCK_SIZE.y)))
	);

	// Launch addition kernel (operationID = 0 for addition)
	Kernels::matrixOperations << <GRID_SIZE, BLOCK_SIZE >> > (m_data_ptr, matrix.m_data_ptr, 
		store_in.m_data_ptr, m_rows, m_cols, 0);
	CUDA_CHECK(cudaGetLastError());
}

// Subtracts another matrix from this matrix and stores result in store_in
void Matrix::subtractInto(Matrix& matrix, Matrix& store_in) {
	assert(matrix.m_cols == m_cols && matrix.m_rows == m_rows
		&& store_in.m_cols == m_cols && store_in.m_rows == m_rows); // Ensure dimensions match

	// Define CUDA block and grid dimensions
	dim3 BLOCK_SIZE(16, 16);
	dim3 GRID_SIZE(
		(ceil(m_cols / static_cast<float>(BLOCK_SIZE.x))),
		(ceil(m_rows / static_cast<float>(BLOCK_SIZE.y)))
	);

	// Launch subtraction kernel (operationID = 1 for subtraction)
	Kernels::matrixOperations << <GRID_SIZE, BLOCK_SIZE >> > (m_data_ptr, matrix.m_data_ptr,
		store_in.m_data_ptr, m_rows, m_cols, 1);
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());
}

// Performs matrix multiplication: this * mat -> store_in
void Matrix::MatMulInto(Matrix& mat, Matrix& store_in){
	assert(store_in.m_rows == m_rows && store_in.m_cols == mat.m_cols); // Ensure output shape is correct
	assert(m_cols == mat.m_rows); // Ensure inner dimensions match for multiplication

	// Define CUDA block and grid dimensions
	dim3 BLOCK_SIZE(16, 16);
	dim3 GRID_SIZE(
		(ceil(mat.m_cols / static_cast<float>(BLOCK_SIZE.x))),
		(ceil(m_rows / static_cast<float>(BLOCK_SIZE.y)))
	);

	// Launch CUDA matmul kernel
	Kernels::matmul << <GRID_SIZE, BLOCK_SIZE >> > (m_data_ptr, mat.data(), store_in.data(),
		m_rows, mat.m_cols, m_cols);
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());
}

//Set all the elements of Matrix to the specified constant
void Matrix::setConstant(float constant) {
	// Define CUDA block and grid dimensions
	dim3 BLOCK_SIZE(16, 16);
	dim3 GRID_SIZE(
		(ceil(m_cols / static_cast<float>(BLOCK_SIZE.x))),
		(ceil(m_rows / static_cast<float>(BLOCK_SIZE.y)))
	);
	// Launch CUDA constant kernel
	Kernels::constant << <GRID_SIZE, BLOCK_SIZE >> > (m_data_ptr, constant, m_rows, m_cols);
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());
}

// Performs element-wise multiplication b/w this and passed matrix and store the result in store_in
void Matrix::elementWiseProduct(Matrix& matrix, Matrix& store_in) {
	assert(store_in.m_rows == m_rows && store_in.m_cols == m_cols);  // Check output shape
	assert(matrix.m_rows == m_rows && matrix.m_cols == m_cols);		// Ensure input matrices match in size

	// Define CUDA block and grid dimensions
	dim3 BLOCK_SIZE(16, 16);
	dim3 GRID_SIZE(
		(ceil(m_cols / static_cast<float>(BLOCK_SIZE.x))),
		(ceil(m_rows / static_cast<float>(BLOCK_SIZE.y)))
	);

	// Launch CUDA kernel for element-wise product
	Kernels::elementWiseProduct << <GRID_SIZE, BLOCK_SIZE >> > (m_data_ptr, matrix.m_data_ptr, store_in.m_data_ptr, m_rows, m_cols);
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());
}

// Adds a row vector to each row of the matrix (broadcasting across rows)
void Matrix::rowBroadcast(Matrix& vector, Matrix& store_in) {
	assert(vector.m_cols == m_cols && vector.m_rows == 1     // Ensure vector is 1xN
		&& store_in.m_rows==m_rows && store_in.m_cols==m_cols);  // Ensure output shape matches this matrix
	
	// Define CUDA block and grid dimensions
	dim3 BLOCK_SIZE(16, 16);
	dim3 GRID_SIZE(
		(ceil(m_cols / static_cast<float>(BLOCK_SIZE.x))),
		(ceil(m_rows / static_cast<float>(BLOCK_SIZE.y)))
	);
	// Launch CUDA kernel to perform row-wise broadcasting
	Kernels::rowBroadcast << <GRID_SIZE, BLOCK_SIZE >> > (vector.data(), m_data_ptr,
		store_in.data(), m_rows, m_cols);

	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());
}

// Computes the derivative (gradient) of the activation function element-wise
// and stores the result in 'store_in'.
void Matrix::activationPrimes(Matrix& store_in, const std::string& activation_fn) {
	assert(store_in.m_rows == m_rows && store_in.m_cols == m_cols);  // Shapes must match
	// Define CUDA block and grid dimensions
	dim3 BLOCK_SIZE(16, 16);
	dim3 GRID_SIZE(
		(ceil(m_cols / static_cast<float>(BLOCK_SIZE.x))),
		(ceil(m_rows / static_cast<float>(BLOCK_SIZE.y)))
	);
	//sigmoid = 0, relu = 1, leaky_relu = 2, tanh = 3
	int activation_id{};
	if (activation_fn == "sigmoid") activation_id = 0;
	else if (activation_fn == "relu") activation_id = 1;
	else if (activation_fn == "leaky_relu") activation_id = 2;
	else if (activation_fn == "tanh") activation_id = 3;
	else if (activation_fn == "linear") {
		store_in.setConstant(1.0f);
		return;
	}
	// Launch CUDA kernel to compute the element-wise activation prime
	Kernels::activationPrime << <GRID_SIZE, BLOCK_SIZE >> > (m_data_ptr, store_in.m_data_ptr, activation_id, m_rows, m_cols);
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());

}

// Broadcasts a column vector (1D) across each column of the matrix and stores result in 'store_in'
// Each element in a row gets the corresponding vector value added to it
void Matrix::colBroadcast(Matrix& vector, Matrix& store_in) {
	assert(vector.m_rows == m_rows && vector.m_cols == 1    // Vector should have same row count
		&& store_in.m_rows == m_rows && store_in.m_cols == m_cols);    // Output matrix shape must match

	// Define CUDA block and grid dimensions
	dim3 BLOCK_SIZE(16, 16);
	dim3 GRID_SIZE(
		(ceil(m_cols / static_cast<float>(BLOCK_SIZE.x))),
		(ceil(m_rows / static_cast<float>(BLOCK_SIZE.y)))
	);

	// Launch kernel to add the broadcasted column vector to each column of the matrix
	Kernels::colBroadcast << <GRID_SIZE, BLOCK_SIZE >> > (vector.data(), m_data_ptr,
		store_in.data(), m_rows, m_cols);

	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());
}

//Add Up All the Columns Vertically
void Matrix::rowWiseSum(Matrix& store_in) {
	assert(store_in.m_cols == m_cols && store_in.m_rows==1); //store_in must be of dims (1, N)

	//Define block size and threads per block
	int THREADS{ 256 };
	int BLOCK_SIZE = ceil(m_cols / static_cast<float>(THREADS));
	
	//launch kernel
	Kernels::rowWiseSum << <BLOCK_SIZE, THREADS >> > (m_data_ptr, store_in.m_data_ptr, m_rows, m_cols);
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());
}

// Applies a scalar operation element-wise to the matrix and stores the result in 'store_in'.
// Supported operations: '+', '-', '*', '/'.
void Matrix::scalarOperation(float scalar, const char& operation, Matrix& store_in) { 
	 
	assert(store_in.m_rows == m_rows && store_in.m_cols == m_cols);  // Ensure dimensions match

	dim3 BLOCK_SIZE(16, 16);
	dim3 GRID_SIZE(
		(ceil(m_cols / static_cast<float>(BLOCK_SIZE.x))),
		(ceil(m_rows / static_cast<float>(BLOCK_SIZE.y)))
	);

	int operation_id{};

	// Map operation character to operation ID
	if (operation == '+') operation_id = 0;
	else if (operation == '-') operation_id = 1;
	else if (operation == '/') operation_id = 2;
	else if (operation == '*') operation_id = 3;
	else throw "Invalid Operation";  // Throw error for unsupported ops

	// Launch CUDA kernel to perform the scalar operation
	Kernels::scalarOperations << <GRID_SIZE, BLOCK_SIZE >> > (m_data_ptr, store_in.m_data_ptr, scalar, m_rows, m_cols, operation_id);
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());
}

//Apply relu activation on the matrix
void Matrix::relu(Matrix& store) {
	assert(store.m_rows == m_rows && store.m_cols == m_cols); //check the opt matrix dims

	//Configure grid and block sizes
	dim3 BLOCK_SIZE(16, 16);
	dim3 GRID_SIZE(
		(ceil(m_cols / static_cast<float>(BLOCK_SIZE.x))),
		(ceil(m_rows / static_cast<float>(BLOCK_SIZE.y)))
	);

	//launch kernel to apply relu
	Kernels::relu<<<GRID_SIZE, BLOCK_SIZE>>>(m_data_ptr, store.m_data_ptr, m_rows, m_cols);
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());
}

//Apply sigmoid activation on the matrix
void Matrix::sigmoid(Matrix& store) {
	assert(store.m_rows == m_rows && store.m_cols == m_cols); //check the opt matrix dims

	//Configure grid and block sizes
	dim3 BLOCK_SIZE(16, 16);
	dim3 GRID_SIZE(
		(ceil(m_cols / static_cast<float>(BLOCK_SIZE.x))),
		(ceil(m_rows / static_cast<float>(BLOCK_SIZE.y)))
	);

	//launch kernel to apply sigmoid
	Kernels::sigmoid << <GRID_SIZE, BLOCK_SIZE >> > (m_data_ptr, store.m_data_ptr, m_rows, m_cols);
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());
}

//Apply leaky_relu activation on the matrix
void Matrix::leaky_relu(Matrix& store) {
	assert(store.m_rows == m_rows && store.m_cols == m_cols); //check the opt matrix dims

	//Configure grid and block sizes
	dim3 BLOCK_SIZE(16, 16);
	dim3 GRID_SIZE(
		(ceil(m_cols / static_cast<float>(BLOCK_SIZE.x))),
		(ceil(m_rows / static_cast<float>(BLOCK_SIZE.y)))
	);

	//launch kernel to apply leaky_relu
	Kernels::leaky_relu << <GRID_SIZE, BLOCK_SIZE >> > (m_data_ptr, store.m_data_ptr, m_rows, m_cols);
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());
}

//Apply tanh activation on the matrix
void Matrix::tanh(Matrix& store) {
	assert(store.m_rows == m_rows && store.m_cols == m_cols); //check the opt matrix dims

	//Configure grid and block sizes
	dim3 BLOCK_SIZE(16, 16);
	dim3 GRID_SIZE(
		(ceil(m_cols / static_cast<float>(BLOCK_SIZE.x))),
		(ceil(m_rows / static_cast<float>(BLOCK_SIZE.y)))
	);
	//launch kernel to apply tanh
	Kernels::tanh << <GRID_SIZE, BLOCK_SIZE >> > (m_data_ptr, store.m_data_ptr, m_rows, m_cols);
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());
}

// Applies softmax activation across each row of the matrix and stores the result in 'store'
void Matrix::softmax(Matrix& store) { 
	assert(store.m_rows == m_rows && store.m_cols == m_cols); // Ensure output dimensions match
	
	// Allocate temporary GPU memory for row-wise max logits and exp sums
	float* max_logit{allocate(m_rows)};
	float* exp_sum{allocate(m_rows)};

	// Launch 1D kernel to compute max logits and exp sums for numerical stability
	int BLOCK_SIZE_{ 256 };
	int GRID_SIZE_ = (m_rows + BLOCK_SIZE_ - 1) / BLOCK_SIZE_;

	Kernels::calculate_max_and_sum_exp << <GRID_SIZE_, BLOCK_SIZE_ >> > (m_data_ptr,
		exp_sum, max_logit, m_rows, m_cols);
	
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());

	// Launch 2D kernel to compute final softmax values
	dim3 BLOCK_SIZE(16, 16);
	dim3 GRID_SIZE(
		(ceil(m_cols / static_cast<float>(BLOCK_SIZE.x))),
		(ceil(m_rows / static_cast<float>(BLOCK_SIZE.y)))
	);

	Kernels::softmax << <GRID_SIZE, BLOCK_SIZE >> > (m_data_ptr, store.data(),
		exp_sum, max_logit, m_rows, m_cols);

	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());

	// Free temporary buffers
	cudaFree(max_logit);
	cudaFree(exp_sum);
}

//reset the dimensions of the matrix
void Matrix::setDims(int rows, int cols) {
	assert(rows * cols <= m_elements); //row * col should be less or equal to the m_elements
	m_rows = rows;
	m_cols = cols;
}

// Generates a shuffled array of indices on the host and transfers it to the GPU.
// - seed: for deterministic shuffling
// - rows: number of indices to generate (0 to rows-1)
// Returns a device pointer to the shuffled indices array.
int* make_device_indices(unsigned int seed, int rows) {
	std::vector<int> indices(rows); // Host vector to hold indices
	// Allocate device memory
	int* d_indices;
	auto alloc{ cudaMalloc(&d_indices, sizeof(int) * indices.size()) };
	CUDA_CHECK(alloc);
	// Fill with sequential values [0, 1, 2, ..., rows-1]
	std::iota(indices.begin(), indices.end(), 0);
	// Shuffle using given seed for reproducibility
	std::shuffle(indices.begin(), indices.end(), std::mt19937(seed));

	// Copy shuffled indices from host to device
	auto cpy{ cudaMemcpy((void**)d_indices, (void**)indices.data(),
	sizeof(int) * indices.size(), cudaMemcpyHostToDevice) };
	CUDA_CHECK(cpy);

	return d_indices; // Return device pointer
}

// Generates a shuffled array of indices on the host and transfers it to the GPU.
// returns a device pointer to the shuffled indices array, but doesnt require a seed argument
int* make_device_indices(int rows) {
	std::vector<int> indices(rows);  // host vector to hold indices
	int* d_indices;  // Allocate device memory
	auto alloc{ cudaMalloc(&d_indices, sizeof(int) * indices.size()) };
	CUDA_CHECK(alloc);

	std::random_device rd;  //seed

	std::iota(indices.begin(), indices.end(), 0);  // Fill with sequential values [0, 1, 2, ..., rows-1]
	std::shuffle(indices.begin(), indices.end(), std::mt19937(rd())); // Shuffle using given seed for reproducibility
	// Copy shuffled indices from host to device
	auto cpy{ cudaMemcpy((void**)d_indices, (void**)indices.data(),
	sizeof(int) * indices.size(), cudaMemcpyHostToDevice) };
	CUDA_CHECK(cpy);

	return d_indices;
}

// Launches CUDA kernel to compute dC/da (partial derivative of cost w.r.t. activations)
// depending on the specified loss function
// Supports: MSE (0), Binary Cross-Entropy (1), Cross-Entropy (2)
void compute_dC_da_GPU(Matrix activations, Matrix Y_train, Matrix errors, const std::string& loss_function) {

	int lossFn_id{};
	if (loss_function == "MSE") lossFn_id = 0;
	else if (loss_function == "binary_cross_entropy") lossFn_id = 1;
	else if (loss_function == "cross_entropy") lossFn_id = 2;
	
	// Configure CUDA thread block and grid dimensions
	dim3 BLOCK_SIZE(16, 16);
	dim3 GRID_SIZE(
		(ceil(errors.getCols() / static_cast<float>(BLOCK_SIZE.x))),
		(ceil(errors.getRows() / static_cast<float>(BLOCK_SIZE.y)))
	);

	// Launch CUDA kernel to compute dC/da
	Kernels::compute_dC_da << <GRID_SIZE, BLOCK_SIZE >> > (activations.data(), 
		Y_train.data(), errors.data(), lossFn_id, errors.getRows(), errors.getCols());
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());
}

//Calculate Cross Entropy Loss
void cross_entropy_GPU(Matrix activations, Matrix Y_train, float* d_loss_ptr) {

	//Define blocks and threads per block
	int THREADS = 256;
	int BLOCKS = std::ceil(activations.getRows() / static_cast<float>(THREADS));
	//launch the Cross Entropy Kernel
	Kernels::cross_entropy_loss << <BLOCKS, THREADS >> > (activations.data(), Y_train.data(),
		d_loss_ptr, activations.getRows(), activations.getCols());
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());
}

//Calculate Binary Cross Entropy Loss
void binary_cross_entropy_GPU(Matrix activations, Matrix Y_train, float* d_loss_ptr) {
	//Define blocks and threads per block
	int THREADS = 256;
	int BLOCKS = std::ceil(activations.getRows() / static_cast<float>(THREADS));
	//launch the Binary Cross Entropy Kernel
	Kernels::binary_cross_entropy << <BLOCKS, THREADS >> > (activations.data(), Y_train.data(),
		d_loss_ptr, activations.getRows(), activations.getCols());
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());
}

//Calculate MSE Loss
void MSE_GPU(Matrix activations, Matrix Y_train, float* d_loss_ptr) {
	//Define blocks and threads per block
	int THREADS = 256;
	int BLOCKS = std::ceil(activations.getRows() / static_cast<float>(THREADS));
	//launch the MSE Kernel
	Kernels::MSE << <BLOCKS, THREADS >> > (activations.data(), Y_train.data(),
		d_loss_ptr, activations.getRows(), activations.getCols());
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());
}
