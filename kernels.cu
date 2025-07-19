#include "KERNELS.cuh"
#include <cstdio>
#include <cmath>

//CUDA Kernel To Print "Matrix" Objects
__global__ void Kernels::print_data(float* data, int rows, int cols) {

	printf("Matrix( [");
	for (int row{}; row < rows; ++row) {

		if (row > 0) printf("%9s", "");
		for (int col{}; col < cols; ++col) {
			int idx{ row * cols + col };

			printf("%f", data[idx]);

			if (row == (rows - 1) && col == (cols - 1)) printf("");
			else printf(", ");
		}
		if (row != rows - 1) printf("\n");
	}
	printf("] )\n");
}

// CUDA kernel for performing element-wise matrix operations on two matrices
// Supported operations:
// operationID == 0 -> Element-wise addition: matrix_c = matrix_a + matrix_b
// operationID == 1 -> Element-wise subtraction: matrix_c = matrix_a - matrix_b
__global__ void Kernels::matrixOperations(float* matrix_a, float* matrix_b, float* matrix_c, int rows, int cols, int operationID) {
	int row = blockDim.y * blockIdx.y + threadIdx.y; // Calculate row index
	int col = blockDim.x * blockIdx.x + threadIdx.x; // Calculate col index
	int idx = row * cols + col; // Flattened index for 2D array

	// Ensure thread is within matrix bounds
	if (row < rows && col < cols) {
		if(operationID==0) matrix_c[idx] = matrix_a[idx] + matrix_b[idx]; // Element-wise addition
		else if(operationID==1) matrix_c[idx] = matrix_a[idx] - matrix_b[idx]; // Element-wise subtraction
	}
}

// CUDA kernel for performing element-wise scalar operations on a matrix
// mat_b = mat_a (op) scalar
// Supported operations based on operationID:
// 0 -> Addition:       mat_b = mat_a + scalar
// 1 -> Subtraction:    mat_b = mat_a - scalar
// 2 -> Division:       mat_b = mat_a / scalar
// 3 -> Multiplication: mat_b = mat_a * scalar
__global__ void Kernels::scalarOperations(float* mat_a, float* mat_b, float scalar, int rows, int cols, int operationID) {
	int row = blockDim.y * blockIdx.y + threadIdx.y; // Calculate row index
	int col = blockDim.x * blockIdx.x + threadIdx.x; // Calculate col index
	int  idx = row * cols + col; // Flattened index for 2D array

	// Make sure the thread is within bounds
	if (row < rows && col < cols) {
		if (operationID==0) mat_b[idx] = mat_a[idx] + scalar; //Addition
		else if (operationID==1) mat_b[idx] = mat_a[idx] - scalar; //Substraction
		else if (operationID == 2) mat_b[idx] = mat_a[idx] / scalar; //Division
		else if (operationID==3) mat_b[idx] = mat_a[idx] * scalar;	//Multiplication
	}
}

// CUDA kernel to compute the transpose of a matrix
// Input:
// - matrix     : Pointer to the original matrix (row-major)
// - matrix_t   : Pointer to the transposed matrix output
// - rows       : Number of rows in the original matrix
// - cols       : Number of columns in the original matrix
//
// Each thread handles one element: it reads (row, col) from the input matrix
// and writes it to (col, row) in the transposed output
__global__ void Kernels::transpose(float* matrix, float* matrix_t, int rows, int cols) {
	int row = blockDim.y * blockIdx.y + threadIdx.y;  // Compute row index
	int col = blockDim.x * blockIdx.x + threadIdx.x;  // Compute col index
	int idx_nt = row * cols + col;	// Index in the original matrix
	int idx_t = col * rows + row;	// Index in the transposed matrix

	if (row < rows && col < cols) {
		matrix_t[idx_t] = matrix[idx_nt];  // Write transposed value
	}
}

// CUDA kernel to shuffle the rows of a matrix based on a provided index mapping
// Input:
// - original : Pointer to the original matrix (row-major order)
// - shuffled : Pointer to the output matrix where shuffled data will be stored
// - indices  : Pointer to an array of shuffled row indices
// - rows     : Number of rows in the matrix
// - cols     : Number of columns in the matrix
//
// Each thread copies one element from the original matrix to the shuffled matrix
// according to the shuffled row index given in 'indices'

__global__ void Kernels::shuffle(float* original, float* shuffled, int* indices, int rows, int cols) {
	int rowID = blockDim.y * blockIdx.y + threadIdx.y; // Compute row index
	int colID = blockDim.x * blockIdx.x + threadIdx.x; // Compute col index
	if (rowID < rows && colID < cols) {
		// Copy the element from the original matrix based on shuffled row index
		shuffled[rowID * cols + colID] = original[indices[rowID] * cols + colID];
	}
}

// CUDA kernel to set all elements of a matrix to a constant value
// Input:
// - mat      : Pointer to the matrix in device memory (row-major order)
// - constant : The constant value to assign to each element
// - rows     : Number of rows in the matrix
// - cols     : Number of columns in the matrix
//
// Each thread sets one element of the matrix to the specified constant
__global__ void Kernels::constant(float* mat, float constant, int rows, int cols) {
	int row = blockDim.y * blockIdx.y + threadIdx.y; // Row index for current thread
	int col = blockDim.x * blockIdx.x + threadIdx.x; // Column index for current thread
	
	if (row < rows && col < cols) {
		mat[row * cols + col] = constant; // Set matrix element to constant
	}
}

// CUDA kernel to perform row-wise broadcasting of a vector over a matrix
// It adds the corresponding element from the vector to each element in the matrix row-wise
// 
// Input:
// - vec      : Pointer to the 1D vector of size [cols], stored in device memory
// - mat      : Pointer to the 2D matrix [rows x cols], stored in row-major format
// - store_in : Pointer to the output matrix of the same size as `mat`
// - rows     : Number of rows in the matrix
// - cols     : Number of columns in the matrix
//
// Each thread handles one element in the output matrix.
__global__ void Kernels::rowBroadcast(float* vec, float* mat, float* store_in, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y; //Row Index
	int col = blockIdx.x * blockDim.x + threadIdx.x; //Col Index
	if (row<rows && col<cols) {
		// Add corresponding vector element to each matrix element (broadcasting)
		store_in[row * cols + col] =  mat[row*cols+col] + vec[col];
	}
}

// CUDA kernel to perform column-wise broadcasting of a vector over a matrix
// It adds the corresponding element from the vector to each element in the matrix column-wise
//
// Input:
// - vec      : Pointer to a 1D vector of size [rows], stored in device memory
// - mat      : Pointer to the 2D matrix [rows x cols], stored in row-major format
// - store_in : Pointer to the output matrix of the same size as `mat`
// - rows     : Number of rows in the matrix
// - cols     : Number of columns in the matrix
//
// Each thread handles one element in the output matrix

__global__ void Kernels::colBroadcast(float* vec, float* mat, float* store_in, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y; //Row Index
	int col = blockIdx.x * blockDim.x + threadIdx.x; //Col Index
	if (row < rows && col < cols) {
		// Add the row-specific vector value to each column element in that row
		store_in[row * cols + col] = mat[row * cols + col] + vec[row];
	}
}

// CUDA kernel for matrix multiplication: C = A * B
// A: [M x K], B: [K x N], C: [M x N]
// All matrices are in row-major order
//
// Each thread computes one element of matrix C by taking the dot product of
// one row of A and one column of B.
//
// Parameters:
// - A: Input matrix A of size M x K
// - B: Input matrix B of size K x N
// - C: Output matrix C of size M x N
// - M: Number of rows in matrix A (and C)
// - N: Number of columns in matrix B (and C)
// - K: Number of columns in A and rows in B

__global__ void Kernels::matmul(const float* A, const float* B, float* C, int M, int N, int K) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;// Row index of C
	int col = blockIdx.x * blockDim.x + threadIdx.x; // Row index of C

	if (row < M && col < N) {
		float sum = 0.0f;
		for (int k = 0; k < K; ++k) {
			sum += A[row * K + k] * B[k * N + col]; // Dot product
		}
		C[row * N + col] = sum;
	}
}

// CUDA kernel to apply ReLU activation function element-wise
// ReLU(x) = max(0, x)
// 
// Parameters:
// - preActivations: Input matrix of pre-activation values [rows x cols]
// - activations: Output matrix to store activated values [rows x cols]
// - rows: Number of rows in the input/output matrices
// - cols: Number of columns in the input/output matrices
//
// Each thread computes the ReLU for one element

__global__ void Kernels::relu(float* preActivations, float* activations, int rows, int cols) {
	int row = blockDim.y * blockIdx.y + threadIdx.y; //Row Index
	int col = blockDim.x * blockIdx.x + threadIdx.x; //Col Index

	if (row < rows && col < cols) {
		activations[row * cols + col] = std::fmaxf(0.0f, preActivations[row * cols + col]); //apply relu
	}
}

// CUDA kernel to apply the Sigmoid activation function element-wise.
// Sigmoid(x) = 1 / (1 + exp(-x))
//
// Parameters:
// - preActivations: Input matrix of pre-activation values [rows x cols]
// - activations: Output matrix to store activated values [rows x cols]
// - rows: Number of rows in the input/output matrices
// - cols: Number of columns in the input/output matrices
//
// Each thread computes the Sigmoid activation for one element.
__global__ void Kernels::sigmoid(float* preActivations, float* activations, int rows, int cols) {
	int row = blockDim.y * blockIdx.y + threadIdx.y;//Row Index
	int col = blockDim.x * blockIdx.x + threadIdx.x;//Col Index

	if (row < rows && col < cols) {
		activations[row * cols + col] = 1.0f / (1.0f + std::expf(-preActivations[row * cols + col])); //apply sigmoid
	}
}

// CUDA kernel to apply the Leaky ReLU activation function element-wise
// Leaky ReLU(x) = x        if x > 0
//                 0.1 * x  if x <= 0
//
// Parameters:
// - preActivations: Input matrix of pre-activation values [rows x cols]
// - activations: Output matrix to store activated values [rows x cols]
// - rows: Number of rows in the input/output matrices
// - cols: Number of columns in the input/output matrices
//
// Each thread processes one element and applies the Leaky ReLU activation
__global__ void Kernels::leaky_relu(float* preActivations, float* activations, int rows, int cols) {
	int row = blockDim.y * blockIdx.y + threadIdx.y;//Row Index
	int col = blockDim.x * blockIdx.x + threadIdx.x;//Col Index
	int idx = row * cols + col;

	if (row < rows && col < cols) {
		activations[idx] = preActivations[idx] > 0.0f ? preActivations[idx] : 0.1f * preActivations[idx]; //apply leaky relu
	}
}

// CUDA kernel to apply the Tanh activation function element-wise
// Tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
// It squashes input values to the range [-1, 1]
//
// Parameters:
// - preActivations: Input matrix of pre-activation values [rows x cols]
// - activations: Output matrix to store activated values [rows x cols]
// - rows: Number of rows in the input/output matrices
// - cols: Number of columns in the input/output matrices
//
// Each thread computes the Tanh activation for one element
__global__ void Kernels::tanh(float* preActivations, float* activations, int rows, int cols) {
	int row = blockDim.y * blockIdx.y + threadIdx.y;//Row Index
	int col = blockDim.x * blockIdx.x + threadIdx.x;//Col Index
	int idx = row * cols + col;

	if (row < rows && col < cols) {
		activations[idx] = std::tanhf(preActivations[idx]); //apply tanh
	}
}

// CUDA kernel to compute the exponential (e^x) of each element in the input matrix
//
// Parameters:
// - activations: Input matrix [rows x cols], contains the values to exponentiate
// - store_in: Output matrix [rows x cols], where the exponentiated values will be stored
// - rows: Number of rows in the input/output matrices
// - cols: Number of columns in the input/output matrices
//
// Each thread computes the exponential of a single element using std::expf for float precision
__global__ void Kernels::calculate_exps(float* activations, float* store_in, int rows, int cols) {
	int row = blockDim.y * blockIdx.y + threadIdx.y;//Row Index
	int col = blockDim.x * blockIdx.x + threadIdx.x;//Col Index
	int idx = row * cols + col;

	if (row < rows && col < cols) {
		store_in[idx] = std::expf(activations[idx]); 
	}
}

// CUDA kernel to compute the max and the sum of exponentials for each row of the input matrix
// This is used in a numerically stable softmax operation
// Each thread handles one row
//
// Parameters:
// - preActivations: Input matrix [rows x cols]
// - store_exp_sum: Output array [rows], stores the sum of exp(x - max) for each row
// - store_max: Output array [rows], stores the max value in each row
// - rows: Number of rows in the matrix
// - cols: Number of columns in the matrix
__global__ void Kernels::calculate_max_and_sum_exp(float* preActivations,
	float* store_exp_sum, float* store_max, int rows, int cols){
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	if (row < rows) {
		//Find max element in this row
		float max{ preActivations[row * cols]};
		float exp_sum{};
		for (int i {}; i<cols; ++i){
			int idx = row * cols + i;
			if (preActivations[idx] > max) max = preActivations[idx];
		}
		//Compute sum of exp(x - max) for numerical stability
		for (int i{}; i < cols; ++i) {
			int idx = row * cols + i;
			exp_sum += std::expf(preActivations[idx] - max);
		}
		store_max[row] = max;
		store_exp_sum[row] = exp_sum;
	}
}

// CUDA kernel to compute the softmax activation row-wise
// Uses precomputed max and sum of exponentials for numerical stability
//
// Parameters:
// - preActivations: Input matrix [rows x cols] (raw scores from last layer)
// - activations: Output matrix [rows x cols] (softmax probabilities)
// - exp_sum: Array [rows], holds sum of exp(x - max) per row (from previous kernel)
// - max: Array [rows], holds max(x) per row (from previous kernel)
// - rows: Number of rows in the matrix (batch size)
// - cols: Number of output classes (number of neurons in the final layer)
__global__ void Kernels::softmax(float* preActivations, float* activations, float* exp_sum, float* max, int rows, int cols) {
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int idx = row * cols + col;
	float eps = 1e-7f; // Small constant to avoid division by zero

	if (row < rows && col < cols) {
		// Subtract max for numerical stability
		// Then divide by sum of exponentials for proper normalization
		activations[idx] = std::expf(preActivations[idx] - max[row]) / (exp_sum[row]+eps);
	}
}

// CUDA kernel to perform element-wise multiplication of two matrices
// Each element: store_in[i][j] = matrix_A[i][j] * matrix_B[i][j]
//
// Parameters:
// - matrix_A: Input matrix A [rows x cols]
// - matrix_B: Input matrix B [rows x cols]
// - store_in: Output matrix to store the result [rows x cols]
// - rows: Number of rows in the matrices
// - cols: Number of columns in the matrices
__global__ void Kernels::elementWiseProduct(float* matrix_A, float* matrix_B, float* store_in, int rows, int cols) {
	int row = blockDim.y * blockIdx.y + threadIdx.y; //Row Index
	int col = blockDim.x * blockIdx.x + threadIdx.x; //Col Index
	int idx = row * cols + col;

	if (row < rows && col < cols) {
		store_in[idx] = matrix_A[idx] * matrix_B[idx]; //Do Multiplication
	}
}

// CUDA kernel to compute the derivative (gradient) of activation functions
// The result is stored in 'store_in' and used during backpropagation
//
// Parameters:
// - preActivations: Input matrix (pre-activation values) [rows x cols]
// - store_in: Output matrix to store the computed gradients [rows x cols]
// - activation_id: Specifies which activation function to use:
//       0 = Sigmoid
//       1 = ReLU
//       2 = Leaky ReLU
//       3 = Tanh
// - rows: Number of rows in the matrices
// - cols: Number of columns in the matrices
__global__ void Kernels::activationPrime(float* preActivations, float* store_in, int activation_id, int rows, int cols) {
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int idx = row * cols + col;

	if (row < rows && col < cols) {
		float element = preActivations[idx];
		// Sigmoid derivative: σ'(x) = σ(x) * (1 - σ(x))
		if (activation_id == 0) {
			float sigmoid = 1.0f / (1.0f + std::expf(-element));
			store_in[idx] = sigmoid * (1.0f - sigmoid);
		}
		// ReLU derivative: 1 if x > 0, else 0
		else if (activation_id == 1) store_in[idx] = element > 0.0f ? 1.0f : 0.0f;
		// Leaky ReLU derivative: 1 if x > 0, else 0.1
		else if (activation_id == 2) store_in[idx] = element > 0.0f ? 1.0f : 0.1f;
		// Tanh derivative: 1 - tanh^2(x)
		else if (activation_id == 3) store_in[idx] = 1 - std::tanhf(element) * std::tanhf(element);
	}
}

// CUDA kernel to compute the sum of each column in a matrix
// The result is stored in 'store_in', where store_in[col] = sum of elements in column 'col'
//
// Parameters:
// - matrix: Input matrix of shape [rows x cols]
// - store_in: Output vector of size [cols], where each element stores the column sum
// - rows: Number of rows in the matrix
// - cols: Number of columns in the matrix
__global__ void Kernels::rowWiseSum(float* matrix, float* store_in, int rows, int cols) {
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	if (col < cols) {
		float sum{};
		// Iterate over all rows for the given column
		for (int row{}; row < rows; ++row) {
			sum += matrix[row * cols + col];
		}
		// Store the sum of the column
		store_in[col] = sum;
	}
}

// CUDA kernel to compute the weighted sum for a layer in an MLP
// Each thread computes one element of the preActivation matrix
//
// Parameters:
// - input: Input matrix of shape [M x K]
// - weights_transposed: Transposed weights matrix of shape [K x N]
// - preActivations: Output matrix to store weighted sums, shape [M x N]
// - biases: Bias vector of shape [N]
// - M: Number of input samples (batch size)
// - N: Number of neurons in the current layer
// - K: Number of inputs to each neuron (input dimension)
__global__ void Kernels::computeWeightedSum(float* input, float* weights_transposed,
	float* preActivations, float* biases, int M, int N, int K) {

	int row = blockIdx.y * blockDim.y + threadIdx.y; // Current sample
	int col = blockIdx.x * blockDim.x + threadIdx.x;  // Current neuron

	// Bounds check
	if (row < M && col < N) {
		float sum = 0.0f;
		// Compute dot product: input[row] · weights_T[:, col]
		for (int k{}; k < K; ++k) {
			sum += input[row * K + k] * weights_transposed[k * N + col];
		}
		// Add bias and store result in preActivations
		preActivations[row * N + col] = sum + biases[col];
	}
}

// CUDA kernel to compute the derivative of the cost w.r.t. activations (dC/da)
// Supports MSE, BCE, and CE loss functions.
//
// Parameters:
// - activations: Output of the activation function (a), shape [rows x cols]
// - Y_train: Ground truth labels (y), shape [rows x cols]
// - errors: Output buffer to store dC/da, shape [rows x cols]
// - lossFn_id: ID representing the loss function used
//      0 -> MSE, 1 -> Binary Cross Entropy, 2 -> Cross Entropy
// - rows, cols: Dimensions of the input
__global__ void Kernels::compute_dC_da(float* activations, float* Y_train, float* errors,
	int lossFn_id, int rows, int cols) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = row * cols + col;

	// Bounds check
	if (row < rows && col < cols) {
		float y = Y_train[idx], a = activations[idx], eps = 1e-7f;
		// MSE loss: dC/da = a - y
		if (lossFn_id == 0) errors[idx] = a-y;
		// Binary Cross Entropy: dC/da = (a - y) / [(a + eps)(1 - a + eps)]
		else if (lossFn_id == 1) errors[idx] = (a - y) / ((a + eps) * (1.0f - a + eps));
		// Cross Entropy: dC/da = -y / (a + eps), Only valid if not using softmax activation
		else if (lossFn_id == 2) errors[idx] = -y / (a + eps);
	}

}

// CUDA kernel to compute total cross-entropy loss over a batch
// The result is accumulated into a single float using atomicAdd
//
// Parameters:
// - activations: Predicted probabilities from softmax, shape [rows x cols]
// - Y_train: One-hot encoded true labels, shape [rows x cols]
// - loss: Pointer to a single float to store total loss (use cudaMallocManaged or cudaMemcpy to access on host)
// - rows: Number of samples (batch size)
// - cols: Number of classes (output neurons)
__global__ void Kernels::cross_entropy_loss(float* activations, float* Y_train, float* loss, int rows, int cols) {
	// Each thread processes one row (sample)
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows) {
		float local_loss{};
		for (int col{}; col < cols; ++col) {
			int idx = row * cols + col;
			// Compute y * log(a)
			local_loss += Y_train[idx] * std::logf(activations[idx]);
		}
		// Accumulate the negative of the local loss into the shared loss variable
		atomicAdd(loss, -local_loss);
	}

}

// CUDA kernel to compute total binary cross-entropy loss for a batch
// The result is accumulated into a single float using atomicAdd
//
// Parameters:
// - activations: Predicted probabilities (from sigmoid), shape [rows x cols]
// - Y_train: True binary labels, shape [rows x cols]
// - loss: Pointer to a float that stores total loss
// - rows: Number of samples
// - cols: Number of outputs per sample
__global__ void Kernels::binary_cross_entropy(float* activations, float* Y_train, float* loss, int rows, int cols) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows) {
		float local_loss{};
		for (int col{}; col < cols; ++col) {
			int idx = row * cols + col;
			const float eps = 1e-7f;
			float a{ activations[idx] }, y{Y_train[idx]};
			// BCE formula per element
			local_loss += y * std::logf(a+eps) + (1.0f - y) * std::logf(1.0f - a+eps);
		}
		// Add the loss from this row to the total loss atomically
		atomicAdd(loss, -local_loss);
	}
}

// CUDA kernel to compute total Mean Squared Error (MSE) loss
// Each thread computes the squared error for one row/sample
//
// Parameters:
// - activations: Predicted outputs [rows x cols]
// - Y_train: Ground truth labels [rows x cols]
// - loss: Pointer to float that accumulates total MSE (to be averaged later)
// - rows: Number of data samples
// - cols: Number of output features per sample
__global__ void Kernels::MSE(float* activations, float* Y_train, float* loss, int rows, int cols) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows) {
		float local_loss{};
		for (int col{}; col < cols; ++col) {
			int idx = row * cols + col;
			float a{ activations[idx] }, y{ Y_train[idx] };
			local_loss += (y - a) * (y - a); // Square error for each output
		}
		// Add the loss from this row to the total loss atomically
		atomicAdd(loss, local_loss);
	}
}
