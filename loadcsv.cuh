#pragma once

#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <numeric>
#include "typedefs.cuh"

struct EigenDataT {
    std::vector<std::string> header;
    EigenMatrix X_train;
    EigenVector Y_train;
    EigenMatrix X_test;
    EigenVector Y_test;
};

// Converts EigenVector to MatrixXf with shape (n, 1)
static EigenMatrix toMatrixXf(const EigenVector& vec) {
    return vec;
}

// Normalize features: zero-mean and unit-norm per column
static void normalizeMatrix(EigenMatrix& matrix) {
    matrix = matrix.rowwise() - matrix.colwise().mean();
    matrix = matrix.array().rowwise() / matrix.colwise().norm().array();
}

// Load CSV into EigenDataT and split into train/test sets
static EigenDataT load_csv_eigen(
    const std::string& filename,
    const std::string& target_column,
    float training_ratio = 0.8f // default 80% training
) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    EigenDataT data;
    std::string line;

    // Parse header
    if (!std::getline(file, line)) {
        throw std::runtime_error("Empty file or missing header.");
    }

    std::stringstream header_stream(line);
    std::string column;
    while (std::getline(header_stream, column, ',')) {
        data.header.push_back(column);
    }

    int target_index = -1;
    for (size_t i = 0; i < data.header.size(); ++i) {
        if (data.header[i] == target_column) {
            target_index = static_cast<int>(i);
            break;
        }
    }

    if (target_index == -1) {
        throw std::runtime_error("Target column '" + target_column + "' not found.");
    }

    std::vector<std::vector<float>> rows;

    // Read data
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<float> row;

        while (std::getline(ss, value, ',')) {
            row.push_back(value.empty() ? 0.0f : std::stof(value));
        }

        if (row.size() != data.header.size()) {
            throw std::runtime_error("Row size doesn't match header size.");
        }

        rows.push_back(std::move(row));
    }

    file.close();

    size_t num_rows = rows.size();
    size_t num_features = data.header.size() - 1;

    EigenMatrix X(num_rows, num_features);
    EigenVector Y(num_rows);

    for (size_t i = 0; i < num_rows; ++i) {
        int col_idx = 0;
        for (size_t j = 0; j < rows[i].size(); ++j) {
            if (static_cast<int>(j) == target_index) {
                Y(i) = rows[i][j];
            }
            else {
                X(i, col_idx++) = rows[i][j];
            }
        }
    }

    // Split into train/test
    size_t train_size = static_cast<size_t>(training_ratio * num_rows);
    data.X_train = X.topRows(train_size);
    data.Y_train = Y.head(train_size);
    data.X_test = X.bottomRows(num_rows - train_size);
    data.Y_test = Y.tail(num_rows - train_size);

    return data;
}

//one hot encode labels (for multiclass classification)
EigenMatrix toOneHot(EigenVector& labels, int num_labels){
    EigenMatrix one_hot {EigenMatrix::Zero(labels.rows(), num_labels)};
    for (std::size_t i {}; i<labels.rows(); ++i){
        one_hot(i, static_cast<int>(labels(i))) = 1.0;
    }
    return one_hot;
}

