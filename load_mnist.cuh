#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <Eigen/Dense>

struct MNISTData {
    EigenMatrix X_train;
    EigenMatrix Y_train;  // one-hot encoded
    EigenMatrix X_test;
    EigenMatrix Y_test;
};

inline int readInt(std::ifstream& stream) {
    unsigned char bytes[4];
    stream.read(reinterpret_cast<char*>(bytes), 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

inline void readImages(const std::string& filename, EigenMatrix& images, int& num_images, int& rows, int& cols) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open image file");

    int magic = readInt(file);
    num_images = readInt(file);
    rows = readInt(file);
    cols = readInt(file);

    images.resize(num_images, rows * cols);
    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < rows * cols; ++j) {
            unsigned char pixel;
            file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
            images(i, j) = static_cast<float>(pixel) / 255.0;  // normalize to [0,1]
        }
    }
}

inline void readLabels(const std::string& filename, EigenMatrix& labels, int num_labels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open label file");

    int magic = readInt(file);
    int total_labels = readInt(file);

    labels = EigenMatrix::Zero(num_labels, 10);
    for (int i = 0; i < num_labels; ++i) {
        unsigned char label;
        file.read(reinterpret_cast<char*>(&label), sizeof(label));
        labels(i, static_cast<int>(label)) = 1.0;  // one-hot
    }
}

inline MNISTData load_mnist(const std::string& train_img, const std::string& train_lbl,
    const std::string& test_img, const std::string& test_lbl) {
    MNISTData data;
    int train_size, test_size, rows, cols;

    readImages(train_img, data.X_train, train_size, rows, cols);
    readLabels(train_lbl, data.Y_train, train_size);

    readImages(test_img, data.X_test, test_size, rows, cols);
    readLabels(test_lbl, data.Y_test, test_size);

    return data;
}
