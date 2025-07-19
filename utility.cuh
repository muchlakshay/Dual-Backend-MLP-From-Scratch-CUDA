#pragma once

#include <Eigen/Dense>
#include <random>
#include "typedefs.cuh"


//some useful utilities
namespace utility {
    //shuffle training data -> X_train and Y_train
    static void shuffleData(EigenMatrix& X_train, EigenMatrix& Y_train) {

        //Set up PRNG
        std::random_device rd;
        std::mt19937 mt{rd()};

        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(X_train.rows());
        perm.setIdentity();
        //shuffle the indices
        shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size(), mt);

        //Shuffle
        X_train = perm * X_train;
        Y_train = perm * Y_train;
    }

    //Flatten A EigenMatrix Into A std::vector
    static std::vector<float> flatten(const EigenMatrix& mat) {
        std::vector<float> flat(mat.rows() * mat.cols());   //vector to store flattened matrix

        //flatten
        for (int i{}; i < mat.rows(); ++i) {
            for (int j{}; j < mat.cols(); ++j) flat[i * mat.cols() + j] = mat(i, j);
        }
        return flat; //return the vector
    }

   //convert a EigenVector into std::vector
   static std::vector<float> flatten(const EigenVector& vec) {
        std::vector<float> flat(vec.size()); //vector to store the EigenVector elements
        for (int i{}; i < vec.size(); ++i) {
            flat[i] = vec(i); //store
        }
        return flat; //returns the vector
    }

}
