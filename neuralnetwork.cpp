#include "neuralnetwork.h"
#include <cmath>
#include "DatasetLoader.h"
#include <ctime>
#include <random>
#include <iostream>
#include <algorithm>

NeuralNetwork::NeuralNetwork() {
    getRandomWeight();
    std::string basePath = "C:/Users/ghost/Documents/HandwrittenDigitRecognizer/Dataset/";
    bool imagesLoaded = DatasetLoader::loadImages(basePath + "train-images.idx3-ubyte", images);
    bool labelsLoaded = DatasetLoader::loadLabels(basePath + "train-labels.idx1-ubyte", labels);
    if (!imagesLoaded || !labelsLoaded) {
        std::cout << "CRITICAL: Failed to load dataset files!" << std::endl;
        return;
    }

}

float NeuralNetwork::randomNumber(int inputSize) {
    static std::random_device rd;
    static std::mt19937 gen(rd());

    float stddev = std::sqrt(2.0f / inputSize);
    std::normal_distribution<float> d(0.0f, stddev);
    return d(gen);
}

void NeuralNetwork::getRandomWeight() {

    convWeights.resize(numFilters, std::vector<std::vector<float>>(filterSize, std::vector<float>(filterSize)));
    convBiases.resize(numFilters, 0.0f);

    for( int f =0 ; f< numFilters;f++){
        for( int i =0; i<filterSize; i++){
            for( int j=0 ; j<filterSize ; j++){
                convWeights[f][i][j]=randomNumber(filterSize * filterSize);
            }
        }
    }

    int flatSize = numFilters * poolOutputDim * poolOutputDim; // 16 x 13 X 13 [2704 inputs][10 outputs]
    denseWeights.resize(flatSize, std::vector<float>(10,0.0f));
    denseBiases.resize(10, 0.0f);

    for( int i =0 ; i< flatSize ; i++){
        for( int j =0 ; j< 10 ;j++){
            denseWeights[i][j]=randomNumber(flatSize);
        }
    }
        outputNeuron.resize(10 , 0.0f);

}

inline float NeuralNetwork::ReLU(float x) {
    return (x > 0) ? x : 0;
}

void NeuralNetwork::forward(const std::vector<float>& inputImage) {

    //Reshape the input to from a 1D to 2D (28 x 28)
    lastInput2D.assign(28, std::vector<float>(28));
    for( int i=0; i< 28 ; i++){
        for ( int j=0 ; j< 28 ; j++){
            lastInput2D[i][i]= inputImage[i*28+j];
        }
    }

    // Convolution Layer (28 x 28 -> 16 feature maps of 26 x 26)
    lastConvOutput.assign(numFilters, std::vector<std::vector<float>>(convOutputDim, std::vector<float>(convOutputDim, 0.0f)));
    for(int f =0 ; f< numFilters; f++){
        for ( int i =0 ; i< convOutputDim ; i++){
            for( int j=0; j<convOutputDim ; j++){
                float sum =0.0f;
                for(int ki =0 ; ki < filterSize; ki++){
                    for ( int kj =0; kj < filterSize; kj++){
                        sum+=lastInput2D[i+ki][j+kj] * convWeights[f][ki][kj];
                    }
                }
                lastConvOutput[f][i][j] = ReLU(sum + convBiases[f]);
            }
        }
    }

    // Max Pooling Layer (2x2 pool window: 26x26 -> 13x13)



}

void NeuralNetwork::train(ProgressCallback onProgress) {
    std::cout << "Training..." << std::endl;

    int totalImages=images.size();
    if(totalImages==0) return;

    for (int i = 0; i < 60000; i++) {
        if (i % 10000 == 0) {
            std::cout << "Processed " << i << " training images..." << std::endl;
        }

        if(onProgress && (i%600 ==0 || i == totalImages-1)){
            float progress=static_cast<float>(i+1)/static_cast<float>(totalImages);
            onProgress(progress);
        }

        inputNeuron = images[i];
        forward();

        std::vector<float> output_deltas(10);
        std::vector<float> hidden2_deltas(64);
        std::vector<float> hidden1_deltas(128);

        // --- STEP 1: Compute all Deltas first ---

        // Output Deltas
        for (int j = 0; j < 10; j++) {
            float target_j = (labels[i] == j) ? 1.0f : 0.0f;
            output_deltas[j] = outputNeuron[j] - target_j;
        }

        // Hidden 2 Deltas (Using OLD WHO weights)
        for (int j = 0; j < 64; j++) {
            float error_sum = 0.0f;
            for (int k = 0; k < 10; k++) {
                error_sum += WHO[j][k] * output_deltas[k];
            }
            float relu_deriv = (hidden2Neuron[j] > 0.0f) ? 1.0f : 0.0f;
            hidden2_deltas[j] = error_sum * relu_deriv;
        }

        // Hidden 1 Deltas (Using OLD WHH weights)
        for (int h1 = 0; h1 < 128; h1++) {
            float error_sum = 0.0f;
            for (int j = 0; j < 64; j++) {
                error_sum += WHH[h1][j] * hidden2_deltas[j];
            }
            float relu_deriv = (hidden1Neuron[h1] > 0.0f) ? 1.0f : 0.0f;
            hidden1_deltas[h1] = error_sum * relu_deriv;
        }

        // --- STEP 2: Apply all weight updates ---

        // Update WHO & biasOutput
        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 64; k++) {
                float gradient = output_deltas[j] * hidden2Neuron[k];
                WHO[k][j] -= alpha * gradient;
            }
            biasOutput[j] -= alpha * output_deltas[j];
        }

        // Update WHH & biasHidden2
        for (int j = 0; j < 64; j++) {
            for (int h1 = 0; h1 < 128; h1++) {
                float gradient = hidden2_deltas[j] * hidden1Neuron[h1];
                WHH[h1][j] -= alpha * gradient;
            }
            biasHidden2[j] -= alpha * hidden2_deltas[j];
        }

        // Update WIH & biasHidden1
        for (int h1 = 0; h1 < 128; h1++) {
            for (int pixel = 0; pixel < 784; pixel++) {
                float gradient = hidden1_deltas[h1] * inputNeuron[pixel];
                WIH[pixel][h1] -= alpha * gradient;
            }
            biasHidden1[h1] -= alpha * hidden1_deltas[h1];
        }
    }
    std::cout << "Training Done!!!" << std::endl;

}



float NeuralNetwork::evaluate() {
    std::cout << "Running evaluation..." << std::endl;

    std::string basePath = "C:/Users/ghost/Documents/HandwrittenDigitRecognizer/Dataset/";

    std::vector<std::vector<float>> testImages;
    std::vector<int> testLabels;

    bool imagesLoaded = DatasetLoader::loadImages(basePath + "t10k-images.idx3-ubyte", testImages);
    bool labelsLoaded = DatasetLoader::loadLabels(basePath + "t10k-labels.idx1-ubyte", testLabels);

    if (!imagesLoaded || !labelsLoaded) {
        std::cout << "CRITICAL: Failed to load test dataset files for evaluation!" << std::endl;
        return 0;
    }

    int correctPredictions = 0;
    int totalTestSamples = testImages.size();

    std::cout << "Starting evaluation on " << totalTestSamples << " test images..." << std::endl;

    for (int i = 0; i < totalTestSamples; i++) {
        inputNeuron = testImages[i];
        forward();

        int predictedLabel = 0;
        float maxProbability = outputNeuron[0];

        for (int j = 1; j < 10; j++) {
            if (outputNeuron[j] > maxProbability) {
                maxProbability = outputNeuron[j];
                predictedLabel = j;
            }
        }

        if (predictedLabel == testLabels[i]) {
            correctPredictions++;
        }
    }

    float accuracy = (static_cast<float>(correctPredictions) / totalTestSamples) * 100.0f;
    std::cout << "Evaluation Complete!" << std::endl;
    std::cout << "Correct: " << correctPredictions << " / " << totalTestSamples << std::endl;
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;

    return accuracy;
}