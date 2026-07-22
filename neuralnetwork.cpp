#include "neuralnetwork.h"
#include <cmath>
#include "DatasetLoader.h"
#include <ctime>
#include <random>
#include <iostream>

NeuralNetwork::NeuralNetwork() {
    getRandomWeight();
    std::string basePath = "C:/Users/ghost/Documents/HandwrittenDigitRecognizer/Dataset/";
    bool imagesLoaded = DatasetLoader::loadImages(basePath + "train-images.idx3-ubyte", images);
    bool labelsLoaded = DatasetLoader::loadLabels(basePath + "train-labels.idx1-ubyte", labels);
    if (!imagesLoaded || !labelsLoaded) {
        std::cout << "CRITICAL: Failed to load dataset files!" << std::endl;
        return;
    }
    std::cout << "Training started..." << std::endl;
    train();
    std::cout << "Training finished! Running evaluation..." << std::endl;
    evaluate();
}

float NeuralNetwork::randomNumber(int inputSize) {
    static std::random_device rd;
    static std::mt19937 gen(rd());

    float stddev = std::sqrt(2.0 / inputSize);
    std::normal_distribution<float> d(0.0, stddev);
    return d(gen);
}

void NeuralNetwork::getRandomWeight() {
    // WIH: [784 (input)][128 (hidden1)]
    for (int i = 0; i < 784; i++) {
        for (int j = 0; j < 128; j++) {
            WIH[i][j] = randomNumber(784);
        }
    }
    // WHH: [128 (hidden1)][64 (hidden2)]
    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 64; j++) {
            WHH[i][j] = randomNumber(128);
        }
    }
    // WHO: [64 (hidden2)][10 (output)]
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 10; j++) {
            WHO[i][j] = randomNumber(64);
        }
    }

    for (int i = 0; i < 128; i++) {
        biasHidden1[i] = 0.0f;
    }
    for (int i = 0; i < 64; i++) {
        biasHidden2[i] = 0.0f;
    }
    for (int i = 0; i < 10; i++) {
        biasOutput[i] = 0.0f;
    }
}

float NeuralNetwork::ReLU(float x) {
    return (x > 0) ? x : 0;
}

void NeuralNetwork::train() {
    for (int i = 0; i < 60000; i++) {
        if (i % 10000 == 0) {
            std::cout << "Processed " << i << " training images..." << std::endl;
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
}

void NeuralNetwork::forward() {
    // Input -> Hidden 1 (WIH is [784][128])
    for (int i = 0; i < 128; i++) {
        hidden1Neuron[i] = 0;
        for (int j = 0; j < 784; j++) {
            hidden1Neuron[i] += WIH[j][i] * inputNeuron[j];
        }
        hidden1Neuron[i] += biasHidden1[i];
        hidden1Neuron[i] = ReLU(hidden1Neuron[i]);
    }

    // Hidden 1 -> Hidden 2 (WHH is [128][64])
    for (int i = 0; i < 64; i++) {
        hidden2Neuron[i] = 0;
        for (int j = 0; j < 128; j++) {
            hidden2Neuron[i] += WHH[j][i] * hidden1Neuron[j];
        }
        hidden2Neuron[i] += biasHidden2[i];
        hidden2Neuron[i] = ReLU(hidden2Neuron[i]);
    }

    // Hidden 2 -> Output (WHO is [64][10])
    float max_logit = -1e9;
    for (int i = 0; i < 10; i++) {
        outputNeuron[i] = 0;
        for (int j = 0; j < 64; j++) {
            outputNeuron[i] += WHO[j][i] * hidden2Neuron[j];
        }
        outputNeuron[i] += biasOutput[i];
        if (outputNeuron[i] > max_logit) {
            max_logit = outputNeuron[i];
        }
    }

    double sum_exp = 0;
    for (int i = 0; i < 10; i++) {
        outputNeuron[i] = std::exp(outputNeuron[i] - max_logit);
        sum_exp += outputNeuron[i];
    }
    for (int i = 0; i < 10; i++) {
        outputNeuron[i] = outputNeuron[i] / sum_exp;
    }
}

void NeuralNetwork::evaluate() {
    std::string basePath = "C:/Users/ghost/Documents/HandwrittenDigitRecognizer/Dataset/";

    std::vector<std::vector<float>> testImages;
    std::vector<int> testLabels;

    bool imagesLoaded = DatasetLoader::loadImages(basePath + "t10k-images.idx3-ubyte", testImages);
    bool labelsLoaded = DatasetLoader::loadLabels(basePath + "t10k-labels.idx1-ubyte", testLabels);

    if (!imagesLoaded || !labelsLoaded) {
        std::cout << "CRITICAL: Failed to load test dataset files for evaluation!" << std::endl;
        return;
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
}