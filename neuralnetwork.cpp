#include "neuralnetwork.h"
#include <cmath>
#include "DatasetLoader.h"
#include <ctime>
#include <random>
#include <iostream>
#include <algorithm>
#include <numeric> // Added for std::iota

NeuralNetwork::NeuralNetwork() {
    // Note: Dimensions like numFilters, filterSize, convOutputDim, poolOutputDim
    // must be initialized before calling getRandomWeight().
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
    // 1. Initialize Network Weights
    convWeights.resize(numFilters, std::vector<std::vector<float>>(filterSize, std::vector<float>(filterSize)));
    convBiases.resize(numFilters, 0.01f); // Initialize with small positive bias to prevent dead ReLUs

    for(int f = 0; f < numFilters; f++) {
        for(int i = 0; i < filterSize; i++) {
            for(int j = 0; j < filterSize; j++) {
                convWeights[f][i][j] = randomNumber(filterSize * filterSize);
            }
        }
    }

    int flatSize = numFilters * poolOutputDim * poolOutputDim;
    denseWeights.resize(flatSize, std::vector<float>(10, 0.0f));
    denseBiases.resize(10, 0.01f);

    for(int i = 0; i < flatSize; i++) {
        for(int j = 0; j < 10; j++) {
            denseWeights[i][j] = randomNumber(flatSize);
        }
    }

    outputNeuron.resize(10, 0.0f);

    // 2. Pre-allocate ALL Forward Pass memory (No heap allocations in loop)
    lastInput2D.resize(28, std::vector<float>(28, 0.0f));
    lastConvOutput.resize(numFilters, std::vector<std::vector<float>>(convOutputDim, std::vector<float>(convOutputDim, 0.0f)));
    lastPoolOutput.resize(numFilters, std::vector<std::vector<float>>(poolOutputDim, std::vector<float>(poolOutputDim, 0.0f)));
    lastFlattened.resize(flatSize, 0.0f);

    // 3. Pre-allocate ALL Backward Pass memory (No heap allocations in loop)
    output_deltas.resize(10, 0.0f);
    flattened_deltas.resize(flatSize, 0.0f);
    pool_deltas.resize(numFilters, std::vector<std::vector<float>>(poolOutputDim, std::vector<float>(poolOutputDim, 0.0f)));
    conv_deltas.resize(numFilters, std::vector<std::vector<float>>(convOutputDim, std::vector<float>(convOutputDim, 0.0f)));
    filter_grads.resize(filterSize, std::vector<float>(filterSize, 0.0f));
}

inline float NeuralNetwork::ReLU(float x) {
    return (x > 0) ? x : 0;
}

void NeuralNetwork::forward(const std::vector<float>& inputImage) {
    // 1. Reshape 1D to 2D (28 x 28) AND Normalize pixels to [0, 1]
    for(int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            lastInput2D[i][j] = inputImage[i * 28 + j] / 255.0f;
        }
    }

    // 2. Convolution Layer
    for(int f = 0; f < numFilters; f++) {
        for (int i = 0; i < convOutputDim; i++) {
            for(int j = 0; j < convOutputDim; j++) {
                float sum = 0.0f;
                for(int ki = 0; ki < filterSize; ki++) {
                    for (int kj = 0; kj < filterSize; kj++) {
                        sum += lastInput2D[i + ki][j + kj] * convWeights[f][ki][kj];
                    }
                }
                lastConvOutput[f][i][j] = ReLU(sum + convBiases[f]);
            }
        }
    }

    // 3. Max Pooling Layer (2x2)
    for(int f = 0; f < numFilters; f++) {
        for (int i = 0; i < poolOutputDim; i++) {
            for (int j = 0; j < poolOutputDim; j++) {
                float maxVal = -1e9f;
                for(int pi = 0; pi < 2; pi++) {
                    for(int pj = 0; pj < 2; pj++) {
                        float val = lastConvOutput[f][i * 2 + pi][j * 2 + pj];
                        if (val > maxVal) maxVal = val;
                    }
                }
                lastPoolOutput[f][i][j] = maxVal;
            }
        }
    }

    // 4. Flattening Layer
    int idx = 0;
    for(int f = 0; f < numFilters; f++) {
        for (int i = 0; i < poolOutputDim; i++) {
            for(int j = 0; j < poolOutputDim; j++) {
                lastFlattened[idx++] = lastPoolOutput[f][i][j];
            }
        }
    }

    // 5. Dense Layer + Softmax Output
    float max_logit = -1e9f;
    int flatSize = numFilters * poolOutputDim * poolOutputDim;

    // Calculate logits (dot product)
    for(int i = 0; i < 10; i++) {
        float logit = 0.0f;
        for(int j = 0; j < flatSize; j++) {
            logit += lastFlattened[j] * denseWeights[j][i];
        }
        logit += denseBiases[i];
        outputNeuron[i] = logit;

        if (logit > max_logit) {
            max_logit = logit;
        }
    }

    // Compute stable softmax probabilities
    double sum_exp = 0.0;
    for(int i = 0; i < 10; i++) {
        outputNeuron[i] = std::exp(outputNeuron[i] - max_logit);
        sum_exp += outputNeuron[i];
    }
    for(int i = 0; i < 10; i++) {
        outputNeuron[i] = outputNeuron[i] / sum_exp;
    }
}

void NeuralNetwork::train(ProgressCallback onProgress) {
    std::cout << "Training..." << std::endl;

    int totalImages = images.size();
    if(totalImages == 0) return;

    int flatSize = numFilters * poolOutputDim * poolOutputDim;

    // Random number generator for shuffling
    std::random_device rd;
    std::mt19937 rng(rd());

    for(int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "Starting Epoch " << epoch + 1 << "/" << epochs << std::endl;

        // Shuffle indices at the start of each epoch
        std::vector<int> indices(totalImages);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        for (int step = 0; step < totalImages; step++) {
            int i = indices[step]; // Use the shuffled index

            if (step % 10000 == 0 && step > 0) {
                std::cout << "Processed " << step << " training images..." << std::endl;
            }

            if(onProgress && (step % 600 == 0 || step == totalImages - 1)) {
                float progress = static_cast<float>((step + 1) + (epoch * totalImages)) / static_cast<float>(totalImages * epochs);
                onProgress(progress);
            }

            forward(images[i]);

            // Output Deltas (Softmax cross-entropy derivative)
            for(int j = 0; j < 10; j++) {
                float target_j = (labels[i] == j) ? 1.0f : 0.0f;
                output_deltas[j] = outputNeuron[j] - target_j;
            }

            // Backpropagate to Dense Layer & Update Weights/Biases
            for(int j = 0; j < flatSize; j++) {
                float error_sum = 0.0f;
                for(int k = 0; k < 10; k++) {
                    float grad = output_deltas[k] * lastFlattened[j];

                    // Apply L2 regularization
                    grad += l2 * denseWeights[j][k];

                    // Accumulate error for the layer below BEFORE updating the weight
                    error_sum += denseWeights[j][k] * output_deltas[k];

                    // Update weight (clipping removed)
                    denseWeights[j][k] -= alpha * grad;
                }
                flattened_deltas[j] = error_sum;
            }

            for(int k = 0; k < 10; k++) {
                float bias_grad = output_deltas[k];
                // Update bias (clipping removed)
                denseBiases[k] -= alpha * bias_grad;
            }

            // Reshape Flattened Deltas back to MaxPool Shape
            int idx = 0;
            for(int f = 0; f < numFilters; f++) {
                for(int r = 0; r < poolOutputDim; r++) {
                    for(int c = 0; c < poolOutputDim; c++) {
                        pool_deltas[f][r][c] = flattened_deltas[idx++];
                    }
                }
            }

            // CRITICAL: Zero out conv_deltas because max pooling route is sparse
            for(int f = 0; f < numFilters; f++) {
                for(int r = 0; r < convOutputDim; r++) {
                    std::fill(conv_deltas[f][r].begin(), conv_deltas[f][r].end(), 0.0f);
                }
            }

            // Route MaxPool errors back to winning locations
            for(int f = 0; f < numFilters; f++) {
                for(int r = 0; r < poolOutputDim; r++) {
                    for(int c = 0; c < poolOutputDim; c++) {
                        float maxVal = -1e9f;
                        int best_pi = 0;
                        int best_pj = 0;
                        for(int pi = 0; pi < 2; pi++) {
                            for(int pj = 0; pj < 2; pj++) {
                                float val = lastConvOutput[f][r * 2 + pi][c * 2 + pj];
                                if(val > maxVal) {
                                    maxVal = val;
                                    best_pi = pi;
                                    best_pj = pj;
                                }
                            }
                        }
                        conv_deltas[f][r * 2 + best_pi][c * 2 + best_pj] = pool_deltas[f][r][c];
                    }
                }
            }

            // ReLU derivative
            for(int f = 0; f < numFilters; f++) {
                for(int r = 0; r < convOutputDim; r++) {
                    for(int c = 0; c < convOutputDim; c++) {
                        if(lastConvOutput[f][r][c] <= 0.0f) {
                            conv_deltas[f][r][c] = 0.0f;
                        }
                    }
                }
            }

            // Update Convolution Filters and Biases
            for(int f = 0; f < numFilters; f++) {
                float bias_grad_sum = 0.0f;

                // 1. Zero out the temporary gradient accumulator for this filter
                for(int ki = 0; ki < filterSize; ki++) {
                    std::fill(filter_grads[ki].begin(), filter_grads[ki].end(), 0.0f);
                }

                // 2. Sum up the gradients across the entire spatial map
                for(int r = 0; r < convOutputDim; r++) {
                    for(int c = 0; c < convOutputDim; c++) {
                        float delta = conv_deltas[f][r][c];
                        bias_grad_sum += delta;

                        for(int ki = 0; ki < filterSize; ki++) {
                            for(int kj = 0; kj < filterSize; kj++) {
                                float input_val = lastInput2D[r + ki][c + kj];
                                filter_grads[ki][kj] += delta * input_val;
                            }
                        }
                    }
                }

                // 3. Apply the accumulated gradient ONCE per image
                for(int ki = 0; ki < filterSize; ki++) {
                    for(int kj = 0; kj < filterSize; kj++) {
                        float w_grad = filter_grads[ki][kj];
                        // Update filter weight (clipping removed)
                        convWeights[f][ki][kj] -= alpha * w_grad;
                    }
                }

                // Apply accumulated bias gradient (clipping removed)
                convBiases[f] -= alpha * bias_grad_sum;
            }
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
        forward(testImages[i]);

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