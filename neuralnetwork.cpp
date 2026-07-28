#include "neuralnetwork.h"
#include <cmath>
#include "DatasetLoader.h"
#include <ctime>
#include <random>
#include <iostream>
#include <algorithm>
#include <numeric>          // for std::iota

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
    float stddev = std::sqrt(2.0f / inputSize);   // He initialization
    std::normal_distribution<float> d(0.0f, stddev);
    return d(gen);
}

void NeuralNetwork::getRandomWeight() {
    // Convolutional weights & biases
    convWeights.resize(numFilters, std::vector<std::vector<float>>(filterSize, std::vector<float>(filterSize)));
    convBiases.resize(numFilters, 0.01f);

    for (int f = 0; f < numFilters; f++) {
        for (int i = 0; i < filterSize; i++) {
            for (int j = 0; j < filterSize; j++) {
                convWeights[f][i][j] = randomNumber(filterSize * filterSize);
            }
        }
    }

    // Dense weights & biases
    int flatSize = numFilters * poolOutputDim * poolOutputDim;
    denseWeights.resize(flatSize, std::vector<float>(10, 0.0f));
    denseBiases.resize(10, 0.01f);

    for (int i = 0; i < flatSize; i++) {
        for (int j = 0; j < 10; j++) {
            denseWeights[i][j] = randomNumber(flatSize);
        }
    }

    outputNeuron.resize(10, 0.0f);

    // Pre-allocate forward buffers
    lastInput2D.resize(28, std::vector<float>(28, 0.0f));
    lastConvOutput.resize(numFilters, std::vector<std::vector<float>>(convOutputDim, std::vector<float>(convOutputDim, 0.0f)));
    lastPoolOutput.resize(numFilters, std::vector<std::vector<float>>(poolOutputDim, std::vector<float>(poolOutputDim, 0.0f)));
    lastFlattened.resize(flatSize, 0.0f);

    // Pre-allocate backward buffers
    output_deltas.resize(10, 0.0f);
    flattened_deltas.resize(flatSize, 0.0f);
    pool_deltas.resize(numFilters, std::vector<std::vector<float>>(poolOutputDim, std::vector<float>(poolOutputDim, 0.0f)));
    conv_deltas.resize(numFilters, std::vector<std::vector<float>>(convOutputDim, std::vector<float>(convOutputDim, 0.0f)));
    filter_grads.resize(filterSize, std::vector<float>(filterSize, 0.0f));
}

inline float NeuralNetwork::ReLU(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

void NeuralNetwork::forward(const std::vector<float>& inputImage) {
    // 1. Reshape + normalize to [0, 1]
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            lastInput2D[i][j] = inputImage[i * 28 + j] / 255.0f;
        }
    }

    // 2. Convolution + ReLU
    for (int f = 0; f < numFilters; f++) {
        for (int i = 0; i < convOutputDim; i++) {
            for (int j = 0; j < convOutputDim; j++) {
                float sum = 0.0f;
                for (int ki = 0; ki < filterSize; ki++) {
                    for (int kj = 0; kj < filterSize; kj++) {
                        sum += lastInput2D[i + ki][j + kj] * convWeights[f][ki][kj];
                    }
                }
                lastConvOutput[f][i][j] = ReLU(sum + convBiases[f]);
            }
        }
    }

    // 3. 2×2 Max Pooling
    for (int f = 0; f < numFilters; f++) {
        for (int i = 0; i < poolOutputDim; i++) {
            for (int j = 0; j < poolOutputDim; j++) {
                float maxVal = -1e9f;
                for (int pi = 0; pi < 2; pi++) {
                    for (int pj = 0; pj < 2; pj++) {
                        float val = lastConvOutput[f][i * 2 + pi][j * 2 + pj];
                        if (val > maxVal) maxVal = val;
                    }
                }
                lastPoolOutput[f][i][j] = maxVal;
            }
        }
    }

    // 4. Flatten
    int idx = 0;
    for (int f = 0; f < numFilters; f++) {
        for (int i = 0; i < poolOutputDim; i++) {
            for (int j = 0; j < poolOutputDim; j++) {
                lastFlattened[idx++] = lastPoolOutput[f][i][j];
            }
        }
    }

    // 5. Dense + stable Softmax
    float max_logit = -1e9f;
    int flatSize = numFilters * poolOutputDim * poolOutputDim;

    for (int i = 0; i < 10; i++) {
        float logit = denseBiases[i];
        for (int j = 0; j < flatSize; j++) {
            logit += lastFlattened[j] * denseWeights[j][i];
        }
        outputNeuron[i] = logit;
        if (logit > max_logit) max_logit = logit;
    }

    double sum_exp = 0.0;
    for (int i = 0; i < 10; i++) {
        outputNeuron[i] = static_cast<float>(std::exp(outputNeuron[i] - max_logit));
        sum_exp += outputNeuron[i];
    }
    for (int i = 0; i < 10; i++) {
        outputNeuron[i] /= static_cast<float>(sum_exp);
    }
}

void NeuralNetwork::train(ProgressCallback onProgress) {
    std::cout << "Training..." << std::endl;
    int totalImages = static_cast<int>(images.size());
    if (totalImages == 0) return;

    int flatSize = numFilters * poolOutputDim * poolOutputDim;

    // Indices for shuffling
    std::vector<int> indices(totalImages);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(std::random_device{}());

    for (int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "Starting Epoch " << (epoch + 1) << "/" << epochs << std::endl;

        // Shuffle every epoch
        std::shuffle(indices.begin(), indices.end(), rng);

        for (int t = 0; t < totalImages; t++) {
            int i = indices[t];

            if (t % 10000 == 0 && t > 0) {
                std::cout << "Processed " << t << " training images..." << std::endl;
            }

            if (onProgress && (t % 600 == 0 || t == totalImages - 1)) {
                float progress = static_cast<float>((t + 1) + (epoch * totalImages))
                / static_cast<float>(totalImages * epochs);
                onProgress(progress);
            }

            forward(images[i]);

            // Softmax + Cross-Entropy deltas
            for (int j = 0; j < 10; j++) {
                float target = (labels[i] == j) ? 1.0f : 0.0f;
                output_deltas[j] = outputNeuron[j] - target;
            }

            // Dense layer back-prop + update (with L2)
            for (int j = 0; j < flatSize; j++) {
                float error_sum = 0.0f;
                for (int k = 0; k < 10; k++) {
                    float grad = output_deltas[k] * lastFlattened[j];
                    error_sum += denseWeights[j][k] * output_deltas[k];
                    denseWeights[j][k] -= alpha * (grad + l2 * denseWeights[j][k]);
                }
                flattened_deltas[j] = error_sum;
            }

            for (int k = 0; k < 10; k++) {
                denseBiases[k] -= alpha * output_deltas[k];
            }

            // Reshape flattened deltas → pool shape
            int idx = 0;
            for (int f = 0; f < numFilters; f++) {
                for (int r = 0; r < poolOutputDim; r++) {
                    for (int c = 0; c < poolOutputDim; c++) {
                        pool_deltas[f][r][c] = flattened_deltas[idx++];
                    }
                }
            }

            // Zero conv deltas (max-pool is sparse)
            for (int f = 0; f < numFilters; f++) {
                for (int r = 0; r < convOutputDim; r++) {
                    std::fill(conv_deltas[f][r].begin(), conv_deltas[f][r].end(), 0.0f);
                }
            }

            // Route max-pool errors to the winning locations
            for (int f = 0; f < numFilters; f++) {
                for (int r = 0; r < poolOutputDim; r++) {
                    for (int c = 0; c < poolOutputDim; c++) {
                        float maxVal = -1e9f;
                        int best_pi = 0, best_pj = 0;
                        for (int pi = 0; pi < 2; pi++) {
                            for (int pj = 0; pj < 2; pj++) {
                                float val = lastConvOutput[f][r * 2 + pi][c * 2 + pj];
                                if (val > maxVal) {
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
            for (int f = 0; f < numFilters; f++) {
                for (int r = 0; r < convOutputDim; r++) {
                    for (int c = 0; c < convOutputDim; c++) {
                        if (lastConvOutput[f][r][c] <= 0.0f) {
                            conv_deltas[f][r][c] = 0.0f;
                        }
                    }
                }
            }

            // Update convolutional filters & biases
            for (int f = 0; f < numFilters; f++) {
                float bias_grad_sum = 0.0f;

                // Zero temporary filter gradient
                for (int ki = 0; ki < filterSize; ki++) {
                    std::fill(filter_grads[ki].begin(), filter_grads[ki].end(), 0.0f);
                }

                // Accumulate gradients over the spatial map
                for (int r = 0; r < convOutputDim; r++) {
                    for (int c = 0; c < convOutputDim; c++) {
                        float delta = conv_deltas[f][r][c];
                        bias_grad_sum += delta;
                        for (int ki = 0; ki < filterSize; ki++) {
                            for (int kj = 0; kj < filterSize; kj++) {
                                filter_grads[ki][kj] += delta * lastInput2D[r + ki][c + kj];
                            }
                        }
                    }
                }

                // Apply accumulated gradients once
                for (int ki = 0; ki < filterSize; ki++) {
                    for (int kj = 0; kj < filterSize; kj++) {
                        convWeights[f][ki][kj] -= alpha * filter_grads[ki][kj];
                    }
                }
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
        return 0.0f;
    }

    int correctPredictions = 0;
    int totalTestSamples = static_cast<int>(testImages.size());
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