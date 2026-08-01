#include "neuralnetwork.h"
#include <cmath>
#include "DatasetLoader.h"
#include <ctime>
#include <random>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <QtConcurrent/QtConcurrentMap>
#include <QThread>

NeuralNetwork::NeuralNetwork(bool loadTrainingData) {
    // Note: Dimensions like numFilters, filterSize, convOutputDim, poolOutputDim
    // must be initialized before calling getRandomWeight().
    getRandomWeight();

    if (!loadTrainingData) {
        return;
    }

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
    denseWeights.resize(flatSize, std::vector<float>(hiddenSize, 0.0f));
    denseBiases.resize(hiddenSize, 0.01f);
    outputWeights.resize(hiddenSize, std::vector<float>(10, 0.0f));
    outputBiases.resize(10, 0.01f);

    for(int i = 0; i < flatSize; i++) {
        for(int j = 0; j < hiddenSize; j++) {
            denseWeights[i][j] = randomNumber(flatSize);
        }
    }
    for (int i = 0; i < hiddenSize; ++i) {
        for (int j = 0; j < 10; ++j) {
            outputWeights[i][j] = randomNumber(hiddenSize);
        }
    }

    outputNeuron.resize(10, 0.0f);

    // 2. Pre-allocate ALL Forward Pass memory (No heap allocations in loop)
    lastInput2D.resize(28, std::vector<float>(28, 0.0f));
    lastConvOutput.resize(numFilters, std::vector<std::vector<float>>(convOutputDim, std::vector<float>(convOutputDim, 0.0f)));
    lastPoolOutput.resize(numFilters, std::vector<std::vector<float>>(poolOutputDim, std::vector<float>(poolOutputDim, 0.0f)));
    lastFlattened.resize(flatSize, 0.0f);
    lastHidden.resize(hiddenSize, 0.0f);
    augmentedInput.resize(28 * 28, 0.0f);
    boldenedInput.resize(28 * 28, 0.0f);

    // 3. Pre-allocate ALL Backward Pass memory (No heap allocations in loop)
    output_deltas.resize(10, 0.0f);
    hidden_deltas.resize(hiddenSize, 0.0f);
    flattened_deltas.resize(flatSize, 0.0f);
    pool_deltas.resize(numFilters, std::vector<std::vector<float>>(poolOutputDim, std::vector<float>(poolOutputDim, 0.0f)));
    conv_deltas.resize(numFilters, std::vector<std::vector<float>>(convOutputDim, std::vector<float>(convOutputDim, 0.0f)));
    filter_grads.resize(filterSize, std::vector<float>(filterSize, 0.0f));
}

inline float NeuralNetwork::ReLU(float x) {
    return (x > 0) ? x : 0;
}

void NeuralNetwork::forward(const std::vector<float>& inputImage) {
    // All callers supply normalized pixels in the [0, 1] range.  The IDX
    // loader performs this conversion once, so doing it again here would make
    // every image nearly black and prevents the model from learning.
    for(int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            lastInput2D[i][j] = std::clamp(inputImage[i * 28 + j], 0.0f, 1.0f);
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

    // 5. Non-linear hidden layer + softmax output
    float max_logit = -1e9f;
    int flatSize = numFilters * poolOutputDim * poolOutputDim;

    for (int i = 0; i < hiddenSize; ++i) {
        float logit = 0.0f;
        for (int j = 0; j < flatSize; ++j) {
            logit += lastFlattened[j] * denseWeights[j][i];
        }
        logit += denseBiases[i];
        lastHidden[i] = ReLU(logit);
    }

    for(int i = 0; i < 10; i++) {
        float logit = outputBiases[i];
        for (int j = 0; j < hiddenSize; ++j) {
            logit += lastHidden[j] * outputWeights[j][i];
        }
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

void NeuralNetwork::trainOne(const std::vector<float>& inputImage, int label) {
    forward(inputImage);
    const int flatSize = numFilters * poolOutputDim * poolOutputDim;

    for (int j = 0; j < 10; ++j) {
        output_deltas[j] = outputNeuron[j] - (label == j ? 1.0f : 0.0f);
    }

    for (int j = 0; j < hiddenSize; ++j) {
        float errorSum = 0.0f;
        for (int k = 0; k < 10; ++k) {
            const float oldWeight = outputWeights[j][k];
            errorSum += oldWeight * output_deltas[k];
            outputWeights[j][k] -= alpha * (output_deltas[k] * lastHidden[j] + l2 * oldWeight);
        }
        hidden_deltas[j] = lastHidden[j] > 0.0f ? errorSum : 0.0f;
    }

    for (int k = 0; k < 10; ++k) {
        outputBiases[k] -= alpha * output_deltas[k];
    }

    for (int j = 0; j < flatSize; ++j) {
        float errorSum = 0.0f;
        for (int k = 0; k < hiddenSize; ++k) {
            const float oldWeight = denseWeights[j][k];
            errorSum += oldWeight * hidden_deltas[k];
            denseWeights[j][k] -= alpha * (hidden_deltas[k] * lastFlattened[j] + l2 * oldWeight);
        }
        flattened_deltas[j] = errorSum;
    }

    for (int k = 0; k < hiddenSize; ++k) {
        denseBiases[k] -= alpha * hidden_deltas[k];
    }

    int index = 0;
    for (int f = 0; f < numFilters; ++f) {
        for (int r = 0; r < poolOutputDim; ++r) {
            for (int c = 0; c < poolOutputDim; ++c) {
                pool_deltas[f][r][c] = flattened_deltas[index++];
            }
        }
    }

    for (int f = 0; f < numFilters; ++f) {
        for (int r = 0; r < convOutputDim; ++r) {
            std::fill(conv_deltas[f][r].begin(), conv_deltas[f][r].end(), 0.0f);
        }
    }

    for (int f = 0; f < numFilters; ++f) {
        for (int r = 0; r < poolOutputDim; ++r) {
            for (int c = 0; c < poolOutputDim; ++c) {
                int bestRow = 0;
                int bestColumn = 0;
                float maxValue = -1e9f;
                for (int pi = 0; pi < 2; ++pi) {
                    for (int pj = 0; pj < 2; ++pj) {
                        const float value = lastConvOutput[f][r * 2 + pi][c * 2 + pj];
                        if (value > maxValue) {
                            maxValue = value;
                            bestRow = pi;
                            bestColumn = pj;
                        }
                    }
                }
                conv_deltas[f][r * 2 + bestRow][c * 2 + bestColumn] = pool_deltas[f][r][c];
            }
        }
    }

    for (int f = 0; f < numFilters; ++f) {
        for (int r = 0; r < convOutputDim; ++r) {
            for (int c = 0; c < convOutputDim; ++c) {
                if (lastConvOutput[f][r][c] <= 0.0f) {
                    conv_deltas[f][r][c] = 0.0f;
                }
            }
        }
    }

    for (int f = 0; f < numFilters; ++f) {
        float biasGradient = 0.0f;
        for (int ki = 0; ki < filterSize; ++ki) {
            std::fill(filter_grads[ki].begin(), filter_grads[ki].end(), 0.0f);
        }

        for (int r = 0; r < convOutputDim; ++r) {
            for (int c = 0; c < convOutputDim; ++c) {
                const float delta = conv_deltas[f][r][c];
                biasGradient += delta;
                for (int ki = 0; ki < filterSize; ++ki) {
                    for (int kj = 0; kj < filterSize; ++kj) {
                        filter_grads[ki][kj] += delta * lastInput2D[r + ki][c + kj];
                    }
                }
            }
        }

        for (int ki = 0; ki < filterSize; ++ki) {
            for (int kj = 0; kj < filterSize; ++kj) {
                convWeights[f][ki][kj] -= alpha * filter_grads[ki][kj];
            }
        }
        convBiases[f] -= alpha * biasGradient;
    }
}

void NeuralNetwork::trainOneAugmented(const std::vector<float>& inputImage, int label,
                                      int shiftX, int shiftY, float angleDegrees, float scale,
                                      bool boldenStroke) {
    if (shiftX == 0 && shiftY == 0 && angleDegrees == 0.0f && scale == 1.0f && !boldenStroke) {
        trainOne(inputImage, label);
        return;
    }

    // A hand-drawn digit is rarely positioned, sized, or slanted exactly like
    // an MNIST scan.  Create an inexpensive affine variation for each training
    // example while keeping the network's 28x28 input contract unchanged.
    std::fill(augmentedInput.begin(), augmentedInput.end(), 0.0f);
    constexpr float center = 13.5f;
    constexpr float pi = 3.14159265358979323846f;
    const float radians = angleDegrees * pi / 180.0f;
    const float cosine = std::cos(radians);
    const float sine = std::sin(radians);

    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            const float destinationX = static_cast<float>(x) - center - shiftX;
            const float destinationY = static_cast<float>(y) - center - shiftY;
            const float sourceX = (cosine * destinationX + sine * destinationY) / scale + center;
            const float sourceY = (-sine * destinationX + cosine * destinationY) / scale + center;

            if (sourceX < 0.0f || sourceX >= 27.0f || sourceY < 0.0f || sourceY >= 27.0f) {
                continue;
            }

            const int left = static_cast<int>(sourceX);
            const int top = static_cast<int>(sourceY);
            const float horizontal = sourceX - left;
            const float vertical = sourceY - top;
            const float topValue = inputImage[top * 28 + left] * (1.0f - horizontal)
                + inputImage[top * 28 + left + 1] * horizontal;
            const float bottomValue = inputImage[(top + 1) * 28 + left] * (1.0f - horizontal)
                + inputImage[(top + 1) * 28 + left + 1] * horizontal;
            augmentedInput[y * 28 + x] = topValue * (1.0f - vertical) + bottomValue * vertical;
        }
    }
    if (!boldenStroke) {
        trainOne(augmentedInput, label);
        return;
    }

    // A canvas pen produces wider, cleaner strokes than many scanned MNIST
    // samples.  Train a quarter of examples with a one-pixel max dilation so
    // the classifier recognises both stroke weights.
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            float maximum = 0.0f;
            for (int offsetY = -1; offsetY <= 1; ++offsetY) {
                const int sourceY = y + offsetY;
                if (sourceY < 0 || sourceY >= 28) {
                    continue;
                }
                for (int offsetX = -1; offsetX <= 1; ++offsetX) {
                    const int sourceX = x + offsetX;
                    if (sourceX >= 0 && sourceX < 28) {
                        maximum = std::max(maximum, augmentedInput[sourceY * 28 + sourceX]);
                    }
                }
            }
            boldenedInput[y * 28 + x] = maximum;
        }
    }
    trainOne(boldenedInput, label);
}

void NeuralNetwork::copyParametersFrom(const NeuralNetwork& source) {
    convWeights = source.convWeights;
    convBiases = source.convBiases;
    denseWeights = source.denseWeights;
    denseBiases = source.denseBiases;
    outputWeights = source.outputWeights;
    outputBiases = source.outputBiases;
}

void NeuralNetwork::mergeWorkerUpdates(const std::vector<NeuralNetwork>& workers,
                                       int activeWorkerCount) {
    // Each worker begins with this model.  Summing its parameter deltas applies
    // the same first-order update magnitude as processing all batch samples
    // serially, while calculations themselves run on separate CPU cores.
    for (int f = 0; f < numFilters; ++f) {
        for (int ki = 0; ki < filterSize; ++ki) {
            for (int kj = 0; kj < filterSize; ++kj) {
                float merged = -(activeWorkerCount - 1) * convWeights[f][ki][kj];
                for (int worker = 0; worker < activeWorkerCount; ++worker) {
                    merged += workers[worker].convWeights[f][ki][kj];
                }
                convWeights[f][ki][kj] = merged;
            }
        }
        float mergedBias = -(activeWorkerCount - 1) * convBiases[f];
        for (int worker = 0; worker < activeWorkerCount; ++worker) {
            mergedBias += workers[worker].convBiases[f];
        }
        convBiases[f] = mergedBias;
    }

    for (std::size_t j = 0; j < denseWeights.size(); ++j) {
        for (int k = 0; k < hiddenSize; ++k) {
            float merged = -(activeWorkerCount - 1) * denseWeights[j][k];
            for (int worker = 0; worker < activeWorkerCount; ++worker) {
                merged += workers[worker].denseWeights[j][k];
            }
            denseWeights[j][k] = merged;
        }
    }
    for (int k = 0; k < hiddenSize; ++k) {
        float merged = -(activeWorkerCount - 1) * denseBiases[k];
        for (int worker = 0; worker < activeWorkerCount; ++worker) {
            merged += workers[worker].denseBiases[k];
        }
        denseBiases[k] = merged;
    }

    for (int j = 0; j < hiddenSize; ++j) {
        for (int k = 0; k < 10; ++k) {
            float merged = -(activeWorkerCount - 1) * outputWeights[j][k];
            for (int worker = 0; worker < activeWorkerCount; ++worker) {
                merged += workers[worker].outputWeights[j][k];
            }
            outputWeights[j][k] = merged;
        }
    }
    for (int k = 0; k < 10; ++k) {
        float merged = -(activeWorkerCount - 1) * outputBiases[k];
        for (int worker = 0; worker < activeWorkerCount; ++worker) {
            merged += workers[worker].outputBiases[k];
        }
        outputBiases[k] = merged;
    }
}

void NeuralNetwork::fineTune(const std::vector<float>& inputImage, int label) {
    if (inputImage.size() != 28 * 28 || label < 0 || label > 9) {
        return;
    }

    // A labelled canvas sample is far more valuable for this app than another
    // generic MNIST sample.  Learn it with gentle variants, so one "teach"
    // action helps the nearby ways the same person writes that digit.
    constexpr int variations = 24;
    for (int pass = 0; pass < variations; ++pass) {
        const unsigned int seed = static_cast<unsigned int>((pass + 1) * 747796405u)
            + static_cast<unsigned int>(label * 2891336453u);
        const int shiftX = static_cast<int>(seed % 3u) - 1;
        const int shiftY = static_cast<int>((seed / 3u) % 3u) - 1;
        const float angle = static_cast<float>((seed / 9u) % 13u) - 6.0f;
        const float scale = 0.95f + 0.025f * static_cast<float>((seed / 117u) % 5u);
        const bool boldenStroke = ((seed / 585u) % 3u) == 0u;
        trainOneAugmented(inputImage, label, shiftX, shiftY, angle, scale, boldenStroke);
    }
}

void NeuralNetwork::train(ProgressCallback onProgress) {
    const int totalImages = static_cast<int>(images.size());
    if (totalImages == 0 || labels.size() != images.size()) {
        std::cerr << "Training data is missing or inconsistent." << std::endl;
        return;
    }

    const int idealThreadCount = QThread::idealThreadCount();
    const int workerCount = std::max(1, std::min(8, idealThreadCount > 1 ? idealThreadCount - 1 : 1));
    constexpr int samplesPerWorker = 16;
    std::cout << "Training with " << workerCount << " worker threads." << std::endl;

    std::vector<NeuralNetwork> workers;
    workers.reserve(workerCount);
    for (int worker = 0; worker < workerCount; ++worker) {
        workers.emplace_back(false);
    }

    std::vector<int> workerIndexes(workerCount);
    std::iota(workerIndexes.begin(), workerIndexes.end(), 0);
    std::random_device rd;
    std::mt19937 rng(rd());

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::vector<int> indices(totalImages);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        int processed = 0;
        while (processed < totalImages) {
            const int activeWorkers = std::min(workerCount,
                (totalImages - processed + samplesPerWorker - 1) / samplesPerWorker);
            const int batchStart = processed;

            for (int worker = 0; worker < activeWorkers; ++worker) {
                workers[worker].copyParametersFrom(*this);
            }

            QtConcurrent::blockingMap(workerIndexes.begin(), workerIndexes.begin() + activeWorkers,
                [this, &workers, &indices, batchStart, totalImages, epoch](int worker) {
                    const int begin = batchStart + worker * samplesPerWorker;
                    const int end = std::min(begin + samplesPerWorker, totalImages);
                    for (int position = begin; position < end; ++position) {
                        const int imageIndex = indices[position];
                        // Deterministic per-sample offsets avoid sharing a RNG
                        // between worker threads while covering every -2..2 shift.
                        const unsigned int seed = static_cast<unsigned int>(position * 1103515245u)
                            + static_cast<unsigned int>(epoch * 12345u);
                        const int shiftX = static_cast<int>(seed % 5u) - 2;
                        const int shiftY = static_cast<int>((seed / 5u) % 5u) - 2;
                        const float angle = static_cast<float>((seed / 25u) % 25u) - 12.0f;
                        const float scale = 0.90f + 0.05f * static_cast<float>((seed / 625u) % 5u);
                        const bool boldenStroke = ((seed / 3125u) % 4u) == 0u;
                        workers[worker].trainOneAugmented(images[imageIndex], labels[imageIndex],
                                                          shiftX, shiftY, angle, scale, boldenStroke);
                    }
                });

            int batchSize = 0;
            for (int worker = 0; worker < activeWorkers; ++worker) {
                batchSize += std::min(samplesPerWorker,
                    totalImages - (batchStart + worker * samplesPerWorker));
            }
            mergeWorkerUpdates(workers, activeWorkers);
            processed += batchSize;

            if (onProgress && (processed % 512 == 0 || processed == totalImages)) {
                const float progress = static_cast<float>(epoch * totalImages + processed)
                    / static_cast<float>(epochs * totalImages);
                onProgress(progress);
            }
        }
    }
    std::cout << "Training complete." << std::endl;
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
