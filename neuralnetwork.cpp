#include "neuralnetwork.h"
#include <cmath>
#include "datasetloader.h"

#include <random>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <QtConcurrent/QtConcurrentMap>
#include <QThread>
#include <QDir>
#include <QCoreApplication>

namespace {
constexpr float BatchNormEpsilon = 1e-5f;

QDir findDataset(const QString &imageFile, const QString &labelFile) {
    if (QCoreApplication::instance()) {
        const QDir appDir(QCoreApplication::applicationDirPath());
        for (const char *relativePath : {"Dataset", "../Dataset", "../../Dataset"}) {
            const QDir candidate(appDir.filePath(relativePath));
            if (candidate.exists(imageFile) && candidate.exists(labelFile)) {
                return candidate;
            }
        }
    }
    return QDir("Dataset");
}
}

NeuralNetwork::NeuralNetwork(bool loadTrainingData) {
    getRandomWeight();

    if (!loadTrainingData) {
        return;
    }

    const QString imageFile = "train-images.idx3-ubyte";
    const QString labelFile = "train-labels.idx1-ubyte";
    const QDir dataset = findDataset(imageFile, labelFile);
    bool imagesLoaded = DatasetLoader::loadImages(dataset.filePath(imageFile).toStdString(), images);
    bool labelsLoaded = DatasetLoader::loadLabels(dataset.filePath(labelFile).toStdString(), labels);

    if (!imagesLoaded || !labelsLoaded) {
        std::cout << "CRITICAL: Failed to load dataset files from path: "
                  << dataset.absolutePath().toStdString() << std::endl;
        return;
    }
}

NeuralNetwork::ConvParameters::ConvParameters(ConvShape shape)
    : weights(shape.filters * shape.channels * Kernel * Kernel),
      scale(shape.filters), offset(shape.filters) {}

NeuralNetwork::ConvCache::ConvCache(ConvShape shape)
    : activations(shape.filters * shape.convSide() * shape.convSide()),
      pooled(shape.filters * shape.poolSide() * shape.poolSide()),
      winners(pooled.size()),
      gradient(activations.size()), poolGradient(pooled.size()) {}

float NeuralNetwork::randomNumber(int inputSize) {
    static thread_local std::mt19937 generator(std::random_device{}());
    std::normal_distribution<float> distribution(0.0f, std::sqrt(2.0f / inputSize));
    return distribution(generator);
}

void NeuralNetwork::getRandomWeight() {
    const auto initialize = [](std::vector<float>& weights, int fanIn) {
        for (float& weight : weights) {
            weight = randomNumber(fanIn);
        }
    };
    initialize(parameters.first.weights, First.channels * Kernel * Kernel);
    initialize(parameters.second.weights, Second.channels * Kernel * Kernel);
    initialize(parameters.hiddenWeights, FlatSize);
    initialize(parameters.outputWeights, HiddenSize);

    for (ConvParameters* layer : {&parameters.first, &parameters.second}) {
        std::fill(layer->scale.begin(), layer->scale.end(), 1.0f);
        std::fill(layer->offset.begin(), layer->offset.end(), 0.0f);
    }
    parameters.hiddenBias.fill(0.01f);
    parameters.outputBias.fill(0.01f);

    for (BatchStats* stats : {&firstRunningStats, &secondRunningStats}) {
        std::fill(stats->mean.begin(), stats->mean.end(), 0.0f);
        std::fill(stats->variance.begin(), stats->variance.end(), 1.0f);
        std::fill(stats->inverseStd.begin(), stats->inverseStd.end(),
                  1.0f / std::sqrt(1.0f + BatchNormEpsilon));
    }
    std::fill(outputNeuron.begin(), outputNeuron.end(), 0.0f);
}

void NeuralNetwork::convolve(ConvShape shape, const ConvParameters& layer,
                              const std::vector<float>& input, ConvCache& cache) {
    const int inputSide = shape.inputSide;
    const int side = shape.convSide();
    const int area = side * side;
    const int inputArea = inputSide * inputSide;
    std::fill(cache.activations.begin(), cache.activations.end(), 0.0f);
    // The BatchNorm offset supplies the bias, so neither convolution needs one.
    for (int filter = 0; filter < shape.filters; ++filter) {
        float* destination = cache.activations.data() + filter * area;
        for (int channel = 0; channel < shape.channels; ++channel) {
            const float* source = input.data() + channel * inputArea;
            const float* weights = layer.weights.data() + (filter * shape.channels + channel) * 9;
            for (int row = 0; row < side; ++row) {
                const float* top = source + row * inputSide;
                const float* middle = top + inputSide;
                const float* bottom = middle + inputSide;
                float* output = destination + row * side;
                for (int col = 0; col < side; ++col) {
                    output[col] += top[col] * weights[0] + top[col + 1] * weights[1]
                        + top[col + 2] * weights[2] + middle[col] * weights[3]
                        + middle[col + 1] * weights[4] + middle[col + 2] * weights[5]
                        + bottom[col] * weights[6] + bottom[col + 1] * weights[7]
                        + bottom[col + 2] * weights[8];
                }
            }
        }
    }
}

void NeuralNetwork::normalizeAndPool(ConvShape shape, const ConvParameters& layer,
                                     Batch& batch, ConvCache Sample::* member,
                                     BatchStats& stats, bool useBatchStats) {
    const int side = shape.convSide();
    const int area = side * side;
    const int poolSide = shape.poolSide();
    const float observations = static_cast<float>(batch.count * area);

    for (int filter = 0; filter < shape.filters; ++filter) {
        if (useBatchStats) {
            double sum = 0.0;
            double squareSum = 0.0;
            for (int i = 0; i < batch.count; ++i) {
                const float* values = (batch.samples[i].*member).activations.data() + filter * area;
                for (int j = 0; j < area; ++j) {
                    const double value = values[j];
                    sum += value;
                    squareSum += value * value;
                }
            }
            const double mean = sum / observations;
            stats.mean[filter] = static_cast<float>(mean);
            stats.variance[filter] = static_cast<float>(std::max(0.0, squareSum / observations - mean * mean));
            stats.inverseStd[filter] = 1.0f / std::sqrt(stats.variance[filter] + BatchNormEpsilon);
        }

        for (int i = 0; i < batch.count; ++i) {
            auto& cache = batch.samples[i].*member;
            const int base = filter * area;
            float* values = cache.activations.data() + base;
            for (int j = 0; j < area; ++j) {
                values[j] = (values[j] - stats.mean[filter]) * stats.inverseStd[filter];
            }

            for (int row = 0; row < poolSide; ++row) {
                for (int col = 0; col < poolSide; ++col) {
                    const int start = 2 * row * side + 2 * col;
                    int winner = start;
                    float maximum = 0.0f;
                    for (int index : {start, start + 1, start + side, start + side + 1}) {
                        const float activation = layer.scale[filter] * values[index] + layer.offset[filter];
                        if (activation > maximum) {
                            maximum = activation;
                            winner = index;
                        }
                    }
                    const int index = (filter * poolSide + row) * poolSide + col;
                    cache.pooled[index] = maximum;
                    cache.winners[index] = base + winner;
                }
            }
        }
    }
}

void NeuralNetwork::forwardBatch(Batch& batch, bool useBatchStats) const {
    if (!useBatchStats) {
        batch.firstStats = firstRunningStats;
        batch.secondStats = secondRunningStats;
    }
    for (int i = 0; i < batch.count; ++i) {
        auto& sample = batch.samples[i];
        convolve(First, parameters.first, sample.input, sample.first);
    }
    normalizeAndPool(First, parameters.first, batch, &Sample::first, batch.firstStats, useBatchStats);
    for (int i = 0; i < batch.count; ++i) {
        auto& sample = batch.samples[i];
        convolve(Second, parameters.second, sample.first.pooled, sample.second);
    }
    normalizeAndPool(Second, parameters.second, batch, &Sample::second, batch.secondStats, useBatchStats);

    for (int i = 0; i < batch.count; ++i) {
        auto& sample = batch.samples[i];
        sample.hidden = parameters.hiddenBias;
        for (int j = 0; j < FlatSize; ++j) {
            const float value = sample.second.pooled[j];
            const float* weights = parameters.hiddenWeights.data() + j * HiddenSize;
            for (int k = 0; k < HiddenSize; ++k) {
                sample.hidden[k] += value * weights[k];
            }
        }
        sample.probabilities = parameters.outputBias;
        for (int j = 0; j < HiddenSize; ++j) {
            sample.hidden[j] = std::max(0.0f, sample.hidden[j]);
            for (int k = 0; k < Classes; ++k) {
                sample.probabilities[k] += sample.hidden[j] * parameters.outputWeights[j * Classes + k];
            }
        }
        const float maximum = *std::max_element(sample.probabilities.begin(), sample.probabilities.end());
        float total = 0.0f;
        for (float& probability : sample.probabilities) {
            probability = std::exp(probability - maximum);
            total += probability;
        }
        for (float& probability : sample.probabilities) {
            probability /= total;
        }
    }
}

void NeuralNetwork::forward(const std::vector<float>& inputImage) {
    if (inputImage.size() != InputSize * InputSize) {
        throw std::invalid_argument("CNN input must contain 784 pixels");
    }
    inference.count = 1;
    auto& sample = inference.samples.front();
    std::transform(inputImage.begin(), inputImage.end(), sample.input.begin(),
                   [](float value) { return std::clamp(value, 0.0f, 1.0f); });
    forwardBatch(inference, false);
    std::copy(sample.probabilities.begin(), sample.probabilities.end(), outputNeuron.begin());
}

void NeuralNetwork::normalizeBackward(ConvShape shape, const ConvParameters& layer,
                                      Batch& batch, ConvCache Sample::* member,
                                      const BatchStats& stats, ConvParameters& gradient,
                                      bool useBatchStats) {
    const int area = shape.convSide() * shape.convSide();
    const float observations = static_cast<float>(batch.count * area);
    for (int filter = 0; filter < shape.filters; ++filter) {
        const int start = filter * area;
        float sum = 0.0f;
        float normalizedSum = 0.0f;
        for (int i = 0; i < batch.count; ++i) {
            const auto& cache = batch.samples[i].*member;
            for (int j = start; j < start + area; ++j) {
                sum += cache.gradient[j];
                normalizedSum += cache.gradient[j] * cache.activations[j];
            }
        }
        gradient.offset[filter] = sum;
        gradient.scale[filter] = normalizedSum;
        const float factor = layer.scale[filter] * stats.inverseStd[filter];
        const float meanGradient = useBatchStats ? sum / observations : 0.0f;
        const float normalizedMean = useBatchStats ? normalizedSum / observations : 0.0f;
        for (int i = 0; i < batch.count; ++i) {
            auto& cache = batch.samples[i].*member;
            for (int j = start; j < start + area; ++j) {
                cache.gradient[j] = factor * (cache.gradient[j] - meanGradient
                                               - cache.activations[j] * normalizedMean);
            }
        }
    }
}

void NeuralNetwork::convolutionBackward(ConvShape shape, const ConvParameters& layer,
                                         const std::vector<float>& input, const ConvCache& cache,
                                         ConvParameters& gradient, ConvCache* previousPool) {
    const int inputSide = shape.inputSide;
    const int side = shape.convSide();
    const int area = side * side;
    const int inputArea = inputSide * inputSide;
    std::vector<float>* previousGradient = nullptr;
    if (previousPool) {
        previousGradient = &previousPool->poolGradient;
        std::fill(previousGradient->begin(), previousGradient->end(), 0.0f);
    }

    for (int filter = 0; filter < shape.filters; ++filter) {
        const float* deltas = cache.gradient.data() + filter * area;
        for (int channel = 0; channel < shape.channels; ++channel) {
            const float* values = input.data() + channel * inputArea;
            const int weightBase = (filter * shape.channels + channel) * 9;
            for (int kr = 0; kr < Kernel; ++kr) {
                for (int kc = 0; kc < Kernel; ++kc) {
                    const int index = weightBase + kr * Kernel + kc;
                    const int origin = kr * inputSide + kc;
                    float weightGradient = 0.0f;
                    for (int row = 0; row < side; ++row) {
                        const float* source = values + origin + row * inputSide;
                        const float* delta = deltas + row * side;
                        for (int col = 0; col < side; ++col) {
                            weightGradient += source[col] * delta[col];
                        }
                    }
                    gradient.weights[index] += weightGradient;
                    if (previousGradient) {
                        const float weight = layer.weights[index];
                        float* destination = previousGradient->data() + channel * inputArea + origin;
                        for (int row = 0; row < side; ++row) {
                            float* rowGradient = destination + row * inputSide;
                            const float* delta = deltas + row * side;
                            for (int col = 0; col < side; ++col) {
                                rowGradient[col] += weight * delta[col];
                            }
                        }
                    }
                }
            }
        }
    }
    if (previousPool) {
        std::fill(previousPool->gradient.begin(), previousPool->gradient.end(), 0.0f);
        for (std::size_t i = 0; i < previousPool->pooled.size(); ++i) {
            if (previousPool->pooled[i] > 0.0f) {
                previousPool->gradient[previousPool->winners[i]] = (*previousGradient)[i];
            }
        }
    }
}

void NeuralNetwork::backwardBatch(Batch& batch, bool useBatchStats) const {
    auto& gradient = batch.gradients;
    for (auto* values : {&gradient.first.weights, &gradient.second.weights,
                         &gradient.hiddenWeights, &gradient.outputWeights}) {
        std::fill(values->begin(), values->end(), 0.0f);
    }
    gradient.hiddenBias.fill(0.0f);
    gradient.outputBias.fill(0.0f);

    for (int i = 0; i < batch.count; ++i) {
        auto& sample = batch.samples[i];
        auto outputGradient = sample.probabilities;
        outputGradient[sample.label] -= 1.0f;
        std::array<float, HiddenSize> hiddenGradient{};
        for (int k = 0; k < Classes; ++k) {
            gradient.outputBias[k] += outputGradient[k];
        }
        for (int j = 0; j < HiddenSize; ++j) {
            if (sample.hidden[j] <= 0.0f) {
                continue;
            }
            float delta = 0.0f;
            for (int k = 0; k < Classes; ++k) {
                const int index = j * Classes + k;
                gradient.outputWeights[index] += sample.hidden[j] * outputGradient[k];
                delta += parameters.outputWeights[index] * outputGradient[k];
            }
            hiddenGradient[j] = delta;
            gradient.hiddenBias[j] += delta;
        }

        auto& second = sample.second;
        std::fill(second.gradient.begin(), second.gradient.end(), 0.0f);
        for (int j = 0; j < FlatSize; ++j) {
            const float value = second.pooled[j];
            if (value <= 0.0f) {
                continue;
            }
            float delta = 0.0f;
            for (int k = 0; k < HiddenSize; ++k) {
                const int index = j * HiddenSize + k;
                gradient.hiddenWeights[index] += value * hiddenGradient[k];
                delta += parameters.hiddenWeights[index] * hiddenGradient[k];
            }
            second.gradient[second.winners[j]] = delta;
        }
    }

    normalizeBackward(Second, parameters.second, batch, &Sample::second,
                      batch.secondStats, gradient.second, useBatchStats);
    for (int i = 0; i < batch.count; ++i) {
        auto& sample = batch.samples[i];
        convolutionBackward(Second, parameters.second, sample.first.pooled, sample.second,
                            gradient.second, &sample.first);
    }
    normalizeBackward(First, parameters.first, batch, &Sample::first,
                      batch.firstStats, gradient.first, useBatchStats);
    for (int i = 0; i < batch.count; ++i) {
        auto& sample = batch.samples[i];
        convolutionBackward(First, parameters.first, sample.input, sample.first, gradient.first, nullptr);
    }
}

void NeuralNetwork::applyUpdates(const std::vector<Batch>& batches, int activeWorkers, bool updateRunningStats) {
    int sampleCount = 0;
    for (int i = 0; i < activeWorkers; ++i) {
        sampleCount += batches[i].count;
    }
    const float decay = 1.0f - alpha * l2 * sampleCount;
    for (auto* weights : {&parameters.hiddenWeights, &parameters.outputWeights}) {
        for (float& weight : *weights) {
            weight *= decay;
        }
    }
    const auto apply = [this](auto& values, const auto& gradient) {
        for (std::size_t j = 0; j < values.size(); ++j) {
            values[j] -= alpha * gradient[j];
        }
    };
    for (int i = 0; i < activeWorkers; ++i) {
        const auto& gradient = batches[i].gradients;
        apply(parameters.first.weights, gradient.first.weights);
        apply(parameters.first.scale, gradient.first.scale);
        apply(parameters.first.offset, gradient.first.offset);
        apply(parameters.second.weights, gradient.second.weights);
        apply(parameters.second.scale, gradient.second.scale);
        apply(parameters.second.offset, gradient.second.offset);
        apply(parameters.hiddenWeights, gradient.hiddenWeights);
        apply(parameters.hiddenBias, gradient.hiddenBias);
        apply(parameters.outputWeights, gradient.outputWeights);
        apply(parameters.outputBias, gradient.outputBias);
    }

    if (!updateRunningStats) {
        return;
    }

    const auto updateStats = [&](BatchStats& running, BatchStats Batch::* member) {
        constexpr float momentum = 0.05f;
        for (std::size_t filter = 0; filter < running.mean.size(); ++filter) {
            float sum = 0.0f;
            float squareSum = 0.0f;
            for (int i = 0; i < activeWorkers; ++i) {
                const auto& stats = batches[i].*member;
                const float mean = stats.mean[filter];
                sum += batches[i].count * mean;
                squareSum += batches[i].count * (stats.variance[filter] + mean * mean);
            }
            const float mean = sum / sampleCount;
            const float variance = std::max(0.0f, squareSum / sampleCount - mean * mean);
            running.mean[filter] += momentum * (mean - running.mean[filter]);
            running.variance[filter] += momentum * (variance - running.variance[filter]);
            running.inverseStd[filter] = 1.0f / std::sqrt(running.variance[filter] + BatchNormEpsilon);
        }
    };
    updateStats(firstRunningStats, &Batch::firstStats);
    updateStats(secondRunningStats, &Batch::secondStats);
}


void NeuralNetwork::augment(const std::vector<float>& inputImage, std::vector<float>& output,
                            std::vector<float>& augmentedInput, int shiftX, int shiftY,
                            float angleDegrees, float scale, bool boldenStroke,
                            bool flipVertical) {
    if (shiftX == 0 && shiftY == 0 && angleDegrees == 0.0f && scale == 1.0f
        && !boldenStroke && !flipVertical) {
        std::copy(inputImage.begin(), inputImage.end(), output.begin());
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
            float sourceX = (cosine * destinationX + sine * destinationY) / scale + center;
            float sourceY = (-sine * destinationX + cosine * destinationY) / scale;
            sourceY = (flipVertical ? -sourceY : sourceY) + center;

            // Keep boundary pixels, including at exact right-angle rotations.
            constexpr float tolerance = 1e-4f;
            if (sourceX < -tolerance || sourceX > 27.0f + tolerance
                || sourceY < -tolerance || sourceY > 27.0f + tolerance) {
                continue;
            }
            sourceX = std::clamp(sourceX, 0.0f, 27.0f);
            sourceY = std::clamp(sourceY, 0.0f, 27.0f);

            const int left = static_cast<int>(sourceX);
            const int top = static_cast<int>(sourceY);
            const int right = std::min(left + 1, 27);
            const int bottom = std::min(top + 1, 27);
            const float horizontal = sourceX - left;
            const float vertical = sourceY - top;
            const float topValue = inputImage[top * 28 + left] * (1.0f - horizontal)
                + inputImage[top * 28 + right] * horizontal;
            const float bottomValue = inputImage[bottom * 28 + left] * (1.0f - horizontal)
                + inputImage[bottom * 28 + right] * horizontal;
            augmentedInput[y * 28 + x] = topValue * (1.0f - vertical) + bottomValue * vertical;
        }
    }
    if (!boldenStroke) {
        std::copy(augmentedInput.begin(), augmentedInput.end(), output.begin());
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
            output[y * 28 + x] = maximum;
        }
    }
}

void NeuralNetwork::fineTune(const std::vector<float>& inputImage, int label) {
    if (inputImage.size() != InputSize * InputSize || label < 0 || label >= Classes) {
        return;
    }

    // Keep population statistics from MNIST when adapting to one labeled drawing.
    constexpr int variations = 24;
    std::vector<Batch> batches;
    batches.emplace_back(SamplesPerWorker);
    auto& batch = batches.front();
    for (int start = 0; start < variations; start += SamplesPerWorker) {
        batch.count = std::min(SamplesPerWorker, variations - start);
        for (int i = 0; i < batch.count; ++i) {
            const unsigned int seed = static_cast<unsigned int>((start + i + 1) * 747796405u)
                + static_cast<unsigned int>(label * 2891336453u);
            const int shiftX = static_cast<int>(seed % 3u) - 1;
            const int shiftY = static_cast<int>((seed / 3u) % 3u) - 1;
            const float angle = static_cast<float>((seed / 9u) % 13u) - 6.0f;
            const float scale = 0.95f + 0.025f * static_cast<float>((seed / 117u) % 5u);
            const bool boldenStroke = ((seed / 585u) % 3u) == 0u;
            auto& sample = batch.samples[i];
            sample.label = label;
            augment(inputImage, sample.input, batch.augmentationScratch,
                    shiftX, shiftY, angle, scale, boldenStroke);
        }
        forwardBatch(batch, false);
        backwardBatch(batch, false);
        applyUpdates(batches, 1, false);
    }
}

void NeuralNetwork::train(ProgressCallback onProgress) {
    const int totalImages = static_cast<int>(images.size());
    if (totalImages == 0 || labels.size() != images.size()
        || std::any_of(labels.begin(), labels.end(), [](int label) { return label < 0 || label >= Classes; })
        || std::any_of(images.begin(), images.end(), [](const auto& image) {
            return image.size() != InputSize * InputSize;
        })) {
        std::cerr << "Training data is missing or inconsistent." << std::endl;
        return;
    }

    const int workerCount = std::clamp(QThread::idealThreadCount() - 1, 1, 8);
    std::cout << "Training with " << workerCount << " worker threads." << std::endl;
    std::vector<Batch> batches;
    batches.reserve(workerCount);
    for (int i = 0; i < workerCount; ++i) {
        batches.emplace_back(SamplesPerWorker);
    }
    std::vector<int> workerIndexes(workerCount);
    std::iota(workerIndexes.begin(), workerIndexes.end(), 0);
    std::vector<int> indices(totalImages);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(std::random_device{}());

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);
        for (int processed = 0; processed < totalImages;) {
            const int batchStart = processed;
            const int activeWorkers = std::min(workerCount,
                (totalImages - batchStart + SamplesPerWorker - 1) / SamplesPerWorker);

            // Each worker normalizes its own group of images and computes gradients
            // against the same model. Only the caller updates model parameters.
            QtConcurrent::blockingMap(workerIndexes.begin(), workerIndexes.begin() + activeWorkers,
                [this, &batches, &indices, batchStart, totalImages, epoch](int worker) {
                    auto& batch = batches[worker];
                    const int begin = batchStart + worker * SamplesPerWorker;
                    batch.count = std::min(SamplesPerWorker, totalImages - begin);
                    for (int i = 0; i < batch.count; ++i) {
                        const int position = begin + i;
                        const int imageIndex = indices[position];
                        const unsigned int seed = static_cast<unsigned int>(position * 1103515245u)
                            + static_cast<unsigned int>(epoch * 12345u);
                        int shiftX = static_cast<int>(seed % 5u) - 2;
                        int shiftY = static_cast<int>((seed / 5u) % 5u) - 2;
                        float angle = static_cast<float>((seed / 25u) % 25u) - 12.0f;
                        float scale = 0.90f + 0.05f * static_cast<float>((seed / 625u) % 5u);
                        bool boldenStroke = ((seed / 3125u) % 4u) == 0u;
                        // Keep most samples close to ordinary handwriting.
                        // Large rotations and reflections can make digit labels ambiguous.
                        const unsigned int variant = (seed / 12500u) % 20u;
                        const bool flipVertical = variant == 1u;
                        if (variant == 0u) {
                            shiftX = shiftY = 0;
                            angle = 0.0f;
                            scale = 1.0f;
                            boldenStroke = false;
                        } else if (variant >= 2u && variant <= 4u) {
                            angle = static_cast<float>((seed / 25u) % 181u) - 90.0f;
                            scale = std::min(scale, 0.95f);
                        }
                        auto& sample = batch.samples[i];
                        sample.label = labels[imageIndex];
                        augment(images[imageIndex], sample.input, batch.augmentationScratch,
                                shiftX, shiftY, angle, scale, boldenStroke, flipVertical);
                    }
                    forwardBatch(batch, true);
                    backwardBatch(batch, true);
                });

            applyUpdates(batches, activeWorkers, true);
            processed += std::min(activeWorkers * SamplesPerWorker, totalImages - processed);
            if (onProgress && (processed % 512 == 0 || processed == totalImages)) {
                const float progress = static_cast<float>(epoch * totalImages + processed)
                    / static_cast<float>(epochs * totalImages);
                onProgress(progress);
            }
        }
        std::cout << "Epoch " << epoch + 1 << " / " << epochs << " complete." << std::endl;
    }
    std::cout << "Training complete." << std::endl;
}


float NeuralNetwork::evaluate() {
    std::cout << "Running evaluation..." << std::endl;

    const QString imageFile = "t10k-images.idx3-ubyte";
    const QString labelFile = "t10k-labels.idx1-ubyte";
    const QDir dataset = findDataset(imageFile, labelFile);

    std::vector<std::vector<float>> testImages;
    std::vector<int> testLabels;

    bool imagesLoaded = DatasetLoader::loadImages(dataset.filePath(imageFile).toStdString(), testImages);
    bool labelsLoaded = DatasetLoader::loadLabels(dataset.filePath(labelFile).toStdString(), testLabels);

    if (!imagesLoaded || !labelsLoaded || testImages.empty() || testImages.size() != testLabels.size()
        || testImages.front().size() != 28 * 28) {
        std::cerr << "Failed to load matching MNIST test images and labels from: "
                  << dataset.absolutePath().toStdString() << std::endl;
        return EvaluationUnavailable;
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
