#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <array>
#include <functional>
#include <vector>

class NeuralNetwork
{
public:
    using ProgressCallback = std::function<void(float)>;
    static constexpr float EvaluationUnavailable = -1.0f;

    int epochs = 5;
    float alpha = 0.001f;
    float l2 = 1e-4f;
    std::vector<std::vector<float>> images;
    std::vector<int> labels;
    std::vector<float> outputNeuron = std::vector<float>(10);

    explicit NeuralNetwork(bool loadTrainingData = true);
    void getRandomWeight();
    void train(ProgressCallback onProgress = nullptr);
    void forward(const std::vector<float>& inputImage);
    float evaluate();
    bool fineTune(const std::vector<float>& inputImage, int label);

private:
    static constexpr int InputSize = 28;
    static constexpr int Kernel = 3;
    static constexpr int HiddenSize = 64;
    static constexpr int Classes = 10;
    static constexpr int SamplesPerWorker = 8;

    struct ConvShape {
        int channels;
        int inputSide;
        int filters;
        constexpr int convSide() const { return inputSide - Kernel + 1; }
        constexpr int poolSide() const { return convSide() / 2; }
    };
    static constexpr ConvShape First{1, InputSize, 16};
    static constexpr int FirstPoolSize = (InputSize - Kernel + 1) / 2;
    static constexpr ConvShape Second{First.filters, FirstPoolSize, 32};
    static constexpr int SecondPoolSize = (FirstPoolSize - Kernel + 1) / 2;
    static constexpr int FlatSize = Second.filters * SecondPoolSize * SecondPoolSize;

    struct ConvParameters {
        std::vector<float> weights;
        std::vector<float> scale;
        std::vector<float> offset;
        explicit ConvParameters(ConvShape shape);
    };

    struct Parameters {
        ConvParameters first{First};
        ConvParameters second{Second};
        std::vector<float> hiddenWeights = std::vector<float>(FlatSize * HiddenSize);
        std::array<float, HiddenSize> hiddenBias{};
        std::vector<float> outputWeights = std::vector<float>(HiddenSize * Classes);
        std::array<float, Classes> outputBias{};
    };

    struct BatchStats {
        std::vector<float> mean;
        std::vector<float> variance;
        std::vector<float> inverseStd;
        explicit BatchStats(int channels)
            : mean(channels), variance(channels), inverseStd(channels) {}
    };

    struct ConvCache {
        std::vector<float> activations;
        std::vector<float> pooled;
        std::vector<int> winners;
        std::vector<float> gradient;
        std::vector<float> poolGradient;
        explicit ConvCache(ConvShape shape);
    };

    struct Sample {
        std::vector<float> input = std::vector<float>(InputSize * InputSize);
        ConvCache first{First};
        ConvCache second{Second};
        std::array<float, HiddenSize> hidden{};
        std::array<float, Classes> probabilities{};
        int label = 0;
    };

    struct Batch {
        std::vector<Sample> samples;
        std::vector<float> augmentationScratch = std::vector<float>(InputSize * InputSize);
        Parameters gradients;
        BatchStats firstStats{First.filters};
        BatchStats secondStats{Second.filters};
        int count = 0;
        explicit Batch(int capacity) : samples(capacity) {}
    };

    Parameters parameters;
    BatchStats firstRunningStats{First.filters};
    BatchStats secondRunningStats{Second.filters};
    Batch inference{1};

    struct TeachingExample {
        std::vector<float> input;
        std::array<float, HiddenSize> hidden{};
        std::array<float, Classes> originalProbabilities{};
    };
    // A small, balanced replay set and bounded correction history for this session.
    std::array<std::vector<TeachingExample>, Classes> teachingReplay;
    std::array<std::vector<TeachingExample>, Classes> teachingCorrections;
    bool teachingReplayReady = false;
    void clearTeachingState();
    std::array<float, Classes> classifyHidden(const std::array<float, HiddenSize>& hidden) const;

    static float randomNumber(int inputSize);
    static void convolve(ConvShape shape, const ConvParameters& layer,
                         const std::vector<float>& input, ConvCache& cache);
    static void normalizeAndPool(ConvShape shape, const ConvParameters& layer,
                                 Batch& batch, ConvCache Sample::* cache,
                                 BatchStats& stats, bool useBatchStats);
    static void normalizeBackward(ConvShape shape, const ConvParameters& layer,
                                  Batch& batch, ConvCache Sample::* cache,
                                  const BatchStats& stats, ConvParameters& gradient,
                                  bool useBatchStats);
    static void convolutionBackward(ConvShape shape, const ConvParameters& layer,
                                    const std::vector<float>& input, const ConvCache& cache,
                                    ConvParameters& gradient, ConvCache* previousPool);
    void forwardBatch(Batch& batch, bool useBatchStats) const;
    void backwardBatch(Batch& batch, bool useBatchStats) const;
    void applyUpdates(const std::vector<Batch>& batches, int activeWorkers, bool updateRunningStats);
    static void augment(const std::vector<float>& input, std::vector<float>& output,
                        std::vector<float>& scratch, int shiftX, int shiftY,
                        float angleDegrees, float scale, bool boldenStroke,
                        bool flipVertical = false);
};

#endif

