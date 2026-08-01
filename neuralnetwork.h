#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <ctime>
#include <functional>
#include <cstddef>

class NeuralNetwork
{
public:
    using ProgressCallback = std::function<void(float)>;

    // Three shuffled passes are enough to exceed 95% on MNIST with this CNN.
    // Keeping this small makes the in-app training finish in a few minutes.
    int epochs = 3;
    float alpha = 0.001f;
    float l2 = 1e-4f;             // Small L2 on dense weights

    std::vector<std::vector<float>> images;
    std::vector<int> labels;

    int numFilters = 16;
    int filterSize = 3;
    int convOutputDim = 26;       // 28 - 3 + 1
    int poolOutputDim = 13;       // 26 / 2
    int hiddenSize = 64;

    std::vector<std::vector<std::vector<float>>> convWeights;
    std::vector<float> convBiases;

    std::vector<std::vector<float>> denseWeights; // flattened features -> hidden layer
    std::vector<float> denseBiases;
    std::vector<std::vector<float>> outputWeights; // hidden layer -> 10 digits
    std::vector<float> outputBiases;

    // Forward-pass cache
    std::vector<std::vector<float>> lastInput2D;
    std::vector<std::vector<std::vector<float>>> lastConvOutput;
    std::vector<std::vector<std::vector<float>>> lastPoolOutput;
    std::vector<float> lastFlattened;
    std::vector<float> lastHidden;
    std::vector<float> outputNeuron;

    // Backward-pass buffers
    std::vector<float> output_deltas;
    std::vector<float> hidden_deltas;
    std::vector<float> flattened_deltas;
    std::vector<std::vector<std::vector<float>>> pool_deltas;
    std::vector<std::vector<std::vector<float>>> conv_deltas;
    std::vector<std::vector<float>> filter_grads;
    std::vector<float> augmentedInput;
    std::vector<float> boldenedInput;

    // Workers used during parallel training do not need to load a second copy
    // of the 60,000-image training set.
    explicit NeuralNetwork(bool loadTrainingData = true);

    float randomNumber(int inputSize);
    void getRandomWeight();
    inline float ReLU(float x);

    void train(ProgressCallback onProgress = nullptr);
    void forward(const std::vector<float>& inputImage);
    float evaluate();
    void fineTune(const std::vector<float>& inputImage, int label);

private:
    void trainOne(const std::vector<float>& inputImage, int label);
    void trainOneAugmented(const std::vector<float>& inputImage, int label,
                           int shiftX, int shiftY, float angleDegrees, float scale,
                           bool boldenStroke);
    void copyParametersFrom(const NeuralNetwork& source);
    void mergeWorkerUpdates(const std::vector<NeuralNetwork>& workers,
                            int activeWorkerCount);
};

#endif
