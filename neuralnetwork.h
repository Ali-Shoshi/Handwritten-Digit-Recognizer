#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <ctime>
#include <functional>

class NeuralNetwork
{
public:
    using ProgressCallback = std::function<void(float)>;

    int epochs = 15;
    float alpha = 0.001f;         // Lowered to stabilize pure SGD
    float l2 = 1e-4f;             // Small L2 on dense weights

    std::vector<std::vector<float>> images;
    std::vector<int> labels;

    int numFilters = 16;
    int filterSize = 3;
    int convOutputDim = 26;       // 28 - 3 + 1
    int poolOutputDim = 13;       // 26 / 2

    std::vector<std::vector<std::vector<float>>> convWeights;
    std::vector<float> convBiases;

    std::vector<std::vector<float>> denseWeights;
    std::vector<float> denseBiases;

    // Forward-pass cache
    std::vector<std::vector<float>> lastInput2D;
    std::vector<std::vector<std::vector<float>>> lastConvOutput;
    std::vector<std::vector<std::vector<float>>> lastPoolOutput;
    std::vector<float> lastFlattened;
    std::vector<float> outputNeuron;

    // Backward-pass buffers
    std::vector<float> output_deltas;
    std::vector<float> flattened_deltas;
    std::vector<std::vector<std::vector<float>>> pool_deltas;
    std::vector<std::vector<std::vector<float>>> conv_deltas;
    std::vector<std::vector<float>> filter_grads;

    NeuralNetwork();

    float randomNumber(int inputSize);
    void getRandomWeight();
    inline float ReLU(float x);

    void train(ProgressCallback onProgress = nullptr);
    void forward(const std::vector<float>& inputImage);
    float evaluate();
};

#endif