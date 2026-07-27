#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include <vector>
#include <ctime>
#include <functional>


class NeuralNetwork
{
public:

    using ProgressCallback = std::function<void(float)>;

    int epochs=10;
    float alpha=0.01;
    std::vector<std::vector<float>> images;
    std::vector<int> labels;

    int numFilters =16;
    int filterSize =3;
    int convOutputDim =26;  // 28 - 3 + 1
    int poolOutputDim = 13; // 26 / 2

    std::vector<std::vector<std::vector<float>>> convWeights; // [numfilters] [3][3]
    std::vector<float> convBiases;                            // [numfilters]

    std::vector<std::vector<float>> denseWeights;   //[2704][10]  16 filters} X  13 height X 13 width = 2,704 values
    std::vector<float>denseBiases;                  // [10]


    //Caching for backpropagatoin
    std::vector<std::vector<float>> lastInput2D;
    std::vector<std::vector<std::vector<float>>> lastConvOutput;
    std::vector<std::vector<std::vector<float>>>lastPoolOutput;
    std::vector<float> lastFlattened;               // [flatSize][10]
    std::vector<float> outputNeuron;                // [10]

    NeuralNetwork();

    float randomNumber(int inputSize);
    void getRandomWeight();
    float ReLU(float x);

    void train(ProgressCallback onProgress = nullptr);
    void forward(const std::vector<float>& inputImage);
    float evaluate();

};

#endif
