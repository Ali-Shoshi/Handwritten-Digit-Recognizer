#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include <vector>
#include <ctime>


class NeuralNetwork
{
public:

    int epochs=10;
    float alpha=0.01;
    std::vector<std::vector<float>> images;
    std::vector<int> labels;

    std::vector<float>inputNeuron{std::vector<float>(784, 0.0f)};
    std::vector<float>hidden1Neuron{std::vector<float>(128, 0.0f)};
    std::vector<float>hidden2Neuron{std::vector<float>(64, 0.0f)};
    std::vector<float>outputNeuron{std::vector<float>(10, 0.0f)};

    std::vector<std::vector<float>> WIH{784 , std::vector<float>(128, 0.0f)};
    std::vector<std::vector<float>> WHH{128 , std::vector<float>(64, 0.0f)};
    std::vector<std::vector<float>> WHO{64 , std::vector<float>(10, 0.0f)};

    std::vector<float> biasHidden1{std::vector<float>(128, 0.0f)};
    std::vector<float> biasHidden2{std::vector<float>(64, 0.0f)};
    std::vector<float> biasOutput{std::vector<float>(10, 0.0f)};

    float randomNumber(int inputSize);
    void getRandomWeight();
    float ReLU(float x);
    void train();
    NeuralNetwork();
    void forward();
    float evaluate();


};

#endif // NEURALNETWORK_H
