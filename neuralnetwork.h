#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include <vector>
#include <ctime>


class NeuralNetwork
{
private:
    int epochs=10;
    double alpha=0.01;
    std::vector<std::vector<float>> images;
    std::vector<int> labels;

    std::vector<double>inputNeuron{std::vector<double>(784)};
    std::vector<double>hidden1Neuron{std::vector<double>(128)};
    std::vector<double>hidden2Neuron{std::vector<double>(64)};
    std::vector<double>outputNeuron{std::vector<double>(10)};

    std::vector<std::vector<double>> WIH{784 , std::vector<double>(128)};
    std::vector<std::vector<double>> WHH{128 , std::vector<double>(64)};
    std::vector<std::vector<double>> WHO{64 , std::vector<double>(10)};

    std::vector<double> biasHidden1{std::vector<double>(128)};
    std::vector<double> biasHidden2{std::vector<double>(64)};
    std::vector<double> biasOutput{std::vector<double>(10)};

    double randomNumber(int inputSize);
    void getRandomWeight();
    double ReLU(double x);
    void train();
public:
    NeuralNetwork();
    void forward();
    void evaluate();


};

#endif // NEURALNETWORK_H
