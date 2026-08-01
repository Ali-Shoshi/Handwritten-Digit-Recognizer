#include "neuralnetwork.h"

#include <iostream>

int main()
{
    NeuralNetwork network;
    network.epochs = 3;
    network.train();
    const float accuracy = network.evaluate();
    std::cout << "Smoke-test accuracy after three epochs: " << accuracy << "%" << std::endl;
    return accuracy >= 95.0f ? 0 : 1;
}
