#include "neuralnetwork.h"

#include <QCoreApplication>
#include <iostream>

int main(int argc, char** argv)
{
    QCoreApplication app(argc, argv);
    NeuralNetwork network;
    network.train();
    const float accuracy = network.evaluate();
    std::cout << "Smoke-test accuracy after " << network.epochs << " epochs: "
              << accuracy << "%" << std::endl;
    return accuracy >= 95.0f ? 0 : 1;
}
