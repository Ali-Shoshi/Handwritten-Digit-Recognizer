#include "neuralnetwork.h"
#include "datasetloader.h"
#include <QCoreApplication>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    if (argc != 2) { std::cerr << "Usage: teaching_regression <MNIST Dataset directory>\n"; return 2; }
    const std::string dir = argv[1];
    std::vector<std::vector<float>> trainImages, testImages;
    std::vector<int> trainLabels, testLabels;
    if (!DatasetLoader::loadImages(dir + "/train-images.idx3-ubyte", trainImages)
        || !DatasetLoader::loadLabels(dir + "/train-labels.idx1-ubyte", trainLabels)
        || !DatasetLoader::loadImages(dir + "/t10k-images.idx3-ubyte", testImages)
        || !DatasetLoader::loadLabels(dir + "/t10k-labels.idx1-ubyte", testLabels)) return 2;
    NeuralNetwork network(false);
    std::vector<float> probe(784, 0.1f);
    network.forward(probe);
    const auto untrainedPrediction = network.outputNeuron;
    if (network.fineTune(probe,8)) { std::cerr<<"Teaching without replay data must be refused.\n";return 1; }
    network.forward(probe);
    if(network.outputNeuron != untrainedPrediction)return 1;
    network.images={probe,probe};network.labels={8,8};
    if (network.fineTune(probe,8)) { std::cerr<<"Single-class replay must be refused.\n";return 1; }
    network.images.clear();network.labels.clear();
    std::array<int, 10> counts{};
    for (std::size_t i=0;i<trainImages.size();++i) {
        if (counts[trainLabels[i]]++ < 300) {
            network.images.push_back(trainImages[i]); network.labels.push_back(trainLabels[i]);
        }
    }
    network.epochs=4;
    network.train();
    const auto accuracy=[&]() {
        std::array<int,10> totals{}, correct{}, predicted{};
        for (std::size_t i=0;i<testImages.size();++i) {
            const int label=testLabels[i];
            if (totals[label]>=100) continue;
            ++totals[label]; network.forward(testImages[i]);
            float sum=0;
            for (float p:network.outputNeuron) {
                if (!std::isfinite(p)||p<0) throw std::runtime_error("Invalid probability");
                sum+=p;
            }
            if (std::abs(sum-1)>1e-4) throw std::runtime_error("Probability sum is not 1");
            const int digit=std::max_element(network.outputNeuron.begin(),network.outputNeuron.end())-network.outputNeuron.begin();
            ++predicted[digit]; correct[label]+=digit==label;
        }
        int n=0; for(int value:correct)n+=value;
        std::cout << "Accuracy: " << n/10.0 << "%; correct per digit:";
        for(int value:correct)std::cout << ' ' << value;
        std::cout << "; predictions:";for(int value:predicted)std::cout << ' ' << value;std::cout << std::endl;
        return n/10.0f;
    };
    const float before=accuracy();
    if(before<70) { std::cerr<<"Fixture training did not reach 70%; cannot assess retention.\n"; return 2; }
    // A difficult real sample not used for either training or the accuracy measurement.
    std::size_t teachingIndex=50000; float lowest=1;
    for(std::size_t i=50000;i<trainImages.size();++i)if(trainLabels[i]==8) {
        network.forward(trainImages[i]);
        if(network.outputNeuron[8]<lowest){lowest=network.outputNeuron[8];teachingIndex=i;}
        if(lowest<0.15f)break;
    }
    network.forward(trainImages[teachingIndex]);
    const float confidenceBefore=network.outputNeuron[8];
    for(int click=0;click<10;++click)network.fineTune(trainImages[teachingIndex],8);
    network.forward(trainImages[teachingIndex]);
    const float confidenceAfter=network.outputNeuron[8];
    std::cout<<"Taught digit 8 confidence: "<<confidenceBefore<<" -> "<<confidenceAfter<<std::endl;
    const float after=accuracy();
    if(after<before-3.0f || confidenceAfter<=confidenceBefore) {
        std::cerr<<"FAIL: teaching must improve the example without losing over 3 accuracy points.\n";return 1;
    }
    // Teaching another digit must preserve earlier corrections as well.
    std::size_t secondIndex=50000; float secondBefore=1;
    for(std::size_t i=50000;i<trainImages.size();++i)if(trainLabels[i]==6){
        network.forward(trainImages[i]);
        if(network.outputNeuron[6]<secondBefore){secondBefore=network.outputNeuron[6];secondIndex=i;}
        if(secondBefore<0.15f)break;
    }
    for(int click=0;click<5;++click)network.fineTune(trainImages[secondIndex],6);
    network.forward(trainImages[secondIndex]);
    std::cout<<"Later digit 6 confidence: "<<secondBefore<<" -> "<<network.outputNeuron[6]<<std::endl;
    if(network.outputNeuron[6]<=secondBefore) {std::cerr<<"FAIL: later correction did not improve.\n";return 1;}
    network.forward(trainImages[teachingIndex]);
    if(network.outputNeuron[8]<confidenceBefore || accuracy()<before-3.0f) {
        std::cerr<<"FAIL: a later correction erased earlier learning.\n";return 1;
    }
    const float savedAlpha=network.alpha;
    network.forward(probe);const auto savedPrediction=network.outputNeuron;
    auto badPixels=probe;badPixels[0]=std::numeric_limits<float>::quiet_NaN();
    if(network.fineTune(badPixels,8)||network.fineTune(probe,-1)||network.fineTune({0.0f},8))return 1;
    network.forward(probe);
    if(network.outputNeuron!=savedPrediction||network.alpha!=savedAlpha)return 1;
    // Reset must also clear the cached features/corrections, even when the public
    // training data changes before the next teaching attempt.
    network.getRandomWeight();network.images={probe,probe};network.labels={8,8};
    if(network.fineTune(probe,8))return 1;
    std::cout<<"PASS: corrections learned; other digits and earlier corrections retained.\n";
    return 0;
}
