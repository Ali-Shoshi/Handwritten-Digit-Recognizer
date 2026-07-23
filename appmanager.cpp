#include "AppManager.h"
#include <algorithm>
#include <iostream>

AppManager::AppManager(QObject *parent) : QObject(parent) {
    // NeuralNetwork constructor will run and load/train weights
}

void AppManager::predictFromPixels(const QList<float> &pixelBuffer) {
    if (pixelBuffer.size() != 784) {
        std::cerr << "Error: Expected 784 pixels, got " << pixelBuffer.size() << std::endl;
        return;
    }

    // 1. Load input into neural network
    m_nn.inputNeuron = std::vector<float>(pixelBuffer.begin(), pixelBuffer.end());

    // 2. Run forward pass
    m_nn.forward();

    // 3. Store pairs of (probability, digit_index)
    std::vector<std::pair<float, int>> results;
    m_probabilities.clear();

    for (int i = 0; i < 10; ++i) {
        float prob = m_nn.outputNeuron[i];
        m_probabilities.append(prob);
        results.push_back({prob, i});
    }

    // Sort descending based on probability (highest first)
    std::sort(results.begin(), results.end(), [](const auto &a, const auto &b) {
        return a.first > b.first;
    });

    // Extract Top 3 Digits
    int bestDigit       = results[0].second;
    int secondBestDigit = results[1].second;
    int thirdBestDigit  = results[2].second;

    // Extract Top 3 Probabilities
    float bestProb       = results[0].first;
    float secondBestProb = results[1].first;
    float thirdBestProb  = results[2].first;

    // 4. Update state & notify QML UI
    m_bestPredictedDigit = bestDigit;
    m_secondBestPredictedDigit = secondBestDigit;
    m_thirdBestPredictedDigit = thirdBestDigit;

    m_bestProb = bestProb;
    m_secondBestProb = secondBestProb;
    m_thirdBestProb = thirdBestProb;
    emit predictionChanged();
}

void AppManager::clearPrediction() {

    m_bestPredictedDigit = -1;
    m_secondBestPredictedDigit = -1;
    m_thirdBestPredictedDigit = -1;

    m_bestProb = 0.0f;
    m_secondBestProb = 0.0f;
    m_thirdBestProb = 0.0f;

    m_probabilities.clear();
    emit predictionChanged();
    emit modelEvaluationChanged();
}

void AppManager::trainModel(){

}
void resetModel(){
    m_nn.getRandomWeight();
}
void evaluateModel(){

}