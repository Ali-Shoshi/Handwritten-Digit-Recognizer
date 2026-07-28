#include "AppManager.h"
#include <algorithm>
#include <iostream>
#include <QtConcurrent>
#include <QImage>
#include <QPainter>

AppManager::AppManager(QObject *parent) : QObject(parent) {
    // NeuralNetwork constructor will run and load/train weights
}

void AppManager::predictFromPixels(const QList<float> &pixelBuffer) {
    if (pixelBuffer.size() != 784) {
        std::cerr << "Error: Expected 784 pixels, got " << pixelBuffer.size() << std::endl;
        return;
    }

    // 2. Run forward pass
    m_nn.forward(std::vector<float>(pixelBuffer.begin(), pixelBuffer.end()));

    // 3. Store pairs of (probability, digit_index)
    std::vector<std::pair<float, int>> results;
    m_probabilities.clear();

    for (int i = 0; i < 10; ++i) {
        float prob = m_nn.outputNeuron[i];
        m_probabilities.append(prob);
        results.push_back({prob, i});
    }

    std::sort(results.begin(), results.end(), [](const auto &a, const auto &b) {
        return a.first > b.first;
    });

    int bestDigit       = results[0].second;
    int secondBestDigit = results[1].second;
    int thirdBestDigit  = results[2].second;

    float bestProb       = results[0].first;
    float secondBestProb = results[1].first;
    float thirdBestProb  = results[2].first;


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

    m_isTraining=false;
    m_isEvaluating=false;

    m_probabilities.clear();
    emit predictionChanged();
    emit modelEvaluationChanged();
}

void AppManager::trainModel(){

    if (m_isTraining) {
        return;
    }

    m_isTraining = true;
    m_isEvaluating =false;
    m_trainingProgress= 0.0f;

    std::cout << "Training started" << std::endl;
    emit isTrainingChanged();
    emit trainingProgressChanged();

    auto future = QtConcurrent::run([this]() {
        std::cout << ">>> Background thread started..." << std::endl;

        try {
            if (m_nn.images.empty()) {
                std::cerr << "CRITICAL ERROR: No training images loaded in m_nn!" << std::endl;
            } else {

                // --- PASS THE CALLBACK TO TRAIN ---
                m_nn.train([this](float progress) {
                    QMetaObject::invokeMethod(this, [this, progress]() {
                        m_trainingProgress = progress;
                        emit trainingProgressChanged(); // <--- Triggers QML update!
                    }, Qt::QueuedConnection);
                });
            }
        } catch (const std::exception &e) {
            std::cerr << "EXCEPTION CAUGHT IN TRAIN THREAD: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "UNKNOWN CRASH IN TRAIN THREAD!" << std::endl;
        }

        QMetaObject::invokeMethod(this, [this]() {
            m_isTraining = false;
            m_actionDone = "Model was trained successfully";
            m_modelPerformance=0;
            emit isTrainingChanged();
            emit actionDoneChanged();
        }, Qt::QueuedConnection);
    });

}
void AppManager::resetModel(){
    m_nn.getRandomWeight();
    m_isEvaluating=false;
    m_isTraining=false;
    m_modelPerformance=0.0f;
    m_actionDone="Model reset successfully";
    emit actionDoneChanged();
    emit modelEvaluationChanged();

}
void AppManager::evaluateModel(){
    if (m_isEvaluating) {
        return;
    }

    m_isEvaluating = true;
    m_isTraining=false;
    std::cout << "Evaluating started" << std::endl;
    emit modelEvaluationChanged();

    auto future = QtConcurrent::run([this]() {
        std::cout << ">>> Background thread started..." << std::endl;

        try {
            if (m_nn.images.empty()) {
                std::cerr << "CRITICAL ERROR: No training images loaded in m_nn!" << std::endl;
            } else {
                m_modelPerformance= m_nn.evaluate();
            }
        } catch (const std::exception &e) {
            std::cerr << "EXCEPTION CAUGHT IN TRAIN THREAD: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "UNKNOWN CRASH IN TRAIN THREAD!" << std::endl;
        }

        QMetaObject::invokeMethod(this, [this]() {
            m_isEvaluating = false;
            m_actionDone = "Model was evaluated successfully";

            emit modelEvaluationChanged();
            emit actionDoneChanged();
        }, Qt::QueuedConnection);
    });
}
void AppManager::predictFromImage(const QVariant &imageVariant) {
    QImage img = qvariant_cast<QImage>(imageVariant);
    if (img.isNull()) {
        std::cerr << "Error: Invalid or null QImage passed from QML!" << std::endl;
        return;
    }

    QImage grayImg = img.convertToFormat(QImage::Format_Grayscale8);

    // 1. Find the exact Bounding Box of the drawn pixels
    int minX = grayImg.width(), maxX = 0;
    int minY = grayImg.height(), maxY = 0;
    bool hasPixels = false;

    for (int y = 0; y < grayImg.height(); ++y) {
        const uchar *line = grayImg.scanLine(y);
        for (int x = 0; x < grayImg.width(); ++x) {
            if (line[x] > 10) {
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
                hasPixels = true;
            }
        }
    }

    if (!hasPixels) {
        clearPrediction();
        return;
    }

    // Define croppedImg from the bounding box
    QRect boundingBox(minX, minY, (maxX - minX) + 1, (maxY - minY) + 1);
    QImage croppedImg = grayImg.copy(boundingBox);

    // 2. Crop and scale to fit inside 20x20 keeping aspect ratio
    QImage scaledCropped = croppedImg.scaled(20, 20, Qt::KeepAspectRatio, Qt::SmoothTransformation);

    // 3. Calculate Center of Mass (Centroid) of the scaled digit
    double totalMass = 0.0;
    double sumX = 0.0;
    double sumY = 0.0;

    for (int y = 0; y < scaledCropped.height(); ++y) {
        const uchar *line = scaledCropped.scanLine(y);
        for (int x = 0; x < scaledCropped.width(); ++x) {
            double mass = line[x] / 255.0;
            totalMass += mass;
            sumX += x * mass;
            sumY += y * mass;
        }
    }

    double cx = (totalMass > 0) ? (sumX / totalMass) : (scaledCropped.width() / 2.0);
    double cy = (totalMass > 0) ? (sumY / totalMass) : (scaledCropped.height() / 2.0);

    // Position centroid exactly at (13.5, 13.5) in the 28x28 target
    int offsetX = qRound(13.5 - cx);
    int offsetY = qRound(13.5 - cy);

    offsetX = std::max(0, std::min(offsetX, 28 - scaledCropped.width()));
    offsetY = std::max(0, std::min(offsetY, 28 - scaledCropped.height()));

    // 4. Draw centered to the 28x28 black canvas
    QImage mnistReady(28, 28, QImage::Format_Grayscale8);
    mnistReady.fill(0);

    QPainter painter(&mnistReady);
    painter.drawImage(offsetX, offsetY, scaledCropped);
    painter.end();

    // 5. Normalization and pixel buffer population
    float maxVal = 0.01f;
    for (int y = 0; y < 28; ++y) {
        const uchar *line = mnistReady.scanLine(y);
        for (int x = 0; x < 28; ++x) {
            float val = line[x] / 255.0f;
            if (val > maxVal) maxVal = val;
        }
    }

    QList<float> pixelBuffer;
    pixelBuffer.reserve(784);

    for (int y = 0; y < 28; ++y) {
        const uchar *line = mnistReady.scanLine(y);
        for (int x = 0; x < 28; ++x) {
            float val = (line[x] / 255.0f) / maxVal;
            if (val > 0.35f) {
                val = 1.0f;
            } else if (val < 0.15f) {
                val = 0.0f;
            }
            pixelBuffer.append(val);
        }
    }

    predictFromPixels(pixelBuffer);
}