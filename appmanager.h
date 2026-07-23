#ifndef APPMANAGER_H
#define APPMANAGER_H

#include <QObject>
#include <QVariantList>
#include "NeuralNetwork.h"

class AppManager : public QObject {
    Q_OBJECT

    // Properties exposed to QML
    Q_PROPERTY(int bestPredictedDigit READ bestPredictedDigit NOTIFY predictionChanged)
    Q_PROPERTY(int secondBestPredictedDigit READ secondBestPredictedDigit NOTIFY predictionChanged)
    Q_PROPERTY(int thirdBestPredictedDigit READ thirdBestPredictedDigit NOTIFY predictionChanged)

    Q_PROPERTY(float bestProb READ bestProb NOTIFY predictionChanged)
    Q_PROPERTY(float secondBestProb READ secondBestProb NOTIFY predictionChanged)
    Q_PROPERTY(float thirdBestPredictedDigit READ thirdBestPredictedDigit NOTIFY predictionChanged)

    Q_PROPERTY(float evaluationOfModel READ evaluationOfModel NOTIFY modelEvaluationChanged)


    Q_PROPERTY(QVariantList probabilities READ probabilities NOTIFY predictionChanged)

public:
    explicit AppManager(QObject *parent = nullptr);

    // Getters for properties
    int bestPredictedDigit() const { return m_bestPredictedDigit; }
    int secondBestPredictedDigit() const { return m_secondBestPredictedDigit; }
    int thirdBestPredictedDigit() const { return m_thirdBestPredictedDigit; }

    float bestProb() const { return m_bestProb; }
    float secondBestProb() const { return m_secondBestProb; }
    float thirdBestProb() const { return m_thirdBestProb; }

    float modelEvaluationChanged() const {return m_modelPerformance;}

    QVariantList probabilities() const { return m_probabilities; }

    // Methods callable directly from QML
    Q_INVOKABLE void predictFromPixels(const QList<float> &pixelBuffer);
    Q_INVOKABLE void clearPrediction();
    Q_INVOKABLE void trainModel();
    Q_INVOKABLE void resetModel();
    Q_INVOKABLE void evaluateModel();

signals:
    // Signal emitted when prediction updates
    void predictionChanged();
    void modelEvaluationChanged();

private:
    NeuralNetwork m_nn;

    int m_bestPredictedDigit = -1;
    int m_secondBestPredictedDigit = -1;
    int m_thirdBestPredictedDigit = -1;

    float m_bestProb = 0.0f;
    float m_secondBestProb = 0.0f;
    float m_thirdBestProb = 0.0f;

    float m_modelPerformance = 0.0f ;

    QVariantList m_probabilities;
};
#endif
