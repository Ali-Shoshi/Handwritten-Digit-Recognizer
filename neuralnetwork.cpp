#include "neuralnetwork.h"
#include <cmath>
#include "DatasetLoader.h"
#include <ctime>
#include <random>

NeuralNetwork::NeuralNetwork() {
    getRandomWeight();
    bool imagesLoaded = DatasetLoader::loadImages("Dataset/train-images.idx3-ubyte", images);
    bool labelsLoaded = DatasetLoader::loadLabels("Dataset/train-labels.idx1-ubyte", labels);
    train();
}

double NeuralNetwork::randomNumber(int inputSize) {
    static std::random_device rd;
    static std::mt19937 gen(rd());

    double stddev = std::sqrt(2.0 / inputSize);
    std::normal_distribution<double> d(0.0, stddev);
    return d(gen);
}

void NeuralNetwork::getRandomWeight(){

    for(int i =0 ; i< 784 ; i++){
        for( int j =0; j<128 ; j++){
            WIH[i][j]=randomNumber(784);
        }
    }
    for(int i =0 ; i< 128 ; i++){
        for( int j =0; j<64 ; j++){
            WHH[i][j]=randomNumber(64);
        }
    }
    for(int i =0 ; i< 64 ; i++){
        for( int j =0; j<10 ; j++){
            WHO[i][j]=randomNumber(10);
        }
    }

    for(int i =0 ; i< 128 ; i++){
        biasHidden1[i]=randomNumber(128);
    }
    for(int i =0 ; i< 64 ; i++){
        biasHidden2[i]=randomNumber(64);
    }
    for(int i =0 ; i< 10 ; i++){
        biasOutput[i]=randomNumber(10);
    }
    return;
}

double NeuralNetwork::ReLU(double x){
    return (x>0)?x:0;
}

void NeuralNetwork::train(){


}

void NeuralNetwork::forward(){

    for( int i=0 ; i<128; i++){
        hidden1Neuron[i]=0;
        for ( int j =0  ; j< 784 ; j++){
            hidden1Neuron[i]+=WIH[j][i]*inputNeuron[j];
        }
        hidden1Neuron[i]+=biasHidden1[i];
        hidden1Neuron[i]=ReLU(hidden1Neuron[i]);
    }


    for( int i=0 ; i<64; i++){
        hidden2Neuron[i]=0;
        for ( int j =0  ; j< 128 ; j++){
            hidden2Neuron[i]+=WHH[j][i]*hidden1Neuron[j];
        }
        hidden2Neuron[i]+=biasHidden2[i];
        hidden2Neuron[i]=ReLU(hidden2Neuron[i]);
    }

    double max_logit = -1e9;
    for ( int i =0 ; i< 10 ; i++){
        outputNeuron[i]=0;
        for( int j =0 ; j < 64 ; j++){
            outputNeuron[i]+=WHO[j][i]*hidden2Neuron[j];
        }
        outputNeuron[i]+=biasOutput[i];
        if(outputNeuron[i] > max_logit) {
            max_logit = outputNeuron[i];
        }

    }
    double sum_exp =0;
    for(int i = 0; i < 10; i++) {
        outputNeuron[i] = std::exp(outputNeuron[i]- max_logit);
        sum_exp+=outputNeuron[i];
    }
    for( int i =0 ; i< 10 ; i++){
        outputNeuron[i]=outputNeuron[i]/sum_exp;
    }

}
void NeuralNetwork::evaluate(){

}