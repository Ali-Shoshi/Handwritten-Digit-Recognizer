#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <NeuralNetwork.h>
#include <QQmlContext>
#include "AppManager.h"

int main(int argc, char *argv[])
{
     QGuiApplication app(argc, argv);
     QQmlApplicationEngine engine;

     AppManager appManager;
    engine.rootContext()->setContextProperty("appManager", &appManager);

     QObject::connect(
         &engine,
         &QQmlApplicationEngine::objectCreationFailed,
         &app,
         []() { QCoreApplication::exit(-1); },
         Qt::QueuedConnection);
    engine.loadFromModule("HandwrittenDigitRecognizer", "Main");

    return QGuiApplication::exec();

    // NeuralNetwork nn;
    // std::cout << " Hello"<< std::endl;
    // std::cout << "Press Enter to exit...";
    // std::cin.get();
    // return 0;
}
