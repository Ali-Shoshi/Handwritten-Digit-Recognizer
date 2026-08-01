cmake_minimum_required(VERSION 3.16)

project(HandwrittenDigitRecognizer VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 REQUIRED COMPONENTS Quick Concurrent)

qt_standard_project_setup(REQUIRES 6.10)

# Content of downloaded design from Figma to Qt or similar
if (EXISTS "${CMAKE_CURRENT_LIST_DIR}/importedcontent/CMakeLists.txt")
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/importedcontent)
endif()

qt_add_executable(appHandwrittenDigitRecognizer
    main.cpp
)

qt_add_qml_module(appHandwrittenDigitRecognizer
    URI HandwrittenDigitRecognizer
    QML_FILES
        Main.qml
        SOURCES appmanager.h appmanager.cpp
        SOURCES
        SOURCES
        SOURCES datasetloader.h datasetloader.cpp
        SOURCES neuralnetwork.h neuralnetwork.cpp
)

# Qt for iOS sets MACOSX_BUNDLE_GUI_IDENTIFIER automatically since Qt 6.1.
# If you are developing for iOS or macOS you should consider setting an
# explicit, fixed bundle identifier manually though.
set_target_properties(appHandwrittenDigitRecognizer PROPERTIES
#    MACOSX_BUNDLE_GUI_IDENTIFIER com.example.appHandwrittenDigitRecognizer
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

target_link_libraries(appHandwrittenDigitRecognizer
    PRIVATE Qt6::Quick
    PRIVATE Qt6::Concurrent
)

option(BUILD_TRAINING_SMOKE_TEST "Build the MNIST training smoke test" OFF)
if (BUILD_TRAINING_SMOKE_TEST)
    add_executable(training_smoke
        training_smoke.cpp
        neuralnetwork.cpp
        datasetloader.cpp
    )
    target_link_libraries(training_smoke PRIVATE Qt6::Concurrent)
endif()

install(TARGETS appHandwrittenDigitRecognizer
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
