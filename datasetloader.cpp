#include "DatasetLoader.h"
#include <fstream>
#include <iostream>

int DatasetLoader::reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

bool DatasetLoader::loadImages(const std::string& filename, std::vector<std::vector<float>>& outImages) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open image file: " << filename << std::endl;
        return false;
    }

    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&number_of_images, sizeof(number_of_images));
    file.read((char*)&n_rows, sizeof(n_rows));
    file.read((char*)&n_cols, sizeof(n_cols));

    magic_number = reverseInt(magic_number);
    number_of_images = reverseInt(number_of_images);
    n_rows = reverseInt(n_rows);
    n_cols = reverseInt(n_cols);

    int image_size = n_rows * n_cols;
    outImages.resize(number_of_images, std::vector<float>(image_size));

    for (int i = 0; i < number_of_images; ++i) {
        std::vector<unsigned char> element(image_size);
        file.read((char*)element.data(), image_size);

        // Normalize pixels from 0-255 to 0.0-1.0 for the neural network
        for (int j = 0; j < image_size; ++j) {
            outImages[i][j] = (float)element[j] / 255.0f;
        }
    }
    return true;
}

bool DatasetLoader::loadLabels(const std::string& filename, std::vector<int>& outLabels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open label file: " << filename << std::endl;
        return false;
    }

    int magic_number = 0;
    int number_of_items = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&number_of_items, sizeof(number_of_items));

    magic_number = reverseInt(magic_number);
    number_of_items = reverseInt(number_of_items);

    outLabels.resize(number_of_items);
    std::vector<unsigned char> elements(number_of_items);
    file.read((char*)elements.data(), number_of_items);

    for (int i = 0; i < number_of_items; ++i) {
        outLabels[i] = (int)elements[i];
    }
    return true;
}