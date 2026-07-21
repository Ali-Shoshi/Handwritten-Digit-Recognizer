#ifndef DATASETLOADER_H
#define DATASETLOADER_H

#include <vector>
#include <string>

class DatasetLoader {
private:
    static int reverseInt(int i);

public:
    static bool loadImages(const std::string& filename, std::vector<std::vector<float>>& outImages);
    static bool loadLabels(const std::string& filename, std::vector<int>& outLabels);
};

#endif // DATASETLOADER_H