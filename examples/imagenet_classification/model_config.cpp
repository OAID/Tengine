#include <iostream>
#include <string>
#include <fstream>
#include <cctype>
#include "model_config.hpp"
#include "tengine_c_api.h"

const Model_Config model_list[] = {
    {"squeezenet",
     227,
     227,
     1.f,
     {104.007, 116.669, 122.679},
     "sqz.prototxt",
     "squeezenet_v1.1.caffemodel",
     "synset_words.txt"},
    {"mobilenet",
     224,
     224,
     0.017,
     {104.007, 116.669, 122.679},
     "mobilenet_deploy.prototxt",
     "mobilenet.caffemodel",
     "synset_words.txt"},
    {"mobilenet_v2",
     224,
     224,
     0.017,
     {104.007, 116.669, 122.679},
     "mobilenet_v2_deploy.prototxt",
     "mobilenet_v2.caffemodel",
     "synset_words.txt"},
    {"resnet50",
     224,
     224,
     1.f,
     {104.007, 116.669, 122.679},
     "resnet50.prototxt",
     "resnet50.caffemodel",
     "synset_words.txt"},
    {"alexnet",
     227,
     227,
     1.f,
     {104.007, 116.669, 122.679},
     "alex_deploy.prototxt",
     "alexnet.caffemodel",
     "synset_words.txt"},
    {"googlenet",
     224,
     224,
     1.f,
     {104.007, 116.669, 122.679},
     "googlenet.prototxt",
     "googlenet.caffemodel",
     "synset_words.txt"},
    {"inception_v3",
     395,
     395,
     0.0078,
     {104.007, 116.669, 122.679},
     "deploy_inceptionV3.prototxt",
     "deploy_inceptionV3.caffemodel",
     "synset2015.txt"},
    {"inception_v4",
     299,
     299,
     1 / 127.5f,
     {104.007, 116.669, 122.679},
     "inception_v4.prototxt",
     "inception_v4.caffemodel",
     "synset_words.txt"},
    {"vgg16", 224, 224, 1.f, {104.007, 116.669, 122.679}, "vgg16.prototxt", "vgg16.caffemodel", "synset_words.txt"}};

/*!
 * @brief Get the model config content from the predefined list according to the model name
 * @param model_name The model name
 * @return The pointer to the model config content
 */
const Model_Config* get_model_config(const char* model_name)
{
    std::string name1 = model_name;
    for(unsigned int i = 0; i < name1.size(); i++)
        name1[i] = tolower(name1[i]);

    for(unsigned int i = 0; i < sizeof(model_list) / sizeof(Model_Config); i++)
    {
        std::string name2 = model_list[i].model_name;
        if(name1 == name2)
        {
            return &model_list[i];
        }
    }
    std::cerr << "Not support model name : " << model_name << "\n";
    return NULL;
}
