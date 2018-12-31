#ifndef __MODEL_CONFIG_HPP__
#define __MODEL_CONFIG_HPP__

typedef struct
{
    const char* model_name;
    int img_h;
    int img_w;
    float scale;
    float mean[3];
    const char* proto_file;
    const char* model_file;
    const char* label_file;
} Model_Config;

const Model_Config* get_model_config(const char* model_name);

#endif    // __MODEL_CONFIG_HPP__
