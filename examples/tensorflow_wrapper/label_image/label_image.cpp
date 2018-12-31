/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2018, Open AI Lab
 * Author: jingyou@openailab.com
 */
#include <fstream>
#include <utility>
#include <vector>
#include <iomanip>
#include <opencv2/opencv.hpp>

#include "tensorflow/c/c_api.h"
#include "common.hpp"

using namespace std;

static int ReadEntireFile(const char* fname, vector<char>& buf)
{
    ifstream fs(fname, ios::binary | ios::in);
    if(!fs)
    {
        cout << fname << " does not exist" << endl;
        return -1;
    }

    fs.seekg(0, ios::end);
    int fsize = fs.tellg();

    fs.seekg(0, ios::beg);
    buf.resize(fsize);
    fs.read(buf.data(), fsize);

    fs.close();
    return 0;
}

static float* ReadImageFile(const char* image_file, cv::Mat& img, const int input_height, const int input_width,
                            const float input_mean, const float input_std)
{
    // Read image
    cv::Mat frame = cv::imread(image_file);
    if(!frame.data)
    {
        cout << image_file << " does not exist" << endl;
        return nullptr;
    }

    // Convert BGR to RGB
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

    // Resize the image
    cv::Mat img_resized;
    cv::Size input_geometry = cv::Size(input_width, input_height);
    cv::resize(frame, img_resized, input_geometry);

    // Convert to float 32, channel 3
    img_resized.convertTo(img_resized, CV_32FC3);

    // Normalization
    img = (img_resized - input_mean) / input_std;

    std::vector<cv::Mat> input_channels;
    float* input_data = ( float* )std::malloc(input_height * input_width * 3 * 4);
    float* ptr = input_data;

    for(int i = 0; i < 3; ++i)
    {
        cv::Mat channel(input_height, input_width, CV_32FC1, ptr);
        input_channels.push_back(channel);
        ptr += input_height * input_width;
    }

    cv::split(img, input_channels);

    return input_data;
}

static int ReadLabelsFile(const char* labels_file, vector<string>* result)
{
    ifstream fs(labels_file);
    if(!fs)
    {
        cout << labels_file << " does not exist" << endl;
        return -1;
    }

    result->clear();
    string line;
    while(getline(fs, line))
    {
        result->push_back(line);
    }

    return 0;
}

static TF_Session* LoadGraph(const char* model_file, TF_Graph* graph)
{
    TF_Status* s = TF_NewStatus();

    vector<char> model_buf;

    if(ReadEntireFile(model_file, model_buf) < 0)
        return nullptr;

    TF_Buffer graph_def = {model_buf.data(), model_buf.size(), nullptr};

    TF_ImportGraphDefOptions* import_opts = TF_NewImportGraphDefOptions();
    TF_ImportGraphDefOptionsSetPrefix(import_opts, "");
    TF_GraphImportGraphDef(graph, &graph_def, import_opts, s);

    if(TF_GetCode(s) != TF_OK)
    {
        printf("load graph failed!\n Error: %s\n", TF_Message(s));
        return nullptr;
    }

    TF_SessionOptions* sess_opts = TF_NewSessionOptions();
    TF_Session* session = TF_NewSession(graph, sess_opts, s);
    assert(TF_GetCode(s) == TF_OK);

    TF_DeleteStatus(s);

    return session;
}

static int PrintTopLabels(const vector<TF_Tensor*>& outputs, const char* labels_file)
{
    vector<string> labels;
    int read_labels_status = ReadLabelsFile(labels_file, &labels);
    if(read_labels_status < 0)
        return -1;

    int label_count = labels.size();
    int N = std::min<int>(label_count, 5);

    float* data = ( float* )TF_TensorData(outputs[0]);

    vector<pair<int, float>> scores;
    for(int i = 0; i < label_count; i++)
    {
        scores.push_back(pair<int, float>({i, data[i]}));
    }

    sort(scores.begin(), scores.end(),
         [](const pair<int, float>& left, const pair<int, float>& right) { return left.second > right.second; });

    for(int pos = 0; pos < N; pos++)
    {
        const int label_index = scores[pos].first;
        const float score = scores[pos].second;
        cout << std::fixed << std::setprecision(4) << score << " - \"" << labels[label_index] << "\"" << endl;
    }
    return 0;
}

/* To make tensor release happy...*/
static void dummy_deallocator(void* data, size_t len, void* arg) {}

int main(int argc, char* argv[])
{
    const string root_path = get_root_path();
#ifdef MOBILE_NET
    const string model_file = root_path + "models/frozen_mobilenet_v1_224.pb";
    const string image_file = root_path + "tests/images/cat.jpg";
    const string label_file = root_path + "models/synset_words.txt";
    int input_height = 224;
    int input_width = 224;
    float input_mean = -127;
    float input_std = 127;
#elif RESNET50
    const string model_file = root_path + "models/frozen_resnet50v1.pb";
    const string image_file = root_path + "tests/images/bike.jpg";
    const string label_file = root_path + "models/synset_words.txt";
    int input_height = 224;
    int input_width = 224;
    float input_mean = 0;
    float input_std = 1;
#else
    const string model_file = root_path + "models/inception_v3_2016_08_28_frozen.pb";
    const string image_file = root_path + "tests/images/grace_hopper.jpg";
    const string label_file = root_path + "models/imagenet_slim_labels.txt";
    int input_height = 299;
    int input_width = 299;
    float input_mean = 0;
    float input_std = 255;
#endif

#ifdef MOBILE_NET
    string input_layer = "input";
    string output_layer = "MobilenetV1/Predictions/Softmax";
#elif RESNET50
    string input_layer = "input";
    string output_layer = "resnet_v1_50/predictions/Softmax";
#else
    string input_layer = "input";
    string output_layer = "InceptionV3/Predictions/Softmax";
#endif

    // Load and initialize the model
    TF_Graph* graph = TF_NewGraph();
    TF_Session* session = LoadGraph(model_file.c_str(), graph);
    if(!session)
        return -1;

    // Read image file
    cv::Mat img;
    float* input_data = ReadImageFile(image_file.c_str(), img, input_height, input_width, input_mean, input_std);
    if(!input_data)
        return -1;

    // Create input tensor
    vector<TF_Output> input_names;
    vector<TF_Tensor*> input_values;

    TF_Operation* input_name = TF_GraphOperationByName(graph, input_layer.c_str());
    input_names.push_back({input_name, 0});

    const int64_t dim[4] = {1, input_height, input_width, 3};

    TF_Tensor* input_tensor = TF_NewTensor(TF_FLOAT, dim, 4, input_data, sizeof(float) * input_height * input_width * 3,
                                           dummy_deallocator, nullptr);
    input_values.push_back(input_tensor);

    // Get output value
    vector<TF_Output> output_names;

    TF_Operation* output_name = TF_GraphOperationByName(graph, output_layer.c_str());
    output_names.push_back({output_name, 0});

    vector<TF_Tensor*> output_values(output_names.size(), nullptr);

    // Actually run the image through the model
    TF_Status* s = TF_NewStatus();
    TF_SessionRun(session, nullptr, input_names.data(), input_values.data(), input_names.size(), output_names.data(),
                  output_values.data(), output_names.size(), nullptr, 0, nullptr, s);

    // Do something interesting with the results we've generated
    cout << "---------- Prediction for " << image_file << " ----------" << endl;
    free(input_data);
    int print_status = PrintTopLabels(output_values, label_file.c_str());
    if(print_status < 0)
    {
        return -1;
    }

    return 0;
}
