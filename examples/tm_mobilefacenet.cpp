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
* Copyright (c) 2020, OPEN AI LAB
* Author: qtang@openailab.com
*/

#include <cstdlib>
#include <cstdio>
#include <vector>

#include "common.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"

#define DEFAULT_MEAN1 104.007
#define DEFAULT_MEAN2 116.669
#define DEFAULT_MEAN3 122.679

#define MOBILE_FACE_HEIGHT 110
#define MOBILE_FACE_WIDTH  110

graph_t graph;
tensor_t input_tensor;
tensor_t output_tensor;
int feature_len;

void init(const char* modelfile)
{
    int dims[4] = {1, 3, MOBILE_FACE_HEIGHT, MOBILE_FACE_WIDTH};
    init_tengine();
    fprintf(stderr, "tengine version: %s\n", get_tengine_version());
    graph = create_graph(NULL, "tengine", modelfile);
    if (graph == NULL)
    {
        fprintf(stderr, "grph nullptr\n");
    }
    else
    {
        fprintf(stderr, "success init graph\n");
    }
    input_tensor = get_graph_input_tensor(graph, 0, 0);
    set_tensor_shape(input_tensor, dims, 4);
    int rc = prerun_graph(graph);
    output_tensor = get_graph_output_tensor(graph, 0, 0);
    get_tensor_shape(output_tensor, dims, 4);
    feature_len = dims[1];
    fprintf(stderr, "mobilefacenet prerun %d\n", rc);
}

int getFeature(const char* imagefile, float* feature)
{
    int height = MOBILE_FACE_HEIGHT;
    int width = MOBILE_FACE_WIDTH;
    int img_size = height * width * 3;
    float means[3] = {DEFAULT_MEAN1, DEFAULT_MEAN2, DEFAULT_MEAN3};
    float scales[3] = {1, 1, 1};
    std::vector<float> input_data(img_size);
    get_input_data(imagefile, input_data.data(), height, width, means, scales);
    set_tensor_buffer(input_tensor, input_data.data(), img_size * sizeof(float));
    if (run_graph(graph, 1) < 0)
    {
        fprintf(stderr, "run_graph fail");
        return -1;
    }
    float* data = (float*)get_tensor_buffer(output_tensor);
    int outsize;
    outsize = get_tensor_buffer_size(output_tensor) / sizeof(float);
    for (int i = 0; i < outsize; i++)
        feature[i] = data[i];
    return outsize;
}

void normalize(float* feature, int size)
{
    float norm = 0;
    for (int i = 0; i < size; ++i)
    {
        norm += feature[i] * feature[i];
    }
    for (int i = 0; i < size; ++i)
    {
        feature[i] /= sqrt(norm);
    }
}

void release()
{
    release_graph_tensor(input_tensor);
    release_graph_tensor(output_tensor);
    destroy_graph(graph);
}

void show_usage()
{
    fprintf(stderr, "[Usage]:  [-h]\n    [-m model_file] [-a person_a -b person_b]\n [-t thread_count]\n");
    fprintf(stderr, "\nmobilefacenet example: \n    ./mobilefacenet -m /path/to/mobilenet.tmfile -a "
                    "/path/to/person_a.jpg -b /path/to/person_b.jpg\n");
}

int main(int argc, char* argv[])
{
    char* model_file = NULL;
    char* person_a = NULL;
    char* person_b = NULL;

    int res;
    while ((res = getopt(argc, argv, "m:a:b:h")) != -1)
    {
        switch (res)
        {
        case 'm':
            model_file = optarg;
            break;
        case 'a':
            person_a = optarg;
            break;
        case 'b':
            person_b = optarg;
            break;
        case 'h':
            show_usage();
            return 0;
        default:
            break;
        }
    }

    /* check files */
    if (model_file == NULL)
    {
        fprintf(stderr, "Error: Tengine model file not specified!\n");
        show_usage();
        return -1;
    }

    if (!check_file_exist(model_file) || !check_file_exist(person_a) || !check_file_exist(person_b))
        return -1;

    init(model_file);

    std::vector<float> featurea(feature_len);
    std::vector<float> featureb(feature_len);

    int outputsizea = getFeature(person_a, featurea.data());
    int outputsizeb = getFeature(person_b, featureb.data());

    if (outputsizea != feature_len || outputsizeb != feature_len)
    {
        fprintf(stderr, "getFeature feature out len error");
    }

    normalize(featurea.data(), feature_len);
    normalize(featureb.data(), feature_len);

    float sim = 0;
    for (int i = 0; i < feature_len; ++i)
    {
        sim += featurea[i] * featureb[i];
    }
    fprintf(stderr, "the cosine sim of person_a and person_b is %f\n", sim);

    release();
    return 0;
}
