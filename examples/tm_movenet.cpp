#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "common.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"

#define NUM_JOINTS   17
#define TARGET_H     192                //lightning = 192,thunder = 256
#define TARGET_W     192                //lightning = 192,thunder = 256
#define FEATURE_SIZE 48                 //lightning = 48,thunder = 64
#define KPTS_SCALE   0.0208282470703125 //lightning = 0.0208282470703125,thunder = 0.015625
typedef struct
{
    float x;
    float y;
    float score;
} keypoint;

template<class ForwardIterator>
inline static size_t argmax(ForwardIterator first, ForwardIterator last)
{
    return std::distance(first, std::max_element(first, last));
}

void show_usage()
{
    fprintf(
        stderr,
        "[Usage]:  [-h]\n    [-m model_file] [-i image_file] [-r repeat_count] [-t thread_count]\n");
}

static void get_input_fp32_data(const char* image_file, float* input_data,
                                int letterbox_rows, int letterbox_cols, const float* mean, const float* scale)
{
    cv::Mat sample = cv::imread(image_file, 1);
    cv::Mat img;

    if (sample.channels() == 1)
        cv::cvtColor(sample, img, cv::COLOR_GRAY2RGB);
    else
        cv::cvtColor(sample, img, cv::COLOR_BGR2RGB);

    /* letterbox process to support different letterbox size */
    float scale_letterbox;
    int resize_rows;
    int resize_cols;
    if ((letterbox_rows * 1.0 / img.rows) < (letterbox_cols * 1.0 / img.cols))
    {
        scale_letterbox = letterbox_rows * 1.0 / img.rows;
    }
    else
    {
        scale_letterbox = letterbox_cols * 1.0 / img.cols;
    }
    resize_cols = int(scale_letterbox * img.cols);
    resize_rows = int(scale_letterbox * img.rows);

    cv::resize(img, img, cv::Size(resize_cols, resize_rows));

    img.convertTo(img, CV_32FC3);
    // Generate a gray image for letterbox using opencv

    cv::Mat img_new(letterbox_rows, letterbox_cols, CV_32FC3, cv::Scalar(0, 0, 0));
    int top = (letterbox_rows - resize_rows) / 2;
    int bot = (letterbox_rows - resize_rows + 1) / 2;
    int left = (letterbox_cols - resize_cols) / 2;
    int right = (letterbox_cols - resize_cols + 1) / 2;
    // Letterbox filling
    cv::copyMakeBorder(img, img_new, top, bot, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    float* img_data = (float*)img_new.data;

    /* nhwc to nchw */
    for (int h = 0; h < letterbox_rows; h++)
    {
        for (int w = 0; w < letterbox_cols; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                int in_index = h * letterbox_cols * 3 + w * 3 + c;
                int out_index = c * letterbox_rows * letterbox_cols + h * letterbox_cols + w;
                input_data[out_index] = (img_data[in_index] - mean[c]) * scale[c];
            }
        }
    }
}

static void draw_result(const cv::Mat& bgr, std::vector<keypoint> pose)
{
    int skele_index[][2] = {{0, 1}, {0, 2}, {1, 3}, {2, 4}, {0, 5}, {0, 6}, {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10}, {11, 12}, {5, 11}, {11, 13}, {13, 15}, {6, 12}, {12, 14}, {14, 16}};
    int color_index[][3] = {
        {255, 0, 0},
        {0, 0, 255},
        {255, 0, 0},
        {0, 0, 255},
        {255, 0, 0},
        {0, 0, 255},
        {0, 255, 0},
        {255, 0, 0},
        {255, 0, 0},
        {0, 0, 255},
        {0, 0, 255},
        {0, 255, 0},
        {255, 0, 0},
        {255, 0, 0},
        {255, 0, 0},
        {0, 0, 255},
        {0, 0, 255},
        {0, 0, 255},
    };

    for (int i = 0; i < 18; i++)
    {
        if (pose[skele_index[i][0]].score > 0.3 && pose[skele_index[i][1]].score > 0.3)
            cv::line(bgr, cv::Point(pose[skele_index[i][0]].x, pose[skele_index[i][0]].y), cv::Point(pose[skele_index[i][1]].x, pose[skele_index[i][1]].y), cv::Scalar(color_index[i][0], color_index[i][1], color_index[i][2]), 2);
    }
    for (int i = 0; i < 17; i++)
    {
        if (pose[i].score > 0.3)
            cv::circle(bgr, cv::Point(pose[i].x, pose[i].y), 3, cv::Scalar(255, 0, 255), -1);
    }

    cv::imwrite("movenet_result.jpg", bgr);
}

static std::vector<keypoint> post_process(const float* center_data, const float* kpt_heatmap_data, const float* kpt_regress_data, const float* kpt_offset_data,
                                          std::vector<std::vector<float> > dist_y, std::vector<std::vector<float> > dist_x, int letterbox_rows, int letterbox_cols, int img_h, int img_w)
{
    int top_index = 0;
    float top_score = 0;

    top_index = int(argmax(center_data, center_data + FEATURE_SIZE * FEATURE_SIZE));
    top_score = *std::max_element(center_data, center_data + FEATURE_SIZE * FEATURE_SIZE);

    int ct_y = (top_index / FEATURE_SIZE);
    int ct_x = top_index - ct_y * FEATURE_SIZE;

    std::vector<float> kpt_ys_regress(NUM_JOINTS), kpt_xs_regress(NUM_JOINTS);
    int offset = ct_y * FEATURE_SIZE * NUM_JOINTS * 2 + ct_x * NUM_JOINTS * 2;
    for (size_t i = 0; i < NUM_JOINTS; i++)
    {
        kpt_ys_regress[i] = kpt_regress_data[i + offset] + (float)ct_y;
        kpt_xs_regress[i] = kpt_regress_data[i + offset + NUM_JOINTS] + (float)ct_x;
    }

    cv::Mat scores = cv::Mat(NUM_JOINTS, FEATURE_SIZE * FEATURE_SIZE, CV_32FC1);
    float* scores_data = (float*)scores.data;
    for (int i = 0; i < FEATURE_SIZE; i++)
    {
        for (int j = 0; j < FEATURE_SIZE; j++)
        {
            std::vector<float> score;
            for (int c = 0; c < NUM_JOINTS; c++)
            {
                float y = (dist_y[i][j] - kpt_ys_regress[c]) * (dist_y[i][j] - kpt_ys_regress[c]);
                float x = (dist_x[i][j] - kpt_xs_regress[c]) * (dist_x[i][j] - kpt_xs_regress[c]);
                float dist_weight = std::sqrt(y + x) + 1.8;
                scores_data[c * FEATURE_SIZE * FEATURE_SIZE + i * FEATURE_SIZE + j] = kpt_heatmap_data[i * FEATURE_SIZE * NUM_JOINTS + j * NUM_JOINTS + c] / dist_weight;
            }
        }
    }

    std::vector<int> kpts_ys, kpts_xs;
    for (int i = 0; i < NUM_JOINTS; i++)
    {
        top_index = 0;
        top_score = 0;
        top_index = int(argmax(scores_data + FEATURE_SIZE * FEATURE_SIZE * i, scores_data + FEATURE_SIZE * FEATURE_SIZE * (i + 1)));
        top_score = *std::max_element(scores_data + FEATURE_SIZE * FEATURE_SIZE * i, scores_data + FEATURE_SIZE * FEATURE_SIZE * (i + 1));

        int top_y = (top_index / FEATURE_SIZE);
        int top_x = top_index - top_y * FEATURE_SIZE;
        kpts_ys.push_back(top_y);
        kpts_xs.push_back(top_x);
    }

    float scale_letterbox;
    int resize_rows;
    int resize_cols;
    if ((letterbox_rows * 1.0 / img_h) < (letterbox_cols * 1.0 / img_w))
    {
        scale_letterbox = letterbox_rows * 1.0 / img_h;
    }
    else
    {
        scale_letterbox = letterbox_cols * 1.0 / img_w;
    }
    resize_cols = int(scale_letterbox * img_w);
    resize_rows = int(scale_letterbox * img_h);

    int tmp_h = (letterbox_rows - resize_rows) / 2;
    int tmp_w = (letterbox_cols - resize_cols) / 2;

    float ratio_x = (float)img_h / resize_rows;
    float ratio_y = (float)img_w / resize_cols;

    std::vector<keypoint> pose(NUM_JOINTS);
    for (int i = 0; i < NUM_JOINTS; i++)
    {
        float kpt_offset_x = kpt_offset_data[kpts_ys[i] * FEATURE_SIZE * NUM_JOINTS * 2 + kpts_xs[i] * NUM_JOINTS * 2 + i * 2];
        float kpt_offset_y = kpt_offset_data[kpts_ys[i] * FEATURE_SIZE * NUM_JOINTS * 2 + kpts_xs[i] * NUM_JOINTS * 2 + i * 2 + 1];

        float kpt_x = (kpts_xs[i] + kpt_offset_y) * KPTS_SCALE * letterbox_cols;
        float kpt_y = (kpts_ys[i] + kpt_offset_x) * KPTS_SCALE * letterbox_rows;

        pose[i].x = (kpt_x - tmp_w) * ratio_x;
        pose[i].y = (kpt_y - tmp_h) * ratio_y;
        pose[i].score = kpt_heatmap_data[kpts_ys[i] * FEATURE_SIZE * NUM_JOINTS + kpts_xs[i] * NUM_JOINTS + i];
    }

    return pose;
}

int main(int argc, char** argv)
{
    const char* model_file = nullptr;
    const char* image_file = nullptr;

    int img_c = 3;
    const float mean[3] = {127.5f, 127.5f, 127.5f};
    const float scale[3] = {1 / 127.5f, 1 / 127.5f, 1 / 127.5f};

    // allow none square letterbox, set default letterbox size
    int letterbox_rows = TARGET_H;
    int letterbox_cols = TARGET_W;

    int repeat_count = 1;
    int num_thread = 1;

    int res;
    while ((res = getopt(argc, argv, "m:i:r:t:h:")) != -1)
    {
        switch (res)
        {
        case 'm':
            model_file = optarg;
            break;
        case 'i':
            image_file = optarg;
            break;
        case 'r':
            repeat_count = std::strtoul(optarg, nullptr, 10);
            break;
        case 't':
            num_thread = std::strtoul(optarg, nullptr, 10);
            break;
        case 'h':
            show_usage();
            return 0;
        default:
            break;
        }
    }

    /* check files */
    if (nullptr == model_file)
    {
        fprintf(stderr, "Error: Tengine model file not specified!\n");
        show_usage();
        return -1;
    }

    if (nullptr == image_file)
    {
        fprintf(stderr, "Error: Image file not specified!\n");
        show_usage();
        return -1;
    }

    if (!check_file_exist(model_file) || !check_file_exist(image_file))
        return -1;

    cv::Mat img = cv::imread(image_file, 1);
    if (img.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", image_file);
        return -1;
    }

    /* set runtime options */
    struct options opt;
    opt.num_thread = num_thread;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_FP32;
    opt.affinity = 0;

    /* inital tengine */
    if (init_tengine() != 0)
    {
        fprintf(stderr, "Initial tengine failed.\n");
        return -1;
    }
    fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());

    /* create graph, load tengine model xxx.tmfile */
    graph_t graph = create_graph(nullptr, "tengine", model_file);
    if (graph == nullptr)
    {
        fprintf(stderr, "Create graph failed.\n");
        return -1;
    }

    int img_size = letterbox_rows * letterbox_cols * img_c;
    int dims[] = {1, 3, letterbox_rows, letterbox_cols};
    std::vector<float> input_data(img_size);

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);

    if (input_tensor == nullptr)
    {
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }

    if (set_tensor_shape(input_tensor, dims, 4) < 0)
    {
        fprintf(stderr, "Set input tensor shape failed\n");
        return -1;
    }

    if (set_tensor_buffer(input_tensor, input_data.data(), img_size * 4) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }

    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(graph, opt) < 0)
    {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return -1;
    }

    /* prepare process input data, set the data mem to input tensor */
    get_input_fp32_data(image_file, input_data.data(), letterbox_rows, letterbox_cols, mean, scale);

    /* run graph */
    double min_time = DBL_MAX;
    double max_time = DBL_MIN;
    double total_time = 0.;
    for (int i = 0; i < repeat_count; i++)
    {
        double start = get_current_time();
        if (run_graph(graph, 1) < 0)
        {
            fprintf(stderr, "Run graph failed\n");
            return -1;
        }
        double end = get_current_time();
        double cur = end - start;
        total_time += cur;
        min_time = std::min(min_time, cur);
        max_time = std::max(max_time, cur);
    }
    fprintf(stderr, "Repeat %d times, thread %d, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n", repeat_count, num_thread,
            total_time / repeat_count, max_time, min_time);
    fprintf(stderr, "--------------------------------------\n");

    tensor_t center = get_graph_tensor(graph, "center");
    tensor_t heatmap = get_graph_tensor(graph, "heatmap");
    tensor_t regress = get_graph_tensor(graph, "regress");
    tensor_t offset = get_graph_tensor(graph, "offset");
    float* center_data = (float*)get_tensor_buffer(center);
    float* kpt_heatmap_data = (float*)get_tensor_buffer(heatmap);
    float* kpt_offset_data = (float*)get_tensor_buffer(offset);
    float* kpt_regress_data = (float*)get_tensor_buffer(regress);

    std::vector<std::vector<float> > dist_y, dist_x;
    for (int i = 0; i < FEATURE_SIZE; i++)
    {
        std::vector<float> x, y;
        for (int j = 0; j < FEATURE_SIZE; j++)
        {
            x.push_back(j);
            y.push_back(i);
        }
        dist_y.push_back(y);
        dist_x.push_back(x);
    }
    std::vector<keypoint> pose = post_process(center_data, kpt_heatmap_data, kpt_regress_data, kpt_offset_data,
                                              dist_y, dist_x, letterbox_rows, letterbox_cols, img.rows, img.cols);

    draw_result(img, pose);

    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return 0;
}
