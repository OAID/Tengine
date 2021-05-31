#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "common.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"

#define COCO
#define DEFAULT_REPEAT_COUNT 1
#define DEFAULT_THREAD_COUNT 1

#ifdef MPI
const int POSE_PAIRS[14][2] = {{0, 1},  {1, 2},  {2, 3}, {3, 4},  {1, 5},   {5, 6},   {6, 7},
                               {1, 14}, {14, 8}, {8, 9}, {9, 10}, {14, 11}, {11, 12}, {12, 13}};
// std::string model_file = "models/openpose_mpi.tmfile";
int nPoints = 15;
#endif

#ifdef COCO
const int POSE_PAIRS[17][2] = {{1, 2},  {1, 5},   {2, 3},   {3, 4}, {5, 6},  {6, 7},   {1, 8},  {8, 9},  {9, 10},
                               {1, 11}, {11, 12}, {12, 13}, {1, 0}, {0, 14}, {14, 16}, {0, 15}, {15, 17}};
// std::string model_file = "models/openpose_coco.tmfile";
int nPoints = 18;
#endif

#ifdef BODY25
const int POSE_PAIRS[24][2] = {{1, 2},   {1, 5},   {2, 3},   {3, 4},   {5, 6},   {6, 7},   {1, 8},   {8, 9},
                               {9, 10},  {10, 11}, {11, 24}, {11, 22}, {22, 23}, {8, 12},  {12, 13}, {13, 14},
                               {14, 21}, {14, 19}, {19, 20}, {1, 0},   {0, 15},  {16, 18}, {0, 16},  {15, 17}};
// std::string model_file = "models/openpose_body25.tmfile"
int nPoints = 25;
#endif

void get_input_data_pose(cv::Mat img, float* input_data, int img_h, int img_w)
{
    cv::resize(img, img, cv::Size(img_h, img_w));
    img.convertTo(img, CV_32FC3);

    float* img_data = ( float* )img.data;
    int hw = img_h * img_w;
    double scalefactor = 1.0 / 255;
    float mean[3] = {0, 0, 0};

    for (int h = 0; h < img_h; h++)
    {
        for (int w = 0; w < img_w; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                input_data[c * hw + h * img_w + w] = scalefactor * (*img_data - mean[c]);
                img_data++;
            }
        }
    }
}

void post_process_pose(cv::Mat img, cv::Mat frameCopy, float threshold, float* outdata, int num, int H, int W)
{
    std::vector<cv::Point> points(nPoints);

    int frameWidth = img.rows;
    int frameHeight = img.cols;
    std::cout << "KeyPoints Coordinate:" << std::endl;
    for (int n = 0; n < num; n++)
    {
        cv::Point maxloc;
        int piexlNums = H * W;
        double prob = -1;
        for (int piexl = 0; piexl < piexlNums; ++piexl)
        {
            if (outdata[piexl] > prob)
            {
                prob = outdata[piexl];
                maxloc.y = ( int )piexl / H;
                maxloc.x = ( int )piexl % W;
            }
        }
        cv::Point2f p(-1, -1);
        if (prob > threshold)
        {
            p = maxloc;
            p.y *= ( float )frameWidth / W;
            p.x *= ( float )frameHeight / H;

            cv::circle(frameCopy, cv::Point(( int )p.x, ( int )p.y), 4, cv::Scalar(255, 255, 0), -1);
            cv::putText(frameCopy, cv::format("%d", n), cv::Point(( int )p.x, ( int )p.y), cv::FONT_HERSHEY_PLAIN, 2,
                        cv::Scalar(0, 255, 255), 2);
        }
        points[n] = p;
        std::cout << n << ":" << p << std::endl;
        outdata += piexlNums;
    }

    int nPairs = sizeof(POSE_PAIRS) / sizeof(POSE_PAIRS[0]);

    for (int n = 0; n < nPairs; n++)
    {
        cv::Point2f partA = points[POSE_PAIRS[n][0]];
        cv::Point2f partB = points[POSE_PAIRS[n][1]];

        if (partA.x <= 0 || partA.y <= 0 || partB.x <= 0 || partB.y <= 0)
            continue;

        cv::line(img, partA, partB, cv::Scalar(0, 255, 255), 2);
        cv::circle(img, partA, 4, cv::Scalar(255, 255, 0), -1);
        cv::circle(img, partB, 4, cv::Scalar(255, 255, 0), -1);
    }
}

void show_usage()
{
    fprintf(stderr, "[Usage]:  [-h]\n    [-m model_file] [-i image_file] [-r repeat_count] [-t thread_count]\n");
}

int main(int argc, char* argv[])
{
    const char* model_file = nullptr;
    const char* image_file = nullptr;
    int repeat_count = DEFAULT_REPEAT_COUNT;
    int num_thread = DEFAULT_THREAD_COUNT;
    int img_h = 368;
    int img_w = 368;

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
                repeat_count = atoi(optarg);
                break;
            case 't':
                num_thread = atoi(optarg);
                break;
            case 'h':
                show_usage();
                return 0;
            default:
                break;
        }
    }

    /* check files */
    if (model_file == nullptr)
    {
        fprintf(stderr, "Error: Tengine model file not specified!\n");
        show_usage();
        return -1;
    }

    if (image_file == nullptr)
    {
        fprintf(stderr, "Error: Image file not specified!\n");
        show_usage();
        return -1;
    }

    if (!check_file_exist(model_file) || !check_file_exist(image_file))
        return -1;

    /* set runtime options */
    struct options opt;
    opt.num_thread = num_thread;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_FP32;
    opt.affinity = 0;

    /* inital tengine */
    init_tengine();
    fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());

    /* create graph, load tengine model xxx.tmfile */
    graph_t graph = create_graph(nullptr, "tengine", model_file);
    if (graph == nullptr)
    {
        std::cout << "Create graph0 failed\n";
        return -1;
    }

    /* set the input shape to initial the graph, and prerun graph to infer shape */
    int channel = 3;
    int img_size = img_h * img_w * channel;
    int dims[] = {1, channel, img_h, img_w};    // nchw

    float* input_data = ( float* )malloc(sizeof(float) * img_size);

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

    if (set_tensor_buffer(input_tensor, input_data, img_size * 4) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }    

    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(graph, opt) < 0)
    {
        fprintf(stderr, "Prerun graph failed\n");
        return -1;
    }

    /* prepare process input data, set the data mem to input tensor */
    cv::Mat frame = cv::imread(image_file);
    get_input_data_pose(frame, input_data, img_h, img_w);

    /* run graph */
    double min_time = __DBL_MAX__;
    double max_time = -__DBL_MAX__;
    double total_time = 0.;
    for (int i = 0; i < 1; i++)
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
    fprintf(stderr, "Repeat %d times, thread %d, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n", 1, 1,
            total_time, max_time, min_time);
    fprintf(stderr, "--------------------------------------\n");

    /* get the result of classification */
    tensor_t out_tensor = get_graph_output_tensor(graph, 0, 0);
    int out_dim[4];

    if (get_tensor_shape(out_tensor, out_dim, 4) <= 0)
    {
        return -1;
    }

    float* outdata = ( float* )get_tensor_buffer(out_tensor);
    int num = nPoints;
    int H = out_dim[2];
    int W = out_dim[3];
    float show_threshold = 0.1;
    cv::Mat frameCopy = frame.clone();

    post_process_pose(frame, frameCopy, show_threshold, outdata, num, H, W);

    cv::imwrite("Output-Keypionts.jpg", frameCopy);
    cv::imwrite("Output-Skeleton.jpg", frame);

    /* release tengine */
    free(input_data);
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return 0;
}

