#pragma once
#include "../graph/node.h"
#include "common/c_api.h"
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <opencv2/opencv.hpp>
#include <sys/time.h>

namespace pipe {

struct Box_t {
  int x0;
  int y0;
  int x1;
  int y1;
  int class_idx;
  float score;
};

class PedestrianDetection : public Node<Param<cv::Mat>, Param<cv::Mat>> {
public:
  PedestrianDetection(std::string model_path) {
    /* inital tengine */
    init_tengine();
    fprintf(stderr, "tengine-lite library version: %s\n",
            get_tengine_version());
    m_graph = create_graph(NULL, "tengine", model_path.c_str());
    if (m_graph == nullptr) {
      fprintf(stderr, "create graph failed\n");
      return;
    }

    /* set runtime options */
    struct options opt;
    opt.num_thread = 1;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_FP32;
    opt.affinity = 0;

    /* set the input shape to initial the graph, and prerun graph to infer shape
     */
    int img_size = img_h * img_w * 3;
    int dims[] = {1, 3, img_h, img_w}; // nchw
    if (m_input_data == nullptr) {
      float *input_data = (float *)malloc(img_size * sizeof(float));
      m_input_data = std::unique_ptr<float>(input_data);
    }

    tensor_t input_tensor = get_graph_input_tensor(m_graph, 0, 0);
    if (input_tensor == NULL) {
      fprintf(stderr, "Get input tensor failed\n");
      return;
    }

    if (set_tensor_shape(input_tensor, dims, 4) < 0) {
      fprintf(stderr, "Set input tensor shape failed\n");
      return;
    }

    if (set_tensor_buffer(input_tensor, m_input_data.get(), img_size * 4) < 0) {
      fprintf(stderr, "Set input tensor buffer failed\n");
      return;
    }

    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(m_graph, opt) < 0) {
      fprintf(stderr, "Prerun graph failed\n");
      return;
    }
  }

  Box_t post_process_ssd(const cv::Mat &im, float threshold,
                         const float *outdata, int num) {
    const char *class_names[] = {
        "background", "aeroplane",   "bicycle", "bird",  "boat",
        "bottle",     "bus",         "car",     "cat",   "chair",
        "cow",        "diningtable", "dog",     "horse", "motorbike",
        "person",     "pottedplant", "sheep",   "sofa",  "train",
        "tvmonitor"};

    int raw_h = im.rows;
    int raw_w = im.cols;

    Box_t boxes[num] = {0};
    int box_count = 0;

    fprintf(stderr, "detect result num: %d \n", num);
    for (int i = 0; i < num; i++) {
      if (outdata[1] >= threshold) {
        Box_t box;

        box.class_idx = outdata[0];
        box.score = outdata[1];
        box.x0 = outdata[2] * raw_w;
        box.y0 = outdata[3] * raw_h;
        box.x1 = outdata[4] * raw_w;
        box.y1 = outdata[5] * raw_h;

        boxes[box_count] = box;
        box_count++;

        fprintf(stderr, "%s\t:%.1f%%\n", class_names[box.class_idx],
                box.score * 100);
        fprintf(stderr, "BOX:( %d , %d ),( %d , %d )\n", box.x0, box.y0, box.x1,
                box.y1);
      }
      outdata += 6;
    }

    Box_t max = {0};
    for (int i = 0; i < box_count; i++) {
      if (boxes[i].score > max.score) {
        max = boxes[i];
      }
    }

    return max;
  }

  void exec() override {
    cv::Mat mat;
    auto suc = input<0>()->pop(mat);
    if (not suc or mat.empty()) {
      return;
    }

    auto get_current_time = []() -> double {
      struct timeval tv;
      gettimeofday(&tv, NULL);
      return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    };

    /* prepare process input data, set the data mem to input tensor */
    auto time1 = get_current_time();

    cv::Mat input(img_h, img_w, CV_8UC3);
    cv::resize(mat, input, cv::Size(img_w, img_h));
    cv::cvtColor(input, input, CV_BGR2RGB);
    input.convertTo(input, CV_32FC3);

    float *img_data = (float *)input.data;
    /* nhwc to nchw */
    for (int h = 0; h < img_h; h++) {
      for (int w = 0; w < img_w; w++) {
        for (int c = 0; c < 3; c++) {
          int in_index = h * img_w * 3 + w * 3 + c;
          int out_index = c * img_h * img_w + h * img_w + w;
          m_input_data.get()[out_index] =
              (img_data[in_index] - mean[c]) * scale[c];
        }
      }
    }

    auto time2 = get_current_time();

    if (run_graph(m_graph, 1) < 0) {
      fprintf(stderr, "Run graph failed\n");
      return;
    }

    auto time3 = get_current_time();

    /* process the detection result */
    tensor_t output_tensor =
        get_graph_output_tensor(m_graph, 0, 0); //"detection_out"
    int out_dim[4];
    get_tensor_shape(output_tensor, out_dim, 4);
    float *output_data = (float *)get_tensor_buffer(output_tensor);

    /* postprocess*/
    fprintf(stdout, "out shape [%d %d %d %d]\n", out_dim[0], out_dim[1],
            out_dim[2], out_dim[3]);
    auto result =
        post_process_ssd(mat, show_threshold, output_data, out_dim[1]);

    auto time4 = get_current_time();

    fprintf(stdout, "preproc %.2f,  inference %.2f,  postproc %.2f \n",
            time2 - time1, time3 - time2, time4 - time3);

    if (result.score > show_threshold) {
      cv::Rect rect(std::max(0, result.x0), std::max(0, result.y0),
                    result.x1 - result.x0, result.y1 - result.y0);
      rect.width = std::min(rect.width, mat.cols - rect.x - 1);
      rect.height = std::min(rect.height, mat.rows - rect.y - 1);
      cv::rectangle(mat, rect, cv::Scalar(255, 255, 255), 3);
    }
    output<0>()->try_push(std::move(mat));
  }

  ~PedestrianDetection() {
    /* release tengine */
    postrun_graph(m_graph);
    destroy_graph(m_graph);
    release_tengine();
  }

private:
  graph_t m_graph;
  std::unique_ptr<float> m_input_data;

  int img_h = 300;
  int img_w = 300;
  float mean[3] = {127.5f, 127.5f, 127.5f};
  float scale[3] = {0.007843f, 0.007843f, 0.007843f};
  float show_threshold = 0.5f;
};

} // namespace pipe