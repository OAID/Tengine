#pragma once
#include "../graph/node.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace pipe {

class VideoCamera : public Node<Param<void>, Param<cv::Mat>> {
public:
  VideoCamera(const std::string video_path) : m_path(video_path) {
    m_opened = false;
  }

  void exec() override {
    if (m_opened) {
      return;
    }

    cv::VideoCapture cap(0);
    if (not cap.isOpened()) {
      fprintf(stderr, "cannot open video %s\n", m_path.c_str());
      m_opened = true;
      return;
    }

    double rate = cap.get(CV_CAP_PROP_FPS);
    fprintf(stdout, "rate %lf\n", rate);
    while (true) {
      cv::Mat mat;
      if (not cap.read(mat)) {
        break;
      }

      if (mat.empty()) {
        break;
      }

      auto success = output<0>()->try_push(mat.clone());
      if (not success) {
        fprintf(stdout, "abandon\n");
      }
    }

    m_opened = true;
    cv::waitKey(1000 / std::max(1.0, rate));
  }

private:
  std::string m_path;
  bool m_opened;
};

} // namespace pipe