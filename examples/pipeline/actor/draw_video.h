#pragma once
#include "../graph/node.h"
#include <opencv2/opencv.hpp>

namespace pipe {

class DrawVideo : public Node<Param<cv::Mat>, Param<void>> {
public:
  void exec() override {
    cv::Mat mat;
    while (true) {

      auto suc = input<0>()->pop(mat);
      if (not suc) {
        continue;
      }
      // fprintf(stdout, "show\n");
      cv::imshow("camera", mat);
      cv::waitKey(25);
    }
  }

  ~DrawVideo() { cv::destroyAllWindows(); }
};

} // namespace pipe