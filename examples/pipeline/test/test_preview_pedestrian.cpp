#include "../actor/draw_video.h"
#include "../actor/pedestrian_detection.h"
#include "../actor/video_camera.h"
#include "../graph/graph.h"
#include <chrono>
#include <opencv2/opencv.hpp>
using namespace pipe;

int main() {
  Graph g;
  auto cam = g.add_node<VideoCamera, std::string>("");
  auto draw = g.add_node<DrawVideo>();
  auto detect_ped = g.add_node<PedestrianDetection>("mobilenet_ssd.tmfile");

  auto cam_det = g.add_edge<InstantEdge<cv::Mat>>(100);
  auto det_draw = g.add_edge<InstantEdge<cv::Mat>>(100);

  cam->set_output<0>(cam_det);
  detect_ped->set_input<0>(cam_det);
  detect_ped->set_output<0>(det_draw);
  draw->set_input<0>(det_draw);

  g.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(60000));
}