#include <unistd.h>

#include <iostream>
#include <functional>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sys/time.h>

#include "tengine_c_api.h"
#include "../benchmark/include/common_util.hpp"
#include "tengine_operations.h"

using namespace TEngine;
using namespace std;

const char* model_file = "./models/mnet.25-0000.tmfile";
const char* image_file = "./tests/images/face5.jpg";

const float CONF_THRESH = 0.8;
const float NMS_THRESH = 0.4;
int repeat_count = 1;

typedef struct abox
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    float** land_mark;
} abox;

const char* bbox_name[3] = {"face_rpn_bbox_pred_stride32", "face_rpn_bbox_pred_stride16", "face_rpn_bbox_pred_stride8"};

const char* score_name[3] = {"face_rpn_cls_prob_reshape_stride32", "face_rpn_cls_prob_reshape_stride16",
                             "face_rpn_cls_prob_reshape_stride8"};

const char* landmark_name[3] = {"face_rpn_landmark_pred_stride32", "face_rpn_landmark_pred_stride16",
                                "face_rpn_landmark_pred_stride8"};

const int stride[3] = {32, 16, 8};

float temp[6][4] = {
    // stride = 32
    {-248, -248, 263, 263},    // scale 32
    {-120, -120, 135, 135},    // scale 16
    // stride = 16
    {-56, -56, 71, 71},    // scale 8
    {-24, -24, 39, 39},    // scale 4
    // stride = 8
    {-8, -8, 23, 23},    // scale 2
    {0, 0, 15, 15}    // scale 1
};

template <typename T> std::vector<size_t> sort_index(const std::vector<T>& v)
{
    std::vector<size_t> idx(v.size());
    for(size_t i = 0; i != idx.size(); ++i)
    {
        idx[i] = i;
    }
    std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] > v[i2]; });
    return idx;
}

float* clip_landmark(float* landmark, float scale_width, float scale_height, float im_scale)
{
    if(landmark[0] < 0 || landmark[0] > scale_width)
    {
        landmark[0] = scale_width;
    }
    if(landmark[1] < 0 || landmark[1] > scale_height)
    {
        landmark[1] = scale_height;
    }
    landmark[0] = landmark[0] / im_scale;
    landmark[1] = landmark[1] / im_scale;
    return landmark;
}

void anchor_calculation(int height, int width, int stride_index, int A, float** anchors_reshape)
{
    for(int i = 0; i < height; i++)
    {
        int sh = i * stride[stride_index];
        for(int j = 0; j < width; j++)
        {
            int sw = j * stride[stride_index];
            for(int a = 0; a < A; a++)
            {
                anchors_reshape[i * width * A + j * A + a][0] = temp[a + A * stride_index][0] + sw;
                anchors_reshape[i * width * A + j * A + a][1] = temp[a + A * stride_index][1] + sh;
                anchors_reshape[i * width * A + j * A + a][2] = temp[a + A * stride_index][2] + sw;
                anchors_reshape[i * width * A + j * A + a][3] = temp[a + A * stride_index][3] + sh;
            }
        }
    }
    return;
}

void bbox_delta_data_reshape(int dims_bbox_delta[], float* bbox_data, float** box_delta_reshape, int height, int width)
{
    int sn = dims_bbox_delta[0];
    int sc = dims_bbox_delta[1];
    int sh = dims_bbox_delta[2];
    int sw = dims_bbox_delta[3];
    int bbox_pred_len = 4;
    int hwt = sh * sw;
    int idx = 0;
    int num_col_box_reshape = sn * hwt * sc / bbox_pred_len;
    float* new_bbox = new float[sn * sc * height * width];
    for(int nn = 0; nn < sn; nn++)
    {
        for(int hwc = 0; hwc < hwt; hwc++)
        {
            for(int cc = 0; cc < sc; cc++)
            {
                new_bbox[nn * hwt + hwc * sc + cc] = bbox_data[nn * sc * hwt + cc * hwt + hwc];
            }
        }
    }

    for(int i = 0; i < num_col_box_reshape; i++)
    {
        for(int j = 0; j < bbox_pred_len; j++)
        {
            box_delta_reshape[i][j] = new_bbox[idx];
            idx++;
        }
    }
    delete(new_bbox);
    return;
}

void score_reshape(int dims_bbox_score[], float* score_data, float* new_score)
{
    int idx = 0;
    int sn = dims_bbox_score[0];    // 1
    int sc = dims_bbox_score[1];    // 4
    int sh = dims_bbox_score[2];    // 1504
    int sw = dims_bbox_score[3];    // 1

    int bsc = sc / 2;
    float* score_temp = new float[sn * ( bsc )*sh * sw];
    float* mxnet_score = new float[sn * bsc * sh * sw];

    // NCHW format
    idx = 0;
    for(int n = 0; n < sn; n++)
    {
        for(int c = 0; c < bsc; c++)
        {
            for(int h = 0; h < sh; h++)
            {
                for(int w = 0; w < sw; w++)
                {
                    score_temp[idx] = score_data[n * ( sc )*sh * sw + (c + bsc) * sh * sw + h * sw + w];
                    idx++;
                }
            }
        }
    }

    idx = 0;
    for(int n = 0; n < sn; n++)
    {
        for(int c = 0; c < bsc; c++)
        {
            for(int h = 0; h < sh; h++)
            {
                for(int w = 0; w < sw; w++)
                {
                    mxnet_score[n * bsc * sh * sw + c * sh * sw + h * sw + w] = score_temp[idx];
                    idx++;
                }
            }
        }
    }

    // NHWC format
    int hwt = dims_bbox_score[2] * dims_bbox_score[3];
    for(int nn = 0; nn < sn; nn++)
    {
        for(int hwc = 0; hwc < hwt; hwc++)
        {
            for(int cc = 0; cc < bsc; cc++)
            {
                new_score[nn * hwt * bsc + hwc * bsc + cc] = mxnet_score[nn * bsc * hwt + cc * hwt + hwc];
            }
        }
    }
    delete(mxnet_score);
    delete(score_temp);
    return;
}

void landmark_reshape_function(int dims_bbox_landmark[], float* landmark_data, float*** pred_landmark,
                               float*** landmark_pred_reshape)
{
    int ln = dims_bbox_landmark[0];
    int lc = dims_bbox_landmark[1];
    int lh = dims_bbox_landmark[2];
    int lw = dims_bbox_landmark[3];
    int ld_hwt = lh * lw;
    float* new_landmark = new float[ln * lc * lh * lw];

    for(int nn = 0; nn < ln; nn++)
    {
        for(int hwc = 0; hwc < ld_hwt; hwc++)
        {
            for(int cc = 0; cc < lc; cc++)
            {
                new_landmark[nn * ld_hwt + hwc * lc + cc] = landmark_data[nn * lc * ld_hwt + cc * ld_hwt + hwc];
            }
        }
    }

    int idx = 0;
    for(int i = 0; i < ln * ld_hwt * lc / 10; i++)
    {
        for(int j = 0; j < 5; j++)
        {
            for(int z = 0; z < 2; z++)
            {
                landmark_pred_reshape[i][j][z] = new_landmark[idx];
                idx++;
            }
        }
    }
    delete(new_landmark);
    return;
}

std::vector<float**> landmark_pred(int K, int A, float im_scale, float*** landmark_pred_reshape,
                                   float** anchors_reshape, float*** pred_landmark, std::vector<float**> temp_lmark)
{
    float* local_anchor = new float[4];    // single anchor size

    for(int i = 0; i < K * A; i++)
    {
        memcpy(local_anchor, anchors_reshape[i], sizeof(float) * 4);
        int ld_widths = local_anchor[2] - local_anchor[0] + 1;
        int ld_heights = local_anchor[3] - local_anchor[1] + 1;
        int ld_ctr_x = local_anchor[0] + 0.5 * (ld_widths - 1);
        int ld_ctr_y = local_anchor[1] + 0.5 * (ld_heights - 1);
        for(int j = 0; j < 5; j++)
        {
            pred_landmark[i][j][0] = (landmark_pred_reshape[i][j][0] * ld_widths + ld_ctr_x) / im_scale;
            pred_landmark[i][j][1] = (landmark_pred_reshape[i][j][1] * ld_heights + ld_ctr_y) / im_scale;
        }
        temp_lmark.push_back(pred_landmark[i]);
    }
    delete(local_anchor);
    return temp_lmark;
}

abox bbox_tranform_inv(float* anchor, float** boxs_delta, int i, int imgw, int imgh, float im_scale)
{
    double pred_ctr_x, pred_ctr_y, src_ctr_x, src_ctr_y;
    double dst_ctr_x, dst_ctr_y, dst_scl_x, dst_scl_y;
    double src_w, src_h, pred_w, pred_h;
    src_w = anchor[2] - anchor[0] + 1;
    src_h = anchor[3] - anchor[1] + 1;
    src_ctr_x = anchor[0] + 0.5 * (src_w - 1);
    src_ctr_y = anchor[1] + 0.5 * (src_h - 1);
    dst_ctr_x = boxs_delta[i][0];
    dst_ctr_y = boxs_delta[i][1];
    dst_scl_x = boxs_delta[i][2];
    dst_scl_y = boxs_delta[i][3];
    pred_ctr_x = dst_ctr_x * src_w + src_ctr_x;
    pred_ctr_y = dst_ctr_y * src_h + src_ctr_y;
    pred_w = exp(dst_scl_x) * src_w;
    pred_h = exp(dst_scl_y) * src_h;

    boxs_delta[i][0] = pred_ctr_x - 0.5 * (pred_w - 1);

    if(boxs_delta[i][0] < 0)
        boxs_delta[i][0] = 0;
    if(boxs_delta[i][0] > imgw)
        boxs_delta[i][0] = imgw;

    boxs_delta[i][1] = pred_ctr_y - 0.5 * (pred_h - 1);

    if(boxs_delta[i][1] < 0)
        boxs_delta[i][1] = 0;
    if(boxs_delta[i][1] > imgh)
        boxs_delta[i][1] = imgh;

    boxs_delta[i][2] = pred_ctr_x + 0.5 * (pred_w - 1);

    if(boxs_delta[i][2] < 0)
        boxs_delta[i][2] = 0;
    if(boxs_delta[i][2] > imgw)
        boxs_delta[i][2] = imgw;

    boxs_delta[i][3] = pred_ctr_y + 0.5 * (pred_h - 1);

    if(boxs_delta[i][3] < 0)
        boxs_delta[i][3] = 0;
    if(boxs_delta[i][3] > imgh)
    {
        boxs_delta[i][3] = imgh;
    }

    abox tmp;
    tmp.x1 = boxs_delta[i][0] / im_scale;
    tmp.y1 = boxs_delta[i][1] / im_scale;
    tmp.x2 = boxs_delta[i][2] / im_scale;
    tmp.y2 = boxs_delta[i][3] / im_scale;

    return tmp;
}

void nms(std::vector<abox>& input_boxes, float nms_thresh)
{
    std::vector<float> vArea(input_boxes.size());
    for(int i = 0; i < ( int )input_boxes.size(); ++i)
    {
        vArea[i] =
            (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1) * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for(int i = 0; i < ( int )input_boxes.size(); ++i)
    {
        for(int j = i + 1; j < ( int )input_boxes.size();)
        {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float h = std::max(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if(ovr >= nms_thresh)
            {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}
void draw_target(std::vector<abox> all_pred_boxes, image img)
{
    std::string save_name = "saveTest.jpg";
    const char* class_names[] = {"faces"};

    float** lmark_points = new float*[5];
    for(int i = 0; i < 5; i++)
    {
        lmark_points[i] = new float[2];
    }

    for(int b = 0; b < ( int )all_pred_boxes.size(); b++)
    {
        abox box = all_pred_boxes[b];
        lmark_points = all_pred_boxes[b].land_mark;

        printf("%s\t: %.3f %%\n", class_names[0], box.score * 100);
        printf("BOX:( %g , %g ),( %g , %g )\n", box.x1, box.y1, box.x2, box.y2);

        draw_box(img, box.x1, box.y1, box.x2, box.y2, 2, 0, 0, 0);
        for(int l = 0; l < 5; l++)
        {
            draw_circle(img, lmark_points[l][0], lmark_points[l][1], 3, 50, 125, 25);
        }
    }
    save_image(img, "tengine_example_out");
    std::cout << "======================================\n";
    std::cout << "[DETECTED IMAGE SAVED]:\t" << save_name << "\n";
    std::cout << "======================================\n";

    return;
}

int main(int argc, char* argv[])
{
    int res;
    int ret = -1;

    while((res = getopt(argc, argv, "r:")) != -1)
    {
        switch(res)
        {
            case 'r':
                repeat_count = strtoul(optarg, NULL, 10);
                break;
            default:
                break;
        }
    }
    std::string save_file = "./images/saveTest.jpg";

    image img = imread(image_file);
    //img = rgb2bgr_premute(img);

    init_tengine();

    std::cout << "Tengine version: " << get_tengine_version() << "\n";

    if(request_tengine_version("1.0") < 0)
        return 1;

    // graph_t graph = create_graph(nullptr, "mxnet", text_file, model_file);
	graph_t graph = create_graph(nullptr, "tengine", model_file);
    if(graph == nullptr)
    {
        std::cout << "Create graph0 failed\n";
        return -1;
    }
    int val = 1;
    set_graph_attr(graph, "low_mem_mode", &val, sizeof(val));

    //================================================================
    //===================== Pre dealing with image ===================
    //================================================================
    int target_size = 1024;
    int max_size = 1980;
    int im_size_min = std::min(img.h, img.w);
    int im_size_max = std::max(img.h, img.w);

    double im_scale = double(target_size) / double(im_size_min);
    if(im_scale * im_size_max > max_size)
        im_scale = double(max_size) / double(im_size_max);

    int height = std::round(img.h * im_scale);
    int width = std::round(img.w * im_scale);

    image resImg = resize_image(img, width, height);

    int hw = height * width;
    int img_size = hw * 3;

    float* input_data = ( float* )malloc(sizeof(float) * img_size);
    memcpy(input_data, resImg.data, img_size * sizeof(float));

    //================================================================
    //===================== Set tensor info ==========================
    //================================================================
    const char* input_tensor_names[] = {"data"};

    tensor_t input_tensor1 = get_graph_tensor(graph, input_tensor_names[0]);

    int dims[] = {1, 3, height, width};

    set_tensor_shape(input_tensor1, dims, 4);
    set_tensor_buffer(input_tensor1, input_data, img_size * 4);

    ret = prerun_graph(graph);
    if(ret != 0)
    {
        std::cout << "Prerun graph failed, errno: " << get_tengine_errno() << "\n";
        return 0;
    }

    const char* repeat = std::getenv("REPEAT_COUNT");
    if(repeat)
        repeat_count = std::strtoul(repeat, NULL, 10);

    struct timeval t0, t1;
    float total_time = 0.f;

    for(int i = 0; i < repeat_count; i++)
    {
        gettimeofday(&t0, NULL);
        ret = run_graph(graph, 1);
        if(ret != 0)
        {
            std::cout << "Run graph failed, errno: " << get_tengine_errno() << "\n";
            return 1;
        }
        gettimeofday(&t1, NULL);

        float mytime = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        total_time += mytime;
    }
    std::cout << "run time for " << repeat_count << " cycles:" << total_time
              << " (ms), for single cycle: " << total_time / repeat_count << " (ms) \n"
              << endl;

    std::vector<abox> all_pred_boxes;
    std::vector<abox> proposals_list;
    std::vector<float**> landmark_list;
    std::vector<float> score_list;

    for(int stride_index = 0; stride_index < 3; stride_index++)
    {
        // ==================================================================
        // ========== This part is to get tensor information ================
        // ==================================================================
        tensor_t tensor_bbx = get_graph_tensor(graph, bbox_name[stride_index]);
        tensor_t tensor_score = get_graph_tensor(graph, score_name[stride_index]);
        tensor_t tensor_landmark = get_graph_tensor(graph, landmark_name[stride_index]);

        int dims_bbox_delta[4] = {0};
        int dims_bbox_score[4] = {0};
        int dims_bbox_landmark[4] = {0};

        get_tensor_shape(tensor_bbx, dims_bbox_delta, 4);
        get_tensor_shape(tensor_score, dims_bbox_score, 4);
        get_tensor_shape(tensor_landmark, dims_bbox_landmark, 4);

        float* score_data = ( float* )get_tensor_buffer(tensor_score);
        float* bbox_data = ( float* )get_tensor_buffer(tensor_bbx);
        float* landmark_data = ( float* )get_tensor_buffer(tensor_landmark);

        // ==================================================================
        // ========== This part is to get calculate anchor boxes ============
        // ==================================================================
        int A = 2;    // Scale dimension
        int K = dims_bbox_delta[2] * dims_bbox_delta[3];
        // int hwt = 0;	// size value for reshape
        float** anchors_reshape;
        height = dims_bbox_delta[2];
        width = dims_bbox_delta[3];

        anchors_reshape = ( float** )malloc(sizeof(float*) * K * A);
        for(int i = 0; i < A * K; i++)
        {
            anchors_reshape[i] = ( float* )malloc(sizeof(float) * 4);
        }
        anchor_calculation(height, width, stride_index, A, anchors_reshape);

        // ==================================================================
        // ========== This part is to reshape the bbox_delta value ==========
        // ==================================================================

        int bbox_pred_len = 4;
        int reshape_size = dims_bbox_delta[0] * dims_bbox_delta[1] * dims_bbox_delta[2] * dims_bbox_delta[3];
        int num_col_box_reshape = reshape_size / bbox_pred_len;
        float** box_delta_reshape = new float*[num_col_box_reshape];

        for(int i = 0; i < num_col_box_reshape; i++)
        {
            box_delta_reshape[i] = new float[bbox_pred_len];
        }

        bbox_delta_data_reshape(dims_bbox_delta, bbox_data, box_delta_reshape, height, width);

        // ==================================================================
        // ====== This part is to reshape the score value from tensor =======
        // ==================================================================
        int new_score_size = dims_bbox_score[0] * dims_bbox_score[1] * dims_bbox_score[2] * dims_bbox_score[3] / 2;
        float* new_score = new float[new_score_size];
        score_reshape(dims_bbox_score, score_data, new_score);

        // ==================================================================
        // === This part is to reshape the landmarks value from tensor ======
        // ==================================================================
        int landmark_size =
            dims_bbox_landmark[0] * dims_bbox_landmark[1] * dims_bbox_landmark[2] * dims_bbox_landmark[3];
        float*** landmark_pred_reshape = new float**[landmark_size];
        float*** pred_landmark = new float**[landmark_size];
        for(int i = 0; i < landmark_size / 10; i++)
        {
            landmark_pred_reshape[i] = new float*[5];
            pred_landmark[i] = new float*[5];
            for(int j = 0; j < 5; j++)
            {
                landmark_pred_reshape[i][j] = new float[2];
                pred_landmark[i][j] = new float[2];
            }
        }

        landmark_reshape_function(dims_bbox_landmark, landmark_data, pred_landmark, landmark_pred_reshape);

        // ==================================================================
        // ====== This part is to calculate the proposals boxes and =========
        // ====== sort the score value for predicting targets ===============
        // ==================================================================
        std::vector<abox> pre_boxes;
        abox tempBox;

        float** bbox_delt = ( float** )calloc(A, sizeof(float*));
        for(int k = 0; k < A; k++)
            bbox_delt[k] = ( float* )calloc(4, sizeof(float*));
        float scale_height = img.h * im_scale;
        float scale_width = img.w * im_scale;
        for(int i = 0; i < K; i++)
        {
            for(int j = 0; j < A; j++)
            {
                bbox_delt[j][0] = box_delta_reshape[i * A + j][0];
                bbox_delt[j][1] = box_delta_reshape[i * A + j][1];
                bbox_delt[j][2] = box_delta_reshape[i * A + j][2];
                bbox_delt[j][3] = box_delta_reshape[i * A + j][3];
            }
            for(int j = 0; j < A; j++)
            {
                tempBox =
                    bbox_tranform_inv(anchors_reshape[i * A + j], bbox_delt, j, scale_width, scale_height, im_scale);
                pre_boxes.push_back(tempBox);
            }
        }

        std::vector<int> order;
        int score_range = dims_bbox_score[0] * dims_bbox_score[2] * dims_bbox_score[3] * dims_bbox_score[1] / 2;
        for(int i = 0; i < score_range; i++)
        {
            if(new_score[i] > CONF_THRESH)
            {
                order.push_back(i);
            }
        }

        // ==================================================================
        // = This part is to order the landmarks order based on score order =
        // ==================================================================
        std::vector<float**> temp_lmark;
        temp_lmark = landmark_pred(K, A, im_scale, landmark_pred_reshape, anchors_reshape, pred_landmark, temp_lmark);

        // ==================================================================
        // =========== Finally, push all addressed data into list ===========
        // ==================================================================
        for(unsigned int i = 0; i < order.size(); i++)
        {
            landmark_list.push_back(temp_lmark.at(order.at(i)));
            proposals_list.push_back(pre_boxes.at(order.at(i)));
            score_list.push_back(new_score[order.at(i)]);
        }
    }

    // ====================================================================
    // ===== Based on previous data, binding the proposals, scores, =======
    // ===== landmarks value together, then prepared to filtering =========
    // ====================================================================
    int proposals_size = proposals_list.size();
    std::vector<size_t> sorted_idx;
    sorted_idx = sort_index(score_list);
    std::vector<abox> proposals_box;
    for(int i = 0; i < ( int )sorted_idx.size(); i++)
    {
        proposals_box.push_back(proposals_list.at(( int )sorted_idx.at(i)));
    }

    for(int i = 0; i < proposals_size; i++)
    {
        proposals_box[i].score = score_list.at(( int )sorted_idx.at(i));
        proposals_box[i].land_mark = landmark_list.at(( int )sorted_idx.at(i));
    }
    // ==================================================================
    // ====== Filtering the target boxes =======
    // ==================================================================
    nms(proposals_box, NMS_THRESH);
    for(int k = 0; k < proposals_size; k++)
    {
        if(proposals_box[k].score < CONF_THRESH)
        {
            proposals_box.erase(proposals_box.begin() + k);
        }
        else
            k++;
    }
    if(proposals_box.size() > 0)
    {
        for(int b = 0; b < ( int )proposals_box.size(); b++)
        {
            all_pred_boxes.push_back(proposals_box[b]);
        }
    }
    // ==================================================================
    // ========= draw the rectangle and landmarks for targets ===========
    // ==================================================================

    draw_target(all_pred_boxes, img);

    ret = postrun_graph(graph);
    if(ret != 0)
    {
        std::cout << "Postrun graph failed, errno: " << get_tengine_errno() << "\n";
        return 1;
    }
    destroy_graph(graph);
    release_tengine();
    return 0;
}
