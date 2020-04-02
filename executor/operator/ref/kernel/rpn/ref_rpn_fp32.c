#include "ref_rpn_kernel.h"

int ref_rpn_fp32(const float* score, float* featmap, float* anchors, float* output, struct rpn_param* param)
{
    if(score == nullptr || featmap == nullptr || anchors == nullptr || output == nullptr)
        return false;
    int featmap_size = param->feat_height * param->feat_width * param->feat_chan;
    int max_num_boxes = featmap_size / 4;
    struct RPN_Box* boxes = ( struct RPN_Box* )malloc(max_num_boxes * sizeof(struct RPN_Box));

    bbox_tranform_inv(featmap, anchors, param);

    int num_boxes = 0;
    ref_filter_boxes(boxes, featmap, score, &num_boxes, param);

    sort_rpn_boxes_by_score(boxes, num_boxes);

    if(param->per_nms_topn > 0)
    {
        num_boxes = RPN_MIN(param->per_nms_topn, num_boxes);
    }
    nms_rpn_boxes(boxes, &num_boxes, param->nms_thresh);

    if(param->post_nms_topn > 0)
    {
        num_boxes = RPN_MIN(param->post_nms_topn, num_boxes);
    }
    // inder shape [default batch=1]

    // std::cout<<"num_box "<<num_box<<"\n";
    for(int i = 0; i < num_boxes; i++)
    {
        float* outptr = output + i * 4;
        outptr[0] = boxes[i].x0;
        outptr[1] = boxes[i].y0;
        outptr[2] = boxes[i].x1;
        outptr[3] = boxes[i].y1;
    }

    free(boxes);
    return num_boxes;
}
