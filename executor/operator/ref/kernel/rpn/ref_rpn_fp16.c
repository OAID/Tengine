#include "ref_rpn_kernel.h"

int ref_rpn_fp16(const __fp16* score, __fp16* featmap, float* anchors, __fp16* output, struct rpn_param* param)
{
    if(score == nullptr || featmap == nullptr || anchors == nullptr || output == nullptr)
        return false;
    int featmap_size = param->feat_height * param->feat_width * param->feat_chan;
    int max_num_boxes = featmap_size /4;
    struct RPN_Box* boxes = (struct RPN_Box*)malloc(max_num_boxes * sizeof(struct RPN_Box));

    /*   __fp16 -> float */
    float* featmap_fp32 = (float*)malloc(featmap_size * sizeof(float));
    float* score_fp32 = (float*)malloc(max_num_boxes * sizeof(float));
    for(int i = 0; i < featmap_size; i++)
        featmap_fp32[i] = fp16_to_fp32(featmap[i]);
    for(int i = 0; i < max_num_boxes; i++)
        score_fp32[i] = fp16_to_fp32(score[i]);

    bbox_tranform_inv(featmap_fp32, anchors, param);
    
    int num_boxes = 0;
    ref_filter_boxes(boxes, featmap_fp32, score_fp32, &num_boxes, param);

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
        __fp16* outptr = output + i * 4;
        outptr[0] = fp32_to_fp16(boxes[i].x0);
        outptr[1] = fp32_to_fp16(boxes[i].y0);
        outptr[2] = fp32_to_fp16(boxes[i].x1);
        outptr[3] = fp32_to_fp16(boxes[i].y1);
    }

    free(score_fp32);
    free(featmap_fp32);
    free(boxes);
    return num_boxes;
}
