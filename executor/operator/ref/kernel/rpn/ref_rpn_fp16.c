#include "ref_rpn_kernel.h"

static inline void bbox_tranform_inv_fp16(__fp16* m_box, float* local_anchors, struct rpn_param* param)
{
    int feat_size = param->feat_height * param->feat_width;
    int c_4 = param->feat_chan / 4;
    for(int i = 0; i < c_4; ++i)
    {
        for(int j = 0; j < (2 * feat_size); ++j)
        {
            local_anchors[(i * 4 + 2) * feat_size + j] -= local_anchors[(i * 4 + 0) * feat_size + j] - 1;
            local_anchors[(i * 4 + 0) * feat_size + j] += local_anchors[(i * 4 + 2) * feat_size + j] * 0.5;

            float boxA = fp16_to_fp32(m_box[(i * 4 + 0) * feat_size + j]);
            boxA *= local_anchors[(i * 4 + 2) * feat_size + j];
            boxA += local_anchors[(i * 4 + 0) * feat_size + j];

            float boxB = fp16_to_fp32(m_box[(i * 4 + 2) * feat_size + j]);
            boxB = exp(boxB);
            boxB *= local_anchors[(i * 4 + 2) * feat_size + j];

            m_box[(i * 4 + 0) * feat_size + j] = fp32_to_fp16(boxA);
            m_box[(i * 4 + 2) * feat_size + j] = fp32_to_fp16(boxB);
        }
    }
}

static inline void ref_filter_boxes_fp16(struct RPN_Box* boxes, const __fp16* featmap, const __fp16* score,
                                         int* num_boxes, struct rpn_param* param)
{
    float local_minsize = param->min_size * param->src_scale;
    int c_4 = param->feat_chan / 4;
    int feat_size = param->feat_height * param->feat_width;

    int offset_w, offset_h, offset_x, offset_y, offset_s;

    int num = 0;
    for(int h = 0; h < param->feat_height; h++)
        for(int w = 0; w < param->feat_width; w++)
        {
            offset_x = h * param->feat_width + w;
            offset_y = offset_x + feat_size;
            offset_w = offset_y + feat_size;
            offset_h = offset_w + feat_size;
            offset_s = feat_size * param->num_anchors + offset_x;
            ;
            for(int c = 0; c < c_4; c++)
            {
                float width = fp16_to_fp32(featmap[offset_w]);
                float height = fp16_to_fp32(featmap[offset_h]);
                if((width >= local_minsize) & (height >= local_minsize))
                {
                    boxes[num].x0 = fp16_to_fp32(featmap[offset_x]) - 0.5 * width;
                    boxes[num].y0 = fp16_to_fp32(featmap[offset_y]) - 0.5 * height;
                    boxes[num].x1 = fp16_to_fp32(featmap[offset_x]) + 0.5 * width;
                    boxes[num].y1 = fp16_to_fp32(featmap[offset_y]) + 0.5 * height;
                    boxes[num].x0 = RPN_MIN(RPN_MAX(boxes[num].x0, 0), param->src_width);
                    boxes[num].y0 = RPN_MIN(RPN_MAX(boxes[num].y0, 0), param->src_height);
                    boxes[num].x1 = RPN_MIN(RPN_MAX(boxes[num].x1, 0), param->src_width);
                    boxes[num].y1 = RPN_MIN(RPN_MAX(boxes[num].y1, 0), param->src_height);
                    boxes[num].score = fp16_to_fp32(score[offset_s]);
                    num++;
                }
                offset_x += 4 * feat_size;
                offset_y += 4 * feat_size;
                offset_w += 4 * feat_size;
                offset_h += 4 * feat_size;
                offset_s += feat_size;
            }
        }

    *num_boxes = num;
}

int ref_rpn_fp16(const __fp16* score, __fp16* featmap, float* anchors, __fp16* output, struct rpn_param* param)
{
    if(score == nullptr || featmap == nullptr || anchors == nullptr || output == nullptr)
        return false;
    int featmap_size = param->feat_height * param->feat_width * param->feat_chan;

    int max_num_boxes = featmap_size / 4;

    struct RPN_Box* boxes = ( struct RPN_Box* )malloc(max_num_boxes * sizeof(struct RPN_Box));

    bbox_tranform_inv_fp16(featmap, anchors, param);

    int num_boxes = 0;
    ref_filter_boxes_fp16(boxes, featmap, score, &num_boxes, param);

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

    for(int i = 0; i < num_boxes; i++)
    {
        __fp16* outptr = output + i * 4;
        outptr[0] = fp32_to_fp16(boxes[i].x0);
        outptr[1] = fp32_to_fp16(boxes[i].y0);
        outptr[2] = fp32_to_fp16(boxes[i].x1);
        outptr[3] = fp32_to_fp16(boxes[i].y1);
    }

    free(boxes);
    return num_boxes;
}
