void get_boxes_fp16(std::vector<Box>& boxes, int num_prior, const __fp16* loc_ptr, const __fp16* prior_ptr)
{
    for(int i = 0; i < num_prior; i++)
    {
        const __fp16* loc = loc_ptr + i * 4;
        const __fp16* pbox = prior_ptr + i * 4;
        const __fp16* pvar = pbox + num_prior * 4;
        // pbox [xmin,ymin,xmax,ymax]
        float pbox_w = fp16_to_fp32(pbox[2]) - fp16_to_fp32(pbox[0]);
        float pbox_h = fp16_to_fp32(pbox[3]) - fp16_to_fp32(pbox[1]);
        float pbox_cx = (fp16_to_fp32(pbox[0]) + fp16_to_fp32(pbox[2])) * 0.5f;
        float pbox_cy = (fp16_to_fp32(pbox[1]) + fp16_to_fp32(pbox[3])) * 0.5f;

        // loc []
        float bbox_cx = fp16_to_fp32(pvar[0]) * fp16_to_fp32(loc[0]) * pbox_w + pbox_cx;
        float bbox_cy = fp16_to_fp32(pvar[1]) * fp16_to_fp32(loc[1]) * pbox_h + pbox_cy;
        float bbox_w = pbox_w * exp(fp16_to_fp32(pvar[2]) * fp16_to_fp32(loc[2]));
        float bbox_h = pbox_h * exp(fp16_to_fp32(pvar[3]) * fp16_to_fp32(loc[3]));

        // bbox [xmin,ymin,xmax,ymax]
        boxes[i].x0 = bbox_cx - bbox_w * 0.5f;
        boxes[i].y0 = bbox_cy - bbox_h * 0.5f;
        boxes[i].x1 = bbox_cx + bbox_w * 0.5f;
        boxes[i].y1 = bbox_cy + bbox_h * 0.5f;
    }
}

static inline float intersection_area_fp16(const Box& a, const Box& b)
{
    if(a.x0 > b.x1 || a.x1 < b.x0 || a.y0 > b.y1 || a.y1 < b.y0)
    {
        return 0.f;
    }

    float inter_width = std::min(a.x1, b.x1) - std::max(a.x0, b.x0);
    float inter_height = std::min(a.y1, b.y1) - std::max(a.y0, b.y0);

    return inter_width * inter_height;
}
void nms_sorted_bboxes_fp16(const std::vector<Box>& bboxes, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();
    const int n = bboxes.size();

    std::vector<float> areas(n);
    for(int i = 0; i < n; i++)
    {
        const Box& r = bboxes[i];

        float width = r.x1 - r.x0;
        float height = r.y1 - r.y0;

        areas[i] = width * height;
    }

    for(int i = 0; i < n; i++)
    {
        const Box& a = bboxes[i];

        int keep = 1;
        for(int j = 0; j < ( int )picked.size(); j++)
        {
            const Box& b = bboxes[picked[j]];

            float inter_area = intersection_area_fp16(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if(inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if(keep)
            picked.push_back(i);
    }
}

int ref_DetectionOutput_fp16(const __fp16* location, const __fp16* confidence, const __fp16* priorbox,
                             std::vector<int> dims, ddo_param* param_)
{
    const int num_priorx4 = dims[2];
    const int num_prior = num_priorx4 / 4;
    const int num_classes = param_->num_classes;

    const __fp16* loc_ptr = location;
    const __fp16* conf_ptr = confidence;
    const __fp16* prior_ptr = priorbox;

    std::vector<Box> boxes(num_prior);
    get_boxes_fp16(boxes, num_prior, loc_ptr, prior_ptr);

    std::vector<std::vector<Box>> temp_all_box;
    temp_all_box.clear();
    temp_all_box.resize(num_classes);

    for(int i = 1; i < num_classes; i++)
    {
        std::vector<Box> class_box;
        for(int j = 0; j < num_prior; j++)
        {
            float score = fp16_to_fp32(conf_ptr[j * num_classes + i]);
            if(score > param_->confidence_threshold)
            {
                boxes[j].score = score;
                boxes[j].class_idx = i;
                class_box.push_back(boxes[j]);
            }
        }

        std::sort(class_box.begin(), class_box.end(), [](const Box& a, const Box& b) { return a.score > b.score; });

        if(param_->nms_top_k < ( int )class_box.size())
        {
            class_box.resize(param_->nms_top_k);
        }

        std::vector<int> picked;
        nms_sorted_bboxes_fp16(class_box, picked, param_->nms_threshold);

        for(int j = 0; j < ( int )picked.size(); j++)
        {
            int z = picked[j];
            temp_all_box[i].push_back(class_box[z]);
        }
    }

    param_->bbox_rects.clear();
    for(int i = 0; i < param_->num_classes; i++)
    {
        const std::vector<Box> class_bbox_rects = temp_all_box[i];
        param_->bbox_rects.insert(param_->bbox_rects.end(), class_bbox_rects.begin(), class_bbox_rects.end());
    }

    std::sort(param_->bbox_rects.begin(), param_->bbox_rects.end(),
              [](const Box& a, const Box& b) { return a.score > b.score; });

    if(param_->keep_top_k < ( int )param_->bbox_rects.size())
    {
        param_->bbox_rects.resize(param_->keep_top_k);
    }
    return 0;
}
