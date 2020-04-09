void get_boxes_int8(std::vector<Box>& boxes, int num_prior, const int8_t* loc_ptr, const int8_t* prior_ptr,
                    ddo_param* param_)
{
    for(int i = 0; i < num_prior; i++)
    {
        const int8_t* loc = loc_ptr + i * 4;
        const int8_t* pbox = prior_ptr + i * 4;
        const int8_t* pvar = pbox + num_prior * 4;
        // center size
        // pbox [xmin,ymin,xmax,ymax]
        float pbox_w = (pbox[2] - pbox[0]) * param_->scale[2];
        float pbox_h = (pbox[3] - pbox[1]) * param_->scale[2];
        float pbox_cx = (pbox[0] + pbox[2]) * 0.5f * param_->scale[2];
        float pbox_cy = (pbox[1] + pbox[3]) * 0.5f * param_->scale[2];

        // loc []
        float bbox_cx = pvar[0] * loc[0] * pbox_w * param_->scale[2] * param_->scale[0] + pbox_cx;
        float bbox_cy = pvar[1] * loc[1] * pbox_h * param_->scale[2] * param_->scale[0] + pbox_cy;
        float bbox_w = pbox_w * exp(pvar[2] * loc[2] * param_->scale[2] * param_->scale[0]);
        float bbox_h = pbox_h * exp(pvar[3] * loc[3] * param_->scale[2] * param_->scale[0]);
        // bbox [xmin,ymin,xmax,ymax]
        boxes[i].x0 = bbox_cx - bbox_w * 0.5f;
        boxes[i].y0 = bbox_cy - bbox_h * 0.5f;
        boxes[i].x1 = bbox_cx + bbox_w * 0.5f;
        boxes[i].y1 = bbox_cy + bbox_h * 0.5f;
    }
}
int ref_DetectionOutput_int8(const int8_t* location, const int8_t* confidence, const int8_t* priorbox,
                             std::vector<int> dims, ddo_param* param_)
{
    const int num_priorx4 = dims[2];
    const int num_prior = num_priorx4 / 4;
    const int num_classes = param_->num_classes;

    const int8_t* loc_ptr = location;
    const int8_t* conf_ptr = confidence;
    const int8_t* prior_ptr = priorbox;

    std::vector<Box> boxes(num_prior);
    get_boxes_int8(boxes, num_prior, loc_ptr, prior_ptr, param_);

    std::vector<std::vector<Box>> temp_all_box;
    temp_all_box.resize(num_classes);

    for(int i = 1; i < num_classes; i++)
    {
        std::vector<Box> class_box;
        for(int j = 0; j < num_prior; j++)
        {
            float score = conf_ptr[j * num_classes + i] * param_->scale[1];
            if(score > param_->confidence_threshold)
            {
                boxes[j].score = score;
                boxes[j].class_idx = i;
                class_box.push_back(boxes[j]);
            }
        }
        // sort
        std::sort(class_box.begin(), class_box.end(), [](const Box& a, const Box& b) { return a.score > b.score; });

        if(param_->nms_top_k < ( int )class_box.size())
        {
            class_box.resize(param_->nms_top_k);
        }
        // apply nms

        std::vector<int> picked;
        nms_sorted_bboxes(class_box, picked, param_->nms_threshold);
        // select

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

    // global sort inplace
    std::sort(param_->bbox_rects.begin(), param_->bbox_rects.end(),
              [](const Box& a, const Box& b) { return a.score > b.score; });

    // keep_top_k
    if(param_->keep_top_k < ( int )param_->bbox_rects.size())
    {
        param_->bbox_rects.resize(param_->keep_top_k);
    }

    float max_value = 0.0f;

    for(int i = 0; i < ( int )param_->bbox_rects.size(); i++)
    {
        if(max_value < param_->bbox_rects[i].score)
            max_value = param_->bbox_rects[i].score;
        if(max_value < param_->bbox_rects[i].x0)
            max_value = param_->bbox_rects[i].x0;
        if(max_value < param_->bbox_rects[i].x1)
            max_value = param_->bbox_rects[i].x1;
        if(max_value < param_->bbox_rects[i].y0)
            max_value = param_->bbox_rects[i].y0;
        if(max_value < param_->bbox_rects[i].y1)
            max_value = param_->bbox_rects[i].y1;
    }

    param_->out_scale = max_value / 127;

    return 0;
}
