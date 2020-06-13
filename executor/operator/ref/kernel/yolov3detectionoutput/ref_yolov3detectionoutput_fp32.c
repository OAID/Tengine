float intersection_area(const Box& box1, const Box& box2)
{
    //printf("box1:%f %f %f %f\n", box1.x0, box1.x1, box1.y0, box1.y1);
    //printf("box2:%f %f %f %f\n", box2.x0, box2.x1, box2.y0, box2.y1);
    if(box1.x0 > box2.x1||box1.x1 < box2.x0 ||box1.y0 > box2.y1||box1.y1 < box2.y0)
        return 0;

    float width = std::min(box1.x1, box2.x1) - std::max(box1.x0, box2.x0);
    float height = std::min(box1.y1, box2.y1) - std::max(box1.y0, box2.y0);

    return width * height;
}

static float sigmoid(float x)
{
    return 1.f / (1.f + std::exp(-x));
}

void nms_sorted_box(const std::vector<Box>& box, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();
    const int n = box.size();

    std::vector<float> areas(n);
    for(int i = 0; i < n; i++)
    {
        const Box& r = box[i];
        
        float width = r.x1 - r.x0;
        float height = r.y1 - r.y0;

        areas[i] = width * height;
    }

    for(int i = 0; i < n ;i++)
    {
        const Box& a = box[i];

        int keep = 1;

        for(int j = 0; j < (int)picked.size(); j++)
        {
            const Box& b = box[picked[j]];
            //printf("%d\n", a.class_idx);
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            //printf("%f %f\n", inter_area/union_area, nms_threshold);
            if(inter_area / union_area > nms_threshold)
                keep = 0;
        }
    
        if(keep)
            picked.push_back(i);
    }

}

int ref_YOLOV3DetectionOutput_fp32(float* input, YOLOV3_ddo_parm* param_)
{
    int input_num = param_->anchors_scale.size();
    std::vector<Box> all_boxes;
    int num_box = param_->num_box;
    int num_classes = param_->num_classes;
    int point_counter = 0;
    //printf("input_num is %d\n", param_->anchors_scale.size());
    for(int i = 0; i < input_num; i++)
    {   
        std::vector<std::vector<Box>> all_bboxes;
        all_bboxes.resize(num_box);
        std::vector<int> input_dims = param_->input_dims[i];
        int c = input_dims[1];
        int h = input_dims[2];
        int w = input_dims[3];
        //printf("%d, %d, %d, %d, %d.\n",c ,h, w, num_box, num_classes);
        if( c/num_box != 4+1+num_classes)
            return -1;
        int mask_offset = i * num_box;
        int net_w = (int)(param_->anchors_scale[i] * w);
        int net_h = (int)(param_->anchors_scale[i] * h);
        //printf("%d %d\n", net_w, net_h);
        for(int j = 0; j < num_box; j++)
        {
            int counter = j*(4+1+num_classes);
            int pre_data = static_cast<int>(param_->mask[j+mask_offset]);
            int bias_index = static_cast<int>(*((float*)&pre_data));
            //printf("%d num is %d.\n", pre_data, bias_index);   
            const float bias_w = param_->bias[bias_index * 2];
            const float bias_h = param_->bias[bias_index * 2 + 1];
            //printf("%f %f\n", bias_w, bias_h);
            const float* x_ptr = input + counter * h * w;
            const float* y_ptr = input + (counter + 1) * h * w;
            const float* w_ptr = input + (counter + 2) * h * w;
            const float* h_ptr = input + (counter + 3) * h * w;
            const float* box_score_ptr = input + (counter + 4) * h * w;
            const float* class_score_ptr = input + (counter + 5) * h * w;

            for(int m = 0; m < h; m++)
            {
                for(int n = 0; n < w; n++)
                {
                    //printf("%d %d\n", m, n);
                    float box_score = sigmoid(box_score_ptr[0]);
                    int class_index = 0;
                    float class_score = -9999;
                    for (int cl = 0; cl < num_classes; cl++)
                    {
                        int class_score_offset = cl*h*w + m*w + n;
                        float score = class_score_ptr[class_score_offset];
                        if(score > class_score)
                        {
                            class_index = cl;
                            class_score = score;
                        }
                    }
                    class_score = sigmoid(class_score);
                    //printf( "%d %f %f\n", class_index, box_score, class_score);
                    float cofidence = box_score * class_score;
                    //printf("%f\n", param_->confidence_threshold);
                    if(cofidence >= param_->confidence_threshold)
                    {
                        float bbox_cx = (n + sigmoid(x_ptr[0])) / w;
                        float bbox_cy = (m + sigmoid(y_ptr[0])) / h;
                        float bbox_w = static_cast<float>(std::exp(w_ptr[0]) * bias_w / net_w);
                        float bbox_h = static_cast<float>(std::exp(h_ptr[0]) * bias_h / net_h);

                        float bbox_xmin = bbox_cx - bbox_w * 0.5f;
                        float bbox_ymin = bbox_cy - bbox_h * 0.5f;
                        float bbox_xmax = bbox_cx + bbox_w * 0.5f;
                        float bbox_ymax = bbox_cy + bbox_h * 0.5f;

                        Box c = {bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, class_index, cofidence};
                        //printf("%f %f %f %f %d %f\n",bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax, class_index, cofidence);
                        all_bboxes[j].push_back(c);
                    }

                    x_ptr++;
                    y_ptr++;
                    w_ptr++;
                    h_ptr++;

                    box_score_ptr++;
                }
            }
        }

        input += h*w*c;

        for(int j = 0; j < num_box; j++)
        {
            const std::vector<Box>& bboxes = all_bboxes[j];
            all_boxes.insert(all_boxes.end(), bboxes.begin(), bboxes.end());
            //printf("%d %d.\n", all_boxes.size(), bboxes.size());
        }
    }
    std::sort(all_boxes.begin(), all_boxes.end(),
                [](const Box& a, const Box& b) { return a.score > b.score; });
    //printf("sorted done.\n");
    std::vector<int> picked;
    nms_sorted_box(all_boxes, picked, param_->nms_threshold);
    //printf("nms done.\n");
    for(size_t j = 0; j < picked.size(); j++)
    {
        param_->output_box.push_back(all_boxes[picked[j]]);
    }

    return 0;
}