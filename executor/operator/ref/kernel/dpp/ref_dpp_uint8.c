
int ref_dpp_uint8(const uint8_t* input, const uint8_t* score, const uint8_t* anchor, float* detect_num,
                  float* detect_class, float* detect_score, float* detect_boxes, dpp_param* param)
{
    const int num_classes = param->num_classes + 1;
    const int num_boxes = param->num_boxes;

    /* transform uint8_t to fp32 */
    int input_size = num_boxes * 4;
    int score_size = num_boxes * num_classes;
    float* input_f = ( float* )malloc(input_size * sizeof(float));
    float* score_f = ( float* )malloc(score_size * sizeof(float));
    float* anchor_f = ( float* )malloc(input_size * sizeof(float));
    for(int i = 0; i < input_size; i++)
        input_f[i] = (input[i] - param->zero[0]) * param->quant_scale[0];
    for(int i = 0; i < score_size; i++)
        score_f[i] = score[i] * param->quant_scale[1];
    for(int i = 0; i < input_size; i++)
        anchor_f[i] = (anchor[i] - param->zero[2]) * param->quant_scale[2];

    ref_dpp_common(input_f, score_f, anchor_f, param, detect_num, detect_class, detect_score, detect_boxes);

    free(anchor_f);
    free(score_f);
    free(input_f);

    return 0;
}
