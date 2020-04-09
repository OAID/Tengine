
int ref_dpp_fp16(const __fp16* input, const __fp16* score, const __fp16* anchor, float* detect_num, float* detect_class,
                 float* detect_score, float* detect_boxes, dpp_param* param)
{
    const int num_classes = param->num_classes + 1;
    const int num_boxes = param->num_boxes;

    /* transform __fp16 to fp32 */
    int input_size = num_boxes * 4;
    int score_size = num_boxes * num_classes;
    float* input_f = ( float* )malloc(input_size * sizeof(float));
    float* score_f = ( float* )malloc(score_size * sizeof(float));
    float* anchor_f = ( float* )malloc(input_size * sizeof(float));

#if !defined(__ARM_ARCH) || __ARM_ARCH < 8
    for(int i = 0; i < input_size; i++)
        input_f[i] = fp16_to_fp32(input[i]);
    for(int i = 0; i < input_size; i++)
        score_f[i] = fp16_to_fp32(score[i]);
    for(int i = 0; i < input_size; i++)
        anchor_f[i] = fp16_to_fp32(anchor[i]);
#else
    for(int i = 0; i < input_size; i++)
        input_f[i] = input[i];
    for(int i = 0; i < input_size; i++)
        score_f[i] = score[i];
    for(int i = 0; i < input_size; i++)
        anchor_f[i] = anchor[i];
#endif

    ref_dpp_common(input_f, score_f, anchor_f, param, detect_num, detect_class, detect_score, detect_boxes);
    free(anchor_f);
    free(score_f);
    free(input_f);
    return 0;
}
