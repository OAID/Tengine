
<<<<<<< HEAD
int ref_dpp_fp32(const float* input, const float* score, const float* anchor,
            float* detect_num, float* detect_class, float* detect_score, float* detect_boxes, dpp_param* param)
{
    return ref_dpp_common(input, score, anchor, param, detect_num, detect_class, detect_score, detect_boxes);;
=======
int ref_dpp_fp32(const float* input, const float* score, const float* anchor, float* detect_num, float* detect_class,
                 float* detect_score, float* detect_boxes, dpp_param* param)
{
    return ref_dpp_common(input, score, anchor, param, detect_num, detect_class, detect_score, detect_boxes);
    ;
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074
}
