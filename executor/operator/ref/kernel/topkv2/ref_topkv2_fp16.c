static void swap_fp16(float* p, float* q)
{
    float buf;
    buf = *p;
    *p = *q;
    *q = buf;
    return;
}

static inline void quick_sort_fp16(float* a, int low, int high, std::vector<int>& indexv)
{
    int i = low;
    int j = high;
    float key = a[low];
    if(low >= high)    //如果low >= high说明排序结束了
    {
        return;
    }
    while(low < high)    //该while循环结束一次表示比较了一轮
    {
        while(low < high && key >= a[high])
        {
            --high;    //向前寻找
        }
        if(key < a[high])
        {
            swap_fp16(&a[low], &a[high]);
            std::swap(indexv.at(low), indexv.at(high));
            ++low;
        }
        while(low < high && key <= a[low])
        {
            ++low;    //向后寻找
        }
        if(key > a[low])
        {
            swap_fp16(&a[low], &a[high]);
            std::swap(indexv.at(low), indexv.at(high));
            --high;
        }
    }
    quick_sort_fp16(a, i, low - 1, indexv);    //用同样的方式对分出来的左边的部分进行同上的做法
    quick_sort_fp16(a, low + 1, j, indexv);    //用同样的方式对分出来的右边的部分进行同上的做法
}

static int ref_topkv2_fp16(const __fp16* in_data, __fp16* out_data, int* out_index, struct topkv2_param* param)
{
    int k = param->k;
    int row_size = param->row_size;
    int num_rows = param->num_rows;
    int input_size = row_size * num_rows;
    float* input_f = ( float* )malloc(input_size * sizeof(float));
    float* output_f = ( float* )malloc(input_size * sizeof(float));
    for(int i = 0; i < input_size; i++)
        input_f[i] = fp16_to_fp32(in_data[i]);

    std::vector<int> index;
    for(int i = 0; i < num_rows; ++i)
    {
        int start = i * row_size;
        for(int j = 0; j < row_size; ++j)
            index.push_back(j);

        quick_sort_fp32(&input_f[start], 0, row_size - 1, index);
        memcpy(&output_f[i * k], &input_f[start], k * sizeof(float));
        memcpy(&out_index[i * k], index.data(), k * sizeof(float));
        index.clear();
    }

    for(int i = 0; i < input_size; i++)
        out_data[i] = fp32_to_fp16(output_f[i]);

    free(input_f);
    free(output_f);
    return 0;
}
