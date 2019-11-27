static void swap_uint8(uint8_t* p, uint8_t* q)
{
    uint8_t buf;
    buf = *p;
    *p = *q;
    *q = buf;
    return;
}

static void quick_sort_uint8(uint8_t* a, int low, int high, std::vector<int>& indexv)
{
    int i = low;
    int j = high;
    uint8_t key = a[low];
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
            swap_uint8(&a[low], &a[high]);
            std::swap(indexv.at(low), indexv.at(high));
            ++low;
        }
        while(low < high && key <= a[low])
        {
            ++low;    //向后寻找
        }
        if(key > a[low])
        {
            swap_uint8(&a[low], &a[high]);
            std::swap(indexv.at(low), indexv.at(high));
            --high;
        }
    }
    quick_sort_uint8(a, i, low - 1, indexv);    //用同样的方式对分出来的左边的部分进行同上的做法
    quick_sort_uint8(a, low + 1, j, indexv);    //用同样的方式对分出来的右边的部分进行同上的做法
}

static int ref_topkv2_uint8(uint8_t* in_data, uint8_t* out_data, int* out_index, struct topkv2_param* param)
{
    int k = param->k;
    int row_size = param->row_size;
    int num_rows = param->num_rows;
    std::vector<int> index;

    for(int i = 0; i < num_rows; ++i)
    {
        int start = i * row_size;
        for(int j = 0; j < row_size; ++j)
            index.push_back(j);

        quick_sort_uint8(&in_data[start], 0, row_size - 1, index);
        memcpy(&out_data[i * k], &in_data[start], k * sizeof(uint8_t));
        memcpy(&out_index[i * k], index.data(), k * sizeof(uint8_t));
        index.clear();
    }

    return 0;
}
