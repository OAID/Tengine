int ref_embed_fp32(float* in_data, float* out_data, float* weight_data, float* bias_data, int input_dim, int num_output, int size, int bias_term, float scale, int zero_point)
{
    for(int i = 0; i < size; i++)
    {
        int word_index = in_data[i];
        if(word_index < 0)
            word_index = 0;
        if(word_index >= input_dim)
            word_index = input_dim - 1;
        const float* embed = (const float*) weight_data + num_output * word_index;
        for(int z = 0; z < num_output; z++)
        {
            out_data[i*num_output+z] = embed[z];
            if(bias_term)
                out_data[i*num_output+z] += bias_data[z];
        }
    }
    
    return 0;
}