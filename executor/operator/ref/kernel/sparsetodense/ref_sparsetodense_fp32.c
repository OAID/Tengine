static int ref_sparsetodense_fp32(int* input, int* outout_shape, int* sparse_values, float* output, sparsetodense_param* param)
{
    int output_dim_size = param -> output_dim_size;
    int indices_dim_size = param -> indices_dim_size;
    int sparse_value_size = param -> sparse_values_size;
    float default_value = param -> default_value;

    if (output_dim_size == 1)
    {
        for(int i = 0; i < outout_shape[0]; i++)
        {
            output[i] = default_value;
        }

        if (sparse_value_size == 0)
        {
            if(indices_dim_size == 0)
            {
                output[*input] = *sparse_values;
            }

            else if (indices_dim_size == 1)
            {
                for(int i = 0; i < param->indices_shape[0]; i++)
                {
                    output[input[i]] = *sparse_values;
                }
            }
            
            else
            {
                return -1;
            }
        }    

        else if (sparse_value_size == 1)
        {
            if(indices_dim_size == 0)
            {
                output[*input] = sparse_values[0];
            }

            else if (indices_dim_size == 1)
            {
                for(int i = 0; i < param->indices_shape[0]; i++)
                {
                    output[input[i]] = sparse_values[i];
                }
            }

            else
            {
                return -1;
            }
        }
            
    }

    if (output_dim_size == 2)
    {
        for(int i = 0; i < outout_shape[0] * outout_shape[1]; i++)
        {
            output[i] = default_value;
        }

        if(indices_dim_size != 2)
        {
            return -1;
        }

        if (sparse_value_size == 0)
        {
            for(int i = 0; i < param->indices_shape[0] * 2; i += 2)
            {
                int x = input[i];
                int y = input[i + 1];
                output[outout_shape[1] * x + y] = *sparse_values;
            }
        }
        else if (sparse_value_size == 1)
        {
            for(int i = 0; i < param->indices_shape[0] * 2; i += 2)
            {
                int x = input[i];
                int y = input[i + 1];
                output[outout_shape[1] * x + y] = sparse_values[i / 2];
            }
        }
        
    }

    return 0;
}

