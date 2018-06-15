#ifndef __TENGINE_TEST_API_H__
#define __TENGINE_TEST_API_H__

#ifdef __cplusplus
extern "C" {
#endif


typedef void * test_node_t;

test_node_t create_convolution_test_node(int kernel_h, int kernel_w,
                                 int stride_h, int stride_w, int pad_h0, int pad_h1,
                                 int pad_w0, int pad_w1, int dilation_h, int dilation_w,
                                 int input_channel, int output_channel, int group);
                                 
test_node_t create_fc_test_node(int hidden_number, int output_number);

test_node_t create_pooling_test_node(int pool_method, int kernel_h,
                                 int kernel_w, int stride_h, int stride_w,
                                 int pad_h0, int pad_h1, int pad_w0, int pad_w1,
                                 int global);
                                 

int test_node_set_input(test_node_t node, float * input_data[], int * input_shape[], int input_number);
int test_node_set_output(test_node_t node, float * output_data[], int * output_shape[], int output_number);

int test_node_prerun(test_node_t node);

int test_node_run(test_node_t node);

int test_node_postrun(test_node_t node);

void destroy_test_node(test_node_t node);




#ifdef __cplusplus
}
#endif

#endif
