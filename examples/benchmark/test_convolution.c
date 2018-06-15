#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>


#include "tengine_c_api.h"
#include "cpu_device.h"
#include "tengine_test_api.h"


int input_c=64;
int input_h=256;
int input_w=256;
int input_n=1;
int kernel_h=3;
int kernel_w=3;
int stride_h=1;
int stride_w=1;
int pad_h0=1;
int pad_h1=1;
int pad_w0=1;
int pad_w1=1;
int dilation_h=1;
int dilation_w=1;
int output_c=16;
int group=1;

static unsigned long get_cur_time(void)
{
	struct timeval tv;

	gettimeofday(&tv,NULL);

	return (tv.tv_sec*1000000+tv.tv_usec);

}

int main(int argc, char * argv[])
{
	test_node_t node;
	float * input;
	float * weight;
	float * bias;
	float * output;
	int output_h;
	int output_w;
	int repeat_count=1;

	init_tengine_library();

#if 1
	const struct cpu_info * p_info=get_predefined_cpu("rk3399");
	int cpu_list[]={4,5};

	set_online_cpu((struct cpu_info *)p_info,cpu_list,sizeof(cpu_list)/sizeof(int));
	create_cpu_device("rk3399",p_info);

#endif

#if 1
	if(load_device("CaffeNode","sw.caffe.cpu.node")==0)
	{
		set_default_device("sw.caffe.cpu.node");
		printf("Using caffe node device\n");
	}
#endif

	node=create_convolution_test_node(kernel_h,kernel_w,stride_h,stride_w,pad_h0,pad_h1,
			pad_w0,pad_w1,dilation_h,dilation_w,
			input_c, output_c,group);

	//allocate memory for input/weight/bias/output

	output_h=(input_h+pad_h0+pad_h1-kernel_h)/stride_h+1;
	output_w=(input_w+pad_w0+pad_w1-kernel_w)/stride_w+1;


	input=malloc(sizeof(float)*input_c*input_h*input_w);
	weight=malloc(sizeof(float)*output_c*input_c*kernel_h*kernel_w);
	bias=malloc(sizeof(float)*output_c);
	output=malloc(sizeof(float)*output_c*output_h*output_w);

	//prepare to set  inputs
	float * inputs[3]={input,weight,bias};
	float * outputs[1]={output};
	int    input_shape[4]={1,input_c,input_h,input_w};
	int     weight_shape[4]={output_c,input_c,kernel_h,kernel_w};
	int      bias_shape[1]={output_c};
	int output_shape[4]={1,output_c,output_h,output_w};

	int * input_shapes[]={input_shape,weight_shape,bias_shape};
	int * output_shapes[]={output_shape};


	test_node_set_input(node,inputs,input_shapes,3);
	test_node_set_output(node,outputs,output_shapes,1);

	//optional: set the device to run the node

	test_node_prerun(node);

	//warm up run
	test_node_run(node);

	printf("performance benchmark run: repeat count [%d]\n",repeat_count);

	unsigned long start_time=get_cur_time();

	for(int i=0;i<repeat_count;i++)
		test_node_run(node);

	unsigned long end_time=get_cur_time();

	unsigned long used_time=end_time-start_time;   

	printf("total used: %lu us\n",used_time);

	test_node_postrun(node);

	destroy_test_node(node);

	free(input);
	free(weight);
	free(bias);
	free(output);

	release_tengine_library();

	return 0;  

}





