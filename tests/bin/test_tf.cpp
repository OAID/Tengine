/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2018, Open AI Lab
 * Author: chunyinglv@openailab.com
 */
#include <iostream>
#include <iomanip>
#include <string>
#include "tengine_c_api.h"
#include <sys/time.h>


int main(int argc, char *argv[])
{
    // init tengine
    init_tengine_library();
    if (request_tengine_version("0.1") < 0)
        return 1;

    // load model
    const char *model_name = "test";
    std::string mdl_name ="models/test.pb";
    if (load_model(model_name, "tensorflow", mdl_name.c_str()) < 0)
        return 1;
    std::cout << "load model done!\n";

    // create graph
    graph_t graph = create_runtime_graph("graph", model_name, NULL);
    if (!check_graph_valid(graph))
    {
        std::cout << "create graph0 failed\n";
        return 1;
    }

    
	tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
	int      input_size=get_tensor_buffer_size(input_tensor);
	float * input_data=(float *)malloc(input_size);

	for(unsigned int i=0;i<input_size/sizeof(float);i++)
		input_data[i]=i;


    prerun_graph(graph);

	dump_graph(graph);

    set_tensor_buffer(input_tensor, input_data, input_size);

    run_graph(graph, 1);

    tensor_t output_tensor=get_graph_output_tensor(graph,0,0);

	float * output_data=(float *)get_tensor_buffer(output_tensor);
	int     output_size=get_tensor_buffer_size(output_tensor);

	for(unsigned int i=0;i<output_size/sizeof(float);i++)
		printf("output %d: %g\n",i,output_data[i]);


    free(input_data);

    put_graph_tensor(input_tensor);
    put_graph_tensor(output_tensor);

    postrun_graph(graph);
    destroy_runtime_graph(graph);

    remove_model(model_name);

    std::cout << "ALL TEST DONE\n";

    return 0;
}
