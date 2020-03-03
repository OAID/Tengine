
## 1. Use valgrind to check memory leakage 

    ./valgrind --tool=memcheck <command_to_run> 

common mistakes:
     forget to call put_graph_tensor() on tensor handle, which is gotten by get_graph_tensor()
                    put_graph_node()  on node handle, which is gotten by put_graph_node()


## 2. Add .hidden in assembly code

   In order to hide the function symbol outside dynamic library, add this line in your assembly code

        .global <func_name>
	.hidden <func_name>






