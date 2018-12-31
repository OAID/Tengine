# Create device for new SoC

Tengine can automatically probe a SoC as running device, also can manually create a device for new SoC. 

---
## 1. Predefined SoC
Tengine predefines some SoCs. You can get these SoCs by using `get_predefined_cpu()`.

- rk3399    4xA53 + 2xA72
- a63       4xA53
- rk3288    4xA17
- r40       4xA7
- kirin960  4xA53 + 4xA73
- apq8096   4xA72

---

## 2. Manually set SoC
Below is an examples about how to create a device for SoC(4xA55).

### 2.1. Set new SoC info
The declaration of struct `cpu_cluster` and `cpu_info` is in [cpu_device.h](../core/include/cpu_device.h).
```
    struct cpu_cluster my_cluster = {4, 1520, CPU_A55, ARCH_ARMV8,
                                     32<<10, 52<<10, {0, 1, 2, 3} };

    struct cpu_info my_soc = {"my_soc", "my_board", 1, 0, &my_cluster, -1, NULL};
```

### 2.2. Create device
After init_tengine() has been called
```
    int cpu_list={0,1,2,3};
    set_online_cpu(&my_soc, cpu_list, 4);
    create_cpu_device("my_device", &my_soc);
```

### 2.3. Set the device as default device 
```
   set_default_device("my_device");
```

### 3. Run graph
Tengine can run graph on the created device
```
    prerun_graph(graph);
    run_graph(graph,1);
```

