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
 * Author: haitao@openailab.com
 */

#ifndef __DEVICE_DRIVER_HPP__
#define __DEVICE_DRIVER_HPP__

#include <functional>
#include <string>

#include "compiler.hpp"

#include "graph.hpp"
#include "dev_proposal.hpp"

namespace TEngine {

class Device;
struct DevExecutor;

enum dev_status_t
{
    kDevInvalid,
    kDevNormal,
    kDevStopped,
    kDevSuspend,
    kDevRemoved
};

struct DevWorkload
{
    float past;    // workload percentage on last period
    float current;
    float future;
    int period;    // in us
    float pending_fops;
};

struct GraphPerf
{
    float latency;    // in us
    float power;    // in mW
    float memory;    // in kB
};

using dev_type_t = std::string;
using dev_id_t = std::string;
using dev_graph_cb_t = std::function<void(Subgraph*, bool)>;
using dev_node_cb_t = std::function<void(Node*, bool)>;

/*!
 * @brief Driver class defines the interface between device and device
 *        executor, which is the HAL interface.
 *
 *        TODO: create document to describe the Graph/Node/Tensor data structure
 *        as the device needs to know the structure of the graph
 *
 *        For current supported operator list, please visit
 *           https://github.com/OAID/Tengine/blob/master/doc/operator_ir.md
 */

class Driver
{
public:
    const std::string& GetName(void)
    {
        return name_;
    }
    void SetName(const std::string& name)
    {
        name_ = name;
    }
    bool AutoProbe(void)
    {
        return auto_probe_;
    }

    Driver(void)
    {
        auto_probe_ = true;
    }

    /*!
     * @brief detect the supported H/W in system.
     *         For each detected H/W, identified by dev_id,
     *         a device is created. It is possible for single driver to create
     *         multiple devices.
     *
     * @param None
     * @return how many devices created
     */

    virtual int ProbeDevice(void) = 0;

    /*!
     * @brief detect the device for specific device id
     *
     * @param dev_id the device id to be probed
     * @return if the specific H/W detected or not
     */
    virtual bool ProbeDevice(const dev_id_t& dev_id) = 0;

    /*!
     * @brief release all resource allocated for devices created by this driver
     *        Only a device's status is stopped, it can be released.
     *
     * @param None
     * @return the number of devices was released
     */

    virtual int DestroyDevice(void) = 0;

    /*!
     * @brief release resource of a single device
     *
     * @param device to be destoried
     * @return release done or failed
     */
    virtual bool DestroyDevice(Device* device) = 0;

    /*!
     * @brief get the number of devices created by the driver
     *
     * @param None
     * @return number of devices
     */

    virtual int GetDeviceNum(void) = 0;

    /*!
     * @brief get the device by the index in device list
     *
     * @param idx, the index of the device, cannot be larger than
     *             the number returned by GetDeviceNum()
     * @return pointer of the device object
     */

    virtual Device* GetDevice(int idx) = 0;

    /*!
     * @brief get the device by device name
     *
     * @param device name
     * @return pointer of the device object
     */

    virtual Device* GetDevice(const std::string& name) = 0;

    /*!
     * @brief Initalize H/W resource for the device
     *        Should be called in ProbeDevice()
     * @param device, pointer to the device to be initialied
     * @return true, H/W initialization successed. Otherwise, false
     */

    virtual bool InitializeDevice(Device* device) = 0;

    /*!
     * @brief release H/W resource allocated for the device
     *
     * @param device, pointer to the device H/W resource to be released
     * @return true, H/W release successed. Otherwise, false
     */
    virtual bool ReleaseDevice(Device* device) = 0;

    /*!
     * @brief Get the status of the device. dev_status_t defines the possible
     *        status of a device
     *
     * @param device, pointer to the device
     * @return device status
     */

    virtual dev_status_t GetDeviceStatus(Device* device) = 0;

    /*!
     * @brief Start the H/W and set the device status to kDevNormal
     *
     * @param device to be started
     * @return true, start device successfully
     */

    virtual bool StartDevice(Device* device) = 0;

    /*!
     * @brief Stop the H/W and set the device status to kDevStopped
     *
     * @param device to be stepped
     * @return true, stop device successfully
     */
    virtual bool StopDevice(Device* device) = 0;

    /*!
     * @brief Create a graph handle on a device. Graph handle is a H/W context inside device
     *
     * @param dev, device to create context
     * @param graph, the graph to be run on the device
     *
     * @return graph handle if context created successfully
     *         nullptr, if create failed
     */

    virtual void* CreateGraphHandle(Device* dev, Subgraph* graph) = 0;

    /*!
     * @brief Create a graph handle on a device. Graph handle is a H/W context inside device
     *
     * @param dev, device to create context
     *
     * @return graph handle if context created successfully
     *         nullptr, if create failed
     */

    virtual void* CreateGraphHandle(Device* dev) = 0;

    /*!
     * @brief Release the graph handle which is created on device dev
     *        H/W resource allocated for this graph handle should be released too
     *
     * @param dev, device where the graph handle belongs
     * @param graph handle, pointer of graph handle to be release
     *
     * @return true, release with no error
     */

    virtual bool ReleaseGraphHandle(Device* dev, void* graph_handle) = 0;

    /*!
     * @brief Set the callback hook, when graph bound with graph_handle is done.
     *        It is used by driver to notify the upper layer the work is done.
     *
     * @param dev, the device to be set
     * @param graph_handle, the graph handle
     * @param func, the callback function
     *
     * @return none
     */

    virtual void SetGraphDoneHook(Device* dev, void* graph_handle, dev_graph_cb_t func) = 0;

    /*!
     * @brief Set the callback hook, when a single node executed in graph_handle context
     *        is done. It is used by driver to notify the upper layer node execution is done.
     *
     * @param dev, the device to be set
     * @param graph_handle, the graph handle
     * @param func, the callback function
     *
     * @return none
     */

    virtual void SetNodeDoneHook(Device* dev, void* graph_handle, dev_node_cb_t func) = 0;

    /*!
     * @brief Optimized the upper layer passed graph, and save the optimized graph in graph_handle
     *
     * @param dev, the device
     * @param graph_handle, the graph handle
     * @param graph, the subgraph to be executed
     *
     * @return true, optimizated with no error
     */

    virtual bool OptimizeGraph(Device* dev, void* graph_handle, Subgraph* graph) = 0;

    /*!
     * @brief Optimized the upper layer passed graph, saved in graph_handle.
     *        The optimized graph will be saved in graph_handle too
     *
     * @param dev, the device
     * @param graph_handle, the graph handle
     *
     * @return true, optimizated with no error
     */

    virtual bool OptimizeGraph(Device* dev, void* graph_handle) = 0;

    /*!
     * @brief Prepare the execution of the optimized subgraph, saved in graph_handle
     *        This interface will only be called once for a subgraph, or graph_handle
     *
     * @param dev, the device
     * @param graph_handle, the graph handle
     *
     * @return true, preturn with no error
     */

    virtual bool Prerun(Device* dev, void* graph_handle) = 0;

    /*!
     * @brief Run the subgraph bounded with graph_handle.
     *        This interface may be called repeatedly.
     *        It is supposed to be a non-block interface and
     *        the graph done hook should be called when graph execution is done
     *
     * @param dev, the device
     * @param graph_handle, the graph handle
     *
     * @return true, run the graph with no error
     */

    virtual bool Run(Device* dev, void* graph_handle) = 0;

    /*!
     * @brief Run the subgraph bounded with graph_handle in block mode
     *
     * @param dev, the device
     * @param graph_handle, the graph handle
     *
     * @return true, run the graph with no error
     */

    virtual bool SyncRun(Device* dev, void* graph_handle) = 0;

    /*!
     * @brief Release resource allocated for graph execution
     *
     * @param dev, the device
     * @param graph_handle, the graph_handle
     *
     * @return true, run the graph with no error
     */
    virtual bool Postrun(Device* dev, void* graph_handle) = 0;

    /*!
     * @brief Prepare the execution of a single node, in context of graph_handle
     *        This interface will only be called once for a node
     *
     * @param dev, the device
     * @param graph_handle, the graph handle
     * @param node, the node to be executed
     *
     * @return true, preturn with no error
     */

    virtual bool Prerun(Device* dev, void* graph_handle, Node* node) = 0;

    /*!
     * @brief Run the node in context of graph_handle.
     *        This interface may be called repeatedly.
     *        It is supposed to be a non-block interface and
     *        the node done hook should be called when graph execution is done
     *
     * @param dev, the device
     * @param graph_handle, the graph handle
     * @param node, the node to be executed
     *
     * @return true, run the node with no error
     */

    virtual bool Run(Device* dev, void* graph_handle, Node* node) = 0;

    /*!
     * @brief Run the node in graph_handle in block mode
     *
     * @param dev, the device
     * @param graph_handle, the graph handle
     * @param node, the node to be executed
     *
     * @return true, run the node with no error
     */

    virtual bool SyncRun(Device* dev, void* graph_handle, Node* node) = 0;

    /*!
     * @brief Release resource allocated for the node execution
     *
     * @param dev, the device
     * @param graph_handle, the graph_handle
     * @param node, the node whose resource will be released
     *
     * @return true, run the node with no error
     */

    virtual bool Postrun(Device* dev, void* graph_handle, Node* node) = 0;

    /*!
     * @brief Get the size of memory need to run the graph saved in graph_handle
     *
     * @param dev, the device
     * @param graph_handle, the graph_handle
     * @param mem_size, return the requested memory size
     *
     * @return true, the value in mem_size is trustale
     *         false, the device will allocate memory by itself
     */

    virtual bool GetRunMemorySize(Device* dev, void* graph_handle, unsigned int& mem_size)
    {
        return false;
    }

    /*!
     * @brief Set the memory address to run graph saved in graph_handle
     *
     * @param dev, the device
     * @param graph_handle, the graph_handle
     * @param mem_addr, the address of memory region
     *
     * @return none
     */

    virtual void SetRunMemory(Device* dev, void* graph_handle, void* mem_addr){};

    /*!
     * @brief Configure the device, referenced by attr_name
     *
     * @param dev, the device
     * @param attr_name, the name of the config item
     * @param val, pointer of the argument buffer
     * @param size, the size of the arugment buffer
     *
     * @return true, config done successfully
     */

    virtual bool SetDevAttr(Device* dev, const char* attr_name, const void* val, int size)
    {
        return false;
    }

    /*!
     * @brief Get the configure of the device, referenced by attr_name
     *
     * @param dev, the device
     * @param attr_name, the name of the config item
     * @param val, pointer of the argument buffer
     * @param size, the size of the arugment buffer
     *
     * @return true, get config item successfully
     */

    virtual bool GetDevAttr(Device* dev, const char* attr_name, void* val, int size)
    {
        return false;
    }

    virtual bool SetGraphAttr(Device* dev, void* graph_handle, const char* attr_name, const void* val, int size)
    {
        return false;
    }
    virtual bool GetGraphAttr(Device* dev, void* graph_handle, const char* attr_name, void* val, int size)
    {
        return false;
    }

    /* these interfaces are used by device allocator/scheduler */

    /*!
     * @brief Get the workload of the device
     *
     * @param dev, the device
     * @param load, the workload of the device
     *
     * @return true, the data in load is valid
     */

    virtual bool GetWorkload(Device* dev, DevWorkload& load) = 0;

    /*!
     * @brief Get the performance vector of a sugraph with specific execution policy
     *
     * @param dev, the device
     * @param graph, the subgraph
     * @param policy, the execution policy
     * @param perf, the returned perf vector
     *
     * @return true, the data in perf is valid
     */
    virtual bool GetPerf(Device* dev, Subgraph* graph, int policy, GraphPerf& perf) = 0;

    /*!
     * @brief Get the performance vector of a node with specific execution policy
     *
     * @param dev, the device
     * @param node, the node
     * @param policy, the execution policy
     * @param perf, the returned perf vector
     *
     * @return true, the data in perf is valid
     */
    virtual bool GetPerf(Device* dev, Node* node, int policy, GraphPerf& perf)
    {
        return false;
    }

    /*!
     * @brief Get the float ops rate ( Kfops/second) for the device to execution a graph
     *        if the graph is nullptr, return the nominal fops of the device
     * @param dev, the device
     * @param graph, the graph
     * @param policy, the execution policy
     *
     * @return the float ops rate
     */

    virtual float GetFops(Device* dev, Subgraph* graph, int policy) = 0;

    /*!
     * @brief Get the priority the device claimed for a policy
     *
     * @param dev, the device
     * @param policy, the policy
     *
     * @return int, the priority
     */

    virtual int GetPolicyPriority(Device* dev, int policy) = 0;

    /*!
     * @brief Get the execution propsoal for subgraph on this dev with execution policy
     *        the device should go throught the graph, and set or replace DEV_PROPOSAL_ATTR of a node
     *        only when its level is greater than presetted one
     *
     * @param dev, the device
     * @param graph, the graph
     * @param policy, the execution policy
     *
     * @return true, there is new proposal in graph
     *         false, nothing is changed
     */

    virtual bool GetProposal(Device* dev, Graph* graph, int policy, bool static_assign) = 0;

    virtual Subgraph* GetOptimizedGraph(Device* dev, void* graph_handle)
    {
        return nullptr;
    }

    virtual ~Driver() {}

protected:
    std::string name_;
    bool auto_probe_;
};

class Device
{
public:
    /* Manangement Interface */
    Device(const dev_id_t& dev_id)
    {
        dev_id_ = dev_id;
        policy_ = 0;
    }
    void SetName(const std::string& name)
    {
        name_ = name;
    }
    void BindDriver(Driver* driver)
    {
        driver_ = driver;
    }
    void SetDeviceType(const dev_type_t& dev_type)
    {
        dev_type_ = dev_type;
    }

    const std::string& GetName(void)
    {
        return name_;
    }
    const dev_id_t& GetDeviceID(void)
    {
        return dev_id_;
    }
    const dev_type_t& GetDeviceType(void)
    {
        return dev_type_;
    }
    Driver* GetDriver(void)
    {
        return driver_;
    }

    /* control interface */

    bool InitHW(void)
    {
        return driver_->InitializeDevice(this);
    }
    bool ReleaseHW(void)
    {
        return driver_->ReleaseDevice(this);
    }
    bool Stop(void)
    {
        return driver_->StopDevice(this);
    }
    bool Start(void)
    {
        return driver_->StartDevice(this);
    }

    /* data process interface */
    void* CreateGraphHandle(Subgraph* graph)
    {
        return driver_->CreateGraphHandle(this, graph);
    }
    void* CreateGraphHandle(void)
    {
        return driver_->CreateGraphHandle(this);
    }
    bool ReleaseGraphHandle(void* graph_handle)
    {
        return driver_->ReleaseGraphHandle(this, graph_handle);
    }

    bool OptimizeGraph(void* graph_handle)
    {
        return driver_->OptimizeGraph(this, graph_handle);
    }
    bool OptimizeGraph(void* graph_handle, Subgraph* graph)
    {
        return driver_->OptimizeGraph(this, graph_handle, graph);
    }

    Subgraph* GetOptimizedGraph(void* graph_handle)
    {
        return driver_->GetOptimizedGraph(this, graph_handle);
    }

    void SetGraphDoneHook(void* graph_handle, dev_graph_cb_t func)
    {
        driver_->SetGraphDoneHook(this, graph_handle, func);
    }
    void SetNodeDoneHook(void* graph_handle, dev_node_cb_t func)
    {
        driver_->SetNodeDoneHook(this, graph_handle, func);
    }

    bool Prerun(void* graph_handle)
    {
        return driver_->Prerun(this, graph_handle);
    }
    bool Run(void* graph_handle)
    {
        return driver_->Run(this, graph_handle);
    }
    bool SyncRun(void* graph_handle)
    {
        return driver_->SyncRun(this, graph_handle);
    }
    bool Postrun(void* graph_handle)
    {
        return driver_->Postrun(this, graph_handle);
    }

    bool Prerun(void* graph_handle, Node* node)
    {
        return driver_->Prerun(this, graph_handle, node);
    }
    bool Run(void* graph_handle, Node* node)
    {
        return driver_->Run(this, graph_handle, node);
    }
    bool SyncRun(void* graph_handle, Node* node)
    {
        return driver_->SyncRun(this, graph_handle, node);
    }
    bool Postrun(void* graph_handle, Node* node)
    {
        return driver_->Postrun(this, graph_handle, node);
    }

    bool GetRunMemorySize(void* graph_handle, unsigned int& mem_size)
    {
        return driver_->GetRunMemorySize(this, graph_handle, mem_size);
    }

    void SetRunMemory(void* graph_handle, void* mem_addr)
    {
        driver_->SetRunMemory(this, graph_handle, mem_addr);
    }

    /*device config/query interface */
    bool SetDevAttr(const char* attr_name, const void* val, int size)
    {
        return driver_->SetDevAttr(this, attr_name, val, size);
    }

    bool GetDevAttr(const char* attr_name, void* buf, int size)
    {
        return driver_->GetDevAttr(this, attr_name, buf, size);
    }

    bool SetGraphAttr(void* graph_handle, const char* attr_name, const void* val, int size)
    {
        return driver_->SetGraphAttr(this, graph_handle, attr_name, val, size);
    }

    bool GetGraphAttr(void* graph_handle, const char* attr_name, void* buf, int size)
    {
        return driver_->GetGraphAttr(this, graph_handle, attr_name, buf, size);
    }

    /* query/stats interface */
    dev_status_t GetDeviceStatus(void)
    {
        return driver_->GetDeviceStatus(this);
    }

    void GetWorkload(DevWorkload& load)
    {
        driver_->GetWorkload(this, load);
    }
    bool GetPerf(Subgraph* graph, int policy, GraphPerf& perf)
    {
        return driver_->GetPerf(this, graph, policy, perf);
    };
    float GetFops(Subgraph* graph, int policy)
    {
        return driver_->GetFops(this, graph, policy);
    }
    int GetPolicyPriority(int policy)
    {
        return driver_->GetPolicyPriority(this, policy);
    }

    bool GetProposal(Subgraph* graph, int policy, bool static_assign)
    {
        return driver_->GetProposal(this, graph, policy, static_assign);
    }

    void SetPolicy(int policy)
    {
        policy_ = policy;
    }
    int GetPolicy(void)
    {
        return policy_;
    }

    virtual ~Device(){};

protected:
    dev_id_t dev_id_;
    Driver* driver_;
    std::string name_;
    dev_type_t dev_type_;
    int policy_;
};

class DriverManager;

extern template DriverManager SimpleObjectManagerWithLock<DriverManager, Driver*>::instance;

class DriverManager : public SimpleObjectManagerWithLock<DriverManager, Driver*>
{
public:
    using probe_default_t = void (*)(void);

    static bool RegisterDriver(const std::string& name, Driver* driver);
    static bool UnregisterDriver(const std::string& name);
    static bool UnregisterDriver(Driver* driver);
    static Driver* GetDriver(const std::string& name);
    static Device* GetDevice(const dev_id_t& dev_id);

    static Device* GetDefaultDevice(void);
    static bool GetDefaultDeviceName(std::string& dev_name);
    static bool SetDefaultDevice(const std::string& dev_name);
    static void SetDefaultDevice(Device* device);
    static bool HasDefaultDevice(void);

    static bool LoadDevice(Driver* driver, Device* device);
    static bool LoadDevice(Driver* driver);

    static bool UnloadDevice(Driver* driver);
    static bool UnloadDevice(Driver* driver, Device* device);

    static int ProbeDevice(void);
    static int ReleaseDevice(void);

    static void SetProbeDefault(probe_default_t probe);

    Device* RealGetDevice(const dev_id_t& dev_id);

    DriverManager()
    {
        default_dev = nullptr;
        probe_default = nullptr;
    }

protected:
    Device* default_dev;
    probe_default_t probe_default;
};

}    // namespace TEngine

#endif
