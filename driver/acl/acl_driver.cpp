#include "acl_driver.hpp"
#include "acl_executor.hpp"

#include "arm_compute/runtime/CL/CLScheduler.h"

#include "acl_conv.hpp"
#include "acl_lrn.hpp"

#define  ATTR_NODE_OPS "acl_node_ops"
#define  RK3399_DEV_ID "sw.acl.rk3399.gpu"


namespace TEngine {

void RegisterAllNodeOps(ACLDriver * driver);

ACLDriver::ACLDriver(void)
{
     SetName("ACLDriver");

     dev_id_.push_back(RK3399_DEV_ID);

     arm_compute::CLScheduler::get().default_init();

     RegisterAllNodeOps(this);

}

ACLDriver::~ACLDriver(void)
{
   for(auto item: node_ops_map_)
       delete item.second;

}

bool ACLDriver::Prerun(Device * dev, void * node_handle, Node * node)
{
	Operator * op=node->GetOp();

	if(node_ops_map_.count(op->GetName())==0)
        {
               XLOG_ERROR()<<"ACLDriver does not support operator: "<<op->GetName()<<"\n";
		return false;
        }

	ACLNodeOps * node_ops=node_ops_map_[op->GetName()];

	if(!node_ops->Prerun(node))
		return false;

	node->SetAttr(ATTR_NODE_OPS,node_ops);

	return true;
}

bool ACLDriver::SyncRun(Device * dev, void * node_handle, Node * node)
{
	if(!node->ExistAttr(ATTR_NODE_OPS))
		return false;

	ACLNodeOps * node_ops=any_cast<ACLNodeOps *>(node->GetAttr(ATTR_NODE_OPS));

	return node_ops->Run(node);

}

bool ACLDriver::Run(Device * dev, void * node_handle, Node * node) 
{
	bool ret=SyncRun(dev,node_handle,node);

	DevContext * context=reinterpret_cast<DevContext *>(node_handle);

	if(context->node_cb)
		context->node_cb(node,ret);

	return ret; 
}

bool ACLDriver::Postrun(Device * dev, void * node_handle, Node  * node)
{
	if(!node->ExistAttr(ATTR_NODE_OPS))
		return false;

	ACLNodeOps * node_ops=any_cast<ACLNodeOps *>(node->GetAttr(ATTR_NODE_OPS));

	return node_ops->Postrun(node);
}

bool ACLDriver::InitDev(NodeDevice * device)
{
     if(device->GetDeviceID()==RK3399_DEV_ID)
     {
         arm_compute::CLScheduler::get().set_target(arm_compute::GPUTarget::T800);

     }

     return true;
}

bool ACLDriver::ProbeDevice(const dev_id_t& dev_id) 
{
    ACLDevice * dev=new ACLDevice(dev_id);

    InitializeDevice(dev);
    dev->SetName(dev_id);

    dev_table_.push_back(dev);

    return true;

}


bool ACLDriver::DestroyDevice(Device * device) 
{
	ACLDevice * acl_dev=dynamic_cast<ACLDevice *>(device);

	if(acl_dev->dev_status!=kDevStopped)
		return false;

	ReleaseDevice(acl_dev);

        auto ir=dev_table_.begin();

        while((*ir)!=acl_dev && ir!=dev_table_.end())
        {
             ir++;
        }

        dev_table_.erase(ir);

        delete acl_dev;

	return true;
}

void ACLDriver::RegisterNodeOps(const std::string& op_name,ACLNodeOps *ops)
{
	node_ops_map_[op_name]=ops;
}

void RegisterAllNodeOps(ACLDriver * driver)
{
     ACLConvOps * ops=new ACLConvOps();
     ops->name="Convolution";

     driver->RegisterNodeOps(ops->name,ops);

     ACLLrnOps *lrnops = new ACLLrnOps();
     lrnops->name="LRN";
     driver->RegisterNodeOps(lrnops->name,lrnops);
    

}

//////////////////////////////////////////////

void ACLDriverInit(void)
{
    ACLDriver * acl_driver=new ACLDriver();

    DriverManager::RegisterDriver(acl_driver->GetName(),acl_driver);

    auto dev_executor_factory=DevExecutorFactory::GetFactory();

    int n=acl_driver->GetDevIDTableSize();

    for(int i=0;i<n;i++)
        dev_executor_factory->RegisterInterface<ACLExecutor,const dev_id_t&>
                (acl_driver->GetDevIDByIdx(i));

    LOG_INFO()<<"ACL Driver Initialized\n";
}



} //namespace TEngine


