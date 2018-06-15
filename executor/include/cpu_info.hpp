#ifndef __CPU_INFO_HPP__
#define __CPU_INFO_HPP__

#include <string.h>
#include <stdlib.h>

#include "cpu_device.h"

namespace TEngine {


struct CPUInfo {

	struct cpu_cluster * find_cluster(int cpu_id) const
	{
		int start_idx=0;
		struct cpu_cluster * p_cluster;

		for(int i=0;i<dev.cluster_number;i++)
		{
			p_cluster=&dev.cluster[i];

			if(cpu_id>=start_idx && cpu_id<start_idx+p_cluster->cpu_number)
				return p_cluster;

			start_idx+=p_cluster->cpu_number;
		}

		return NULL;
	}

	int GetCPUModel(int cpu_id) const
	{
		struct cpu_cluster * cluster=find_cluster(cpu_id);
		return cluster->cpu_model;
	}

	int GetCPUArch(int cpu_id) const
	{
		struct cpu_cluster * cluster=find_cluster(cpu_id);
		return cluster->cpu_arch;
	}

	int GetMaxFreq(int cpu_id) const
	{
		struct cpu_cluster * cluster=find_cluster(cpu_id);
		return cluster->max_freq;
	}

	int GetL2Size(int cpu_id) const
	{
		struct cpu_cluster * cluster=find_cluster(cpu_id);
		return cluster->l2_size;
	}


        const char * GetCPUArchString(int cpu_id) const;

        const char * GetCPUModelString(int cpu_id) const;


	int GetMasterCPU(void) const { return master_cpu;}

	int GetCPUNumber(void) const { return dev.online_cpu_number;}

        int GetOnlineCPU(int idx) const { return dev.online_cpu_list[idx];}

	CPUInfo(const struct cpu_info * cpu_dev)
	{

		if(cpu_dev->cpu_name)
			dev.cpu_name=strdup(cpu_dev->cpu_name);
		else
			dev.cpu_name=NULL;

		if(cpu_dev->board_name)
			dev.board_name=strdup(cpu_dev->board_name);
		else
			dev.board_name=NULL;

		dev.l3_size=cpu_dev->l3_size;
		dev.cluster_number=cpu_dev->cluster_number;
		dev.cluster=(struct cpu_cluster *)malloc(sizeof(struct cpu_cluster)*dev.cluster_number);

		for(int i=0;i<cpu_dev->cluster_number;i++)
		{
			memcpy(dev.cluster+i,cpu_dev->cluster+i,sizeof(struct cpu_cluster));
		}

		dev.online_cpu_number=cpu_dev->online_cpu_number;

		if(dev.online_cpu_number>0)
		{
			dev.online_cpu_list=(int *)malloc(sizeof(int)*dev.online_cpu_number);
			memcpy(dev.online_cpu_list,cpu_dev->online_cpu_list,sizeof(int)*dev.online_cpu_number);
		}
		else
		{
			int total_number=0;

			for(int i=0;i<dev.cluster_number;i++)
				total_number+=dev.cluster[i].cpu_number;

			dev.online_cpu_number=total_number;
			dev.online_cpu_list=(int *)malloc(sizeof(int)*total_number);

			for(int i=0;i<total_number;i++)
				dev.online_cpu_list[i]=i;
		}

		int max_freq=0;

		for(int i=0; i<dev.online_cpu_number;i++)
		{
			int cur_max_freq=GetMaxFreq(dev.online_cpu_list[i]);

			if(cur_max_freq>=max_freq)
			{
				master_cpu=dev.online_cpu_list[i];
				max_freq=cur_max_freq;
			}
		}

	}

	~CPUInfo(void)
	{

		if(dev.cpu_name)  free((void *)dev.cpu_name);

		if(dev.board_name) free((void *)dev.board_name);

		free(dev.cluster);
		free(dev.online_cpu_list);
	}


	struct cpu_info dev;
	int master_cpu;
};

} //namespace TEngine




#endif
