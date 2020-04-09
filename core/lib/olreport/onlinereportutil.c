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
 * Copyright (c) 2019, Open AI Lab
 * Author: jjzeng@openailab.com
 */

#include "onlinereportutil.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifdef __ANDROID__
#include <sys/system_properties.h>
#endif

#define CPU_IMPLEMENTER "CPU implementer"
#define CPU_ARCH "CPU architecture"
#define CPU_VARIANT "CPU variant"
#define CPU_PART "CPU part"
#define CPU_VERSION "CPU revision"
#define MEMORY_TOTAL "MemTotal"

int get_rand()
{
    return rand() +1;
}

void strtrim(char* data)
{
	int idx = strlen(data) - 1;
	while( idx >= 0 && data[idx] == ' ' )
	{
		--idx;
	}
	data[idx+1] = '\0';
}

uint32_t get_arch()
{
    #ifdef CONFIG_ARCH_ARM64
    return 64;
    #else
    return 32;
    #endif
}

uint32_t get_totoal_memory()
{
    uint32_t res = 0;
    FILE* fp = fopen("/proc/meminfo","r");
    if( fp == NULL)
    {
        return res;
    }

    char buff[1024] = {'\0'};
    if( fgets( buff,1024,fp ) )
    {
        const char* pa = strstr(buff,MEMORY_TOTAL);
        if( pa )
        {
            pa += strlen(MEMORY_TOTAL) + 1;
            res = strtoul( pa,NULL,10 );
        }
    }

    return res; 
}

void get_os_info(char* os,int maxLen)
{
#ifdef __ANDROID__
    char sdk_ver_str[PROP_VALUE_MAX +1] = {'\0'};
    __system_property_get("ro.build.version.sdk", sdk_ver_str);
    int sdk_ver = atoi(sdk_ver_str);
    sprintf(os,"Android-%d",sdk_ver); 
#else
    FILE* fp = fopen("/etc/os-release","r");
    if( fp != NULL )
    {
        char buf[512] = {'\0'};
        while(fgets(buf, 512, fp))
        {
            if( memcmp(buf, "PRETTY_NAME=", 12) == 0 )
            {
                int len = strlen(buf) - 12 - 3 ;
                len = len < maxLen?len:(maxLen-1);
                memcpy(os,buf+13,len);
                os[len] = '\0';
                break;
            }
        }

	strtrim(os);
        fclose(fp);
        return ;
    }

    fp = fopen("/etc/issue","r");
    if( fp == NULL)
    {
        get_os_kernel_info(os,maxLen);
        return ;
    }

    char* res= fgets( os,maxLen,fp );
    if( res != NULL )
    {
        char* pl = strstr(os,"\\n");
        if( pl != NULL )
        {
			--pl;
			while( *pl == ' ' && pl > os )
			{
				--pl;
			}

		    *(pl+1) = '\0';
        }                    
    }
    strtrim(os);
    fclose(fp);
#endif
    
}

void get_os_kernel_info(char* os,int maxlen)
{
    FILE* fp = fopen("/proc/version","r");
    if( fp != NULL )
    {
        if( fgets(os,maxlen,fp) ){}

        char* pa = strstr(os,"(");
        if( pa != NULL )
        {
            *pa = '\0';
        }
	strtrim(os);
        fclose(fp);
        return ;
    }

    fp = fopen("/proc/sys/kernel/ostype","r");
    if( fp == NULL)
    {
        return ;
    }

    int offset = fscanf(fp,"%s",os);
    fclose(fp);
    offset = strlen(os);
    os[offset] = ' ';
    offset += 1;
    
    fp = fopen("/proc/sys/kernel/osrelease","r");
    if( fp == NULL)
    {
        return ;
    }

    int rest_len = maxlen - offset;
    if( fgets( os + offset,rest_len,fp ) )
	{
		if( os[ strlen(os)-1 ] == '\n' )
		{
			os[ strlen(os)-1 ] = '\0';
		}
	}

    strtrim(os);
    fclose(fp);
}

uint32_t parse_cpu_param(const char* buf,int base)
{
    while( *buf != ':' && *buf != '\0' )
    {
        ++buf;
    }

    if( *buf == '\0' )
    {
        return 0;
    }
    return strtoul(buf+1, NULL, base);
}

void parse_cpuid(FILE* fp,char* cpu_id)
{
    char buf[256] = {'\0'};
    int flag = 0;
    uint32_t cpu_implementer= 0,cpu_arch=0,cpu_variant=0,cpu_part=0,cpu_version=0;

    while( fgets(buf,256,fp) )
    {
        if( memcmp(buf, CPU_IMPLEMENTER, strlen(CPU_IMPLEMENTER) ) == 0  )
        {
            ++flag;
            cpu_implementer = parse_cpu_param(buf + strlen(CPU_IMPLEMENTER),16);
        }
        else if( memcmp(buf, CPU_ARCH, strlen(CPU_ARCH) ) == 0  )
        {
            ++flag;
            cpu_arch = parse_cpu_param(buf + strlen(CPU_ARCH),10);                
        }
        else if( memcmp(buf, CPU_VARIANT, strlen(CPU_VARIANT) ) == 0  )
        {
            ++flag;
            cpu_variant = parse_cpu_param(buf + strlen(CPU_VARIANT),16);                
        }
        else if( memcmp(buf, CPU_PART, strlen(CPU_PART) ) == 0  )
        {
            ++flag;
            cpu_part = parse_cpu_param(buf + strlen(CPU_PART),16);                
        } 
        else if( memcmp(buf, CPU_VERSION, strlen(CPU_VERSION) ) == 0  )
        {
            ++flag;
            cpu_version = parse_cpu_param(buf + strlen(CPU_VERSION),10);                
        }

        if( flag >= 5 )
        {
            break;
        }
    }

    sprintf(cpu_id,"0x%x:%u:0x%x:0x%x:%u",cpu_implementer,cpu_arch,cpu_variant,cpu_part,cpu_version);

}

void get_cpu_param_info(int query_cpuid, char* cpu_id,uint32_t* cpu_freq,uint32_t *total_cpu_nums)
{
    char cpu_path[256] = {'\0'};
    sprintf(cpu_path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", query_cpuid);

    struct stat stat_buf;
    if(stat(cpu_path, &stat_buf) >= 0)
    {
        FILE* fp = fopen(cpu_path, "r");

        if(fp != NULL)
        {
            if(fscanf(fp, "%d", cpu_freq) < 0)
            {
                fclose(fp);
            }

            fclose(fp);                
        }

    }

    FILE* fp = fopen("/proc/cpuinfo", "rb");
    if( fp == NULL )
    {
        return ;
    }

     char buf[256];
     *total_cpu_nums = 0;
     while(fgets(buf, 256, fp)) 
     {
        if(memcmp(buf, "processor", 9) == 0)
        {
            if( *total_cpu_nums == query_cpuid )
            {
                parse_cpuid(fp,cpu_id);
            }
            ++(*total_cpu_nums);               
        }
     }    

     fclose(fp);
}

void get_cur_process_info(uint32_t* id,char *name)
{
    pid_t pid = getpid();
    *id = pid;
    char process_path[1024] ={'\0'};
    if( readlink("/proc/self/exe",process_path,1024) < 0 )
    {
        return ;
    }

    const char* pe = strrchr(process_path,'/');
    if( pe == NULL )
    {
        return ;
    }

    ++pe;
    strcpy(name,pe);
}
