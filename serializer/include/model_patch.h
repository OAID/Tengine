#ifndef __MODEL_PATCH_HPP__
#define __MODEL_PATCH_HPP__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_MODEL_NAME_LEN (64-1)

struct model_patch {
	char model_name[MAX_MODEL_NAME_LEN+1];
	uint32_t  vendor_id;
	uint32_t  nn_id;
	uint32_t  total_size;
	uint32_t  patch_off;
	uint32_t  patch_size;
	void *    addr;
};

#ifdef __cplusplus	
};
#endif



#endif
