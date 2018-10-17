#ifndef __PATCH_SERIALIZER_HPP__
#define __PATCH_SERIALIZER_HPP__

#include "src_serializer.hpp"

namespace TEngine {

class PatchSerializer: public SrcSerializer {
public:
     virtual   bool encrypt(const void * text, int text_size, void **crypt_addr, int * crypt_size) override;
     virtual   bool decrypt(const void * crypt_addr, int crypt_size, uint32_t vendor_id, uint32_t nn_id,
		                                    void **text_addr, int* text_size) override;

     virtual   bool SaveToSource(const char * model_name,void * addr, int size) override;

     virtual   std::vector<struct model_patch> GetModelByName(const std::string& model_name) override;

     virtual   int  GetPatchNumber(void)=0;
     virtual  void GenerateIOV(void * buf, uint32_t vendor_id,uint32_t nn_id);
     virtual   uint32_t GetVendorId(void)=0;
     virtual   uint32_t GetNNId(void)=0;
     virtual   const char * GetRegisterFuncName(void);


     virtual ~PatchSerializer(void) {};

};	


} //namespace TEngine


#endif
