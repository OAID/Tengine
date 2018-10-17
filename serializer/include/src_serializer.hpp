#ifndef __SRC_SERIALIZER_HPP__
#define __SRC_SERIALIZER_HPP__

#include "model_patch.h"
#include "serializer.hpp"

namespace TEngine {


class SrcSerializer: public Serializer {
public:
		bool LoadModel(const std::vector<std::string>& file_list, StaticGraph * static_graph) override; 
		bool SaveModel(const std::vector<std::string>& file_list, Graph *graph) override;

		unsigned int GetFileNum(void) final {return 1;}

		bool LoadConstTensor(const std::string& fname, StaticTensor * const_tensor) override {return false; }
		bool LoadConstTensor(int fd, StaticTensor * const_tensor) override { return false; }

		virtual   bool encrypt(const void * text, int text_size, void **crypt_addr, int * crypt_size)=0;

		virtual   bool decrypt(const void * crypt_addr, int crypt_size, uint32_t vendor_id, uint32_t nn_id,
						                 void **text_addr, int* text_size)=0;

		virtual   bool compress(const void * origin, int orign_size,
						       void **compressed_addr, int * compressed_size ) { return false;}
		virtual   bool decompress(const void * compressed, int comp_size,
						       void ** decompressed_addr, int * decompressed_size) {return false;}

		virtual   bool SaveToSource(const char * model_name,void * addr, int size)=0;

		virtual   std::vector<struct model_patch> GetModelByName(const std::string& model_name)=0;

		virtual ~SrcSerializer() {}

protected:
		Serializer * backend_;

};



} //namespace TEngine

#endif
