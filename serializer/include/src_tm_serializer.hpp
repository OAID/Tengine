#ifndef __SRC_TM_SERIALIZER_HPP__
#define __SRC_TM_SERIALIZER_HPP__

#include "patch_serializer.hpp"

namespace TEngine {

class SrcTmSerializer: public PatchSerializer {

public:
      SrcTmSerializer(void);
      int  GetPatchNumber(void) final;
      uint32_t GetVendorId(void) final;
      uint32_t GetNNId(void) final;
};


} //namespace TEngine


#endif
