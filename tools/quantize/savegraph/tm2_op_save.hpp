#ifndef __TM2_OP_SAVE_HPP__
#define __TM2_OP_SAVE_HPP__

#include <functional>
extern "C" {
    #include "utility/vector.h"
    #include "tm2_format.h"
    #include "tm2_generate.h"
    #include "graph/node.h"
    
    #include "op_include.h"
}


using op_save_t = std::function<tm_uoffset_t(void* const, tm_uoffset_t*, struct node*)>;
op_save_t SaveTmOpFunc(uint32_t op_type);

#endif