#ifndef __TM2_OP_SAVE_HPP__
#define __TM2_OP_SAVE_HPP__

#include <functional>
#include <cstdlib>
extern "C" {
#include "utility/vector.h"
#include "serializer/tmfile/tm2_format.h"
#include "tm2_generate.h"
#include "graph/node.h"

#include "op_include.h"
}

using op_save_t = std::function<tm_uoffset_t(void* const, tm_uoffset_t*, ir_node_t*)>;
op_save_t SaveTmOpFunc(uint32_t op_type);

#endif