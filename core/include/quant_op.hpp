#ifndef __QUANT_OP_HPP__
#define __QUANT_OP_HPP__

#include <vector>

namespace TEngine {

struct quant_param {
   int   quant_method;
   int   quant_width;
   bool  data_quanted;
   float min;
   float max;
   float scale;
   float float_zero;
   int   quant_zero;

};

struct quant_op {

   quant_op() { enabled=false;}

   bool enabled;
   bool quant_input;
   bool dequant_output;
   std::vector<quant_param> params;
};



} //namespace TEngine

#endif
