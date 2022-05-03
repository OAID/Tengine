

extern void ocl_OP_UPSAMPLE_creator();
extern void ocl_OP_CONV_creator();
extern void ocl_OP_CONCAT_creator();
extern void ocl_OP_POOL_creator();
extern void ocl_OP_RELU_creator();
extern void ocl_OP_DROPOUT_creator();
extern void ocl_OP_RELU1_creator();
extern void ocl_OP_RELU6_creator();
extern void ocl_OP_FLATTEN_creator();
extern void ocl_OP_FC_creator();
extern void ocl_OP_ELTWISE_creator();
extern void ocl_OP_INTERP_creator();
//
//
void register_all_ocl_creator(void)
{
    ocl_OP_CONCAT_creator();
    ocl_OP_CONV_creator();
    ocl_OP_POOL_creator();
    ocl_OP_RELU_creator();
    ocl_OP_UPSAMPLE_creator();
    ocl_OP_DROPOUT_creator();
    ocl_OP_FLATTEN_creator();
    ocl_OP_FC_creator();
    ocl_OP_RELU1_creator();
    ocl_OP_RELU6_creator();
    ocl_OP_ELTWISE_creator();
    ocl_OP_INTERP_creator();
}
