/*
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _DLA_INTERFACE_H_
#define _DLA_INTERFACE_H_

/**
 * @ingroup Processors
 * @name DLA Processors
 * Processor modules in DLA engine. Each processor has it's
 * own operation a.k.a. HW layer. Network is formed using
 * graph of these operations
 * @{
 */
#define DLA_OP_BDMA		0
#define DLA_OP_CONV		1
#define DLA_OP_SDP		2
#define DLA_OP_PDP		3
#define DLA_OP_CDP		4
#define DLA_OP_RUBIK		5
/** @} */

/**
 * @ingroup Processors
 * @name Maximum number of processors
 * @brief DLA ash 6 processors
 * @{
 */
#define DLA_OP_NUM		6
/** @} */

/**
 * @ingroup Processors
 * @name Number of groups
 * @brief Each processor has 2 groups of registers
 * @{
 */
#define DLA_NUM_GROUPS		2
/** @} */

/**
 * Network descriptor
 *
 * Contains all information to execute a network
 *
 * @op_head: Index of first operation of each type in operations list
 * @num_rois: Number of ROIs
 * @num_operations: Number of operations in one list
 * @num_luts: Number of LUTs
 */
struct dla_network_desc {
	int16_t operation_desc_index;
	int16_t surface_desc_index;

	int16_t dependency_graph_index;
	int16_t lut_data_index;

	int16_t roi_array_index;
	int16_t surface_index;

	int16_t stat_list_index;
	int16_t reserved1;

	int16_t op_head[DLA_OP_NUM];

	uint16_t num_rois;
	uint16_t num_operations;

	uint16_t num_luts;
	uint16_t num_addresses;

	int16_t input_layer;
	uint8_t dynamic_roi; /* bool to indicate whether network.roi_array_index */
	uint8_t reserved0;
} __attribute__ ((packed, aligned(4)));

/**
 * @name Memory types
 * @brief DLA engnine can read/write to/from 3 memory types
 * @{
 */
#define DLA_MEM_MC			0 /* External DRAM */
#define DLA_MEM_CV			1 /* CV-SRAM */
#define DLA_MEM_HW			2 /* DLA sub-module */
/** @} */

/**
 * @ingroup Events
 * @name Operation events
 * @brief Different events triggered by an operations
 * @{
 */
#define DLA_EVENT_OP_COMPLETED		1
#define DLA_EVENT_OP_PROGRAMMED		2
#define DLA_EVENT_OP_ENABLED		3
#define DLA_EVENT_CDMA_WT_DONE		4
#define DLA_EVENT_CDMA_DT_DONE		5
/** @} */

struct dla_consumer {
	int16_t index; /* the index of dla_common_op_desc in dep_graph_addr */
	uint8_t event;
	uint8_t res;
} __attribute__ ((packed, aligned(4)));

struct dla_common_op_desc {
	int16_t index; /* set by ucode */
	int8_t roi_index;
	uint8_t op_type;

	uint8_t dependency_count;
	uint8_t reserved0[3];

	struct dla_consumer consumers[DLA_OP_NUM];
	struct dla_consumer fused_parent;
} __attribute__ ((packed, aligned(4)));

struct dla_roi_array_desc {
	uint32_t array_length;

	uint32_t array_reserved;
} __attribute__ ((packed, aligned(4)));

struct dla_roi_desc {
	uint32_t left;

	uint32_t top;

	uint32_t right;

	uint32_t bottom;
} __attribute__ ((packed, aligned(4)));

/**
 * @ingroup BDMA
 * @name Maximum BDMA transfers
 * @brief BDMA supports multiple transfers in operation. This indicates
 *        maximum number of transfers possible in one operation.
 * @{
 */
#define NUM_MAX_BDMA_OPS	20
/** @} */

struct dla_bdma_transfer_desc {
	int16_t source_address;
	int16_t destination_address;

	uint32_t line_size;

	uint32_t line_repeat;

	uint32_t source_line;

	uint32_t destination_line;

	uint32_t surface_repeat;

	uint32_t source_surface;

	uint32_t destination_surface;
} __attribute__ ((packed, aligned(4)));

struct dla_bdma_surface_desc {
	uint8_t source_type;
	uint8_t destination_type;
	uint16_t num_transfers;

	struct dla_bdma_transfer_desc transfers[NUM_MAX_BDMA_OPS];
} __attribute__ ((packed, aligned(4)));

struct dla_bdma_op_desc {
	uint16_t num_transfers;
	uint16_t reserved0;
} __attribute__ ((packed, aligned(4)));

struct dla_bdma_stat_desc {
	uint32_t read_stall;
	uint32_t write_stall;
	uint32_t runtime;
} __attribute__ ((packed, aligned(4)));

/**
 * @ingroup Convolution
 * @name Convolution mode
 * @brief Convolution modes support by DLA
 * @{
 */
#define CONV_MODE_DIRECT	0
#define CONV_MODE_WINOGRAD	1
/** @} */

/**
 * @ingroup Processors
 * @name Precision BPE mapping
 * @brief Precision formats and Bit Per Elements mapping
 * @{
 */
#define BPE_PRECISION_INT8		1
#define BPE_PRECISION_INT16		2
#define BPE_PRECISION_FP16		2
/** @} */


/**
 * @ingroup Processors
 * @name Precision types
 * @brief Precision formats supported by DLA engine
 * @{
 */
#define PRECISION_INT8		0
#define PRECISION_INT16		1
#define PRECISION_FP16		2
/** @} */

/**
 * @ingroup Processors
 * @name Data formats
 * @brief Data formats supported by DLA engine
 * @{
 */
#define FORMAT_T_R8			0
#define FORMAT_T_R10			1
#define FORMAT_T_R12			2
#define FORMAT_T_R16			3
#define FORMAT_T_R16_I			4
#define FORMAT_T_R16_F			5
#define FORMAT_T_A16B16G16R16		6
#define FORMAT_T_X16B16G16R16		7
#define FORMAT_T_A16B16G16R16_F		8
#define FORMAT_T_A16Y16U16V16		9
#define FORMAT_T_V16U16Y16A16		10
#define FORMAT_T_A16Y16U16V16_F		11
#define FORMAT_T_A8B8G8R8		12
#define FORMAT_T_A8R8G8B8		13
#define FORMAT_T_B8G8R8A8		14
#define FORMAT_T_R8G8B8A8		15
#define FORMAT_T_X8B8G8R8		16
#define FORMAT_T_X8R8G8B8		17
#define FORMAT_T_B8G8R8X8		18
#define FORMAT_T_R8G8B8X8		19
#define FORMAT_T_A2B10G10R10		20
#define FORMAT_T_A2R10G10B10		21
#define FORMAT_T_B10G10R10A2		22
#define FORMAT_T_R10G10B10A2		23
#define FORMAT_T_A2Y10U10V10		24
#define FORMAT_T_V10U10Y10A2		25
#define FORMAT_T_A8Y8U8V8			26
#define FORMAT_T_V8U8Y8A8			27
#define FORMAT_T_Y8___U8V8_N444		28
#define FORMAT_T_Y8___V8U8_N444		29
#define FORMAT_T_Y10___U10V10_N444	30
#define FORMAT_T_Y10___V10U10_N444	31
#define FORMAT_T_Y12___U12V12_N444	32
#define FORMAT_T_Y12___V12U12_N444	33
#define FORMAT_T_Y16___U16V16_N444	34
#define FORMAT_T_Y16___V16U16_N444	35
#define FORMAT_FEATURE			36
/** @} */

/**
 * @ingroup Convolution
 * @name Pixel mapping
 * @brief Pixel mapping formats supported for image input in Convolution
 * @{
 */
#define MAP_PITCH_LINEAR		0
/** @} */

/**
 * @ingroup Convolution
 * @name Weight formats
 * @brief Weight data formats supported in Convolution
 * @{
 */
#define WEIGHT_FORMAT_UNCOMPRESSED	0
#define WEIGHT_FORMAT_COMPRESSED	1
/** @} */

/**
 * @ingroup Convolution
 * @name Mean data format
 * @brief Mean data formats supported in Convolution
 * @{
 */
#define MEAN_FORMAT_DISABLE     0
#define MEAN_FORMAT_ENABLE      1
/** @} */

struct dla_cvt_param {
	int16_t  scale;
	uint8_t  truncate;
	uint8_t  enable;

	int32_t  offset;
} __attribute__ ((packed, aligned(4)));

struct dla_data_cube {
	uint16_t type; /* dla_mem_type */
	int16_t address; /* offset to the actual IOVA in task.address_list */

	uint32_t offset; /* offset within address */
	uint32_t size;

	/* cube dimensions */
	uint16_t width;
	uint16_t height;

	uint16_t channel;
	uint16_t reserved0;

	/* stride information */
	uint32_t line_stride;
	uint32_t surf_stride;

	/* For Rubik only */
	uint32_t plane_stride;
} __attribute__ ((packed, aligned(4)));

#define PIXEL_OVERRIDE_UINT 0
#define PIXEL_OVERRIDE_INT  1

struct dla_conv_surface_desc {
	/* Data cube */
	struct dla_data_cube weight_data;
	struct dla_data_cube wmb_data;
	struct dla_data_cube wgs_data;
	struct dla_data_cube src_data;
	struct dla_data_cube dst_data;

	/*
	 * u_addr = input_data.source_addr + offset_u
	 * this field should be set when YUV is not interleave format
	 * */
	int64_t offset_u;

	/* line stride for 2nd plane, must be 32bytes aligned */
	uint32_t in_line_uv_stride;
} __attribute__ ((packed, aligned(4)));

struct dla_conv_op_desc {
	/* Performance parameters */

	/* dla_conv_mode */
	uint8_t conv_mode;
	uint8_t data_reuse;
	uint8_t weight_reuse;
	uint8_t skip_data_rls;

	uint8_t skip_weight_rls;
	uint8_t reserved0;
	uint16_t entry_per_slice;

	uint8_t data_format; /* dla_data_format */
	uint8_t pixel_mapping; /* dla_pixel_mapping */
	uint16_t fetch_grain; /* number of free slices before fetch */

	uint8_t reserved_b[8];

	uint8_t batch; /* batch_num */
	uint8_t weight_format; /* dla_weight_format */
	uint8_t data_bank;
	uint8_t weight_bank;

	uint32_t batch_stride; /* the offset in bytes of each data cube in a batch */

	uint8_t post_extension;
	uint8_t pixel_override;
	uint16_t release; /* number of slices need to be released */

	 /* The input cube dimension for CSC */
	uint16_t input_width_csc;
	uint16_t input_height_csc;

	uint16_t input_channel_csc;
	uint16_t kernel_width_csc;

	uint16_t kernel_height_csc;
	uint16_t kernel_channel_csc;

	/* The input cube dimension for CMAC */
	uint16_t input_width_cmac;
	uint16_t input_height_cmac;

	uint32_t bytes_per_kernel; /* actual size in bytes */

	/* Algorithm parameters */

	int16_t mean_ry; /* mean value for red in RGB or Y in YUV */
	int16_t mean_gu; /* mean value for green in RGB or U in YUV */

	int16_t mean_bv; /* mean value for blue in RGB or V in YUV */
	int16_t mean_ax;

	uint8_t mean_format; /* dla_mean_format */
	uint8_t conv_stride_x;
	uint8_t conv_stride_y;
	uint8_t pad_x_left;

	uint8_t pad_x_right;
	uint8_t pad_y_top;
	uint8_t pad_y_bottom;
	uint8_t dilation_x;

	uint8_t dilation_y;
	uint8_t reserved2[2];

	/* Precision parameters */
	uint8_t pra_truncate;

	uint8_t in_precision;
	uint8_t out_precision; /* The output precision from CONV, it's the MAC processing precison */
	int16_t pad_val;

	struct dla_cvt_param in_cvt; /* input convertor parameters */
	struct dla_cvt_param out_cvt; /* output convertor parameters, support truncate only */

} __attribute__ ((packed, aligned(4)));

struct dla_conv_stat_desc {
	uint32_t data_read_stall;
	uint32_t weight_read_stall;
	uint32_t data_read_latency;
	uint32_t weight_read_latency;
	uint32_t saturation_count;
	uint32_t nan_data_num;
	uint32_t nan_weight_num;
	uint32_t inf_data_num;
	uint32_t inf_weight_num;
	uint32_t runtime;
} __attribute__ ((packed, aligned(4)));

/**
 * @ingroup SDP
 * @name Activation functions
 * @brief Activation functions supported in SDP
 * @{
 */
#define ACTIVATION_NONE		0
#define ACTIVATION_RELU		1
#define ACTIVATION_LUT		2
#define ACTIVATION_PRELU	3
/** @} */

/**
 * @ingroup LUT
 * @name LUT size
 * @brief LUT sizes for linear and exponentila LUT
 * @{
 */
#define LUT_LINEAR_EXP_TABLE_ENTRY_LOG2		6
#define LUT_LINEAR_ONLY_TABLE_ENTRY_LOG2	8
/** @} */

/**
 * @ingroup LUT
 * @name LUT types
 * @brief DLA supports two types of LUT, linear and exonential
 * @{
 */
#define LUT_LINEAR_EXP_TABLE		0
#define LUT_LINEAR_ONLY_TABLE		1
/** @} */

/**
 * @ingroup LUT
 * @name LUT methods
 * @brief DLA supports two types of LUT, linear and exonential
 * @{
 */
#define LUT_METHOD_EXPONENTIAL		0
#define LUT_METHOD_LINEAR		1
/** @} */

/**
 * @ingroup LUT
 * @name LUT
 * @brief DLA supports two types of LUT, linear and exonential
 * @{
 */
#define LUT_PRI_LINEAR_EXP		0
#define LUT_PRI_LINEAR_ONLY		1
/** @} */

union dla_lut_offset {
	/**
	 * Number should be substracted on log domain before look up
	 * exponetial table it has the same definition as hardware
	 * thus input scaling should also take into account when
	 * set this field.
	 */
	int8_t exp_offset;
	/**
	 * Number of bits should be right shift before looking
	 * up linear table
	 */
	int8_t frac_bits;
	uint16_t reserved0;
};

/**
 * This struct is used to represent floating point values by INT
 * suppose we have a float point number fp_x, it will be represented
 * as:
 *
 * fp_x = scale_int_x>>(shifter_x)
 *
 * This is very useful for INT pipeline;
 */
struct dla_float_data {
	int16_t scale;
	int8_t shifter;
	uint8_t reserved0;
} __attribute__ ((packed, aligned(4)));

/**
 * For INT pipeline, we use the struct above to represent a floating number;
 * For FP16 pipeline, we should store the FP16 encoded value into a uint16_t
 * container
 */
union dla_slope {
	struct dla_float_data data_i;

	uint16_t data_f;
};

struct dla_lut_param {
	/**
	 * value of expression ((1<<LUT_LINEAR_EXP_TABLE_ENTRY_LOG2)+1) is 65,
	 * ((1<<LUT_LINEAR_ONLY_TABLE_ENTRY_LOG2)+1) is 257, and int16_t is of
	 * 2Byte. And below two statement's combined memory size is 644 Byte.
	 *
	 * NOTE: below two declaration combined size should always be multiple
	 * of 4.
	 */
	int16_t linear_exp_table[(1<<LUT_LINEAR_EXP_TABLE_ENTRY_LOG2)+1];
	int16_t linear_only_table[(1<<LUT_LINEAR_ONLY_TABLE_ENTRY_LOG2)+1];

	union dla_lut_offset linear_exp_offset;
	union dla_lut_offset linear_only_offset;

	/* The start and end point of raw table, valid when raw_method=LINEAR only */
	uint64_t linear_exp_start;
	uint64_t linear_exp_end;
	uint64_t linear_only_start;
	uint64_t linear_only_end;

	union dla_slope linear_exp_underflow_slope;
	union dla_slope linear_exp_overflow_slope;
	union dla_slope linear_only_underflow_slope;
	union dla_slope linear_only_overflow_slope;

	/**
	 * dla_lut_priority, when both lut are hit(or one overflow, the other underflow),
	 * which one should be selected as output
	 */
	uint8_t hybrid_priority;
	uint8_t underflow_priority;
	uint8_t overflow_priority;
	uint8_t method; /* dla_lut_method */
} __attribute__ ((packed, aligned(4)));

struct dla_sdp_surface_desc {
	/* Data cube */
	/* source input cube, available when SDP working on offline mode */
	struct dla_data_cube src_data;

	/* X1 input cube */
	struct dla_data_cube x1_data;

	/* X2 input cube */
	struct dla_data_cube x2_data;

	/* Y input cube */
	struct dla_data_cube y_data;

	/* Output cube */
	struct dla_data_cube dst_data;
} __attribute__ ((packed, aligned(4)));

#define SDP_OP_NONE		0
#define SDP_OP_MUL		1
#define SDP_OP_ADD		2
#define SDP_OP_BOTH		3

#define SDP_ALU_OP_MAX		0
#define SDP_ALU_OP_MIN		1
#define SDP_ALU_OP_SUM		2
#define SDP_ALU_OP_EQL		3

#define SDP_OP_PER_LAYER	0
#define SDP_OP_PER_KERNEL	1
#define SDP_OP_PER_POINT	2

struct dla_sdp_cvt {
	struct dla_cvt_param alu_cvt;
	struct dla_cvt_param mul_cvt;
} __attribute__ ((packed, aligned(4)));

struct dla_sdp_op {
	uint8_t enable;
	uint8_t alu_type; /* dla_sdp_alu_op_type */
	uint8_t type; /* dla_sdp_op_type */
	uint8_t mode; /* dla_sdp_op_mode */

	uint8_t act; /* dla_act_type */
	uint8_t shift_value;	// left shift
	uint8_t truncate;
	uint8_t precision;

	int32_t alu_operand;
	int32_t mul_operand;

	struct dla_sdp_cvt  cvt;
} __attribute__ ((packed, aligned(4)));

struct dla_sdp_op_desc {
	/* Precision parameters */
	/* dla_precision */
	uint8_t src_precision;
	uint8_t dst_precision;
	int16_t lut_index;

	struct dla_cvt_param out_cvt;

	/* Performance parameters */
	/* dla_conv_mode */
	uint8_t conv_mode;
	uint8_t batch_num;
	uint16_t reserved0;

	uint32_t batch_stride;	// will be used when batch_num > 1

	/* Algorithm parameters */
	struct dla_sdp_op x1_op;
	struct dla_sdp_op x2_op;
	struct dla_sdp_op y_op;
} __attribute__ ((packed, aligned(4)));

struct dla_sdp_stat_desc {
	uint32_t nan_input_num;
	uint32_t inf_input_num;
	uint32_t nan_output_num;
	uint32_t wdma_write_stall;
	uint32_t lut_underflow;
	uint32_t lut_overflow;
	uint32_t lut_hybrid;
	uint32_t lut_le_hit;
	uint32_t lut_lo_hit;
	uint32_t saturation_count;
	uint32_t runtime;
} __attribute__ ((packed, aligned(4)));

#define POOL_MODE_AVG		0
#define POOL_MODE_MAX		1
#define POOL_MODE_MIN		2

#define POOL_SIZE_1		0
#define POOL_SIZE_2		1
#define POOL_SIZE_3		2
#define POOL_SIZE_4		3
#define POOL_SIZE_5		4
#define POOL_SIZE_6		5
#define POOL_SIZE_7		6
#define POOL_SIZE_8		7

#define PDP_PAD_VAL_NUM	7

struct dla_pdp_surface_desc {
	/* Data cube */
	struct dla_data_cube src_data;

	struct dla_data_cube dst_data;
} __attribute__ ((packed, aligned(4)));

struct dla_pdp_op_desc {
	/* Performance parameters */
	uint16_t  partial_in_width_first;
	uint16_t  partial_in_width_mid;

	uint16_t  partial_in_width_last;
	uint16_t  partial_width_first;

	uint16_t  partial_width_mid;
	uint16_t  partial_width_last;

	uint8_t   split_num;

	/* Algorithm parameters */
	uint8_t  pool_mode; /* dla_pool_mode */
	uint8_t  pool_width; /* dla_pool_width */
	uint8_t  pool_height; /* dla_pool_height */

	uint8_t  stride_x;
	uint8_t  stride_y;

	/* The left/right padding size, pad_right might be less than pad_left */
	uint8_t  pad_left;
	uint8_t  pad_right;

	/* The top/bottom padding size */
	uint8_t  pad_top;
	uint8_t  pad_bottom;

	/* Precision parameters */
	uint8_t  precision; /* dla_precision */
	uint8_t  reserved0;
	/**
	 * if input has non-zero "offset", this value should be set
	 * There'll be 7 different paddding values, the relationship between
	 * those versions are:
	 * padding_value[0] = -offset*scaling;
	 * padding_value[1] = 2*padding_value[0]
	 * padding_value[2] = 3*padding_value[0]
	 * ...
	 * The purpose is to avoid ucode implement FP16 multiplier(for FP16 mode)
	 */
	int32_t  padding_value[PDP_PAD_VAL_NUM];
} __attribute__ ((packed, aligned(4)));

struct dla_pdp_stat_desc {
	uint32_t inf_input_num;
	uint32_t nan_input_num;
	uint32_t nan_output_num;
	uint32_t write_stall;
	uint32_t runtime;
} __attribute__ ((packed, aligned(4)));

struct dla_cdp_surface_desc {
	/* Data cube */
	struct dla_data_cube src_data;

	struct dla_data_cube dst_data;
} __attribute__ ((packed, aligned(4)));

struct dla_cdp_op_desc {
	/* Precision parameters */

	/* dla_precision */
	uint8_t  in_precision;
	uint8_t  out_precision;
	int16_t  lut_index;

	struct dla_cvt_param in_cvt;
	struct dla_cvt_param out_cvt;

	/* Performance parameters */

	/* Algorithm parameters */
	uint8_t  local_size;
	uint8_t  bypass_sqsum;
	uint8_t  bypass_out_mul;
	uint8_t  reserved0;
} __attribute__ ((packed, aligned(4)));

struct dla_cdp_stat_desc {
	uint32_t nan_input_num;
	uint32_t inf_input_num;
	uint32_t nan_output_num;
	uint32_t write_stall;
	uint32_t lut_uflow;
	uint32_t lut_oflow;
	uint32_t lut_hybrid;
	uint32_t lut_le_hit;
	uint32_t lut_lo_hit;
	uint32_t saturation_count;
	uint32_t runtime;
} __attribute__ ((packed, aligned(4)));

struct dla_rubik_surface_desc {
	/* Data cube */
	struct dla_data_cube src_data;

	struct dla_data_cube dst_data;
} __attribute__ ((packed, aligned(4)));

/* rubik mode */
#define RUBIK_MODE_CONTRACT	0
#define RUBIK_MODE_SPLIT	1
#define RUBIK_MODE_MERGE	2

struct dla_rubik_op_desc {
	/* Precision parameters */
	uint8_t mode;
	uint8_t precision;
	uint8_t stride_x;
	uint8_t stride_y;
} __attribute__ ((packed, aligned(4)));

struct dla_rubik_stat_desc {
	uint32_t read_stall;
	uint32_t write_stall;
	uint32_t runtime;
} __attribute__ ((packed, aligned(4)));

union dla_surface_container {
	struct dla_bdma_surface_desc bdma_surface;
	struct dla_conv_surface_desc conv_surface;
	struct dla_sdp_surface_desc sdp_surface;
	struct dla_pdp_surface_desc pdp_surface;
	struct dla_cdp_surface_desc cdp_surface;
	struct dla_rubik_surface_desc rubik_surface;
};

union dla_operation_container {
	struct dla_bdma_op_desc bdma_op;
	struct dla_conv_op_desc conv_op;
	struct dla_sdp_op_desc sdp_op;
	struct dla_pdp_op_desc pdp_op;
	struct dla_cdp_op_desc cdp_op;
	struct dla_rubik_op_desc rubik_op;
};

union dla_stat_container {
	struct dla_bdma_stat_desc bdma_stat;
	struct dla_conv_stat_desc conv_stat;
	struct dla_sdp_stat_desc sdp_stat;
	struct dla_pdp_stat_desc pdp_stat;
	struct dla_cdp_stat_desc cdp_stat;
	struct dla_rubik_stat_desc rubik_stat;
};

/**
 * status notifier structure
 *
 * @address: 64-bit timestamp representing the time at which
 * the notifier was written
 * @status_engine: status work captured from HW engine
 * @subframe: NA
 * @status_task: status word as configured from an action list
 */
struct dla_task_status {
	uint64_t timestamp;

	uint32_t status_engine;

	uint16_t subframe;
	uint16_t status_task;
} __attribute__ ((packed, aligned(4)));

#endif
