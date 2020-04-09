#define __TENGINE_ARRT_ENV__  __attribute__ ((section(".tengine")))

#ifdef CONFIG_KERNEL_FP32
#define ST_FP32 "Y"
#else
#define ST_FP32 "N"
#endif

#ifdef CONFIG_KERNEL_FP16
#define ST_FP16 "Y"
#else
#define ST_FP16 "N"
#endif

#ifdef CONFIG_KERNEL_INT8
#define ST_INT8 "Y"
#else
#define ST_INT8 "N"
#endif

#ifdef CONFIG_KERNEL_UINT8
#define ST_UINT8 "Y"
#else
#define ST_UINT8 "N"
#endif

#ifdef ENABLE_ONLINE_REPORT
#define ST_OLREPORT "Y"
#else
#define ST_OLREPORT "N"
#endif

char tengine_field_info[1024] __TENGINE_ARRT_ENV__ = {
	"TENGINE_FIELD_INFO_" ST_FP32 ST_FP16 ST_INT8 ST_UINT8 ST_OLREPORT
};

