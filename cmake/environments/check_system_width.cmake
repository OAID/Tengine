# this function could get target system bit width in bytes

MACRO(TENGINE_CHECK_SYS_WIDTH _bit_width)
    MATH(EXPR ${_bit_width} "8 * ${CMAKE_SIZEOF_VOID_P}" OUTPUT_FORMAT DECIMAL)
ENDMACRO()
