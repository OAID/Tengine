# this function will check if gcc is hard float point

FUNCTION(TENGINE_CHECK_ARM32_HARD_FP _flag)
    SET (_compile_flag "-march=armv7-a -mfpu=neon -mfloat-abi=hard")
    CHECK_C_COMPILER_FLAG (${_compile_flag} ${_flag})
ENDFUNCTION()
