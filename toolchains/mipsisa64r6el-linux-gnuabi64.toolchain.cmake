# set cross-compiled system type, it's better not use the type which cmake cannot recognized.
SET (CMAKE_SYSTEM_NAME Linux)
SET (CMAKE_SYSTEM_PROCESSOR mips64r2)

# when mips-linux-gnu-gcc was installed, toolchain was available as below:
SET (CMAKE_C_COMPILER   "mipsisa64r6el-linux-gnuabi64-gcc")
SET (CMAKE_CXX_COMPILER "mipsisa64r6el-linux-gnuabi64-g++")

# set searching rules for cross-compiler
SET (CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET (CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET (CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# other needed options
SET (TENGINE_TOOLCHAIN_FLAG -march=mips64r6 -mmsa -mhard-float -mfp64 -mnan=2008)

# do not skip OpenMP check as default
SET (TENGINE_FORCE_SKIP_OPENMP OFF)
