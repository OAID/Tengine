# arm-himix400: Hi3520DV500 Hi3521DV200

# set cross-compiled system type, it's better not use the type which cmake cannot recognized.
SET (CMAKE_SYSTEM_NAME Linux)
SET (CMAKE_SYSTEM_PROCESSOR arm)

# when HiSilicon SDK was installed, toolchain was installed in the path as below:
SET (CMAKE_FIND_ROOT_PATH "/opt/hisi-linux/x86-arm/arm-himix400-linux")

# then set compiler
SET (CMAKE_ASM_COMPILER   "/opt/hisi-linux/x86-arm/arm-himix400-linux/bin/arm-himix400-linux-gcc")
SET (CMAKE_C_COMPILER     "/opt/hisi-linux/x86-arm/arm-himix400-linux/bin/arm-himix400-linux-gcc")
SET (CMAKE_CXX_COMPILER   "/opt/hisi-linux/x86-arm/arm-himix400-linux/bin/arm-himix400-linux-g++")

# set searching rules for cross-compiler
SET (CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET (CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET (CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# other needed options
SET (TENGINE_TOOLCHAIN_FLAG -march=armv7-a -mfloat-abi=softfp -mfpu=neon-vfpv4)

# it's no need to skip OpenMP for multi-core processor Hi3520DV500 or Hi3521DV200
SET (TENGINE_FORCE_SKIP_OPENMP OFF)
