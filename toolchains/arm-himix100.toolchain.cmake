# arm-himix100: Hi3516DV200 Hi3516EV200 Hi3516EV300 Hi3518EV300; Hi3556V200 Hi3559V200

# set cross-compiled system type, it's better not use the type which cmake cannot recognized.
SET (CMAKE_SYSTEM_NAME Linux)
SET (CMAKE_SYSTEM_PROCESSOR arm)

# when HiSilicon SDK was installed, toolchain was installed in the path as below:
SET (CMAKE_FIND_ROOT_PATH "/opt/hisi-linux/x86-arm/arm-himix100-linux")

# then set compiler
SET (CMAKE_ASM_COMPILER   "/opt/hisi-linux/x86-arm/arm-himix100-linux/bin/arm-himix100-linux-gcc")
SET (CMAKE_C_COMPILER     "/opt/hisi-linux/x86-arm/arm-himix100-linux/bin/arm-himix100-linux-gcc")
SET (CMAKE_CXX_COMPILER   "/opt/hisi-linux/x86-arm/arm-himix100-linux/bin/arm-himix100-linux-g++")

# set searching rules for cross-compiler
SET (CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET (CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET (CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# other needed options
SET (TENGINE_TOOLCHAIN_FLAG -march=armv7-a -mfloat-abi=softfp -mfpu=neon-vfpv4)

# it may be a good choice for this low cost series products to skip OpenMP check
SET (TENGINE_FORCE_SKIP_OPENMP ON)
