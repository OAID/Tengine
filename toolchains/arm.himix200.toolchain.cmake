# arm-himix200: Hi3516AV300 Hi3516CV500 Hi3516DV300 Hi3519AV100

# set cross-compiled system type, it's better not use the type which cmake cannot recognized.
SET ( CMAKE_SYSTEM_NAME Linux )
SET ( CMAKE_SYSTEM_PROCESSOR arm )

# when hislicon SDK was installed, toolchain was installed in the path as below:
SET ( CMAKE_FIND_ROOT_PATH "/opt/hisi-linux/x86-arm/arm-himix200-linux" )
SET ( CMAKE_C_COMPILER     "/opt/hisi-linux/x86-arm/arm-himix200-linux/bin/arm-himix200-linux-gcc" )
SET ( CMAKE_CXX_COMPILER   "/opt/hisi-linux/x86-arm/arm-himix200-linux/bin/arm-himix200-linux-g++" )

# set searching rules for cross-compiler
SET ( CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER )
SET ( CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY )
SET ( CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY )

# other needed options
SET (TENGINE_TOOLCHIN_FLAG "-march=armv7-a -mfloat-abi=softfp -mfpu=neon-vfpv4")
SET (TENGINE_FORCE_SKIP_OPENMP OFF)
