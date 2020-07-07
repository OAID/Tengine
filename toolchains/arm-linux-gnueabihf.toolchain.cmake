# set cross-compiled system type, it's better not use the type which cmake cannot recognized.
SET ( CMAKE_SYSTEM_NAME Linux )
SET ( CMAKE_SYSTEM_PROCESSOR arm )

# when gcc-arm-linux-gnueabihf was installed, toolchain was available as below:
SET ( CMAKE_C_COMPILER "arm-linux-gnueabihf-gcc" )
SET ( CMAKE_CXX_COMPILER "arm-linux-gnueabihf-g++" )

# set searching rules for cross-compiler
SET ( CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER )
SET ( CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY )
SET ( CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY )

# other needed options
SET (TENGINE_TOOLCHIN_FLAG "-march=armv7-a -mfloat-abi=hard -mfpu=neon-vfpv4")
SET (TENGINE_FORCE_SKIP_OPENMP OFF)
