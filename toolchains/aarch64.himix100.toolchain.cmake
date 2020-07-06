# aarch64-himix100: Hi3556AV100

# set cross-compiled system type, it's better not use the type which cmake cannot recognized.
SET ( CMAKE_SYSTEM_NAME Linux )
SET ( CMAKE_SYSTEM_PROCESSOR aarch64 )

# when hislicon SDK was installed, toolchain was installed in the path as below:
SET ( CMAKE_FIND_ROOT_PATH "/opt/hisi-linux/x86-arm/aarch64-himix100-linux" )
SET ( CMAKE_C_COMPILER     "/opt/hisi-linux/x86-arm/aarch64-himix100-linux/bin/aarch64-himix100-linux-gcc" )
SET ( CMAKE_CXX_COMPILER   "/opt/hisi-linux/x86-arm/aarch64-himix100-linux/bin/aarch64-himix100-linux-g++" )

# set searching rules for cross-compiler
SET ( CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER )
SET ( CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY )
SET ( CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY )

# other needed options
SET (TENGINE_TOOLCHIN_FLAG "-march=armv8-a")
SET (TENGINE_FORCE_SKIP_OPENMP OFF)
