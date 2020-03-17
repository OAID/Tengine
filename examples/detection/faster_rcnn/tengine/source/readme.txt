# Build Tengine example for Android & Linux

1. Build example for android
Set the correct ANDROID_NDK, API_LEVEL and ABI in 'android_build.sh'
Set the correct TENGINE_INCLUDE_PATH and TENGINE_LIB_PATH in 'android_build.sh'
------------------------------------------------------------------------------
build :
./android_build.sh
-------------------------------------------------------------------------------

2. Build example for android
Set the correct TENGINE_INCLUDE_PATH and TENGINE_LIB_PATH in 'linux_build.sh'
Set the correct EMBEDDED_CROSS_ROOT and TOOL_CHAIN_PREFIX in 'linux_build.sh' for cross compile
------------------------------------------------------------------------------
build :
./linux_build.sh
-------------------------------------------------------------------------------


