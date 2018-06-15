include_directories(driver/include)
include_directories(driver/cpu)
include_directories(core/include)

FILE(GLOB_RECURSE ALL_LIB_CPP_SRCS driver/plugin/init.cpp driver/cpu/*.cpp)

list(APPEND TENGINE_LIB_SRCS ${ALL_LIB_CPP_SRCS})




