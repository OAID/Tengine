include_directories(driver/include)
include_directories(driver/cpu)
include_directories(core/include)

FILE(GLOB_RECURSE ALL_LIB_CPP_SRCS driver/plugin/init.cpp driver/cpu/*.cpp)


IF(CONFIG_ACL_GPU)
  include_directories(driver/acl_graph)
  FILE(GLOB_RECURSE ALL_LIB_CPP_SRCS driver/plugin/init.cpp driver/cpu/*.cpp driver/acl_graph/*.cpp)
ENDIF(CONFIG_ACL_GPU)

list(APPEND TENGINE_LIB_SRCS ${ALL_LIB_CPP_SRCS})
