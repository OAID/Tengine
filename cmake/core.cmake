<<<<<<< HEAD
FILE(GLOB_RECURSE core_src "core/lib/*.cpp")
=======
FILE(GLOB core_src "core/lib/*.cpp" "core/lib/*.c" "core/lib/logger/*.cpp" "core/lib/logger/*.c")

if( CONFIG_ONLINE_REPORT )
    include_directories(core/include/olreport)
    FILE(GLOB online_src "core/lib/olreport/*.cpp" "core/lib/olreport/*.c")
    list(APPEND core_src ${online_src})
endif()
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074

include_directories(core/include)
include_directories(include)

FOREACH(src ${core_src})
list(APPEND TENGINE_LIB_SRCS ${src})
ENDFOREACH()
<<<<<<< HEAD

=======
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074
