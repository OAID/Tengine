FILE(GLOB_RECURSE core_src "core/lib/*.cpp" "core/lib/*.c" )

if( CONFIG_ONLINE_REPORT )
    include_directories(core/include/olreport)
else()
	list(FILTER core_src EXCLUDE REGEX ".*olreport.*")
endif()

include_directories(core/include)
include_directories(include)

FOREACH(src ${core_src})
list(APPEND TENGINE_LIB_SRCS ${src})
ENDFOREACH()
