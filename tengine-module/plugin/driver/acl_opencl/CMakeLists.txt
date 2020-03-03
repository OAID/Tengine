if(NOT DEFINED ACL_ROOT)
    message(FATAL_ERROR "please set ACL_ROOT for acl opencl")
endif()

include_directories(${ACL_ROOT} ${ACL_ROOT}/include)

set(DRIVER_TARGET acl_opencl)

FILE(GLOB_RECURSE DRIVER_SRCS "*.cpp")

ADD_LIBRARY(${DRIVER_TARGET} SHARED ${DRIVER_SRCS})

if(ANDROID)
 	if(("${ANDROID_ABI}" STREQUAL "armeabi-v7a"))
		set(ACL_LIBRARY ${ACL_ROOT}/build_32/libarm_compute_core.so ${ACL_ROOT}/build_32/libarm_compute.so)
	else()
		set(ACL_LIBRARY ${ACL_ROOT}/build_64/libarm_compute_core.so ${ACL_ROOT}/build_64/libarm_compute.so)
	endif()
else()
	set(ACL_LIBRARY ${ACL_ROOT}/build/libarm_compute_core.so ${ACL_ROOT}/build/libarm_compute.so)
	if( "${ARCH_TYPE}" STREQUAL  "Arm32"  )
		add_definitions(-march=armv7-a -mfpu=neon -mfp16-format=ieee -mfpu=neon-fp16)
	endif()
endif()

target_link_libraries(${DRIVER_TARGET} ${TENGINE_LIBRARY} ${ACL_LIBRARY})

SET_TARGET_PROPERTIES(${DRIVER_TARGET} PROPERTIES VERSION ${DRIVER_VERSION} SOVERSION ${DRIVER_SO_VERSION})

install(TARGETS ${DRIVER_TARGET} DESTINATION lib)
install(FILES ${ACL_LIBRARY} DESTINATION lib)

#below is for static library

ADD_LIBRARY(${DRIVER_TARGET}-static STATIC ${DRIVER_SRCS})
SET_TARGET_PROPERTIES(${DRIVER_TARGET}-static PROPERTIES OUTPUT_NAME  ${DRIVER_TARGET})

ADD_DEPENDENCIES(${DRIVER_TARGET} ${DRIVER_TARGET}-static)

install(TARGETS ${DRIVER_TARGET}-static DESTINATION lib)

