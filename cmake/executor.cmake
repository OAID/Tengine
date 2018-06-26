include_directories(executor/include executor/operator/include)

FILE(GLOB_RECURSE COMMON_LIB_CPP_SRCS executor/engine/*.cpp executor/lib/*.cpp executor/plugin/*.cpp)
FILE(GLOB COMMON_CPP_SRCS  executor/operator/common/*.cpp executor/operator/common/fused/*.cpp)
if(CONFIG_ARCH_BLAS)
    FILE(GLOB COMMON_BLAS_SRCS  executor/operator/common/blas/*.cpp)
    list(APPEND COMMON_CPP_SRCS ${COMMON_BLAS_SRCS})
endif()

list(APPEND TENGINE_LIB_SRCS ${COMMON_LIB_CPP_SRCS})
list(APPEND TENGINE_LIB_SRCS ${COMMON_CPP_SRCS})

include_directories(driver/cpu)

#add openblas include
if(CONFIG_ARCH_BLAS)
     if(ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7-a"))
         include_directories(${BLAS_DIR}/arm32/include)
     endif()
     if(ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64"))
         include_directories(${BLAS_DIR}/arm64/include)
     endif()
endif()


# Now, handle the .S file
if(CONFIG_ARCH_ARM64)
    FILE(GLOB_RECURSE ARCH64_LIB_CPP_SRCS executor/operator/arm64/*.cpp)
    include_directories(executor/operator/arm64/include)

    FOREACH(file ${ARCH64_LIB_CPP_SRCS})
       set(ACL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/executor/operator/arm64/conv/conv_2d_acl")
       STRING(REGEX MATCH ${ACL_PREFIX} skip_file2 ${file})

      if( NOT skip_file2)
	      list(APPEND ARCH_LIB_CPP_SRCS ${file})
      endif()

    endforeach()
endif()


list(APPEND TENGINE_LIB_SRCS ${ARCH_LIB_CPP_SRCS})

# Now, handle the .S file

if( CONFIG_ARCH_ARM64)

set(src_path executor/operator/arm64)
FILE(GLOB TARGET_ARCH_FILES ${src_path}/*.S ${src_path}/fc/*.S 
                            ${src_path}/conv/*.S                                
                            ${src_path}/fused/*.S)
endif()

FOREACH( file ${TARGET_ARCH_FILES})

string(REPLACE "\.S" "\.s" PREPROCESS_FILE0 ${file})
string(REPLACE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} PREPROCESS_FILE ${PREPROCESS_FILE0})

get_filename_component(dest_bin_dir ${PREPROCESS_FILE} DIRECTORY)

ADD_CUSTOM_COMMAND(
      OUTPUT ${PREPROCESS_FILE}
      COMMAND mkdir -p ${dest_bin_dir}
      COMMAND ${CMAKE_C_COMPILER} -E ${file} -o ${PREPROCESS_FILE}
      DEPENDS ${file}
)

#message(${file} --> ${PREPROCESS_FILE})

list(APPEND TENGINE_LIB_SRCS ${PREPROCESS_FILE})
list(APPEND ASM_FILES ${PREPROCESS_FILE})

SET_SOURCE_FILES_PROPERTIES ( ${PREPROCESS_FILE} PROPERTIES  GENERATED  1)
set_property(SOURCE ${PREPROCESS_FILE} PROPERTY LANGUAGE C)

ENDFOREACH()


ADD_CUSTOM_TARGET(KERNEL_ASM_TARGET DEPENDS ${ASM_FILES})


