include_directories(executor/include executor/operator/include)

FILE(GLOB_RECURSE COMMON_LIB_CPP_SRCS executor/engine/*.cpp executor/lib/*.cpp executor/plugin/*.cpp)
FILE(GLOB COMMON_CPP_SRCS executor/operator/init.cpp)
FILE(GLOB_RECURSE REF_CPP_SRCS executor/operator/ref/*.cpp)

if(CONFIG_AUTH_DEVICE)
include_directories(hclarm/auth)
include_directories(${AUTH_HEADER})
FILE(GLOB_RECURSE HCL_AUTH_SRCS hclarm/*.cpp hclarm/*.c)
list(APPEND TOPERATOR_LIB_SRCS ${HCL_AUTH_SRCS})

if (CONFIG_AUTHENICATION)
    add_definitions(-DCONFIG_AUTHENICATION=1)
endif()

# For different settings, please change the COMPILE_FLAGS
# Please refers to hclarm/auth/auth.config 
FOREACH (file ${HCL_AUTH_SRCS})
SET_SOURCE_FILES_PROPERTIES ( ${file} PROPERTIES  COMPILE_FLAGS "-DCONFIG_INTERN_TRIAL -DCONFIG_TIME_LIMIT=7200")

ENDFOREACH()
 
endif()

list(APPEND TENGINE_LIB_SRCS ${COMMON_LIB_CPP_SRCS})
list(APPEND TOPERATOR_LIB_SRCS ${COMMON_CPP_SRCS})
list(APPEND TOPERATOR_LIB_SRCS ${REF_CPP_SRCS})

include_directories(driver/cpu)
if(CONFIG_ARCH_X86)
    FILE(GLOB_RECURSE ARCH_LIB_CPP_SRCS executor/operator/x86/*.cpp)
    include_directories(executor/operator/x86/include)
endif()

if(CONFIG_ARCH_ARM32)
    FILE(GLOB_RECURSE ARCH_LIB_CPP_SRCS executor/operator/arm32/*.cpp)
    FILE(GLOB_RECURSE TARGET_ARCH_FILES executor/operator/arm32/*.S)
    include_directories(executor/operator/arm32/include)
endif()

if(CONFIG_ARCH_ARM64)
    FILE(GLOB_RECURSE ARCH_LIB_CPP_SRCS executor/operator/arm64/*.cpp)
    FILE(GLOB_RECURSE TARGET_ARCH_FILES executor/operator/arm64/*.S)
    include_directories(executor/operator/arm64/include)
endif()

if(CONFIG_ARCH_ARM8_2)
    FILE(GLOB_RECURSE ARCH_LIB_CPP_SRCS_8_2 executor/operator/arm8_2/*.cpp)
    FILE(GLOB_RECURSE TARGET_ARCH_FILES_8_2 executor/operator/arm8_2/*.S)
    include_directories(executor/operator/arm8_2/include)
    list(APPEND ARCH_LIB_CPP_SRCS ${ARCH_LIB_CPP_SRCS_8_2})
    list(APPEND TARGET_ARCH_FILES ${TARGET_ARCH_FILES_8_2})
endif()

FOREACH(file ${ARCH_LIB_CPP_SRCS})
	set_property(SOURCE ${file} PROPERTY  COMPILE_FLAGS  "-fvisibility=hidden")
ENDFOREACH()

list(APPEND TOPERATOR_LIB_SRCS ${ARCH_LIB_CPP_SRCS})

# Now, handle the .S file
FOREACH( file ${TARGET_ARCH_FILES})
    string(REPLACE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} PREPROCESS_FILE ${file})
    get_filename_component(dest_bin_dir ${PREPROCESS_FILE} DIRECTORY)

    ADD_CUSTOM_COMMAND(
            OUTPUT ${PREPROCESS_FILE}
            COMMAND mkdir -p ${dest_bin_dir}
            COMMAND ${CMAKE_C_COMPILER} -E ${file} -o ${PREPROCESS_FILE}
            DEPENDS ${file}
    )

    #message(${file} --> ${PREPROCESS_FILE})
    list(APPEND TOPERATOR_LIB_SRCS ${PREPROCESS_FILE})
    list(APPEND ASM_FILES ${PREPROCESS_FILE})

    SET_SOURCE_FILES_PROPERTIES ( ${PREPROCESS_FILE} PROPERTIES  GENERATED  1)
    set_property(SOURCE ${PREPROCESS_FILE} PROPERTY LANGUAGE C)
ENDFOREACH()

ADD_CUSTOM_TARGET(KERNEL_ASM_TARGET DEPENDS ${ASM_FILES})
