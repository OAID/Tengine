find_package(CUDA REQUIRED)
# find the library
if(CUDA_FOUND)
    message(CUDA FOUND)
    find_cuda_helper_libs(cudnn)
    set(CUDNN_LIBRARY ${CUDA_cudnn_LIBRARY} CACHE FILEPATH "location of the cuDNN library")
    message("CUDNN_LIBRARY: ${CUDNN_LIBRARY}")
    unset(CUDA_cudnn_LIBRARY CACHE)
endif()

# find the include
if(CUDNN_LIBRARY)
    find_path(CUDNN_INCLUDE_DIR
            cudnn.h
            PATHS ${CUDA_TOOLKIT_INCLUDE}
            DOC "location of cudnn.h"
            NO_DEFAULT_PATH
            )

    if(NOT CUDNN_INCLUDE_DIR)
        find_path(CUDNN_INCLUDE_DIR
                cudnn.h
                DOC "location of cudnn.h"
                )
    endif()
endif()

# extract version from the include
if(CUDNN_INCLUDE_DIR)
    file(READ "${CUDNN_INCLUDE_DIR}/cudnn.h" CUDNN_H_CONTENTS)

    string(REGEX MATCH "define CUDNN_MAJOR ([0-9]+)" _ "${CUDNN_H_CONTENTS}")
    set(CUDNN_MAJOR_VERSION ${CMAKE_MATCH_1} CACHE INTERNAL "")
    string(REGEX MATCH "define CUDNN_MINOR ([0-9]+)" _ "${CUDNN_H_CONTENTS}")
    set(CUDNN_MINOR_VERSION ${CMAKE_MATCH_1} CACHE INTERNAL "")
    string(REGEX MATCH "define CUDNN_PATCHLEVEL ([0-9]+)" _ "${CUDNN_H_CONTENTS}")
    set(CUDNN_PATCH_VERSION ${CMAKE_MATCH_1} CACHE INTERNAL "")

    set(CUDNN_VERSION
            "${CUDNN_MAJOR_VERSION}.${CUDNN_MINOR_VERSION}.${CUDNN_PATCH_VERSION}"
            CACHE
            STRING
            "cuDNN version"
            )

    unset(CUDNN_H_CONTENTS)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDNN
        FOUND_VAR CUDNN_FOUND
        REQUIRED_VARS
        CUDNN_LIBRARY
        CUDNN_INCLUDE_DIR
        VERSION_VAR CUDNN_VERSION
        )

if(CUDNN_FOUND)
    set(CUDNN_LIBRARIES ${CUDNN_LIBRARY})
    set(CUDNN_INCLUDE_DIRS ${CUDNN_INCLUDE_DIR})
endif()

mark_as_advanced(
        CUDNN_LIBRARY
        CUDNN_INCLUDE_DIR
        CUDNN_VERSION
)
