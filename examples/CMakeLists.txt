# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# License); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# Copyright (c) 2021, OPEN AI LAB
# Author: lswang@openailab.com
#

# generate tengine header file
FILE (MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tengine)
FILE (COPY ${CMAKE_SOURCE_DIR}/source/api/c_api.h DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tengine)
#FILE (COPY ${CMAKE_SOURCE_DIR}/source/api/c_api.h DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
#FILE (RENAME ${CMAKE_CURRENT_BINARY_DIR}/c_api.h ${CMAKE_CURRENT_BINARY_DIR}/tengine_c_api.h)

# macro for adding test
FUNCTION (TENGINE_EXAMPLE name file)
    ADD_EXECUTABLE (${name} "${CMAKE_CURRENT_SOURCE_DIR}/${file}" "${CMAKE_CURRENT_SOURCE_DIR}/common/tengine_operations.c")

    TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${PROJECT_SOURCE_DIR}/examples/common")

    TARGET_LINK_LIBRARIES (${name} PRIVATE ${CMAKE_PROJECT_NAME} m)
    INSTALL (TARGETS ${name} DESTINATION bin)
ENDFUNCTION()

# add c++ api examples
IF (TENGINE_BUILD_CPP_API)
    TENGINE_EXAMPLE (cpp_tm_classification       cpp_tm_classification.cpp)
    TENGINE_EXAMPLE (cpp_tm_mobilenet_ssd        cpp_tm_mobilenet_ssd.cpp)
ENDIF()

# add c api examples
TENGINE_EXAMPLE (tm_classification           tm_classification.c)
TENGINE_EXAMPLE (tm_classification_int8      tm_classification_int8.c)
TENGINE_EXAMPLE (tm_classification_uint8     tm_classification_uint8.c)
TENGINE_EXAMPLE (tm_classification_vulkan    tm_classification_vulkan.c)
TENGINE_EXAMPLE (tm_classification_acl       tm_classification_acl.c)
TENGINE_EXAMPLE (tm_classification_timvx     tm_classification_timvx.c)
TENGINE_EXAMPLE (tm_classification_trt       tm_classification_trt.cpp)
TENGINE_EXAMPLE (tm_classification_cuda      tm_classification_cuda.cpp)
TENGINE_EXAMPLE (tm_mobilenet_ssd            tm_mobilenet_ssd.c)
TENGINE_EXAMPLE (tm_mobilenet_ssd_uint8      tm_mobilenet_ssd_uint8.cpp)
TENGINE_EXAMPLE (tm_mobilenet_ssd_acl        tm_mobilenet_ssd_acl.c)
TENGINE_EXAMPLE (tm_retinaface               tm_retinaface.cpp)
TENGINE_EXAMPLE (tm_yolov3                   tm_yolov3.cpp)
TENGINE_EXAMPLE (tm_yolov3_tiny              tm_yolov3_tiny.cpp)
TENGINE_EXAMPLE (tm_yolov3_uint8             tm_yolov3_uint8.cpp)
TENGINE_EXAMPLE (tm_landmark                 tm_landmark.cpp)
TENGINE_EXAMPLE (tm_landmark_uint8           tm_landmark_uint8.cpp)
TENGINE_EXAMPLE (tm_landmark_timvx           tm_landmark_timvx.cpp)
TENGINE_EXAMPLE (tm_mobilefacenet            tm_mobilefacenet.cpp)
TENGINE_EXAMPLE (tm_mobilefacenet_uint8      tm_mobilefacenet_uint8.cpp)
TENGINE_EXAMPLE (tm_yolov4                   tm_yolov4.cpp)
TENGINE_EXAMPLE (tm_yolov5                   tm_yolov5.cpp)
TENGINE_EXAMPLE (tm_yolofastest              tm_yolofastest.cpp)
TENGINE_EXAMPLE (tm_ultraface                tm_ultraface.cpp)

IF (${TENGINE_TARGET_PROCESSOR} MATCHES "ARM" AND TENGINE_TARGET_PROCESSOR_64Bit)
    TENGINE_EXAMPLE (tm_classification_fp16      tm_classification_fp16.c)
ENDIF()


# add examples with opencv
# set(OpenCV_DIR /mnt/d/ubuntu/opencv_install/linux-armv7/lib/cmake/opencv4)
FIND_PACKAGE(OpenCV QUIET)

IF (OpenCV_FOUND AND ${TENGINE_TARGET_PROCESSOR} MATCHES "X86")
    # macro for adding examples
    FUNCTION (TENGINE_EXAMPLE_CV name file)
        ADD_EXECUTABLE (${name} "${CMAKE_CURRENT_SOURCE_DIR}/${file}" "${CMAKE_CURRENT_SOURCE_DIR}/common/tengine_operations.c")

        TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
        TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
        TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${PROJECT_SOURCE_DIR}/examples/common")

        TARGET_LINK_LIBRARIES (${name} ${CMAKE_PROJECT_NAME} ${OpenCV_LIBS})
        INSTALL (TARGETS ${name} DESTINATION bin)
    ENDFUNCTION()

    TENGINE_EXAMPLE_CV (tm_openpose              tm_openpose.cpp)
    TENGINE_EXAMPLE_CV (tm_yolact                tm_yolact.cpp)
    TENGINE_EXAMPLE_CV (tm_crnn                  tm_crnn.cpp)
    TENGINE_EXAMPLE_CV (tm_alphapose             tm_alphapose.cpp)
    TENGINE_EXAMPLE_CV (tm_yolov4_tiny           tm_yolov4_tiny.cpp)
    TENGINE_EXAMPLE_CV (tm_yolov4_tiny_timvx     tm_yolov4_tiny_timvx.cpp)
    TENGINE_EXAMPLE_CV (tm_yolov5s               tm_yolov5s.cpp)
    TENGINE_EXAMPLE_CV (tm_yolov5s_timvx         tm_yolov5s_timvx.cpp)
ELSE()
    MESSAGE (WARNING "OpenCV not found, some examples won't be built")
ENDIF()

#if (TENGINE_ENABLE_NNIE)
#    function (tengine_nnie_example name file)
#        add_executable(${name}
#                    "${CMAKE_CURRENT_SOURCE_DIR}/${file}"
#                    "${CMAKE_CURRENT_SOURCE_DIR}/common/tengine_operations.c")
#
#        target_link_libraries(${name} PRIVATE ${CMAKE_PROJECT_NAME})
#
#        include_directories("${PROJECT_SOURCE_DIR}/src/dev/nnie/include")
#        include_directories("${PROJECT_SOURCE_DIR}/src/dev/nnie/common")
#        include_directories("${PROJECT_SOURCE_DIR}/3rdparty/nnie/include")
#
#        install (TARGETS ${name} DESTINATION bin)
#    endfunction()
#
#    # add c api examples
#    tengine_nnie_example(tm_yolov3_nnie             tm_yolov3_nnie.cpp)
#endif()
