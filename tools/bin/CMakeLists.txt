
cmake_minimum_required (VERSION 2.8)
project(YOLOV2)
set(TENGINE_DIR "/home/firefly/my_tengine/tengine/")

set( INSTALL_DIR ${TENGINE_DIR}/install/ )
set( TENGINE_LIBS tengine)

set( CODE_SRCS test_yolov2.cpp)
set( BIN_EXE TEST_YOLO)

#set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH} /usr/local/AID/opencv3.3.0/share/OpenCV/")
#opencv
find_package(OpenCV REQUIRED)



#flag
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -o3  -Wall")

#include
include_directories(${INSTALL_DIR}/include
		   ${TENGINE_DIR}/include
		   ${TENGINE_DIR}/core/include
		   ${TENGINE_DIR}/operator/include)

#lib
link_directories( ${INSTALL_DIR}/lib )

#exe
add_executable(${BIN_EXE} ${CODE_SRCS})
if( ANDROID)
   target_link_libraries(${BIN_EXE} ${TENGINE_LIBS} ${PROTOBUF_LIB} ${OpenCV_LIBS} ${BLAS_LIB})
else()
   target_link_libraries(${BIN_EXE} ${TENGINE_LIBS} ${OpenCV_LIBS})
endif()


