
cmake_minimum_required (VERSION 2.8)
project(label_image)

set( INSTALL_DIR ${TENGINE_DIR}/install/ )
set( TENGINE_LIBS wrapper tengine boost_system boost_thread )
set( CODE_SRC label_image.cpp ../../common/common.cpp )

#opencv
find_package(OpenCV REQUIRED)

#include
include_directories(${INSTALL_DIR}/wrapper_include
                    ${INSTALL_DIR}/include
                    ${TENGINE_DIR}/examples/common)


set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wall")

#lib
link_directories(${INSTALL_DIR}/lib)

#exe
add_executable(label_image_inceptionv3 ${CODE_SRC})
target_link_libraries(label_image_inceptionv3 ${TENGINE_LIBS} ${OpenCV_LIBS})

add_executable(label_image_mobilenet ${CODE_SRC})
target_compile_definitions(label_image_mobilenet PUBLIC -DMOBILE_NET)
target_link_libraries(label_image_mobilenet ${TENGINE_LIBS} ${OpenCV_LIBS})

add_executable(label_image_resnet50 ${CODE_SRC})
target_compile_definitions(label_image_resnet50 PUBLIC -DRESNET50)
target_link_libraries(label_image_resnet50 ${TENGINE_LIBS} ${OpenCV_LIBS})

