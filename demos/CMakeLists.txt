# generate tengine header file
FILE (MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tengine)
FILE (COPY ${CMAKE_SOURCE_DIR}/source/api/c_api.h DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tengine)

find_package(OpenCV)

FUNCTION (TENGINE_DEMO name)
    ADD_EXECUTABLE (${name})

    FOREACH(file IN LISTS ARGN)
        TARGET_SOURCES(${name} PRIVATE ${file})
    ENDFOREACH()

    TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/algorithm")
    TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/utilities")
    TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${OpenCV_INCLUDE_DIRS}")

    TARGET_LINK_LIBRARIES (${name} PRIVATE ${CMAKE_PROJECT_NAME} ${OpenCV_LIBS})

    IF(MSVC)
        TARGET_COMPILE_DEFINITIONS (${name} PRIVATE $<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:NOMINMAX>)

        # add to a virtual project group
        SET_PROPERTY(TARGET ${name} PROPERTY FOLDER "demos")
    ELSE()
        TARGET_LINK_LIBRARIES (${name} PRIVATE m)
    ENDIF()

    INSTALL (TARGETS ${name} DESTINATION demo)
ENDFUNCTION()


if(OpenCV_FOUND)
    TENGINE_DEMO (demo_yolo_camera      demo_yolo_camera.cpp     algorithm/yolo.cpp algorithm/yolo_layer.cpp utilities/timer.cc)
    TENGINE_DEMO (demo_scrfd_camera     demo_scrfd_camera.cc     algorithm/scrfd.cc utilities/letterbox.cc utilities/permute.cc utilities/nms.cc utilities/iou.cc utilities/timer.cc)
    TENGINE_DEMO (demo_scrfd_image      demo_scrfd_image.cc      algorithm/scrfd.cc utilities/letterbox.cc utilities/permute.cc utilities/nms.cc utilities/iou.cc utilities/timer.cc)
    TENGINE_DEMO (demo_arcface_image    demo_arcface_image.cc    algorithm/scrfd.cc algorithm/arcface.cc utilities/permute.cc utilities/timer.cc utilities/affine.cc utilities/letterbox.cc utilities/nms.cc utilities/iou.cc utilities/distance.cc)
    TENGINE_DEMO (demo_scrfd_list       demo_scrfd_list.cc       algorithm/scrfd.cc utilities/letterbox.cc utilities/iou.cc utilities/nms.cc utilities/permute.cc utilities/timer.cc)
    TENGINE_DEMO (demo_arcface_list     demo_arcface_list.cc     algorithm/arcface.cc utilities/affine.cc utilities/distance.cc utilities/permute.cc utilities/timer.cc)
    TENGINE_DEMO (demo_centerface_image demo_centerface_image.cc algorithm/centerface.cc utilities/letterbox.cc utilities/permute.cc utilities/nms.cc utilities/iou.cc utilities/timer.cc)
    TENGINE_DEMO (demo_centerface_list  demo_centerface_list.cc  algorithm/centerface.cc utilities/letterbox.cc utilities/permute.cc utilities/nms.cc utilities/iou.cc utilities/timer.cc)
else()
    message(WARNING "OpenCV not found, some DEMO won't be built")
endif()
