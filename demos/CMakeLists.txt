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
    TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${OpenCV_INCLUDE_DIRS}")

    TARGET_LINK_LIBRARIES (${name} PRIVATE ${CMAKE_PROJECT_NAME} ${OpenCV_LIBS})

    IF(MSVC)
        TARGET_COMPILE_DEFINITIONS (${name} PRIVATE $<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:NOMINMAX>)
    ELSE()
        TARGET_LINK_LIBRARIES (${name} PRIVATE m)
    ENDIF()

    INSTALL (TARGETS ${name} DESTINATION demo)
ENDFUNCTION()


if(OpenCV_FOUND)
    TENGINE_DEMO (demo_yolo_camera     demo_yolo_camera.cpp yolo.cpp yolo_layer.cpp timer.cpp)
else()
    message(WARNING "OpenCV not found, some DEMO won't be built")
endif()
