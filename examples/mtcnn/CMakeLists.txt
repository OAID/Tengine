set( MTCNN mtcnn )

add_executable(${MTCNN} mtcnn.cpp mtcnn_utils.cpp test_mtcnn.cpp)
target_link_libraries(${MTCNN} tengine)

install(TARGETS ${MTCNN} DESTINATION examples)