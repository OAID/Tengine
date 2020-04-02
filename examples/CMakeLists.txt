set( CLASSIFICATION classification )
set( MOBILENET_SSD mobilenet_ssd )

add_executable(${CLASSIFICATION} classification.cpp common.cpp)
target_link_libraries(${CLASSIFICATION} tengine)

add_executable(${MOBILENET_SSD} mobilenet_ssd.cpp common.cpp)
target_link_libraries(${MOBILENET_SSD} tengine)

install(TARGETS ${CLASSIFICATION} DESTINATION examples)
install(TARGETS ${MOBILENET_SSD} DESTINATION examples)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/synset_words.txt DESTINATION examples)
