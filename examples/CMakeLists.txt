set( CLASSIFICATION classification )
set( CLASSIFICATION_OLD classification_old_api )
set( MOBILENET_SSD mobilenet_ssd )
set( MOBILENET_SSD_OLD mobilenet_ssd_old_api )

add_executable(${CLASSIFICATION_OLD} classification_old_api.cpp common.cpp)
target_link_libraries(${CLASSIFICATION_OLD} tengine)

add_executable(${CLASSIFICATION} classification.cpp common.cpp)
target_link_libraries(${CLASSIFICATION} tengine)

add_executable(${MOBILENET_SSD_OLD} mobilenet_ssd_old_api.cpp common.cpp)
target_link_libraries(${MOBILENET_SSD_OLD} tengine)

add_executable(${MOBILENET_SSD} mobilenet_ssd.cpp common.cpp)
target_link_libraries(${MOBILENET_SSD} tengine)

install(TARGETS ${CLASSIFICATION} DESTINATION examples)
install(TARGETS ${MOBILENET_SSD} DESTINATION examples)
install(TARGETS ${CLASSIFICATION_OLD} DESTINATION examples)
install(TARGETS ${MOBILENET_SSD_OLD} DESTINATION examples)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/synset_words.txt DESTINATION examples)
