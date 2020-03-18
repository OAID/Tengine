set( CODE_SRCS classification.cpp common.cpp)
set( BIN_EXE classification )

add_executable(${BIN_EXE} ${CODE_SRCS})
target_link_libraries(${BIN_EXE} tengine)

install(TARGETS ${BIN_EXE} DESTINATION examples)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/synset_words.txt DESTINATION examples)
