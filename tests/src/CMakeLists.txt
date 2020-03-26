include_directories(../include)

function(create_target file_list)
  STRING(REGEX REPLACE ".+/(.+)\\..*" "\\1" fname ${file})
  add_executable(${fname} ${file})
  target_link_libraries(${fname} tengine)

  install(TARGETS ${fname} DESTINATION tests/bin)
endfunction()


FILE(GLOB file_list "*.cpp" "*.c")

foreach(file ${file_list})
	create_target(${file})
endforeach()

