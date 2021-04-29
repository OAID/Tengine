# add source file to list, and add to special visual folder
FUNCTION (TENGINE_SOURCE_GROUP _FOLDER_NAME)
	IF (MSVC)
		STRING (REGEX REPLACE "/" "\\\\" _TARGET_FOLDER "${_FOLDER_NAME}")
    ENDIF()

    IF (ARGN)
        FOREACH (_FILE ${ARGN})
            SOURCE_GROUP ("${_TARGET_FOLDER}" FILES "${_FILE}")
        ENDFOREACH()
    ENDIF()
ENDFUNCTION()
