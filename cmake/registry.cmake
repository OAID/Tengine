# generate needed registry
FUNCTION (GENERATE_REGISTER_HEADER_FILE _REG_LEAD_STRING _DEL_LEAD_STRING _BACK_STRING _CONFIG_FILE _TARGET_FILE)
    SET (_BGN_NOTICE_STR "// code generation start\n")
    SET (_END_NOTICE_STR "// code generation finish\n")

    SET (_GEN_REG_DEF_STR "${_BGN_NOTICE_STR}")
    SET (_GEN_REG_CAL_STR "${_BGN_NOTICE_STR}")
    SET (_GEN_DEL_DEF_STR "${_BGN_NOTICE_STR}")
    SET (_GEN_DEL_CAL_STR "${_BGN_NOTICE_STR}")

    FOREACH (_VAR ${ARGN})
        STRING(REGEX REPLACE ".+/(.+)\\..*" "\\1" _NAME ${_VAR})

        SET (_REG_FUNC "${_REG_LEAD_STRING}${_NAME}${_BACK_STRING}()")
        SET (_DEL_FUNC "${_DEL_LEAD_STRING}${_NAME}${_BACK_STRING}()")

        SET (_GEN_REG_DEF_STR "${_GEN_REG_DEF_STR}extern int ${_REG_FUNC};\n")

        SET (_GEN_REG_CAL_STR "${_GEN_REG_CAL_STR}    ret = ${_REG_FUNC};\n")
        SET (_GEN_REG_CAL_STR "${_GEN_REG_CAL_STR}    if(0 != ret)\n")
        SET (_GEN_REG_CAL_STR "${_GEN_REG_CAL_STR}    {\n")
        SET (_GEN_REG_CAL_STR "${_GEN_REG_CAL_STR}        TLOG_ERR(\"Tengine FATAL: Call %s failed(%d).\\n\", \"${_REG_FUNC}\", ret);\n")
        SET (_GEN_REG_CAL_STR "${_GEN_REG_CAL_STR}    }\n")

        SET (_GEN_DEL_DEF_STR "${_GEN_DEL_DEF_STR}extern int ${_DEL_FUNC};\n")

        SET (_GEN_DEL_CAL_STR "${_GEN_DEL_CAL_STR}    ret = ${_DEL_FUNC};\n")
        SET (_GEN_DEL_CAL_STR "${_GEN_DEL_CAL_STR}    if(0 != ret)\n")
        SET (_GEN_DEL_CAL_STR "${_GEN_DEL_CAL_STR}    {\n")
        SET (_GEN_DEL_CAL_STR "${_GEN_DEL_CAL_STR}        TLOG_ERR(\"Tengine FATAL: Call %s failed(%d).\\n\", \"${_REG_FUNC}\", ret);\n")
        SET (_GEN_DEL_CAL_STR "${_GEN_DEL_CAL_STR}    }\n")
    ENDFOREACH()

    SET (_GEN_REG_DEF_STR "${_GEN_REG_DEF_STR}${_END_NOTICE_STR}")
    SET (_GEN_REG_CAL_STR "${_GEN_REG_CAL_STR}    ${_END_NOTICE_STR}")
    SET (_GEN_DEL_DEF_STR "${_GEN_DEL_DEF_STR}${_END_NOTICE_STR}")
    SET (_GEN_DEL_CAL_STR "${_GEN_DEL_CAL_STR}    ${_END_NOTICE_STR}")

    CONFIGURE_FILE(${_CONFIG_FILE} ${_TARGET_FILE})
ENDFUNCTION()
