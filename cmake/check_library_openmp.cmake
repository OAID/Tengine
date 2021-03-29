# find & use OpenMP
#   CMake FindOpenMP module: see https://cmake.org/cmake/help/latest/module/FindOpenMP.html
#     version    OPENMP_FOUND    OpenMP_<lang>_FOUND    OpenMP_<lang>_FLAGS    OpenMP::OpenMP_<lang>    OpenMP_<lang>_VERSION    OpenMP_<lang>_SPEC_DATE    OpenMP_<lang>_INCLUDE_DIRS    OpenMP_<lang>_LIBRARIES
#       3.0           V                  NA                    C/C++                    NA                       NA                        NA                          NA                           NA
#       3.1           V                  NA                C/C++/Fortran                NA                       NA                        NA                          NA                           NA
#       3.7           V                  NA                C/C++/Fortran                NA                       NA                   C/C++/Fortran                    NA                           NA
#       3.9      OpenMP_FOUND       C/C++/Fortran          C/C++/Fortran           C/C++/Fortran            C/C++/Fortran             C/C++/Fortran                    NA                      C/C++/Fortran
#       3.16     OpenMP_FOUND       C/C++/Fortran          C/C++/Fortran           C/C++/Fortran            C/C++/Fortran             C/C++/Fortran               C/C++/Fortran                C/C++/Fortran
#       3.19
#
#   OpenMP Compilers: see https://www.openmp.org//resources/openmp-compilers-tools
#     

MACRO(TENGINE_CHECK_LIB_OPENMP _has_flag)
    # use find_package to find openmp
    FIND_PACKAGE(OpenMP)

    # OPENMP_FOUND: <  3.9
    # OpenMP_FOUND: >= 3.9
    IF(OPENMP_FOUND OR OpenMP_FOUND OR OpenMP_C_FOUND OR OpenMP_CXX_FOUND)
        SET(${_has_flag} ON)
    ENDIF()
ENDMACRO()


FUNCTION(TENGINE_COMPILE_LIB_OPENMP _target)
    IF (NOT(OpenMP_C_FLAGS OR OpenMP_CXX_FLAGS))
        TENGINE_CHECK_LIB_OPENMP(_has_openmp)
    ELSE()
        TARGET_COMPILE_OPTIONS(${_target} PRIVATE $<$<COMPILE_LANGUAGE:C>:${OpenMP_C_FLAGS}> $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>)
    ENDIF()

    UNSET(_has_openmp)
ENDFUNCTION()


FUNCTION(TENGINE_LINK_LIB_OPENMP _target)
    IF (NOT(OpenMP_C_FOUND OR OpenMP_C_FOUND) AND NOT(OpenMP_C_FLAGS OR OpenMP_CXX_FLAGS))
        TENGINE_CHECK_LIB_OPENMP(_has_openmp)
    ELSE()
        IF(ANDROID_NDK_MAJOR AND (ANDROID_NDK_MAJOR GREATER 20))
            #        list (APPEND TENGINE_LINKING_LIBS "-fopenmp -static-openmp")
            TARGET_LINK_LIBRARIES(${_target} PRIVATE "-fopenmp -static-openmp")
            #        TARGET_LINK_LIBRARIES(${_target} PRIVATE "${OpenMP_C_FLAGS} -static-openmp")
        ELSEIF((TARGET OpenMP::OpenMP_C) OR (TARGET OpenMP::OpenMP_CXX))
            # LINK_LANGUAGE keyword is added to generator expressions in 3.18
            IF((${CMAKE_MAJOR_VERSION} VERSION_EQUAL 3) AND (${CMAKE_MINOR_VERSION} VERSION_EQUAL 18 OR ${CMAKE_MINOR_VERSION} VERSION_GREATER 18))
                TARGET_LINK_LIBRARIES(${_target} PRIVATE $<$<LINK_LANGUAGE:C>:OpenMP::OpenMP_C>)
                TARGET_LINK_LIBRARIES(${_target} PRIVATE $<$<LINK_LANGUAGE:CXX>:OpenMP::OpenMP_CXX>)
            ELSE()
                TARGET_LINK_LIBRARIES(${_target} PRIVATE ${OpenMP_C_FLAGS})
            ENDIF()
        ELSE()
            TARGET_LINK_LIBRARIES(${_target} PRIVATE ${OpenMP_C_FLAGS})
        ENDIF()
    ENDIF()

    UNSET(_has_openmp)
ENDFUNCTION()


FUNCTION(TENGINE_USE_LIB_OPENMP _target)
    TENGINE_COMPILE_LIB_OPENMP(${_target})
    TENGINE_LINK_LIB_OPENMP(${_target})
ENDFUNCTION()
