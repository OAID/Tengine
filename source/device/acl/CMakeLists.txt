# 0. clear var
UNSET (_DEV_ACL_HEADER_PATH)
UNSET (_ACL_BASE_SOURCE)
UNSET (_ACL_OPS_SOURCE)
UNSET (_DEV_ACL_DEVICE_SOURCE)
UNSET (_DEV_ACL_COMPILER_DEFINES)
UNSET (_DEV_ACL_COMPILER_OPTIONS)
UNSET (_DEV_ACL_LINKER_OPTIONS)
UNSET (_DEV_ACL_LINK_LIBRARIES)


# 1.  set source root path
SET(_ACL_ROOT ${CMAKE_SOURCE_DIR}/source/device/acl)


# 2.  add header file path
LIST (APPEND _DEV_ACL_HEADER_PATH      ${_ACL_ROOT})
LIST (APPEND _DEV_ACL_HEADER_PATH      ${CMAKE_SOURCE_DIR}/3rdparty/acl/include)


# 3.  add linking lib searching path
LIST (APPEND _DEV_ACL_LINK_PATH        ${CMAKE_SOURCE_DIR}/3rdparty/acl/lib)


# 4.  add source files
AUX_SOURCE_DIRECTORY("${_ACL_ROOT}"    _ACL_BASE_SOURCE)
AUX_SOURCE_DIRECTORY("${_ACL_ROOT}/op" _ACL_OPS_SOURCE)
LIST (APPEND _DEV_ACL_DEVICE_SOURCE    ${_ACL_BASE_SOURCE})
LIST (APPEND _DEV_ACL_DEVICE_SOURCE    ${_ACL_OPS_SOURCE})


# 5.  add build options for cpu device
# 5.1 is a gcc or clang like compiler
IF (TENGINE_COMPILER_GCC OR TENGINE_COMPILER_CLANG)
    IF (TENGINE_COMPILER_GCC AND (${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER_EQUAL "6.1"))
        LIST (APPEND _DEV_ACL_COMPILER_OPTIONS -Wno-ignored-attributes)
    ENDIF()
ENDIF()


# 5.2 is Microsoft Visual C++
IF (TENGINE_COMPILER_MSVC)
ENDIF()


# 6.  add link options


# 7.  add link libs
LIST (APPEND _DEV_ACL_LINK_LIBRARIES   arm_compute)
LIST (APPEND _DEV_ACL_LINK_LIBRARIES   arm_compute_core)


# 8. set all to cmake cache
SET (TENGINE_ACL_HEADER_PATH       ${_DEV_ACL_HEADER_PATH}        CACHE INTERNAL  "Tengine Arm Compute Library device header files searching path"   FORCE)
SET (TENGINE_ACL_LINK_PATH         ${_DEV_ACL_LINK_PATH}          CACHE INTERNAL  "Tengine Arm Compute Library device link libraries searching path" FORCE)
SET (TENGINE_ACL_DEVICE_SOURCE     ${_DEV_ACL_DEVICE_SOURCE}      CACHE INTERNAL  "Tengine Arm Compute Library device main source files"             FORCE)
SET (TENGINE_ACL_COMPILER_DEFINES  ${_DEV_ACL_COMPILER_DEFINES}   CACHE INTERNAL  "Tengine Arm Compute Library about compiler defines"               FORCE)
SET (TENGINE_ACL_COMPILER_OPTIONS  ${_DEV_ACL_COMPILER_OPTIONS}   CACHE INTERNAL  "Tengine Arm Compute Library about compiler options"               FORCE)
SET (TENGINE_ACL_LINKER_OPTIONS    ${_DEV_ACL_LINKER_OPTIONS}     CACHE INTERNAL  "Tengine Arm Compute Library about linker options"                 FORCE)
SET (TENGINE_ACL_LINK_LIBRARIES    ${_DEV_ACL_LINK_LIBRARIES}     CACHE INTERNAL  "Tengine Arm Compute Library about link libraries"                 FORCE)


# 9. install device option
INSTALL (FILES ${_ACL_ROOT}/acl_define.h DESTINATION include/tengine RENAME acl_device.h)
