# generate tengine header file
FILE (MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tengine)
FILE (COPY ${CMAKE_SOURCE_DIR}/source/api/c_api.h DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tengine)
FILE (COPY ${CMAKE_SOURCE_DIR}/source/api/c_api_ex.h DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tengine)


# macro for adding original test
function (tengine_test name file)
    add_executable (${name} ${CMAKE_CURRENT_SOURCE_DIR}/${file} "${PROJECT_SOURCE_DIR}/tests/common/tengine_operations.cpp")

    target_link_libraries (${name} PRIVATE ${CMAKE_PROJECT_NAME})

    target_include_directories (${name} PRIVATE "${PROJECT_SOURCE_DIR}/source")
    target_include_directories (${name} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    target_include_directories (${name} PRIVATE "${PROJECT_BINARY_DIR}")
    target_include_directories (${name} PRIVATE "${PROJECT_BINARY_DIR}/source")
    target_include_directories (${name} PRIVATE "${PROJECT_SOURCE_DIR}/tests/common")

    if (${TENGINE_TARGET_PROCESSOR} MATCHES "ARM" AND (NOT ANDROID AND NOT OHOS) AND TENGINE_TARGET_PROCESSOR_32Bit)
        target_compile_options (${name} PRIVATE "-mfp16-format=ieee")
    endif()

    # add_test (${name} ${name})
endfunction()

# operator level test
#[[
tengine_test(test_op_conv               op/test_op_conv.c)
tengine_test(test_op_prelu              op/test_op_prelu.c)
tengine_test(test_op_relu               op/test_op_relu.c)
tengine_test(test_op_relu6              op/test_op_relu6.c)
]]

# model level test
tengine_test(tm_classify           models/test_tm_classification.cpp)
tengine_test(tm_darknet_yolo       models/test_tm_darknet_yolo.cpp)
tengine_test(tm_faster_rcnn        models/test_tm_faster_rcnn.cpp)
tengine_test(tm_lighten_cnn        models/test_tm_lighten_cnn.cpp)
tengine_test(tm_mssd               models/test_tm_mssd.cpp)
tengine_test(tm_mssd_quant         models/test_tm_mssd_quant_tflite.cpp)
tengine_test(tm_mtcnn              models/test_tm_mtcnn.cpp)
tengine_test(tm_retinaface_mx      models/test_tm_retinafacedetect_mx.cpp)
tengine_test(tm_squeezenet_ssd     models/test_tm_squeezenet_ssd.cpp)
tengine_test(tm_ssd                models/test_tm_ssd.cpp)
tengine_test(tm_yolov2             models/test_tm_yolov2.cpp)
tengine_test(tm_yufacedetect       models/test_tm_yufacedetect.cpp)
tengine_test(test_tm_landmark      models/test_tm_landmark.cpp)
tengine_test(test_tm_yolact        models/test_tm_yolact.cpp)
tengine_test(tm_mfn                models/test_tm_mfn.cpp)

if (TENGINE_ENABLE_TIM_VX)
    tengine_test(test_op_prelu_timvx        op/test_op_prelu_timvx.c)
endif()


# operator level test using onnx test
find_package(Protobuf)

if(PROTOBUF_FOUND)
    protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS onnx.proto)

    function (tengine_onnx_op_test name file)
        add_executable (${name} ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${ONNX_PROTO_SRCS} ${ONNX_PROTO_HDRS})

        target_link_libraries (${name} PRIVATE ${CMAKE_PROJECT_NAME} ${PROTOBUF_LIBRARIES})

        target_include_directories (${name} PRIVATE "${PROTOBUF_INCLUDE_DIR}")
        target_include_directories (${name} PRIVATE "${PROJECT_SOURCE_DIR}/source")
        target_include_directories (${name} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
        target_include_directories (${name} PRIVATE "${PROJECT_BINARY_DIR}")
        target_include_directories (${name} PRIVATE "${PROJECT_BINARY_DIR}/source")

        add_test (${name} ${name})
    endfunction()

    tengine_onnx_op_test(test_onnx_op_add                           op/test_onnx_op_add.cpp)
    tengine_onnx_op_test(test_onnx_op_averagepool_2d_default        op/test_onnx_op_averagepool_2d_default.cpp)
    tengine_onnx_op_test(test_onnx_op_averagepool_2d_pads           op/test_onnx_op_averagepool_2d_pads.cpp)
    tengine_onnx_op_test(test_onnx_op_basic_conv_with_padding       op/test_onnx_op_basic_conv_with_padding.cpp)
    tengine_onnx_op_test(test_onnx_op_basic_conv_without_padding    op/test_onnx_op_basic_conv_without_padding.cpp)
    # tengine_onnx_op_test(test_onnx_op_concat_1d_axis_0              op/test_onnx_op_concat_1d_axis_0.cpp)
    tengine_onnx_op_test(test_onnx_op_concat_2d_axis_0              op/test_onnx_op_concat_2d_axis_0.cpp)
    tengine_onnx_op_test(test_onnx_op_concat_2d_axis_1              op/test_onnx_op_concat_2d_axis_1.cpp)
    tengine_onnx_op_test(test_onnx_op_concat_3d_axis_0              op/test_onnx_op_concat_3d_axis_0.cpp)
    tengine_onnx_op_test(test_onnx_op_concat_3d_axis_1              op/test_onnx_op_concat_3d_axis_1.cpp)
    tengine_onnx_op_test(test_onnx_op_concat_3d_axis_2              op/test_onnx_op_concat_3d_axis_2.cpp)
    tengine_onnx_op_test(test_onnx_op_conv_with_strides_no_padding  op/test_onnx_op_conv_with_strides_no_padding.cpp)
    tengine_onnx_op_test(test_onnx_op_conv_with_strides_padding     op/test_onnx_op_conv_with_strides_padding.cpp)
    tengine_onnx_op_test(test_onnx_op_div                           op/test_onnx_op_div.cpp)
    tengine_onnx_op_test(test_onnx_op_dropout_default               op/test_onnx_op_dropout_default.cpp)
    tengine_onnx_op_test(test_onnx_op_elu                           op/test_onnx_op_elu.cpp)
    tengine_onnx_op_test(test_onnx_op_exp                           op/test_onnx_op_exp.cpp)
    tengine_onnx_op_test(test_onnx_op_floor                         op/test_onnx_op_floor.cpp)
    tengine_onnx_op_test(test_onnx_op_globalaveragepool             op/test_onnx_op_globalaveragepool.cpp)
    tengine_onnx_op_test(test_onnx_op_gru_defaults                  op/test_onnx_op_gru_defaults.cpp)
    tengine_onnx_op_test(test_onnx_op_gru_seq_length                op/test_onnx_op_gru_seq_length.cpp)
    tengine_onnx_op_test(test_onnx_op_gru_with_initial_bias         op/test_onnx_op_gru_with_initial_bias.cpp)
    tengine_onnx_op_test(test_onnx_op_leakyrelu                     op/test_onnx_op_leakyrelu.cpp)
    tengine_onnx_op_test(test_onnx_op_lstm_defaults                 op/test_onnx_op_lstm_defaults.cpp)
    tengine_onnx_op_test(test_onnx_op_lstm_with_initial_bias        op/test_onnx_op_lstm_with_initial_bias.cpp)
    tengine_onnx_op_test(test_onnx_op_maxpool_2d_default            op/test_onnx_op_maxpool_2d_default.cpp)
    # tengine_onnx_op_test(test_onnx_op_maxpool_2d_dilations          op/test_onnx_op_maxpool_2d_dilations.cpp)
    tengine_onnx_op_test(test_onnx_op_maxpool_2d_pads               op/test_onnx_op_maxpool_2d_pads.cpp)
    tengine_onnx_op_test(test_onnx_op_relu                          op/test_onnx_op_relu.cpp)
    # tengine_onnx_op_test(test_onnx_op_softmax_default_axis          op/test_onnx_op_softmax_default_axis.cpp)
    tengine_onnx_op_test(test_onnx_op_squeeze                       op/test_onnx_op_squeeze.cpp)
    tengine_onnx_op_test(test_onnx_op_tanh                          op/test_onnx_op_tanh.cpp)
    tengine_onnx_op_test(test_onnx_op_unsqueeze_axis_1              op/test_onnx_op_unsqueeze_axis_1.cpp)
else()
    message(WARNING "Protobuf not found, onnx op test won't be built")
endif()
# macro for adding test
function (tengine_onnx_op_test name file)

    target_link_libraries (${name} PRIVATE ${CMAKE_PROJECT_NAME})

    target_include_directories (${name} PRIVATE "${PROJECT_SOURCE_DIR}/source")
    target_include_directories (${name} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    target_include_directories (${name} PRIVATE "${PROJECT_BINARY_DIR}")
    target_include_directories (${name} PRIVATE "${PROJECT_BINARY_DIR}/source")
    target_include_directories (${name} PRIVATE "${PROJECT_SOURCE_DIR}/tests/common")

    if (${TENGINE_TARGET_PROCESSOR} MATCHES "ARM" AND (NOT ANDROID AND NOT OHOS) AND TENGINE_TARGET_PROCESSOR_32Bit)
        target_compile_options (${name} PRIVATE "-mfp16-format=ieee")
    endif()

    add_test (${name} ${name})
endfunction()


# operator level test using opencv
find_package(OpenCV QUIET)

if(OpenCV_FOUND)
    # macro for adding cv tests
    function (tengine_test_cv name file)
        add_executable (${name} ${CMAKE_CURRENT_SOURCE_DIR}/${file} "${PROJECT_SOURCE_DIR}/tests/common/tengine_operations.cpp")

        target_link_libraries (${name} PRIVATE ${CMAKE_PROJECT_NAME} ${OpenCV_LIBS})
    
        target_include_directories (${name} PRIVATE "${PROJECT_SOURCE_DIR}/source")
        target_include_directories (${name} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
        target_include_directories (${name} PRIVATE "${PROJECT_BINARY_DIR}")
        target_include_directories (${name} PRIVATE "${PROJECT_BINARY_DIR}/source")
        target_include_directories (${name} PRIVATE "${PROJECT_SOURCE_DIR}/tests/common")
    
        if (${TENGINE_TARGET_PROCESSOR} MATCHES "ARM" AND (NOT ANDROID AND NOT OHOS) AND TENGINE_TARGET_PROCESSOR_32Bit)
            target_compile_options (${name} PRIVATE "-mfp16-format=ieee")
        endif()
    endfunction()

    #tengine_test_cv(test_tm_yolact_cv      models/test_tm_yolact_cv.cpp)
else()
    message(WARNING "OpenCV not found, some model examples won't be built")
endif()

