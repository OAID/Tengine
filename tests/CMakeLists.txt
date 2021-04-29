# generate tengine header file
FILE (MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tengine)
FILE (COPY ${CMAKE_SOURCE_DIR}/source/api/c_api.h DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tengine)
FILE (COPY ${CMAKE_SOURCE_DIR}/source/api/c_api_ex.h DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tengine)


# macro for adding test
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

    add_test (${name} ${name})
endfunction()

# operator level test
tengine_test(test_op_conv               op/test_op_conv.c)
tengine_test(test_op_prelu              op/test_op_prelu.c)
tengine_test(test_op_relu               op/test_op_relu.c)
tengine_test(test_op_relu6              op/test_op_relu6.c)

if (TENGINE_ENABLE_TIM_VX)
    tengine_test(test_op_prelu_timvx        op/test_op_prelu_timvx.c)
endif()

#[[ add op
tengine_test(test_op_conv               op/test_op_conv.c)
tengine_test(test_op_bias               op/test_op_bias.cpp)
tengine_test(test_op_batchnorm          op/test_op_batchnorm.c)
tengine_test(test_op_ceil               op/test_op_ceil.c)
tengine_test(test_op_clip               op/test_op_clip.c)
tengine_test(test_op_concat             op/test_op_concat.cpp)
tengine_test(test_op_crop               op/test_op_crop.c)
tengine_test(test_op_deconv             op/test_op_deconv.c)
tengine_test(test_op_depthtospace       op/test_op_depth_to_space.c)
tengine_test(test_op_dropout            op/test_op_dropout.c)
tengine_test(test_op_eltwise            op/test_op_eltwise.c)
tengine_test(test_op_expanddims         op/test_op_expanddims.c)
tengine_test(test_op_flatten            op/test_op_flatten.c)
tengine_test(test_op_gather             op/test_op_gather.c)
tengine_test(test_op_lrn                op/test_op_lrn.c)
tengine_test(test_op_permute            op/test_op_permute.cpp)
tengine_test(test_op_pooling            op/test_op_pooling.c)
tengine_test(test_op_relu               op/test_op_relu.c)
tengine_test(test_op_relu6              op/test_op_relu6.c)
tengine_test(test_op_prelu              op/test_op_prelu.c)
tengine_test(test_op_reshape            op/test_op_reshape.c)
tengine_test(test_op_resize             op/test_op_resize.c)
tengine_test(test_op_reverse            op/test_op_reverse.c)
tengine_test(test_op_round              op/test_op_round.c)
tengine_test(test_op_scale              op/test_op_scale.c)
tengine_test(test_op_spacetodepth       op/test_op_spacetodepth.c)
tengine_test(test_op_softmax            op/test_op_softmax.c)
tengine_test(test_op_squareddifference  op/test_op_squareddifference.c)
tengine_test(test_op_squeeze            op/test_op_squeeze.c)
tengine_test(test_op_unary              op/test_op_unary.cpp)
tengine_test(test_op_zeroslike          op/test_op_zeroslike.c)
tengine_test(test_op_tanh               op/test_op_tanh.c)
tengine_test(test_op_sigmoid            op/test_op_sigmoid.c)
tengine_test(test_op_lstm               op/test_op_lstm.cpp)
tengine_test(test_op_gru                op/test_op_gru.cpp)
tengine_test(test_op_reduction          op/test_op_reduction.cpp)
tengine_test(test_op_shufflechannel     op/test_op_shufflechannel.cpp)
tengine_test(test_op_threshold          op/test_op_threshold.cpp)
tengine_test(test_op_hardsigmoid        op/test_op_hardsigmoid.cpp)
tengine_test(test_op_embedding          op/test_op_embedding.cpp)
tengine_test(test_op_instancenorm       op/test_op_instancenorm.cpp)
tengine_test(test_op_mvn                op/test_op_mvn.cpp)
tengine_test(test_op_absval             op/test_op_absval.cpp)
tengine_test(test_op_cast               op/test_op_cast.cpp)
tengine_test(test_op_hardswish          op/test_op_hardswish.cpp)
tengine_test(test_op_selu               op/test_op_selu.cpp)
tengine_test(test_op_elu                op/test_op_elu.cpp)
tengine_test(test_op_broadmul           op/test_op_broadmul.cpp)
tengine_test(test_op_interp             op/test_op_interp.cpp)
tengine_test(test_op_argmax             op/test_op_argmax.cpp)
tengine_test(test_op_argmin             op/test_op_argmin.cpp)
tengine_test(test_op_maximum            op/test_op_maximum.cpp)
tengine_test(test_op_minimum            op/test_op_minimum.cpp)
tengine_test(test_op_mean               op/test_op_mean.cpp)
tengine_test(test_op_add_n              op/test_op_add_n.cpp)
]]

#[[ new add
tengine_test(test_op_gather_tflite              op/test_op_gather_tflite.cpp)
tengine_test(test_op_transpose_tflite             op/test_op_transpose_tflite.cpp)
tengine_test(test_op_psroipooling             op/test_op_psroipooling.cpp)
tengine_test(test_op_roialign           op/test_op_roialign.cpp)
tengine_test(test_op_leakyRelu_onnx           op/test_op_leakyRelu_onnx.cpp)
tengine_test(test_op_clip_onnx           op/test_op_clip_onnx.cpp)
tengine_test(test_op_div_onnx           op/test_op_div_onnx.cpp)
tengine_test(test_op_elu_onnx           op/test_op_elu_onnx.cpp)
tengine_test(test_op_mul_onnx           op/test_op_mul_onnx.cpp)
tengine_test(test_op_transpose_onnx           op/test_op_transpose_onnx.cpp)
]]

# unit test
#[[
tengine_test(test_buddy_mem     utility/test_buddy_mem.c)
tengine_test(test_graph         utility/test_graph.c)
tengine_test(test_log           utility/test_log.cpp)
tengine_test(test_model_load    utility/test_model_load.c)
tengine_test(test_pack          utility/test_pack.cpp)
tengine_test(test_pack_graph    utility/test_pack_graph.c)
tengine_test(test_param         utility/test_param.c)
tengine_test(test_plugin        utility/test_plugin.c)
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

#
#tengine_test(test_tiny.c tiny_graph_generated.c utility/tiny)

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
    message(WARNING "OpenCV not found, some examples won't be built")
endif()

