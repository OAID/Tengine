/*
 * The content of this file is copied from Tensorflow project:
 *     https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.h
 */
#ifndef __TENSORFLOW_C_C_API_H__
#define __TENSORFLOW_C_C_API_H__

#include "tensorflow/c/c_api_internal.h"

#define TF_CAPI_EXPORT

#ifdef __cplusplus
extern "C" {
#endif

// -----------------------------------------------------------------------------
/*!
 * @brief TF_DataType holds the type for a scalar value.
 */
typedef enum TF_DataType
{
    TF_FLOAT = 1,
    TF_DOUBLE = 2,
    TF_INT32 = 3,    // Int32 tensors are always in 'host' memory.
    TF_UINT8 = 4,
    TF_INT16 = 5,
    TF_INT8 = 6,
    TF_STRING = 7,
    TF_COMPLEX64 = 8,    // Single-precision complex
    TF_COMPLEX = 8,    // Old identifier kept for API backwards compatibility
    TF_INT64 = 9,
    TF_BOOL = 10,
    TF_QINT8 = 11,    // Quantized int8
    TF_QUINT8 = 12,    // Quantized uint8
    TF_QINT32 = 13,    // Quantized int32
    TF_BFLOAT16 = 14,    // Float32 truncated to 16 bits.  Only for cast ops.
    TF_QINT16 = 15,    // Quantized int16
    TF_QUINT16 = 16,    // Quantized uint16
    TF_UINT16 = 17,
    TF_COMPLEX128 = 18,    // Double-precision complex
    TF_HALF = 19,
    TF_RESOURCE = 20,
} TF_DataType;

// -----------------------------------------------------------------------------
/*!
 * @brief TF_Code holds an error code.
 */
typedef enum TF_Code
{
    TF_OK = 0,
    TF_CANCELLED = 1,
    TF_UNKNOWN = 2,
    TF_INVALID_ARGUMENT = 3,
    TF_DEADLINE_EXCEEDED = 4,
    TF_NOT_FOUND = 5,
    TF_ALREADY_EXISTS = 6,
    TF_PERMISSION_DENIED = 7,
    TF_UNAUTHENTICATED = 16,
    TF_RESOURCE_EXHAUSTED = 8,
    TF_FAILED_PRECONDITION = 9,
    TF_ABORTED = 10,
    TF_OUT_OF_RANGE = 11,
    TF_UNIMPLEMENTED = 12,
    TF_INTERNAL = 13,
    TF_UNAVAILABLE = 14,
    TF_DATA_LOSS = 15,
} TF_Code;

/*!
 * @brief TF_Status holds error information.  It either has an OK code, or
 *        else an error code with an associated error message.
 */
typedef struct TF_Status TF_Status;

/*!
 * @brief Return a new status object.
 * @return Pointer to a new status object.
 */
TF_CAPI_EXPORT extern TF_Status* TF_NewStatus();

/*!
 * @brief Delete a previously created status object.
 * @param Pointer to the status object to be deleted.
 */
TF_CAPI_EXPORT extern void TF_DeleteStatus(TF_Status*);

/*!
 * @brief Return the code record in *s.
 * @param Pointer to the status object.
 * @return The code record.
 */
TF_CAPI_EXPORT extern TF_Code TF_GetCode(const TF_Status* s);

/*!
* @brief Return a pointer to the (null-terminated) error message in *s.
* @param Pointer to the status object.
* @return Pointer to memory that is only usable until the next mutation to *s.
          Always returns an empty string if TF_GetCode(s) is TF_OK.
*/
TF_CAPI_EXPORT extern const char* TF_Message(const TF_Status* s);

// -----------------------------------------------------------------------------
/*!
 * @brief TF_Buffer holds a pointer to a block of data, its associated length
 *        and the `data_deallocator` function pointer.
 */
typedef struct TF_Buffer
{
    const void* data;
    size_t length;
    void (*data_deallocator)(void* data, size_t length);
} TF_Buffer;

// -----------------------------------------------------------------------------
/*!
 * @brief Operation that has been added to the graph. Valid until the graph is
 *        deleted.
 */
typedef struct TF_Operation TF_Operation;

/*!
 * @brief Represents a specific input of an operation.
 */
typedef struct TF_Input
{
    TF_Operation* oper;
    int index;    // The index of the input within oper.
} TF_Input;

/*!
 * @brief Represents a specific output of an operation.
 */
typedef struct TF_Output
{
    TF_Operation* oper;
    int index;    // The index of the output within oper.
} TF_Output;

/*!
 * @brief Returns the operation in the graph with `oper_name`.
 * @param Pointer to the graph object.
 * @param Pointer to the string of operation name.
 * @return Pointer to the operation.  Returns nullptr if no operation found.
 */
TF_CAPI_EXPORT extern TF_Operation* TF_GraphOperationByName(TF_Graph* graph, const char* oper_name);

// -----------------------------------------------------------------------------
/*!
 * @brief Holds a multi-dimensional array of elements of a single data type.
 */
typedef struct TF_Tensor TF_Tensor;

/*!
 * @brief Return a new tensor that holds the bytes data[0,len-1].
 * @param Type of the tensor data.
 * @param Array of dims.
 * @param Number of dims.
 * @param Pointer to the tensor data.
 * @param Length of the tensor data.
 * @param Custom deallocator function.  The data will be deallocated by a
 *        subsequent call to TF_DeleteTensor via:
 *        (*deallocator)(data, len, deallocator_arg)
 * @param Parameters of the custom deallocator function.
 * @return Pointer to the new tensor.
 */
TF_CAPI_EXPORT extern TF_Tensor* TF_NewTensor(TF_DataType, const int64_t* dims, int num_dims, void* data, size_t len,
                                              void (*deallocator)(void* data, size_t len, void* arg),
                                              void* deallocator_arg);

/*!
 * @brief Destroy a tensor.
 * @param Pointer to the tensor to be deleted.
 */
TF_CAPI_EXPORT extern void TF_DeleteTensor(TF_Tensor*);

/*!
 * @brief Return the length of the tensor in the "dim_index" dimension.
 * @param Pointer to the tensor.
 * @param Index of the dim. REQUIRES: 0 <= dim_index < TF_NumDims(tensor)
 * @return Length of the tensor in the "dim_index" dimension.
 */
TF_CAPI_EXPORT extern int64_t TF_Dim(const TF_Tensor* tensor, int dim_index);

/*!
 * @brief Return a pointer to the underlying data buffer.
 * @param Pointer to the tensor.
 * @return Pointer to the data buffer.
 */
TF_CAPI_EXPORT extern void* TF_TensorData(const TF_Tensor*);

// -----------------------------------------------------------------------------
/*!
 * @brief Represents a computation graph.  Graphs may be shared between sessions.
 *        Graphs are thread-safe when used as directed below.
 */
typedef struct TF_Graph TF_Graph;

/*!
 * @brief Return a new graph object.
 * @return Pointer to a new graph object.
 */
TF_CAPI_EXPORT extern TF_Graph* TF_NewGraph();

/*!
 * @brief Destroy a graph object.
 * @param Pointer to the graph object to be deleted.
 */
TF_CAPI_EXPORT extern void TF_DeleteGraph(TF_Graph*);

// -----------------------------------------------------------------------------
/*!
 * @brief TF_ImportGraphDefOptions holds options that can be passed to
 *        TF_GraphImportGraphDef.
 */
typedef struct TF_ImportGraphDefOptions TF_ImportGraphDefOptions;

/*!
 * @brief Return a new graph option object.
 * @return Pointer to a new graph option object.
 */
TF_CAPI_EXPORT extern TF_ImportGraphDefOptions* TF_NewImportGraphDefOptions();

/*!
 * @brief Set the prefix to be prepended to the names of nodes in `graph_def`
 *        that will be imported into `graph`.
 * @param Pointer to the graph option.
 * @param Pointer to the prefix string.
 */
TF_CAPI_EXPORT extern void TF_ImportGraphDefOptionsSetPrefix(TF_ImportGraphDefOptions* opts, const char* prefix);

/*!
 * @brief Import the graph serialized in `graph_def` into `graph`.
 *        Convenience function for when no return outputs have been added.
 * @param Pointer to the graph object.
 * @param Pointer to the block of data.
 * @param Pointer to the graph option.
 * @param Pointer to the status object.
 */
TF_CAPI_EXPORT extern void TF_GraphImportGraphDef(TF_Graph* graph, const TF_Buffer* graph_def,
                                                  const TF_ImportGraphDefOptions* options, TF_Status* status);

// -----------------------------------------------------------------------------
/*!
 * @brief Holds options that can be passed during session creation.
 */
typedef struct TF_SessionOptions TF_SessionOptions;

/*!
 * @brief Return a new session option object.
 * @return Pointer to a new session option object.
 */
TF_CAPI_EXPORT extern TF_SessionOptions* TF_NewSessionOptions();

// -----------------------------------------------------------------------------
/*!
 * @brief Represents a session for driving Graph execution.
 */
typedef struct TF_Session TF_Session;

/*!
 * @brief Return a new execution session with the associated graph, or NULL
 *        on error.  *graph must be a valid graph (not deleted or nullptr).
 *        This function will prevent the graph from being deleted until
 *        TF_DeleteSession() is called.  Does not take ownership of opts.
 * @param Pointer to the graph object.
 * @param Pointer to the session option.
 * @param Pointer to the status object.
 * @return Pointer to a new execution session.
 */
TF_CAPI_EXPORT extern TF_Session* TF_NewSession(TF_Graph* graph, const TF_SessionOptions* opts, TF_Status* status);

/*!
 * @brief Close a session.  May not be called after TF_DeleteSession().
 * @param Pointer to the execution session.
 * @param Pointer to the status object.
 */
TF_CAPI_EXPORT extern void TF_CloseSession(TF_Session*, TF_Status* status);

/*!
 * @brief Destroy a session object.
 *        Even if error information is recorded in *status, this call discards
 *        all local resources associated with the session.
 *        The session may not be used during or after this call (and the session
 *        drops its reference to the corresponding graph).
 * @param Pointer to the execution session.
 * @param Pointer to the status object.
 */
TF_CAPI_EXPORT extern void TF_DeleteSession(TF_Session*, TF_Status* status);

/*!
 * @brief Run the graph associated with the session starting with the supplied inputs
 *        (inputs[0,ninputs-1] with corresponding values in input_values[0,ninputs-1]).
 *        On success, the tensors corresponding to outputs[0,noutputs-1] are placed in
 *        output_values[]. Ownership of the elements of output_values[] is transferred
 *        to the caller, which must eventually call TF_DeleteTensor on them.
 *        On failure, output_values[] contains NULLs.
 * @param Pointer to the execution session.
 * @param Pointer to the run options.
 *        `run_options` may be NULL, in which case it will be ignored; or
 *        non-NULL, in which case it must point to a `TF_Buffer` containing the
 *        serialized representation of a `RunOptions` protocol buffer.
 * @param Pointer to the input tensors.
 * @param Pointer to the input values.
 * @param Number of input tensors.
 * @param Pointer to the output tensors.
 * @param Pointer to the output values.
 * @param Number of output tensors.
 * @param Pointer to the target operations.
 * @param Number of target operations.
 * @param Pointer to the run metadata.
 *        `run_metadata` may be NULL, in which case it will be ignored; or
 *        non-NULL, in which case it must point to an empty, freshly allocated
 *        `TF_Buffer` that may be updated to contain the serialized representation
 *        of a `RunMetadata` protocol buffer.
 * @param Pointer to the status.
 */
TF_CAPI_EXPORT extern void TF_SessionRun(TF_Session* session, const TF_Buffer* run_options, const TF_Output* inputs,
                                         TF_Tensor* const* input_values, int ninputs, const TF_Output* outputs,
                                         TF_Tensor** output_values, int noutputs,
                                         const TF_Operation* const* target_opers, int ntargets, TF_Buffer* run_metadata,
                                         TF_Status*);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif    // __TENSORFLOW_C_C_API_H__
