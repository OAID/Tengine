/*
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef NVDLA_PRIV_COMPILER_H
#define NVDLA_PRIV_COMPILER_H

#include <vector>
#include <map>
#include <algorithm>
#include <fstream>

#include "Check.h"
#include "Type.h"

#include "nvdla/ICompiler.h"

#include "CanonicalAST.h"
#include "EngineAST.h"
#include "Memory.h"
#include "priv/Loadable.h"

namespace nvdla {

namespace priv {

class Compiler;
class Loadable;

class CompilerFactory
{
public:
    typedef PrivPair<ICompiler*, Compiler*> CompilerPrivPair;

    static CompilerPrivPair newCompiler();
    static NvDlaError deleteCompiler(ICompiler* compiler);

    static Compiler* priv(ICompiler*);
    static ICompiler* i(Compiler*);
    static ICompiler* self(void* s);

protected:
    static BiMap<ICompiler*, Compiler*> s_priv;
    static BiMap<void*, ICompiler*> s_self;
};

class DLAInterface;

class Compiler : public ICompiler
{
public: // externally facing
    virtual IWisdom* wisdom() const;

    virtual NvDlaError getDataType(DataType::UnderlyingType* d) const;
    virtual NvDlaError compile(const char* profile_name, const char* target_config_name, ILoadable**); // "" := default
    virtual NvDlaError getLoadableImage(const char* profile_name, NvU8* flatbuf);
    virtual NvDlaError getLoadableImageSize(const char* profile_name, NvU64* size);

    virtual NvDlaError compileCheck(const char*, const char*);

public: // internally facing
    NvDlaError emit(engine_ast::Graph* g, LoadableFactory::LoadablePrivPair&);

    Compiler();
    virtual ~Compiler();

    void setWisdom(Wisdom* w)
    {
        m_wisdom = w;
    }

    virtual NvU16 getFactoryType() const;

    inline bool debugVersions() const
    {
        return true;
    }
    inline bool debugTasks() const
    {
        return true;
    }
    inline bool debugMemoryLayout() const
    {
        return true;
    }
    inline bool debugGraphs() const
    {
        return true;
    }
    inline bool debugProfile() const
    {
        return true;
    }

public:
    friend class Wisdom;
    friend class CompilerFactory;

    Wisdom* m_wisdom;

    /**
     * @Purpose:->
     *      Register all unique and shared buffers with the RM
     *      do not reserve spaces for them with the RM since some
     *      buffer sizes might change later for eg: during weight translation
     *      if padding bytes are added
     */
    engine_ast::Graph* registerBuffers(engine_ast::Graph*);

    /**
     * @Purpose:->
     *      Most DNN models are trained on FP32 and the trained data is
     *      not in a form friendly to the DLA (in precision, core-value and layout).
     *      For eg: trained weights for a model could be related to SUB or DIV ops;
     *      but DLA likes ADD and MUL. This api pre-processes the likes of these.
     *      Also, since the max precision available on DLA is FP16, this api also
     *      evaluates if some of the trained weights may not be representable in
     *      the lower precision and tries to contrain them
     */
    engine_ast::Graph* preProcessAuxData(engine_ast::Graph*);

    /**
     * @Purpose:->
     *      Adjacent activation ops can be merged together to save extra memory passes.
     *      Sometimes ops like adjacent addition(s) and/or multiplication(s) can be
     *      combined or translated into a 3rd equation if the combined co-efficients
     *      don't overshoot the compute precision's dynamic range
     *      eg:
     *          - {Scale1, Scale2} = Scale3
     *          - {Bias1, Bias2}   = Bias3
     *          - {NOP, Act1}      = Act1
     *          - {Scale, Bias}    = Batch-Norm
     *          - etc
     */
    engine_ast::Graph* mergeActivationOperations(engine_ast::Graph*);

    /**
     * @Purpose:->
     *      INT8 elementwise operations require both input tensors scaled to
     *      same scaling factor. This pass tries to find input layer which can
     *      be rescaled and updates it's scaling factor. If not layer found
     *      then it will insert scaling layer after one of the input layer
     *      to rescale output of that layer. It is done after merging activation
     *      layers so that math fusion is done before updating factors and before
     *      quantizing aux data so that if scaling factor is updated then new
     *      factor is used to scale data.
     */
    engine_ast::Graph* updateScalingFactors(engine_ast::Graph*);

    /**
     * @Purpose:->
     *      INT8 operations on DLA need quantized activation and trained data.
     *      This phase of the compiler uses per-filter or per-kernel quantization
     *      modes (depending on how the profile is set) and quantizes high precision
     *      trained data (weights/biases/scales/means/variances/etc) in fp16/fp32 to
     *      suitable quantized representations for executing operations in int8 mode
     */
    engine_ast::Graph* quantizeAuxData(engine_ast::Graph*);

    /**
     * @Purpose:->
     *      Adjacent SDP ops can be fused together in SDP SubEngines X1, X2 & Y
     *      to save extra memory passes.
     *      First SDP Op can be configured in X1 while second SDP op in X2 / Y
     *      e.g.
     *          - {Batch-Norm, Eltwise}  = SDPSuperOp X1, X2
     *          - {Batch-Norm, Eltwise, Relu}  = SDPSuperOp X1, X2 with Relu
     *          - {Batch-Norm, Eltwise, Sigmoid}  = SDPSuperOp X1, X2, Y with LUT
     *          - etc
     */
    engine_ast::Graph* fuseSubEngineOps(engine_ast::Graph*);

    /**
     * @Purpose:->
     *      Determine low precision convertor configs for various ops
     */
    engine_ast::Graph* handleLowPrecisionConversions(engine_ast::Graph*);

    /**
     * @Purpose:->
     *      DLA requires auxillary data for each operation to be laid out in a
     *      specific manner. This API translates aux data to a layout suitable for DLA
     */
    engine_ast::Graph* translateAuxData(engine_ast::Graph*);

    /**
     *  @Purpose:->
     *      Reserve sizes of all registered buffers.
     *      Idempotent op.
     */
    engine_ast::Graph* reserveBuffers(engine_ast::Graph*);

    /**
     * @Purpose:->
     *      Some engines dont have memory out port (conv). Operations of such
     *      an engine should be fused with those of an engine which have a write
     *      out potr (sdp). Besides, operations can be fused together if they
     *      are connected over wire in order to save the memory pass latency (sdp+pdp)
     */
    engine_ast::Graph* fuseOnTheFlyNodes(engine_ast::Graph*);

    /**
     * @Purpose:->
     *      Some group of engine operations could be optimized by above compiler
     *      phases and should be executed as atomic such that their execution could
     *      not be intercepted by those from another batch/roi. Group them together
     */
    engine_ast::Graph* groupAtomicOperations(engine_ast::Graph*);

    /**
     * @Purpose:->
     *      Engines have limited sized caching buffers (conv, pdp); its not
     *      always possible to fetch the entire src data before operating on it.
     *      Such operations have to be split and later concatenated using software
     *      magic
     */
    engine_ast::Graph* splitNodes(engine_ast::Graph*);

    /**
     * @Purpose:-> //xxx: mostly can this phase as the duty is spread out to other ops
     *      As a result of splits, fusions and introducing multiple ROIs,
     *      the graph might have several nodes that need special treatment:
     *          - intra-roi nodes might need joins
     *          - nodes around a split node might have to be split themselves
     *          - etc (as we find)
     *      Massage the graph to bound/tighten the nodes from either sides.
     */
    engine_ast::Graph* boundGraph(engine_ast::Graph*);

    /**
     * @Purpose:->
     *      All the compute related decisions made for single batch so far should be now
     *      leveraged to provision for multiple batches. Handle all the operation and
     *      memory related nuances for the multiple batches.
     */
    engine_ast::Graph* handleMultiBatch(engine_ast::Graph*);

    /**
     * @Purpose:->
     *      Flatten the engine graph such that all the multi-ops super nodes
     *      are replaced by their respective nested graph contents.
     */
    engine_ast::Graph* flattenGraph(engine_ast::Graph*);

    /**
     * @Purpose:->
     *      # Resolve all types of dependencies:
     *          - data dependencies between nodes exchanging tensor
     *          - compute dependencies between nodes of same engine
     *          - software dependencies around software (no h/w required) ops like
     *            split & concat
     *      # Determine task boundaries (DLA/EMU/DLA/etc)
     *      # And generate node annotation order within each task so that it
     *        represents functional data-flow among nodes and allows chronological
     *        memory allocation for each of them
     *      # Finally, inherit the dependency graph state generated for 1 batch
     *        into that of the 'N' multiple batches (if N > 1).
     */
    engine_ast::Graph* generateDependencyParams(engine_ast::Graph*, engine_ast::NodeSequence&);

    /**
     * @Purpose:->
     *      Schedule/reserve memory resources.
     */
    engine_ast::Graph* resolveMemory(engine_ast::Graph*, const engine_ast::NodeSequence&);

    /**
     * @Purpose:->
     *      This compilation phase adds debug bdmas to copy intermediate
     *      surfaces to sysmem.  Each such surface is presented
     *      as a bindable debug buffer to the runtime.
     */
    engine_ast::Graph* enableCopyOutDebugSurfaces(engine_ast::Graph*);

    DLAInterface* getTargetDLAInterface(Profile*);
    EMUInterface* getTargetEMUInterface(Profile*);

    /**
     * Internal functions which are unsafe and external interfaces wraps them
     * to catch possible exception thrown.
     **/
    NvDlaError compileInternal(const char*, const char*, ILoadable**, bool);
    NvDlaError compileInternal(Profile*, TargetConfig*, ILoadable**, bool);
    NvDlaError getLoadableImageInternal(const char* profile_name, NvU8* flatbuf);
    NvDlaError getLoadableImageSizeInternal(const char* profile_name, NvU64* size);

private:
    NvDlaError getLoadableFromWisdom(const char* test_point_name,
                                     ILoadable** i);
};

class DumpGraphBase
{
public:
    DumpGraphBase(const std::string& filename, const std::string& graphId)
        : _m_filename(filename), _m_graph_id(graphId)
    {
    } // don't write file by default

    virtual ~DumpGraphBase()
    {
    }

    virtual void setFilename(const std::string s)
    {
        _m_filename = s;
    }
    virtual void setGraphId(const std::string i)
    {
        _m_graph_id = i;
    }
    virtual std::ofstream& out()
    {
        return _m_file;
    }

protected:
    std::string _m_filename;
    std::string _m_graph_id;
    std::ofstream _m_file;
    std::string _m_delim;
};

} // namespace priv

} // namespace nvdla

#endif // NVDLA_PRIV_COMPILER_H
