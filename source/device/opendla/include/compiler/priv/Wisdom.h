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

#ifndef NVDLA_PRIV_WISDOM_H
#define NVDLA_PRIV_WISDOM_H

#include "Type.h"

#include "nvdla/IWisdom.h"

#include "Compiler.h"
#include "Layer.h"
#include "Network.h"
#include "Profiler.h"
#include "Tensor.h"
#include "priv/Loadable.h"

namespace nvdla {

class INetwork;
class ILayer;
class ITensor;

namespace priv {

class WisdomContainer;

class SymbolTable
{
public:
    bool insertNetwork(INetwork* net, const std::string& sym);
    bool insertLayer(ILayer* layer, const std::string& sym);
    bool insertTensor(ITensor* tensor, const std::string& sym);
    bool insertLoadable(ILoadable* loadable, const std::string& sym);
    bool insertProfile(IProfile* profile, const std::string& sym);

    INetwork* findNetwork(const std::string& sym);
    bool findNetwork(Network*, std::string& sym);

    ILayer* findLayer(const std::string& sym);
    bool findLayer(Layer* l, std::string& sym);

    ITensor* findTensor(const std::string& sym);
    bool findTensor(Tensor* t, std::string& sym);

    ILoadable* findLoadable(const std::string& sym);
    bool findLoadable(Loadable* l, std::string& sym);

    IProfile* findProfile(const std::string& sym);
    bool findProfile(Profile* p, std::string& sym);

    //    bool networkSymbolAssigned(const std::string &sym) const;
    //    bool layerSymbolAssigned(const std::string &sym) const;
    //    bool tensorSymbolAssigned(const std::string &sym) const;

protected:
    typedef BiMap<std::string, INetwork*>::left_iterator SymNetIter;
    typedef BiMap<std::string, INetwork*>::right_iterator NetSymIter;

    //    typedef std::map<std::string, INetwork *>::iterator SymNetIter;
    //    typedef std::map<INetwork *, std::string>::iterator NetSymIter;

    typedef std::map<std::string, ILayer*>::iterator SymLayerIter;
    typedef std::map<ILayer*, std::string>::iterator LayerSymIter;

    typedef std::map<std::string, ITensor*>::iterator SymTensorIter;
    typedef std::map<ITensor*, std::string>::iterator TensorSymIter;

    typedef std::map<std::string, ILoadable*>::iterator SymLoadableIter;
    typedef std::map<ILoadable*, std::string>::iterator LoadableSymIter;

    typedef std::map<std::string, IProfile*>::iterator SymProfileIter;
    typedef std::map<IProfile*, std::string>::iterator ProfileSymIter;

    BiMap<std::string, INetwork*> m_sym_net;
    //    std::map<std::string, INetwork *> m_sym_net;
    //    std::map<INetwork *, std::string> m_net_sym;

    std::map<std::string, ILayer*> m_sym_layer;
    std::map<ILayer*, std::string> m_layer_sym;

    std::map<std::string, ITensor*> m_sym_tensor;
    std::map<ITensor*, std::string> m_tensor_sym;

    std::map<std::string, ILoadable*> m_sym_loadable;
    std::map<ILoadable*, std::string> m_loadable_sym;

    std::map<std::string, IProfile*> m_sym_profile;
    std::map<IProfile*, std::string> m_profile_sym;
};

class WisdomFactory
{
public:
    typedef PrivPair<IWisdom*, Wisdom*> WisdomPrivPair;

    static WisdomPrivPair newWisdom();
    static NvDlaError deleteWisdom(IWisdom* wisdom);

    static Wisdom* priv(IWisdom*);
    static IWisdom* i(Wisdom*);
    static IWisdom* self(void* s);

    static IWisdom* deserializeFrom(WisdomContainerEntry*);

protected:
    static BiMap<IWisdom*, Wisdom*> s_priv;
    static BiMap<void*, IWisdom*> s_self;

    //    static IWisdom *deserializeWisdom(WisdomContainerEntry *);
};

class Wisdom : public IWisdom
{
public:
    Wisdom();
    virtual ~Wisdom();

    //     virtual bool open(IWisdomContainer *);
    virtual bool open(const std::string& uri);

    //    const INetwork *getNetwork(); // AST, original input
    //    int getNumTestPoints();
    //    void save(INetwork *, NVDLANetwork *);
    //    void save(const OpList &, const WisdomTestPoint &);
    //    WisdomTestPoint &getTestPoint(int);

    virtual void close();

    virtual IWisdomContainerEntry* getRootEntry();

    virtual bool setNetwork(INetwork*);
    virtual bool setNetworkTransient(INetwork*);

    virtual INetwork* getNetwork();

    virtual IProfiler* getProfiler();

    virtual ICompiler* getCompiler();

    //
    // Dictionary/symbol table interfaces. For the following:
    // find -> look-up, only
    // get -> look-up and, if not found, instantiate
    //    may result in object with dangling symbolic references...
    // assign-> produce a unique symbol name (first, try object "name")
    //    then insert into the table.
    //
    virtual bool insertNetworkSymbol(INetwork*, const std::string&);
    virtual INetwork* findNetworkSymbol(const std::string&);

    virtual bool insertLayerSymbol(ILayer*, const std::string&);
    virtual ILayer* findLayerSymbol(const std::string&);
    virtual ILayer* getLayerFromSymbol(const std::string&);
    virtual bool assignLayerSymbol(Layer*, std::string&);
    virtual bool setLayer(Layer*); // error for it not to have been assigned a symbol

    virtual bool insertTensorSymbol(ITensor*, const std::string&);
    virtual ITensor* findTensorSymbol(const std::string&);
    virtual ITensor* getTensorFromSymbol(const std::string&);
    virtual bool assignTensorSymbol(Tensor*, std::string&);
    virtual bool setTensor(Tensor*); // error for it not to have been assigned a symbol

    virtual bool insertLoadableSymbol(ILoadable*, const std::string&);
    virtual ILoadable* findLoadableSymbol(const std::string&);
    //    virtual ILoadable *getLoadableFromSymbol(const std::string &);
    //    virtual bool assignLoadableSymbol(Loadable *, std::string &);
    //    virtual bool setLoadable(Loadable *); // error for it not to have been assigned a symbol

    virtual bool insertProfileSymbol(IProfile*, const std::string&);
    virtual IProfile* findProfileSymbol(const std::string&);

    virtual NvDlaError setDataType(DataType::UnderlyingType d);
    virtual NvDlaError getDataType(DataType::UnderlyingType*) const;

public: // internally facing
    virtual bool findITensorSymbol(ITensor*, std::string&);
    virtual bool findTensorSymbol(Tensor*, std::string&);

    virtual bool findILayerSymbol(ILayer*, std::string&);
    virtual bool findLayerSymbol(Layer*, std::string&);

    virtual bool findILoadableSymbol(ILoadable*, std::string&);
    virtual bool findLoadableSymbol(Loadable*, std::string&);

    virtual bool findIProfileSymbol(IProfile*, std::string&);
    virtual bool findProfileSymbol(Profile*, std::string&);

protected:
    WisdomContainer* m_container;
    INetwork* m_network;

    SymbolTable m_symbol_table;

    LayerFactory m_layer_factory;
    NetworkFactory m_network_factory;
    TensorFactory m_tensor_factory;
    LoadableFactory m_loadable_factory;

    ICompiler* m_compiler;
    IProfiler* m_profiler;

    DataType m_data_type;

    /**
     * Internal functions which are unsafe and external interfaces wraps them
     * to catch possible exception thrown.
     **/
    virtual bool openInternal(const std::string& uri);
    virtual void closeInternal();
    virtual bool setNetworkInternal(INetwork*);
    virtual INetwork* getNetworkInternal();
    virtual NvDlaError setDataTypeInternal(DataType::UnderlyingType d);
};

} // namespace priv
} // namespace nvdla

#endif // NVDLA_PRIV_WISDOM_H
