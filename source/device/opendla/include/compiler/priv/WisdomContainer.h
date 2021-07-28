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

#ifndef NVDLA_PRIV_WISDOM_CONTAINER_H
#define NVDLA_PRIV_WISDOM_CONTAINER_H

#include "nvdla/IWisdomContainer.h"

namespace nvdla
{

namespace priv
{

class WisdomContainer;
class Wisdom;

class WisdomContainerEntry : public IWisdomContainerEntry
{
public: // externally facing
    virtual IWisdomContainer *container() const;
    virtual const std::string path() const;
    virtual const std::string name() const;
    virtual EntryType type() const;

    virtual bool writeUInt8(NvU8 v);
    virtual bool writeString(const std::string &v);
    virtual bool writeUInt8Vector(const std::vector<NvU8> &v);
    virtual bool writeUInt8Vector(const NvU8 *v, size_t);
    virtual bool writeUInt32(NvU32 v);
    virtual bool writeUInt64(NvU64 v);
    virtual bool writeInt32(NvS32 v);
    virtual bool writeFloat32(NvF32 v);

    virtual bool readUInt8(NvU8 &) const;
    virtual bool readString(std::string &) const;
    virtual bool readUInt8Vector(std::vector<NvU8> &) const;
    virtual bool readUInt8Vector(NvU8 **v, size_t *s) const;
    virtual bool readUInt32(NvU32 &) const;
    virtual bool readUInt64(NvU64 &) const;
    virtual bool readInt32(NvS32 &) const;
    virtual bool readFloat32(NvF32 &) const;

    virtual bool insertEntry(const std::string &name, EntryType, IWisdomContainerEntry *&);
    virtual bool removeEntry(const std::string &name);

    virtual bool getEntryNames(std::vector<std::string> *);
    virtual bool getEntry(const std::string &name, EntryType, IWisdomContainerEntry *&);


public: // internally facing
    bool insertEntry(const std::string &name, EntryType, WisdomContainerEntry *);
    bool getEntry(const std::string &name, EntryType, WisdomContainerEntry *) const;

    WisdomContainer *container_priv() { return m_container; }

    virtual ~WisdomContainerEntry();
    WisdomContainerEntry();
    WisdomContainerEntry(WisdomContainer *c, const std::string &path,
                         const std::string &name, EntryType type);

    // serialize to and from a new, named entry in an existing OBJECT type container entry.
    bool writeUInt8Enum(const std::string &entry_name, NvU8 v);
    bool writeString(const std::string &entry_name, const std::string &v);
    bool writeUInt8Vector(const std::string &entry_name, const std::vector<NvU8> &v);
    bool writeUInt8Vector(const std::string &entry_name, const NvU8 *data, size_t size);
    bool writeUInt32(const std::string &entry_name, NvU32 v);
    bool writeUInt64(const std::string &entry_name, NvU64 v);
    bool writeInt32(const std::string &entry_name, NvS32 v);
    bool writeFloat32(const std::string &entry_name, NvF32 v);
    bool writeObject(const std::string &entry_name);
    bool writeDims2(const std::string &entry_name, const Dims2 &);
    bool writeDims3(const std::string &entry_name, const Dims3 &);
    bool writeDims4(const std::string &entry_name, const Dims4 &);
    bool writeWeights(const std::string &entry_name, const Weights &);

    bool readUInt8Enum(const std::string &entry_name, NvU8 &) const;
    bool readString(const std::string &entry_name, std::string &) const;
    bool readUInt8Vector(const std::string &entry_name, std::vector<NvU8> &) const;
    bool readUInt8Vector(const std::string &entry_name, const NvU8 **data, size_t *size);
    bool readUInt32(const std::string &entry_name, NvU32 &) const;
    bool readUInt64(const std::string &entry_name, NvU64 &) const;
    bool readInt32(const std::string &entry_name, NvS32 &) const;
    bool readFloat32(const std::string &entry_name, NvF32 &) const;
    bool readObject(const std::string &entry_name) const;
    bool readDims2(const std::string &entry_name, Dims2 &) const;
    bool readDims3(const std::string &entry_name, Dims3 &) const;
    bool readDims4(const std::string &entry_name, Dims4 &) const;
    bool readWeights(const std::string &entry_name, Weights &) const;

    bool insertEntryIfNotPresent(const std::string &name, EntryType, WisdomContainerEntry *);


protected:
    friend class WisdomContainer;


    WisdomContainer *m_container;
    std::string m_path;
    std::string m_name;
    EntryType m_type;
    inline const std::string pathName() const {
        if ( m_name.size() && (m_name != std::string(".")) ) {
            return m_path + "/" + m_name;
        }
        return m_path;
    }

    WisdomContainerEntry &operator =(const WisdomContainerEntry &rhs)
    {
        m_container = rhs.m_container;
        m_path = rhs.m_path;
        m_name = rhs.m_name;
        m_type = rhs.m_type;
        return *this;
    }
};


//
// Ideally this is a base class for various implementation schemes to derive
// from.  For now it's a concrete class with a recursive filesystem implementation.
//

class WisdomContainer : public IWisdomContainer
{
public: // externally facing
    virtual bool open(const std::string &uri); // filename, dirname, dbname, etc.
    virtual bool isOpen();
    virtual void close();

    virtual IWisdom *wisdom();
    virtual IWisdomContainerEntry *root();

    virtual bool insertEntry(const std::string &path,
                             const std::string &name,
                             IWisdomContainerEntry::EntryType,
                             IWisdomContainerEntry *&);
    virtual bool removeEntry(const std::string &path);

    virtual bool getEntryNames(const std::string &path, std::vector<std::string> *);
    virtual bool getEntry(const std::string &path, const std::string &name,
                          IWisdomContainerEntry::EntryType type,
                          IWisdomContainerEntry *&entry);

    virtual bool writeUInt8(const std::string &path, NvU8);
    virtual bool writeString(const std::string &path, const std::string &);
    virtual bool writeUInt8Vector(const std::string &path, const std::vector<NvU8> &);
    virtual bool writeUInt8Vector(const std::string &path, const NvU8 *, size_t);
    virtual bool writeUInt32(const std::string &path, NvU32);
    virtual bool writeUInt64(const std::string &path, NvU64);
    virtual bool writeInt32(const std::string &path, NvS32);
    virtual bool writeFloat32(const std::string &path, NvF32);

    virtual bool readUInt8(const std::string &path, NvU8 &);
    virtual bool readString(const std::string &path, std::string &);
    virtual bool readUInt8Vector(const std::string &path, std::vector<NvU8> &);
    virtual bool readUInt8Vector(const std::string &path, NvU8 **, size_t *);
    virtual bool readUInt32(const std::string &path, NvU32 &);
    virtual bool readUInt64(const std::string &path, NvU64 &);
    virtual bool readInt32(const std::string &path, NvS32 &);
    virtual bool readFloat32(const std::string &path, NvF32 &);


public: // internally facing

    WisdomContainer(Wisdom *);
    virtual ~WisdomContainer();

    WisdomContainerEntry *root_priv();
    Wisdom               *wisdom_priv();
    virtual bool insertEntry(const std::string &path,
                             const std::string &name,
                             IWisdomContainerEntry::EntryType,
                             WisdomContainerEntry *);

    virtual bool getEntry(const std::string &path, const std::string &name,
                          IWisdomContainerEntry::EntryType type,
                          WisdomContainerEntry *entry);



protected:
    friend class Wisdom;
    friend class WisdomContainerEntry;

    IWisdom *m_wisdom;
    Wisdom *m_wisdom_priv;

    std::string m_root;
    WisdomContainerEntry m_root_entry;

    const std::string entryFilename(const std::string &path, const std::string &name) const;
    const std::string entryFilename(IWisdomContainerEntry *) const;
    const std::string entryFilename(const std::string &name) const;

    const std::string entryDirname(const std::string &path, const std::string &name) const;

};

} // nvdla::priv
} // nvdla

#endif /* NVDLA_WISDOM_CONTAINER_PRIV_H */
