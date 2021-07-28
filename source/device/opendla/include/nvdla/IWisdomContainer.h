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

#ifndef NVDLA_I_WISDOM_CONTAINER_H
#define NVDLA_I_WISDOM_CONTAINER_H

#include <string>
#include <vector>

namespace nvdla
{

class IWisdom;
class IWisdomContainer;

class IWisdomContainerEntry
{
public:

    enum EntryType {
        ENTRY_TYPE_OBJECT = 0,
        ENTRY_TYPE_STRING = 1,
        ENTRY_TYPE_UINT32 = 2,
        ENTRY_TYPE_INT32  = 3,
        ENTRY_TYPE_UINT8_VECTOR  = 4,
        ENTRY_TYPE_FLOAT32 = 5,
        ENTRY_TYPE_UINT64 = 6,
        ENTRY_TYPE_UINT8   = 7,
    };

    virtual IWisdomContainer *container() const = 0;
    virtual const std::string path() const = 0;
    virtual const std::string name() const = 0;
    virtual EntryType type() const = 0;

    // write<EntryType>
    virtual bool writeUInt8(NvU8 v) = 0;
    virtual bool writeString(const std::string &v) = 0;
    virtual bool writeUInt8Vector(const std::vector<NvU8> &v) = 0;
    virtual bool writeUInt32(NvU32 v) = 0;
    virtual bool writeUInt64(NvU64 v) = 0;
    virtual bool writeInt32(NvS32 v) = 0;
    virtual bool writeFloat32(NvF32 v) = 0;


    // read<EntryType>
    virtual bool readUInt8(NvU8 &) const = 0;
    virtual bool readString(std::string &) const = 0;
    virtual bool readUInt8Vector(std::vector<NvU8> &) const = 0;
    virtual bool readUInt32(NvU32 &) const = 0;
    virtual bool readUInt64(NvU64 &) const = 0;
    virtual bool readInt32(NvS32 &) const = 0;
    virtual bool readFloat32(NvF32 &) const = 0;


    // access to sub-elements for the 'object' type
    virtual bool insertEntry(const std::string &name, EntryType, IWisdomContainerEntry *&) = 0;
    virtual bool removeEntry(const std::string &name) = 0;

    virtual bool getEntryNames(std::vector<std::string> *) = 0;
    virtual bool getEntry(const std::string &name, EntryType, IWisdomContainerEntry *&) = 0;

protected:
    IWisdomContainerEntry();
    virtual ~IWisdomContainerEntry();

};


class IWisdomContainer
{
public:

    virtual bool open(const std::string &uri) = 0;

    virtual bool isOpen() = 0;
    virtual void close() = 0;

    virtual IWisdom *wisdom() = 0;
    virtual IWisdomContainerEntry *root() = 0;


    //
    // To be used by the container entries themselves, not by api clients.
    //
    // i.e.:
    // protected:
    //     friend class IWisdomContainerEntry;
    virtual bool insertEntry(const std::string &path,
                             const std::string &name,
                             IWisdomContainerEntry::EntryType,
                             IWisdomContainerEntry *&) = 0;
    virtual bool removeEntry(const std::string &path) = 0;

    virtual bool getEntryNames(const std::string &path, std::vector<std::string> *) = 0;
    virtual bool getEntry(const std::string &path, const std::string &name,
                          IWisdomContainerEntry::EntryType,
                          IWisdomContainerEntry *&entry) = 0;
    //    virtual bool getEntry(const std::string &name, IWisdomContainerEntry *&entry) = 0;

    virtual bool writeUInt8(const std::string &path, NvU8) = 0;
    virtual bool writeString(const std::string &path, const std::string &) = 0;
    virtual bool writeUInt8Vector(const std::string &path, const std::vector<NvU8> &) = 0;
    virtual bool writeUInt32(const std::string &path, NvU32) = 0;
    virtual bool writeUInt64(const std::string &path, NvU64) = 0;
    virtual bool writeInt32(const std::string &path, NvS32) = 0;
    virtual bool writeFloat32(const std::string &path, NvF32) = 0;

    virtual bool readUInt8(const std::string &path, NvU8 &) = 0;
    virtual bool readString(const std::string &path, std::string &) = 0;
    virtual bool readUInt8Vector(const std::string &path, std::vector<NvU8> &) = 0;
    virtual bool readUInt32(const std::string &path, NvU32 &) = 0;
    virtual bool readUInt64(const std::string &path, NvU64 &) = 0;
    virtual bool readInt32(const std::string &path, NvS32 &) = 0;
    virtual bool readFloat32(const std::string &path, NvF32 &) = 0;

protected:
    IWisdomContainer(IWisdom *);
    virtual ~IWisdomContainer();

};

} // nvdla

#endif // NVDLA_I_WISDOM_CONTAINER_H
