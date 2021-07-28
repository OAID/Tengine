/*
 * Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef NVDLA_UTILS_DLAIMAGE_UTILS_H
#define NVDLA_UTILS_DLAIMAGE_UTILS_H

#include <stdbool.h>
#include "dlaerror.h"
#include "dlatypes.h"

#include "DlaImage.h"

NvDlaError PGM2DIMG(std::string inputfilename, NvDlaImage* output, nvdla::IRuntime::NvDlaTensor *tensorDesc);
NvDlaError DIMG2Tiff(const NvDlaImage* input, std::string outputfilename);
NvDlaError DIMG2DIMGFile(const NvDlaImage* input, std::string outputfilename, bool stableHash);
NvDlaError DIMGFile2DIMG(std::string inputfilename, NvDlaImage* output);
NvDlaError JPEG2DIMG(std::string inputfilename, NvDlaImage* output, nvdla::IRuntime::NvDlaTensor *tensorDesc);

#if defined(NVDLA_UTILS_CAFFE) || defined(NVDLA_UTILS_NVCAFFE)
#include <caffe/blob.hpp>
#endif

#if defined(NVDLA_UTILS_CAFFE)
template <typename Dtype>
NvDlaError CaffeBlob2DIMG(const caffe::Blob<Dtype>* blob, NvDlaImage* output);
#endif

#if defined(NVDLA_UTILS_NVCAFFE)
NvDlaError NvCaffeBlob2DIMG(const caffe::Blob* blob, NvDlaImage* output);
#endif

#endif // NVDLA_UTILS_DLAIMAGE_UTILS_H
