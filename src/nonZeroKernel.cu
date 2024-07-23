/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "nonZeroKernel.h"
#include <stdint.h>

__global__ void determineSrcKernel(
        const int64_t*  inds,
        const int32_t numInds,
        uint32_t* idxBuf) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < numInds){
        atomicMax(&idxBuf[inds[idx]], idx);
    }
}

template <typename T>
__global__ void indexPutKernel(
        const T* src,
        const int64_t*  inds,
        const int32_t numInds,
        const uint32_t C,
        const uint32_t* idxBuf,
        T* dst) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < numInds){
        auto ind = inds[idx];
        auto chosen_idx = idxBuf[ind];
        if (idx == chosen_idx){
            // copy data to dst
            uint32_t s1 = static_cast<uint32_t>(ind)*C, s2 = idx*C;
            for(auto i=0; i<C; ++i)
                dst[s1 + i] = src[s2 + i];
        }
    }
}

template <typename T>
void indexPutImpl(const T* src,
        const int64_t*  inds,
        const int32_t numInds,
        const uint32_t C,
        uint32_t* idxBuf, // should be allocated and set to zeros
        T* dst,
        cudaStream_t stream)
{
    constexpr int32_t kBLOCK_SIZE = 256;
    int32_t const blocksPerGrid = (C + kBLOCK_SIZE - 1) / kBLOCK_SIZE;

    // Determine which source index will be used, this will make sure for all duplicated indexes
    // that the last one in the inds array will be used
    determineSrcKernel<<<blocksPerGrid, kBLOCK_SIZE, 0, stream>>>(
        inds, numInds, idxBuf);
    // Copy data
    indexPutKernel<<<blocksPerGrid, kBLOCK_SIZE, 0, stream>>>(
        src, inds, numInds, C, idxBuf, dst);
}

#define INDEXPUT_SPECIALIZED_IMPL(T) \
    template void indexPutImpl<T>(const T* src, const int64_t*  inds, const int32_t numInds, const uint32_t C, \
        uint32_t* idxBuf, T* dst, cudaStream_t stream);

INDEXPUT_SPECIALIZED_IMPL(float)
INDEXPUT_SPECIALIZED_IMPL(half)
