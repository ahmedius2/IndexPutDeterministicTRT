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
#ifndef SAMPLE_NONZERO_KERNEL_H
#define SAMPLE_NONZERO_KERNEL_H

#include <cuda_fp16.h>

#include <cstdint>

template <typename T>
void indexPutImpl(const T* src,
        const int64_t*  inds,
        const int32_t numInds,
        const uint32_t C,
        uint32_t* idxBuf, // should be allocated and set to zeros
        T* dst,
        cudaStream_t stream);

#endif // SAMPLE_NONZERO_KERNEL_Hs
