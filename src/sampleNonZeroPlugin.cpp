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
// Define TRT entrypoints used in common code
#define DEFINE_TRT_ENTRYPOINTS 1
#define DEFINE_TRT_LEGACY_PARSER_ENTRYPOINT 0

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "nonZeroKernel.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include "sampleNonZeroPlugin.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;

using half = __half;


void indexPutHelper(nvinfer1::DataType type, const void* src, const int64_t* inds, const int32_t numInds,
                    const int32_t C, uint32_t* idxBuf, void* dst, cudaStream_t stream)
{
    if (type == nvinfer1::DataType::kFLOAT)
    {
        indexPutImpl<float>(static_cast<const float*>(src), inds,
            numInds, C, idxBuf, static_cast<float*>(dst), stream);
    }
    else if (type == nvinfer1::DataType::kHALF)
    {
        indexPutImpl<half>(static_cast<const half*>(src), inds,
            numInds, C, idxBuf, static_cast<half*>(dst), stream);
    }
    else
    {
        ASSERT(false && "Unsupported data type");
    }
}

class IndexPutPlugin : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime
{
public:
    IndexPutPlugin(IndexPutPlugin const& p) = default;

    IndexPutPlugin(bool dummy)
        : mDummy(dummy)
    {
        initFieldsToSerialize();
    }

    void initFieldsToSerialize()
    {
        mDataToSerialize.clear();
        mDataToSerialize.emplace_back(PluginField("dummy", &mDummy, PluginFieldType::kINT32, 1));
        mFCToSerialize.nbFields = mDataToSerialize.size();
        mFCToSerialize.fields = mDataToSerialize.data();
    }

    // IPluginV3 methods
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override
    {
        try
        {
            if (type == PluginCapabilityType::kBUILD)
            {
                return static_cast<IPluginV3OneBuild*>(this);
            }
            if (type == PluginCapabilityType::kRUNTIME)
            {
                return static_cast<IPluginV3OneRuntime*>(this);
            }
            ASSERT(type == PluginCapabilityType::kCORE);
            return static_cast<IPluginV3OneCore*>(this);
        }
        catch (std::exception const& e)
        {
            sample::gLogError << e.what() << std::endl;
        }
        return nullptr;
    }

    IPluginV3* clone() noexcept override
    {
        auto clone = std::make_unique<IndexPutPlugin>(*this);
        clone->initFieldsToSerialize();
        return clone.release();
    }

    // IPluginV3OneCore methods
    char const* getPluginName() const noexcept override
    {
        return "IndexPutPlugin";
    }

    char const* getPluginVersion() const noexcept override
    {
        return "0";
    }

    char const* getPluginNamespace() const noexcept override
    {
        return "";
    }

    // IPluginV3OneBuild methods
    int32_t getNbOutputs() const noexcept override
    {
        return 1;
    }

    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override
    {
        return 0;
    }

    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        // NOTE accessing other than pos index caused error
        bool typeOk{false};
        if (pos == IOpos::IN_SRC || pos == IOpos::IN_DST || pos == IOpos::OUT_DST)
        {
            typeOk = (inOut[pos].desc.type == DataType::kFLOAT || inOut[pos].desc.type == DataType::kHALF);
        }
        else // 1
        {
            typeOk = inOut[pos].desc.type == DataType::kINT64;
        }

        typeOk = typeOk && (inOut[pos].desc.format == PluginFormat::kLINEAR);
        return typeOk;
    }

    int32_t getOutputDataTypes(
        DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept override
    {
        outputTypes[0] = inputTypes[IOpos::IN_DST];
        return 0;
    }

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept override
    {
        // The input src and dst tensors must be 2-D
        if (inputs[0].nbDims != 2 || inputs[2].nbDims != 2)
        {
            return -1;
        }

//        auto upperBound = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[0], *inputs[0].d[1]);
//        auto optValue = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *upperBound, *exprBuilder.constant(2));
//        auto numNonZeroSizeTensor = exprBuilder.declareSizeTensor(1, *optValue, *upperBound);

        outputs[0].nbDims = 2;
        outputs[0].d[0] = inputs[IOpos::IN_DST].d[0]; // dst
        outputs[0].d[1] = inputs[IOpos::IN_DST].d[1];

        return 0;
    }

    // IPluginV3OneRuntime methods
    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override
    {
        int32_t const numInds = inputDesc[IOpos::IN_INDS].dims.d[0];
        int32_t const C = inputDesc[IOpos::IN_SRC].dims.d[1];

        auto type = inputDesc[0].type;

        if (!(type == nvinfer1::DataType::kHALF || type == nvinfer1::DataType::kFLOAT))
        {
            sample::gLogError << "Unsupported: Sample only supports DataType::kHALF and DataType::FLOAT" << std::endl;
            return -1;
        }

        auto type_bytes = (type == nvinfer1::DataType::kHALF ? 2 : 4);
        cudaMemcpyAsync(outputs[0], inputs[IOpos::IN_DST], inputDesc[IOpos::IN_SRC].dims.d[0] * C * type_bytes, cudaMemcpyDeviceToDevice, stream);
        cudaMemsetAsync(workspace, 0, numInds * sizeof(int32_t), stream);

        indexPutHelper(type, static_cast<const void*>(inputs[0]), static_cast<const int64_t*>(inputs[1]), numInds, C, static_cast<uint32_t*>(workspace),
                outputs[0], stream);

        return 0;
    }

    int32_t onShapeChange(
        PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept override
    {
        return 0;
    }

    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override
    {
        return clone();
    }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override
    {
        return &mFCToSerialize;
    }

    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override
    {
        //inputs[1].max.d[0]; I might want to use this later
        return 120000 * sizeof(int32_t); // considering a maximum of 60000 voxels
    }

private:
    bool mDummy{true};
    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
};


IndexPutPluginCreator::IndexPutPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("dummy", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* IndexPutPluginCreator::getPluginName() const noexcept
{
    return "IndexPutPlugin";
}

char const* IndexPutPluginCreator::getPluginVersion() const noexcept
{
    return "0";
}

PluginFieldCollection const* IndexPutPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV3* IndexPutPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    try
    {
        bool dummy{true};
        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            auto const fieldName(fc->fields[i].name);
            if (std::strcmp(fieldName, "dummy") == 0)
            {
                dummy = *static_cast<bool const*>(fc->fields[i].data);
            }
        }
        return new IndexPutPlugin(dummy);
    }
    catch (std::exception const& e)
    {
        sample::gLogError << e.what() << std::endl;
    }
    return nullptr;
}

char const* IndexPutPluginCreator::getPluginNamespace() const noexcept
{
    return "";
}


SampleIndexPutPlugin::SampleIndexPutPlugin(IndexPutParams const& params)
    : mParams(params)
    , mRuntime(nullptr)
    , mEngine(nullptr)
{
    mSeed = static_cast<uint32_t>(time(nullptr));
}

bool SampleIndexPutPlugin::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
        sample::gLogError << "Builder failed." << std::endl;
    }

    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    if (!network)
    {
        sample::gLogError << "Network creation failed." << std::endl;
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        sample::gLogError << "Create builder config failed." << std::endl;
        return false;
    }

    auto pluginCreator = std::make_unique<IndexPutPluginCreator>();
    getPluginRegistry()->registerCreator(*pluginCreator.get(), "");

    auto constructed = constructNetwork(builder, network, config);
    if (!constructed)
    {
        sample::gLogError << "Construct network failed." << std::endl;
        return false;
    }

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

    SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        sample::gLogError << "Couldn't create plan." << std::endl;
        return false;
    }

    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!mRuntime)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    ASSERT(network->getNbInputs() == 3);
    mInputDims[0] = network->getInput(IOpos::IN_SRC)->getDimensions(); // src
    ASSERT(mInputDims[0].nbDims == 2);
    mInputDims[1] = network->getInput(IOpos::IN_INDS)->getDimensions(); // inds
    ASSERT(mInputDims[1].nbDims == 1);
    mInputDims[2] = network->getInput(IOpos::IN_DST)->getDimensions(); // dst
    ASSERT(mInputDims[2].nbDims == 2);

    ASSERT(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions(); // dst_out
    ASSERT(mOutputDims.nbDims == 2);

    return true;
}

//!
//! \brief Creates a network with a single custom layer containing the NonZero plugin and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the NonZero plugin
//!
//! \param builder Pointer to the engine builder
//!
bool SampleIndexPutPlugin::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config)
{
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }

    int32_t const C = 2;
    auto* src = network->addInput("src", DataType::kFLOAT, {2, {10, C}});
    auto* inds = network->addInput("inds", DataType::kINT64, {1, {10}});
    auto* dst = network->addInput("dst", DataType::kFLOAT, {2, {6, C}});

    sample::gLogInfo << "Added inputs to network" << std::endl;

    ASSERT(src != nullptr && inds != nullptr && dst != nullptr);

    std::vector<PluginField> const vecPF{{"dummy", &mParams.dummy, PluginFieldType::kINT32, 1}};
    PluginFieldCollection pfc{static_cast<int32_t>(vecPF.size()), vecPF.data()};

    auto pluginCreator = static_cast<IPluginCreatorV3One*>(getPluginRegistry()->getCreator("IndexPutPlugin", "0", ""));
    auto plugin = std::unique_ptr<IPluginV3>(pluginCreator->createPlugin("IndexPutPlugin", &pfc, TensorRTPhase::kBUILD));

    sample::gLogInfo << "Plugin got created" << std::endl;

    std::vector<ITensor*> inputsVec{src, inds, dst};
    auto pluginIndexPutLayer = network->addPluginV3(inputsVec.data(), inputsVec.size(), nullptr, 0, *plugin);
    ASSERT(pluginIndexPutLayer != nullptr);
    ASSERT(pluginIndexPutLayer->getInput(0) != nullptr);
    ASSERT(pluginIndexPutLayer->getInput(1) != nullptr);
    ASSERT(pluginIndexPutLayer->getInput(2) != nullptr);
    ASSERT(pluginIndexPutLayer->getOutput(0) != nullptr);

    pluginIndexPutLayer->getOutput(0)->setName("dst_out");

    network->markOutput(*(pluginIndexPutLayer->getOutput(0)));

    sample::gLogInfo << "Plugin added to network" << std::endl;


    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SampleIndexPutPlugin::infer()
{

    // Since the data dependent output size cannot be inferred from the engine denote a sufficient size for the
    // corresponding output buffer (along with the rest of the I/O tensors)
    std::vector<int64_t> ioVolumes = {mInputDims[0].d[0] * mInputDims[0].d[1], // src
                                      mInputDims[1].d[0], // inds
                                      mInputDims[2].d[0] * mInputDims[2].d[1], // dst
                                      mOutputDims.d[0] * mOutputDims.d[1]}; //dst_out

    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, ioVolumes);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    for (int32_t i = 0, e = mEngine->getNbIOTensors(); i < e; ++i)
    {
        auto const name = mEngine->getIOTensorName(i);
        context->setTensorAddress(name, buffers.getDeviceBuffer(name));
    }

    // Read the input data into the managed buffers
    ASSERT(mParams.inputTensorNames.size() == 3);
    if (!processInput(buffers))
    {
        return false;
    }

    // Create CUDA stream for the execution of this inference.
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDeviceAsync(stream);

    bool status = context->enqueueV3(stream);
    if (!status)
    {
        return false;
    }

    // Asynchronously copy data from device output buffers to host output buffers.
    buffers.copyOutputToHostAsync(stream);

    // Wait for the work in the stream to complete.
    CHECK(cudaStreamSynchronize(stream));

    // Release stream.
    CHECK(cudaStreamDestroy(stream));

    // Verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool SampleIndexPutPlugin::processInput(samplesCommon::BufferManager const& buffers)
{
    std::default_random_engine generator(mSeed);
    std::uniform_int_distribution<int32_t> distr(0, 5);
    std::uniform_int_distribution<int64_t> distr64(0, 5);

    sample::gLogInfo << mParams.inputTensorNames[0] << ":" << std::endl;
    float* srcBuf = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    for (int32_t i = 0; i < 10; ++i)
    {
        for (int32_t j = 0; j < 2; ++j)
        {
            srcBuf[i*2 + j] = distr(generator);
            sample::gLogInfo << srcBuf[i*2 + j] << ", ";
        }
        sample::gLogInfo << std::endl;

    }


    int64_t* indsBuf = static_cast<int64_t*>(buffers.getHostBuffer(mParams.inputTensorNames[1]));
    ASSERT(indsBuf != nullptr);
    sample::gLogInfo << mParams.inputTensorNames[1] << ":" << std::endl;
    for (int32_t i = 0; i < 10; ++i)
    {
        indsBuf[i] = distr64(generator);
        sample::gLogInfo << indsBuf[i] << ", ";
    }
    sample::gLogInfo << std::endl;

    sample::gLogInfo << "Dst:" << std::endl;
    float* dstBuf = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[2]));
    for (int32_t i = 0; i < 6; ++i)
    {
        for (int32_t j = 0; j < 2; ++j)
        {
            dstBuf[i*2 + j] = 0;
            sample::gLogInfo << dstBuf[i*2 + j] << ", ";
        }
        sample::gLogInfo << std::endl;
    }

    return true;
}

//!
//! \brief Verify result
//!
//! \return whether the output correctly identifies all (and only) non-zero elements
//!
bool SampleIndexPutPlugin::verifyOutput(samplesCommon::BufferManager const& buffers)
{
    float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));

    sample::gLogInfo << "Output:" << std::endl;
    for (int32_t i = 0; i < 6; ++i)
    {
        for (int32_t j = 0; j < 2; ++j)
        {
            sample::gLogInfo << output[i*2 + j] << ", ";
        }
        sample::gLogInfo << std::endl;
    }

    return true;
}

