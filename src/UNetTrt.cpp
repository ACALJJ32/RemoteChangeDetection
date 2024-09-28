#include "UNetTrt.h"
#include <fstream>
#include <cmath>
#include "trt_logger.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"

#define INPUT_WIDTH         1024
#define INPUT_HEIGHT        1024


bool UNet::loadTrtModel(const std::string model_path)
{
    char *trt_stream = nullptr;
    size_t size = 0;

    // load trt model
    std::ifstream file(model_path, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trt_stream = new char[size];

        if(!trt_stream)
            return false;
        
        file.read(trt_stream, size);
        file.close();
    } else {
        return false;
    }

    logger::Logger trt_logger(logger::Level::INFO);
    runtime_.reset(nvinfer1::createInferRuntime(trt_logger));

    if(!runtime_)
        return false;

    engine_.reset(runtime_->deserializeCudaEngine(trt_stream, size, nullptr));
    if(!engine_)
        return false;

    context_.reset(engine_->createExecutionContext());
    if(!context_)
        return false;

    const nvinfer1::ICudaEngine& trtEngine = context_->getEngine();

    input_index_ = trtEngine.getBindingIndex(INPUT_NAME);
    output_index_ = trtEngine.getBindingIndex(OUTPUT_NAME);

    CUDA_CHECK(cudaMalloc(&buffers_[input_index_], BATCH_SIZE * 2 * INPUT_WIDTH * INPUT_HEIGHT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers_[output_index_], BATCH_SIZE * 1 * INPUT_WIDTH * INPUT_HEIGHT * sizeof(float)));

    input_float_ = new float[BATCH_SIZE * 2 * INPUT_WIDTH * INPUT_HEIGHT];
    output_float_ = new float[BATCH_SIZE * 1 * INPUT_WIDTH * INPUT_HEIGHT];

    delete []trt_stream;
    return true;
}

bool UNet::trt_infer(cv::Mat &input_mat1, cv::Mat &input_mat2, cv::Mat &output)
{
    if(input_mat1.empty() || input_mat2.empty())
        return false;

    if(input_mat1.rows != input_mat2.rows || input_mat1.cols != input_mat2.cols)
        return false;

    if(input_mat1.channels() <= 1 && input_mat2.channels() <= 1) 
        return false;

    int pre_width = input_mat1.cols;
    int pre_height = input_mat1.rows;

    cv::resize(input_mat1, input_mat1, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::INTER_CUBIC);
    cv::resize(input_mat2, input_mat2, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::INTER_CUBIC);

    std::vector<cv::Mat> input_mat1_channels;
    cv::split(input_mat1, input_mat1_channels);

    std::vector<cv::Mat> input_mat2_channels;
    cv::split(input_mat2, input_mat2_channels);

    // [H, W, C] => [C, H, W] && [0.0, 0.1]
    for(int i = 0; i < INPUT_WIDTH; i++) {
        for(int j = 0; j < INPUT_HEIGHT; j++) {
            int idx_c1 = j * INPUT_WIDTH + i;
            int idx_c2 = idx_c1 + INPUT_WIDTH * INPUT_HEIGHT;
            input_float_[idx_c1] = (float)input_mat1_channels[2].data[idx_c1] / 255.0f;
            input_float_[idx_c2] = (float)input_mat2_channels[2].data[idx_c1] / 255.0f;
        }
    }
    
    memset(output_float_, 0, BATCH_SIZE * 1 * INPUT_WIDTH * INPUT_HEIGHT);
    CUDA_CHECK(cudaStreamCreate(&stream_));
    CUDA_CHECK(cudaMemcpyAsync(buffers_[input_index_], input_float_, 
                BATCH_SIZE * 2 * INPUT_WIDTH * INPUT_HEIGHT * sizeof(float), cudaMemcpyHostToDevice, stream_));

    context_->enqueueV2(buffers_, stream_, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output_float_, buffers_[output_index_], 
                BATCH_SIZE * 1 * INPUT_WIDTH * INPUT_HEIGHT * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    cudaStreamSynchronize(stream_);

    // round
    for(int i = 0; i < INPUT_WIDTH; i++) {
        for(int j = 0; j < INPUT_HEIGHT; j++) {
            int index = j * INPUT_WIDTH + i;
            output_float_[index] = std::round(output_float_[index]);
        }
    }

    output = cv::Mat(INPUT_HEIGHT, INPUT_WIDTH, CV_32F, output_float_);
    output *= 255.0;
    output.convertTo(output, CV_8U);
    cv::resize(output, output, cv::Size(pre_width, pre_height), cv::INTER_CUBIC);
    return true;
}

UNet::~UNet()
{
    if(context_) {
        CUDA_CHECK(cudaFree(buffers_[input_index_]));
        CUDA_CHECK(cudaFree(buffers_[output_index_]));
    }

    if(input_float_)
        delete []input_float_;
    
    if(output_float_)
        delete [] output_float_;

    cudaStreamDestroy(stream_);
}