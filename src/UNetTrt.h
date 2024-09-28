#ifndef UNET_TRT_H_
#define UNET_TRT_H_

#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

namespace nvinfer1
{
    class IRuntime;
    class ICudaEngine;
    class IExecutionContext;
}

class UNet
{
public:
    UNet() {};
    ~UNet();
    bool loadTrtModel(const std::string model_path);
    bool trt_infer(cv::Mat &input_mat1, cv::Mat &input_mat2, cv::Mat &output);          // input_mat1: before, input_mat2: after

private:
    std::shared_ptr<nvinfer1::IRuntime>               runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine>            engine_;
    std::shared_ptr<nvinfer1::IExecutionContext>      context_;
    cudaStream_t                                      stream_;

    int input_index_;
    int output_index_;

    const char                  *INPUT_NAME         = "input0";
    const char                  *OUTPUT_NAME        = "output0";
    const int                    BATCH_SIZE         = 1;
    void                        *buffers_[2];
    float                       *input_float_       = nullptr;
    float                       *output_float_      = nullptr;
};

#endif