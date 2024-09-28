#include <iostream>
#include <memory>
#include <chrono>
#include "UNetTrt.h"

int main()
{
    std::shared_ptr<UNet> unet_infer = std::make_shared<UNet>();
    std::string model_path = "/mnt/d/workspace-wsl/UNet_TensorRT_Test/weights/unet_simple_trt.engine";

    if(unet_infer) {
        if(unet_infer->loadTrtModel(model_path))
            std::cout << "UNet Init Successful! \n";
        else 
            std::cout << "UNet Init Failed! \n";
    }

    cv::Mat img1 = cv::imread("/mnt/d/workspace-wsl/UNet_TensorRT_Test/test_images/val_20_A.png");
    cv::Mat img2 = cv::imread("/mnt/d/workspace-wsl/UNet_TensorRT_Test/test_images/val_20_B.png");
    cv::Mat result;

    if(unet_infer->trt_infer(img1, img2, result)) {
        std::cout << "UNet Infer Successfully! \n";
    } else {
        std::cout << "UNet Infer Failed! \n";
    }

    int count = 100;
    int cost = 0;
    for(int i = 0; i < count; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        bool success = unet_infer->trt_infer(img1, img2, result);
        auto end = std::chrono::high_resolution_clock::now();
        cost += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }
    std::cout << "duration: " << (float)(cost) / count << " ms" << std::endl; 

    if(!result.empty()) {
        cv::imwrite("./result.png", result);
    }

    std::cout << "Finished! \n";
    return 0;
}