## Change Detection

- 准备onnx模型
    
    这里需要准备onnx模型，后续再转换TensorRT的engine文件。模型的.pth文件和onnx转换脚本都在PyFiles里，自己搭建一个conda环境运行即可。

- 准备engine文件

    在onnx准备好后，使用trtexec命令转onnx为engine，命令示例如下：

    ```
    trtexec --onnx=unet_simple.onnx --saveEngine=unet_simple.engine  --explicitBatch
    ```

- 构建

1. 构建前记得修改CMakeLists.txt里你的三方依赖路径，比如TensorRT路径一类的。

2. 修改main.cpp中你的输入图像路径，以及engine文件路径。 
    
3. 编译运行
    ```
    mkdir build

    cd build

    cmake ..

    make -j16

    ./change_detect_test
    ```

- 参考链接

    [遥感变换检测](https://github.com/Doufei0/ChangeDetection_GUI.git)