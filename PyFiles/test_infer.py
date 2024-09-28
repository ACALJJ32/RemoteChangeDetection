import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from unet import UNet
import onnx
import onnxsim

img1 = Image.open("./A/val_20.png")
img2 = Image.open("./B/val_20.png")

output_image_path = "result.png"

transform = transforms.Compose([
    transforms.ToTensor()
])

weights = './params/unet.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img1_data = np.array(img1)
img1_data = img1_data[:, :, 0]

img2_data = np.array(img2)
img2_data = img2_data[:, :, 0]

input_image = np.stack([img1_data, img2_data], axis=2)
input_image_tensor = transform(input_image).unsqueeze(0).type(torch.float32).to(device)


def export_norm_onnx(model, file, input):
    torch.onnx.export(
        model         = model, 
        args          = (input,),
        f             = file,
        input_names   = ["input0"],
        output_names  = ["output0"],
        opset_version = 9)

    print("Finished normal onnx export")

    model_onnx = onnx.load(file)
    onnx.checker.check_model(model_onnx)

    # 使用onnx-simplifier来进行onnx的简化。
    print(f"Simplifying with onnx-simplifier {onnxsim.__version__}...")
    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, "assert check failed"
    onnx.save(model_onnx, file)

with torch.no_grad():
    net = UNet(2).to(device)
    net.eval()
    load_models = torch.load(weights)
    net.load_state_dict(torch.load(weights))
    
    out_image = net(input_image_tensor)


    _out_image = out_image[0][0].round().detach().cpu().numpy()
    _out_image = (_out_image * 255).astype(np.uint8)

    result_image = Image.fromarray(_out_image)
    result_image.save(output_image_path)
    export_norm_onnx(net, "./unet_simple.onnx", input_image_tensor)

print("Finished!")