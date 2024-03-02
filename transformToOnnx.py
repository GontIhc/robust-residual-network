import os

import mlconfig
import torch
import torchvision

from core import util

# 设置模型名称和配置文件路径
# model_name = 'WRN-A1.yaml'  # RobustResNet-A1.yaml
model_name = 'RobustResNet-A1.yaml'
config_file = './configs/CIFAR10'
config_file = os.path.join(config_file, model_name)

config = mlconfig.load(config_file)
device = torch.device('cuda')
model = config.model().to(device)
params = model.parameters()
optimizer = torch.optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=5e-4)

# pth模型路径
# checkpoint_path_file = './exps/CIFAR10/checkpoints/WRN-A1'
checkpoint_path_file = './exps/CIFAR10/checkpoints/RobustResNet-A1'
filename = checkpoint_path_file + '.pth'
params = model.parameters()
checkpoint = util.load_model(filename=filename,
                             model=model,
                             optimizer=optimizer,
                             alpha_optimizer=None,
                             scheduler=None)

dummy_input = torch.randn(1, 3, 32, 32, device='cuda')
#  给输入输出取个名字
input_names = ["input_1"]
output_names = ["output_1"]

# 设置输出文件夹路径
output_folder = "./exps/CIFAR10/checkpoints/"
# 如果文件夹不存在，则创建文件夹
os.makedirs(output_folder, exist_ok=True)
# 构造完整的输出文件路径
# output_path = os.path.join(output_folder, "WRN-A1.onnx")
output_path = os.path.join(output_folder, "RobustResNet-A1.onnx")


torch.onnx.export(model, dummy_input, output_path, verbose=True, input_names=input_names, output_names=output_names)
