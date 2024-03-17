import mlconfig
import torch
from models.resnet import PreActResNet, ResNet
from models.resnet_cbam import ResNet_cbam, Bottleneck_cbam
from models.robnet import RobNet
from models.advrush import AdvRush
from models.wide_resnet import Wide_ResNet

# Setup mlconfig
mlconfig.register(torch.optim.SGD)
mlconfig.register(torch.optim.Adam)
mlconfig.register(torch.optim.Adamax)

mlconfig.register(torch.optim.lr_scheduler.MultiStepLR)
mlconfig.register(torch.optim.lr_scheduler.CosineAnnealingLR)
mlconfig.register(torch.optim.lr_scheduler.StepLR)
mlconfig.register(torch.optim.lr_scheduler.ExponentialLR)

mlconfig.register(torch.nn.CrossEntropyLoss)

mlconfig.register(PreActResNet)
mlconfig.register(RobNet)
mlconfig.register(AdvRush)

mlconfig.register(ResNet)  # 基本的ResNet
mlconfig.register(Wide_ResNet)  # 基本的WideResNet

mlconfig.register(ResNet_cbam)

