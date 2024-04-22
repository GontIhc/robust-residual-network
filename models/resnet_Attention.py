import torch
import torch.nn as nn
import torch.nn.functional as F

# 可选的激活函数和归一化的操作,下面是提前定义的两个字典
from models.activation import avaliable_activations
from models.normalization import avaliable_normalizations
from models.resnet import SqueezeExcitationLayer


# avaliable_activations[activation](inplace=True)

# ====================================== 传统ResNet start ======================================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, mhsa=False, heads=4, resolution=None,
                 activation='ReLU', normalization='BatchNorm', se_switch=False, se_reduction=16, cbam=False, D=False):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = avaliable_normalizations[normalization](planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = avaliable_normalizations[normalization](planes)

        # 激活函数
        if activation == 'Swish':
            self.activation = avaliable_activations[activation]()
        else:
            self.activation = avaliable_activations[activation](inplace=True)

        # SE模块
        self.se_switch = se_switch
        if self.se_switch:
            self.se_bn = avaliable_normalizations[normalization](self.expansion * planes)
            self.se = SqueezeExcitationLayer(self.expansion * planes, reduction=se_reduction)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                avaliable_normalizations[normalization](self.expansion * planes)
                # nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.se_switch:
            se_out = self.se(self.se_bn(out))  # SE模块
            # out = out + se_out
            out = se_out

        out = out + self.shortcut(x)
        out = self.activation(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, mhsa=False, heads=4, resolution=None,
                 activation='ReLU', normalization='BatchNorm',
                 se_switch=False, se_reduction=16,  # 是否使用SE模块
                 cbam=False,  # 是否使用cbam模块
                 D=False,  # 是否使用ResNet-D的下采样优化
                 ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)

        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = avaliable_normalizations[normalization](planes)
        self.bn1.weight.data.fill_(0.0)
        nn.BatchNorm2d(10).weight.data.fill_(0.0)

        # 把这个改造成支持注意力机制(中间层)
        if not mhsa:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            self.conv2 = nn.ModuleList()
            self.conv2.append(MHSA(planes, width=int(resolution[0]), height=int(resolution[1]), heads=heads))
            if stride == 2:
                self.conv2.append(nn.AvgPool2d(2, 2))
            self.conv2 = nn.Sequential(*self.conv2)

        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = avaliable_normalizations[normalization](planes)

        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        # self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.bn3 = avaliable_normalizations[normalization](self.expansion * planes)

        # 激活函数
        if activation == 'Swish':
            self.activation = avaliable_activations[activation]()
        else:
            self.activation = avaliable_activations[activation](inplace=True)

        # SE模块
        self.se_switch = se_switch
        if self.se_switch:
            self.se_bn = avaliable_normalizations[normalization](self.expansion * planes)
            self.se = SqueezeExcitationLayer(self.expansion * planes, reduction=se_reduction)

        # 通道和空间卷积
        self.cbam = cbam
        if self.cbam:
            self.ca = ChannelAttention_(planes * 4)
            self.sa = SpatialAttention_()

        self.shortcut = nn.Sequential()

        if stride != 1:
            if D:  # 启用了ResNet-D的优化,修改恒等链接
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=1, bias=False),
                    avaliable_normalizations[normalization](self.expansion * planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    avaliable_normalizations[normalization](self.expansion * planes)
                    # nn.BatchNorm2d(self.expansion * planes)
                )
        elif in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                avaliable_normalizations[normalization](self.expansion * planes)
                # nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.se_switch:
            se_out = self.se(self.se_bn(out))  # SE模块
            out = out + se_out

        # 通道和空间卷积
        if self.cbam:
            out = self.ca(out) * out
            out = self.sa(out) * out

        out = out + self.shortcut(x)
        out = self.activation(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, activation='ReLU', normalization='BatchNorm',
                 resolution=(32, 32), heads1=4, heads2=4, heads3=4, heads4=4, mhsa1=False, mhsa2=False,
                 mhsa3=False, mhsa4=False, se_switch=False, se_reduction=16, cbam=False, D=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        if block == 'BasicBlock':
            block = BasicBlock
        elif block == 'Bottleneck':
            block = Bottleneck
        else:
            raise ('Unknown block: %s' % block)

        self.resolution = list(resolution)  # 注意力机制使用的 (宽, 高)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if self.conv1.stride[0] == 2:
            self.resolution[0] /= 2
        if self.conv1.stride[1] == 2:
            self.resolution[1] /= 2

        # self.bn1 = nn.BatchNorm2d(64)
        self.bn1 = avaliable_normalizations[normalization](64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, heads=heads1,
                                       mhsa=mhsa1, activation=activation, normalization=normalization,
                                       se_switch=se_switch, se_reduction=se_reduction, cbam=cbam, D=D)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, heads=heads2,
                                       mhsa=mhsa2, activation=activation, normalization=normalization,
                                       se_switch=se_switch, se_reduction=se_reduction, cbam=cbam, D=D)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, heads=heads3,
                                       mhsa=mhsa3, activation=activation, normalization=normalization,
                                       se_switch=se_switch, se_reduction=se_reduction, cbam=cbam, D=D)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, heads=heads4,
                                       mhsa=mhsa4, activation=activation, normalization=normalization,
                                       se_switch=se_switch, se_reduction=se_reduction, cbam=cbam, D=D)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        # 激活函数
        if activation == 'Swish':
            self.activation = avaliable_activations[activation]()
        else:
            self.activation = avaliable_activations[activation](inplace=True)

    def _make_layer(self, block, planes, num_blocks, stride, activation, normalization,
                    heads=4, mhsa=False, se_switch=False, se_reduction=16, cbam=False, D=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, activation=activation, normalization=normalization,
                                mhsa=mhsa, heads=heads, resolution=self.resolution, se_switch=se_switch,
                                se_reduction=se_reduction, cbam=cbam, D=D))
            if stride == 2:  # 为注意力机制改变长款输入(因为只有每个stage的中间卷积的步长为2时会改变图片长宽)
                self.resolution[0] /= 2
                self.resolution[1] /= 2
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# def ResNet18():
#     return ResNet(BasicBlock, [2, 2, 2, 2])
# def ResNet34():
#     return ResNet(BasicBlock, [3, 4, 6, 3])
# def ResNet50():
#     return ResNet(Bottleneck, [3, 4, 6, 3])
# def ResNet101():
#     return ResNet(Bottleneck, [3, 4, 23, 3])
# def ResNet152():
#     return ResNet(Bottleneck, [3, 8, 36, 3])

# ======================================传统ResNet end ======================================

# ====================================== 注意力机制模块 start =================================
class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out


# ====================================== 注意力机制模块 end ===================================

class ChannelAttention_(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention_, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention_(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


if __name__ == '__main__':
    model = ResNet(block='Bottleneck', num_blocks=[3, 4, 6, 3], activation='Swish',
                   mhsa4=True, resolution=[32, 32], heads4=8, D=True)

    x = torch.randn(1, 3, 32, 32)
    out = model(x)
    print(out.shape)
    print(model)
