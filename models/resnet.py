import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.activation import avaliable_activations
from models.normalization import avaliable_normalizations

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)


def conv(in_planes, out_planes, stride=1, kernel_size=3, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
                     groups=groups, bias=False)


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C // g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)


class SqueezeExcitationLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SqueezeExcitationLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class PreActBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride, kernel_size=3, activation='ReLU',
                 normalization='BatchNorm', **kwargs):
        super(PreActBasicBlock, self).__init__()
        self.act = avaliable_activations[activation](inplace=True)
        self.bn1 = avaliable_normalizations[normalization](in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
                               bias=False)
        self.bn2 = avaliable_normalizations[normalization](planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=False)
            )

    def forward(self, x):
        out = self.act(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.act(self.bn2(out)))
        return out + shortcut


class RobustResBlock(nn.Module):
    expansion = 4

    # BN + Skip Connection
    def __init__(self, in_planes, planes, stride=1, kernel_size=3, scales=4, base_width=160, cardinality=1,
                 activation='ReLU', normalization='BatchNorm', se_reduction=16, **kwargs):
        super(RobustResBlock, self).__init__()
        width = int(math.floor(planes * (base_width / 160))) * cardinality
        self.act = avaliable_activations[activation](inplace=True)
        self.bn1 = avaliable_normalizations[normalization](in_planes)
        self.conv1 = nn.Conv2d(in_planes, width * scales, kernel_size=1, bias=False)
        if stride > 1:
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        self.bn2 = avaliable_normalizations[normalization](width * scales)
        if scales == 1:
            self.conv2 = conv(width, width, stride=stride, kernel_size=kernel_size, groups=cardinality)
            self.bn3 = avaliable_normalizations[normalization](width)
        else:
            self.conv2 = nn.ModuleList(
                [conv(width, width, stride=stride, kernel_size=kernel_size, groups=cardinality) for _ in
                 range(scales - 1)])
            self.bn3 = nn.ModuleList([avaliable_normalizations[normalization](width) for _ in range(scales - 1)])
        self.conv3 = nn.Conv2d(width * scales, planes * self.expansion, kernel_size=1, bias=False)
        self.se_bn = avaliable_normalizations[normalization](planes * self.expansion)
        self.se = SqueezeExcitationLayer(planes * self.expansion, reduction=se_reduction)

        if stride > 1 or (in_planes != planes * self.expansion):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )
        self.stride = stride
        self.scales = scales

    def forward(self, x):
        out = self.act(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.act(self.bn2(self.conv1(x)))

        if self.scales == 1:
            out = self.act(self.bn3(self.conv2(out)))
        else:
            xs = torch.chunk(out, self.scales, 1)
            ys = []
            for s in range(self.scales - 1):
                if s == 0 or self.stride > 1:  # if stride > 1, acts like normal bottleneck, without adding
                    input = xs[s]
                else:
                    input = xs[s] + ys[-1]
                ys.append(self.act(self.bn3[s](self.conv2[s](input))))
            ys.append(xs[s + 1] if self.stride == 1 else self.pool(xs[s + 1]))
            out = torch.cat(ys, 1)
        out = self.conv3(out)
        return out + shortcut + self.se(self.se_bn(out))
        # return out + shortcut


class NetworkBlock(nn.Module):

    # 一个NetworkBlock就是论文中的一个stage
    def __init__(self, nb_layers, in_planes, out_planes, stride, kernel_size=3, block_type='basic_block',
                 cardinality=8, base_width=64, scales=4, activation='ReLU', normalization='BatchNorm',
                 se_reduction=16, ):
        super(NetworkBlock, self).__init__()
        self.block_type = block_type
        if block_type == 'basic_block':
            block = PreActBasicBlock
        elif block_type == 'robust_res_block':
            block = RobustResBlock
        else:
            raise ('Unknown block: %s' % block_type)

        # 制作网络块
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, kernel_size, activation, normalization,
            cardinality=cardinality, base_width=base_width, scales=scales, se_reduction=se_reduction)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, kernel_size, activation,
                    normalization, cardinality=8, base_width=64, scales=4, se_reduction=16, ):
        layers = []
        for i in range(int(nb_layers)):
            if i == 0:
                in_planes = in_planes
            else:
                if self.block_type == 'robust_res_block':
                    in_planes = out_planes * 4
                else:
                    in_planes = out_planes

            layers.append(block(in_planes, out_planes, i == 0 and stride or 1, kernel_size=kernel_size,
                                activation=activation, normalization=normalization,
                                cardinality=cardinality, base_width=base_width,
                                scales=scales, se_reduction=se_reduction)  # 第一次使用传入的stride，其他步长为0
                          )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class PreActResNet(nn.Module):
    def __init__(self, num_classes=10, channel_configs=(16, 160, 320, 640),
                 depth_configs=(5, 5, 5), drop_rate_config=(0.0, 0.0, 0.0),
                 stride_config=(1, 2, 2), zero_init_residual=False, stem_stride=1,
                 kernel_size_configs=(3, 3, 3),
                 block_types=('basic_block', 'basic_block', 'basic_block'),
                 activations=('ReLU', 'ReLU', 'ReLU'),
                 normalizations=('BatchNorm', 'BatchNorm', 'BatchNorm'),
                 use_init=True, cardinality=8, base_width=64, scales=4,
                 se_reduction=16, pre_process=False):
        super(PreActResNet, self).__init__()
        assert len(channel_configs) - 1 == len(depth_configs) == len(stride_config)
        self.channel_configs = channel_configs
        self.depth_configs = depth_configs
        self.stride_config = stride_config
        self.get_feature = False
        self.get_stem_out = False
        self.block_types = block_types

        self.pre_process = pre_process
        # if True, add data normalization, this is only used for advanced training on CIFAR-10
        if pre_process:
            self.mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
            self.std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
            self.mean_cuda = None
            self.std_cuda = None

        self.stem_conv = nn.Conv2d(3, channel_configs[0], kernel_size=3, stride=stem_stride, padding=1, bias=False)
        self.blocks = nn.ModuleList([])

        out_planes = channel_configs[0]
        for i, stride in enumerate(stride_config):
            self.blocks.append(NetworkBlock(nb_layers=depth_configs[i],
                                            in_planes=out_planes,
                                            out_planes=channel_configs[i + 1],  # channel_configs有四个，第一个是上面普通卷积的输出
                                            stride=stride,
                                            kernel_size=kernel_size_configs[i],
                                            block_type=block_types[i],
                                            activation=activations[i],
                                            normalization=normalizations[i],
                                            cardinality=cardinality,
                                            base_width=base_width,
                                            scales=scales,
                                            se_reduction=se_reduction,
                                            ))
            if block_types[i] == 'robust_res_block':
                out_planes = channel_configs[i + 1] * 4
            else:
                out_planes = channel_configs[i + 1]

        # global average pooling and classifier
        self.norm1 = avaliable_normalizations[normalizations[-1]](out_planes)
        self.act1 = avaliable_activations[activations[-1]](inplace=True)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(out_planes, num_classes)
        self.fc_size = out_planes

        # 初始化神经网络权重的部分
        if use_init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, x):

        # 设定了就会对输入x进行标准化
        if self.pre_process:
            if x.is_cuda:
                if self.mean_cuda is None:
                    self.mean_cuda = self.mean.cuda()
                    self.std_cuda = self.std.cuda()
                x = (x - self.mean_cuda) / self.std_cuda
            else:
                x = (x - self.mean) / self.std

        out = self.stem_conv(x)
        # 三个stage
        for i, block in enumerate(self.blocks):
            out = block(out)
        out = self.act1(self.norm1(out))
        out = self.global_pooling(out)
        out = out.view(-1, self.fc_size)
        out = self.fc(out)
        return out


# ====================================== 传统ResNet start ======================================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, mhsa=False, heads=4, resolution=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, mhsa=False, heads=4, resolution=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # 把这个改造成支持注意力机制(中间层)
        if not mhsa:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            self.conv2 = nn.ModuleList()
            self.conv2.append(MHSA(planes, width=int(resolution[0]), height=int(resolution[1]), heads=heads))
            if stride == 2:
                self.conv2.append(nn.AvgPool2d(2, 2))
            self.conv2 = nn.Sequential(*self.conv2)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, resolution=(32, 32), heads=4, mhsa=False):
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

        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, heads=heads, mhsa=mhsa)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, heads=4, mhsa=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, mhsa=mhsa, heads=heads, resolution=self.resolution))
            if stride == 2:  # 为注意力机制改变长款输入(因为只有每个stage的中间卷积的步长为2时会改变图片长宽)
                self.resolution[0] /= 2
                self.resolution[1] /= 2
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
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


if __name__ == '__main__':
    import util
    from torchprofile import profile_macs

    stride_config = [1, 2, 2]
    activations = ('ReLU', 'ReLU', 'ReLU')
    normalizations = ('BatchNorm', 'BatchNorm', 'BatchNorm')

    # # WRN-28-10 with RobustResBlock
    # depth, width_mult = [4, 4, 4], [10, 10, 10]
    # block_types = ['robust_res_block', 'robust_res_block', 'robust_res_block']
    # scales, base_width, cardinality, se_reduction = 8, 10, 4, 64

    # # WRN-A1
    # depth, width_mult = [14, 14, 7], [5, 7, 3]
    # block_types = ['basic_block', 'basic_block', 'basic_block']
    # scales, base_width, cardinality, se_reduction = None, None, None, None

    # WRN-A4
    depth, width_mult = [27, 28, 13], [10, 14, 6]
    block_types = ['basic_block', 'basic_block', 'basic_block']
    scales, base_width, cardinality, se_reduction = None, None, None, None

    channels = [16, 16 * width_mult[0], 32 * width_mult[1], 64 * width_mult[2]]
    model = PreActResNet(
        num_classes=10,
        channel_configs=channels, depth_configs=depth,
        stride_config=stride_config, stem_stride=1,
        block_types=block_types,
        activations=activations,
        normalizations=normalizations,
        use_init=True,
        cardinality=cardinality,
        base_width=base_width,
        scales=scales,
        se_reduction=se_reduction,
    )
    # print(model)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    data = torch.rand(1, 3, 32, 32)
    out = model(data)
    flops = profile_macs(model, data) / 1e6
    print('depth@{}-{}-width@{}-{}-channels@{}-block@{}-params = {:.3f}, flops = {:.3f}'.format(
        sum(depth), depth, sum(width_mult), width_mult, channels, block_types[0], param_count, flops / 1000))

    # checkpoint = util.load_model(filename="/Users/luzhicha/Dropbox/2023/github/revisit-resnet-adv-robust/exps/"
    #                                       "wrn-a4-advanced-silu-apex-500k/checkpoints/weights-last-new.pt",
    #                              model=model,
    #                              optimizer=None,
    #                              alpha_optimizer=None,
    #                              scheduler=None)
