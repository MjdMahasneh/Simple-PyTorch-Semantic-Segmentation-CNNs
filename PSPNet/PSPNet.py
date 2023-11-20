import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch import optim
from torch.autograd import Variable

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    # 'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    # 'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    # 'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    # 'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    # 'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(BasicBlock, self).__init__()
        # The first conv layer of the BasicBlock.
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=rate, dilation=rate, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        # The second conv layer of the BasicBlock.
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=rate, dilation=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.rate = rate

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, n_channels, block, layers, os=16, pretrained=False, arch=None):

        if pretrained:
            assert n_channels == 3, 'pretrained backbone model is only available for 3 input channels'

        assert arch in ['resnet18', 'resnet34', 'resnet50', 'resnet101'], 'resnet not supported'
        self.arch = arch

        self.inplanes = 64
        super(ResNet, self).__init__()
        if os == 16:
            strides = [1, 2, 2, 1]
            rates = [1, 1, 1, 2]
            blocks = [1, 2, 4]
        elif os == 8:
            strides = [1, 2, 1, 1]
            rates = [1, 1, 2, 2]
            blocks = [1, 2, 1]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], rate=rates[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], rate=rates[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], rate=rates[2])
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], rate=rates[3])

        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks=[1, 2, 4], stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate=blocks[0] * rate, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, rate=blocks[i] * rate))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):

        print('loading pretrained model from {}'.format(model_urls[self.arch]))
        pretrain_dict = model_zoo.load_url(model_urls[self.arch])
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)

        self.load_state_dict(state_dict, strict=True)


def ResNet101(n_channels=3, os=16, pretrained=False):
    model = ResNet(n_channels, Bottleneck, [3, 4, 23, 3], os, pretrained=pretrained, arch='resnet101')
    return model


def ResNet50(n_channels=3, os=16, pretrained=False):
    model = ResNet(n_channels, Bottleneck, [3, 4, 6, 3], os, pretrained=pretrained, arch='resnet50')
    return model


def ResNet34(n_channels=3, os=16, pretrained=False):
    model = ResNet(n_channels, BasicBlock, [3, 4, 6, 3], os, pretrained=pretrained, arch='resnet34')
    return model


def ResNet18(n_channels=3, os=16, pretrained=False):
    model = ResNet(n_channels, BasicBlock, [2, 2, 2, 2], os, pretrained=pretrained, arch='resnet18')
    return model


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, bias=False):
        super(ConvBlock, self).__init__()
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1)) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.conv(x)
        return out


def upsample(input, size=None, scale_factor=None, align_corners=False, bilinear=True):
    if bilinear is not True:
        raise NotImplementedError('only bilinear upsampling is implemented.')
    else:
        upsampling = 'bilinear'

    out = F.interpolate(input, size=size, scale_factor=scale_factor, mode=upsampling, align_corners=align_corners)
    return out


class PyramidPooling(nn.Module):
    def __init__(self, in_channels):
        super(PyramidPooling, self).__init__()
        self.pooling_size = [1, 2, 3, 6]
        self.channels = in_channels // 4

        self.pool1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.pooling_size[0]),
            ConvBlock(in_channels, self.channels, kernel_size=1),
        )

        self.pool2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.pooling_size[1]),
            ConvBlock(in_channels, self.channels, kernel_size=1),
        )

        self.pool3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.pooling_size[2]),
            ConvBlock(in_channels, self.channels, kernel_size=1),
        )

        self.pool4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.pooling_size[3]),
            ConvBlock(in_channels, self.channels, kernel_size=1),
        )

    def forward(self, x):
        out1 = self.pool1(x)
        out1 = upsample(out1, size=x.size()[-2:])

        out2 = self.pool2(x)
        out2 = upsample(out2, size=x.size()[-2:])

        out3 = self.pool3(x)
        out3 = upsample(out3, size=x.size()[-2:])

        out4 = self.pool4(x)
        out4 = upsample(out4, size=x.size()[-2:])

        out = torch.cat([x, out1, out2, out3, out4], dim=1)

        return out


class PSPNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=21, backbone='resnet50', pretrained=True, bilinear=True, os=16):
        super(PSPNet, self).__init__()
        # self.out_channels = 2048

        if bilinear is not True:
            raise NotImplementedError('only bilinear upsampling is implemented for PSPNet')

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        if backbone == 'resnet18':
            self.backbone = ResNet18(n_channels=n_channels, os=os, pretrained=pretrained)
            self.out_channels = 512
        elif backbone == 'resnet34':
            self.backbone = ResNet34(n_channels=n_channels, os=os, pretrained=pretrained)
            self.out_channels = 512
        elif backbone == 'resnet50':
            self.backbone = ResNet50(n_channels=n_channels, os=os, pretrained=pretrained)
            self.out_channels = 2048
        elif backbone == 'resnet101':
            self.backbone = ResNet101(n_channels=n_channels, os=os, pretrained=pretrained)
            self.out_channels = 2048
        else:
            raise NotImplementedError

        self.stem = nn.Sequential(
            *list(self.backbone.children())[:4],
        )
        self.block1 = self.backbone.layer1
        self.block2 = self.backbone.layer2
        self.block3 = self.backbone.layer3
        self.block4 = self.backbone.layer4
        self.low_level_features_conv = ConvBlock(512, 64, kernel_size=3)

        self.depth = self.out_channels // 4
        self.pyramid_pooling = PyramidPooling(self.out_channels)

        self.decoder = nn.Sequential(
            ConvBlock(self.out_channels * 2, self.depth, kernel_size=3),
            nn.Dropout(0.1),
            nn.Conv2d(self.depth, n_classes, kernel_size=1),
        )

        self.aux = nn.Sequential(
            ConvBlock(self.out_channels // 2, self.depth // 2, kernel_size=3),
            nn.Dropout(0.1),
            nn.Conv2d(self.depth // 2, n_classes, kernel_size=1),
        )

        self.semantic_criterion = nn.CrossEntropyLoss(ignore_index=255, weight=None).cuda()
        self.auxiliary_criterion = nn.CrossEntropyLoss(ignore_index=255, weight=None).cuda()

    def forward(self, images, label=None):
        x = images
        out = self.stem(x)
        out1 = self.block1(out)
        out2 = self.block2(out1)
        out3 = self.block3(out2)

        aux_out = self.aux(out3)

        aux_out = upsample(aux_out, size=images.size()[-2:], align_corners=True)

        out4 = self.block4(out3)

        out = self.pyramid_pooling(out4)
        out = self.decoder(out)
        out = upsample(out, size=x.size()[-2:])

        out = upsample(out, size=images.size()[-2:], align_corners=True)

        if label is not None:
            semantic_loss = self.semantic_criterion(out, label)
            aux_loss = self.auxiliary_criterion(aux_out, label)
            total_loss = semantic_loss + 0.4 * aux_loss
            return out, total_loss

        return out


if __name__ == "__main__":
    model = PSPNet(n_channels=3, n_classes=21, backbone='resnet50', pretrained=True, bilinear=True, os=16)
    model.eval()
    image = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        output = model(image)
    print(output.size())

    print(model)
    print('number of trainable parameters = {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    batch_size, n_class, h, w = 10, 21, 160, 160

    # Initialize PSPNet model
    pspnet_model = PSPNet(n_channels=3, n_classes=n_class, backbone='resnet50', pretrained=True)

    # Test output size
    input_tensor = Variable(torch.randn(batch_size, 3, h, w))
    output = pspnet_model(input_tensor)
    assert output.size() == torch.Size([batch_size, n_class, h, w]), "Output size mismatch"

    print("Pass size check")

    # Test a random batch, loss should decrease
    criterion = nn.CrossEntropyLoss()  # Adjust loss function if needed
    optimizer = optim.SGD(pspnet_model.parameters(), lr=1e-3, momentum=0.9)

    for iter in range(10):
        optimizer.zero_grad()
        input_tensor = Variable(torch.randn(batch_size, 3, h, w))
        y = Variable(torch.randint(0, n_class, (batch_size, h, w)), requires_grad=False)  # Random target
        output = pspnet_model(input_tensor)
        loss = criterion(output, y)
        loss.backward()
        print(f"iter {iter}, loss {loss.item()}")

        optimizer.step()
