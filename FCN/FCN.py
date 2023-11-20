# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import model_zoo
from torchvision import models
from torchvision.models.vgg import VGG


class FCN32s(nn.Module):

    def __init__(self, pretrained_net, n_classes, bilinear=False):
        super().__init__()

        self.n_channels = pretrained_net.n_channels  ## for convenience and consistency with other models, might be used from outside the class.
        self.bilinear = bilinear  ## for convenience and consistency with other models, might be used from outside the class.

        self.n_classes = n_classes
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_classes, kernel_size=1)

        if bilinear:
            raise NotImplementedError(
                'Bilinear interpolation not implemented for FCN32s. Only transposed convolutions are used. Set bilinear=False to fix this.')

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']

        score = self.bn1(self.relu(self.deconv1(x5)))  # size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)  # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)


class FCN16s(nn.Module):

    def __init__(self, pretrained_net, n_classes, bilinear=False):
        super().__init__()

        self.n_channels = pretrained_net.n_channels
        self.bilinear = bilinear

        self.n_classes = n_classes
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_classes, kernel_size=1)

        if bilinear:
            raise NotImplementedError(
                'Bilinear interpolation not implemented for FCN16s. Only transposed convolutions are used. Set bilinear=False to fix this.')

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']
        x4 = output['x4']

        score = self.relu(self.deconv1(x5))
        score = self.bn1(score + x4)
        score = self.bn2(self.relu(self.deconv2(score)))
        score = self.bn3(self.relu(self.deconv3(score)))
        score = self.bn4(self.relu(self.deconv4(score)))
        score = self.bn5(self.relu(self.deconv5(score)))
        score = self.classifier(score)

        return score


class FCN8s(nn.Module):

    def __init__(self, pretrained_net, n_classes, bilinear=False):
        super().__init__()

        self.n_channels = pretrained_net.n_channels
        self.bilinear = bilinear

        self.n_classes = n_classes
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_classes, kernel_size=1)

        if bilinear:
            raise NotImplementedError(
                'Bilinear interpolation not implemented for FCN8s. Only transposed convolutions are used. Set bilinear=False to fix this.')

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']
        x4 = output['x4']
        x3 = output['x3']

        score = self.relu(self.deconv1(x5))
        score = self.bn1(score + x4)
        score = self.relu(self.deconv2(score))
        score = self.bn2(score + x3)
        score = self.bn3(self.relu(self.deconv3(score)))
        score = self.bn4(self.relu(self.deconv4(score)))
        score = self.bn5(self.relu(self.deconv5(score)))
        score = self.classifier(score)

        return score


class FCNs(nn.Module):

    def __init__(self, pretrained_net, n_classes, bilinear=False):
        super().__init__()

        self.n_channels = pretrained_net.n_channels
        self.bilinear = bilinear

        self.n_classes = n_classes
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_classes, kernel_size=1)

        if bilinear:
            raise NotImplementedError(
                'Bilinear interpolation not implemented for FCNs. Only transposed convolutions are used. Set bilinear=False to fix this.')

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']
        x4 = output['x4']
        x3 = output['x3']
        x2 = output['x2']
        x1 = output['x1']

        score = self.bn1(self.relu(self.deconv1(x5)))
        score = score + x4
        score = self.bn2(self.relu(self.deconv2(score)))
        score = score + x3
        score = self.bn3(self.relu(self.deconv3(score)))
        score = score + x2
        score = self.bn4(self.relu(self.deconv4(score)))
        score = score + x1
        score = self.bn5(self.relu(self.deconv5(score)))
        score = self.classifier(score)

        return score


class VGGNet(nn.Module):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False,
                 n_channels=3):
        super(VGGNet, self).__init__()

        self.n_channels = n_channels

        if pretrained:
            assert n_channels == 3, "pretrained model is trained on 3 channel images, please use n_channels=3 when using pretrained weights."

        self.features = make_layers(cfg[model], n_channels=n_channels)
        self.ranges = ranges[model]

        if pretrained:

            if model in vgg_model_urls:
                weights = vgg_model_urls[model]
                self.load_state_dict(model_zoo.load_url(weights), strict=False)
            else:
                raise ValueError("Invalid VGG model name")

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        if remove_fc:
            self._remove_fc()

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def _remove_fc(self):
        """
            Removes the fully connected layer (classifier) from the VGGNet model.

            This is useful for models where the fully connected layers are not required,
            such as when adapting VGGNet for a Fully Convolutional Network (FCN).
            The method checks for the existence of the classifier (e.g., built from VGG16 classification model) attribute before
            attempting to delete it to prevent AttributeError.
            """
        if hasattr(self, 'classifier'):
            del self.classifier

    def forward(self, x):
        output = {}
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d" % (idx + 1)] = x
        return output


ranges = {
    'vgg11': ((0, 3), (3, 6), (6, 11), (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

vgg_model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


def make_layers(cfg, batch_norm=False, n_channels=3):
    layers = []
    # in_channels = 3
    in_channels = n_channels
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def get_fcn_model(arch_type, backbone, pretrained, n_classes, requires_grad, bilinear=False, n_channels=3):
    if backbone.lower() in cfg.keys():
        vgg_model = VGGNet(requires_grad=requires_grad, model=backbone, pretrained=pretrained, n_channels=n_channels)
    else:
        raise NotImplementedError(f"{backbone} is not a valid backbone for FCN")

    if arch_type.lower() == 'fcn32s':
        return FCN32s(pretrained_net=vgg_model, n_classes=n_classes, bilinear=bilinear)
    elif arch_type.lower() == 'fcn16s':
        return FCN16s(pretrained_net=vgg_model, n_classes=n_classes, bilinear=bilinear)
    elif arch_type.lower() == 'fcn8s':
        return FCN8s(pretrained_net=vgg_model, n_classes=n_classes, bilinear=bilinear)
    elif arch_type.lower() == 'fcns':
        return FCNs(pretrained_net=vgg_model, n_classes=n_classes, bilinear=bilinear)
    else:
        raise NotImplementedError(f"{arch_type} is not a valid FCN architecture")


if __name__ == "__main__":

    n_classes = 21
    # Create FCN8s model with pre-trained VGG16
    vgg_model = VGGNet(requires_grad=True, model='vgg11', pretrained=True, n_channels=3)
    fcn_model = FCN32s(pretrained_net=vgg_model, n_classes=n_classes, bilinear=False)
    fcn_model.eval()

    image = torch.randn(1, 3, 224, 224)  # Example input
    with torch.no_grad():
        output = fcn_model(image)
    print(output.size())  # Should be torch.Size([1, n_classes, 224, 224])

    print(vgg_model)
    print('-' * 80)
    print(fcn_model)
    print('-' * 80)
    print('vgg16_model trainable parms',
          sum(p.numel() for p in vgg_model.parameters() if p.requires_grad))  ## number of trainable parameters
    print('fcn model trainable parms',
          sum(p.numel() for p in fcn_model.parameters() if p.requires_grad))  ## number of trainable parameters

    print("Pass quick check")

    batch_size, n_class, h, w = 10, 20, 160, 160

    # test output size
    vgg_model = VGGNet(requires_grad=True, model='vgg16')
    input = torch.autograd.Variable(torch.randn(batch_size, 3, 224, 224))
    output = vgg_model(input)
    assert output['x5'].size() == torch.Size([batch_size, 512, 7, 7])

    fcn_model = FCN32s(pretrained_net=vgg_model, n_classes=n_class)
    input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    output = fcn_model(input)
    assert output.size() == torch.Size([batch_size, n_class, h, w])

    fcn_model = FCN16s(pretrained_net=vgg_model, n_classes=n_class)
    input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    output = fcn_model(input)
    assert output.size() == torch.Size([batch_size, n_class, h, w])

    fcn_model = FCN8s(pretrained_net=vgg_model, n_classes=n_class)
    input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    output = fcn_model(input)
    assert output.size() == torch.Size([batch_size, n_class, h, w])

    fcn_model = FCNs(pretrained_net=vgg_model, n_classes=n_class)
    input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    output = fcn_model(input)
    assert output.size() == torch.Size([batch_size, n_class, h, w])

    print("Pass size check")

    # test a random batch, loss should decrease
    fcn_model = FCNs(pretrained_net=vgg_model, n_classes=n_class)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-3, momentum=0.9)
    input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    y = torch.autograd.Variable(torch.randn(batch_size, n_class, h, w), requires_grad=False)
    for iter in range(10):
        optimizer.zero_grad()
        output = fcn_model(input)
        output = nn.functional.sigmoid(output)
        loss = criterion(output, y)
        loss.backward()
        # print("iter{}, loss {}".format(iter, loss.data[0]))
        print("iter{}, loss {}".format(iter, loss.item()))

        optimizer.step()
