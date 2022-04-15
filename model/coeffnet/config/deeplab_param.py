import torch
import collections

deeplab_param = collections.OrderedDict({
    'backbone.conv1': torch.Size([64, 3, 7, 7]),
    'backbone.layer1.0.conv1': torch.Size([64, 64, 3, 3]),
    'backbone.layer1.0.conv2': torch.Size([64, 64, 3, 3]),
    'backbone.layer1.1.conv1': torch.Size([64, 64, 3, 3]),
    'backbone.layer1.1.conv2': torch.Size([64, 64, 3, 3]),
    'backbone.layer2.0.conv1': torch.Size([128, 64, 3, 3]),
    'backbone.layer2.0.conv2': torch.Size([128, 128, 3, 3]),
    'backbone.layer2.0.downsample.0': torch.Size([128, 64, 1, 1]),
    'backbone.layer2.1.conv1': torch.Size([128, 128, 3, 3]),
    'backbone.layer2.1.conv2': torch.Size([128, 128, 3, 3]),
    'backbone.layer3.0.conv1': torch.Size([256, 128, 3, 3]),
    'backbone.layer3.0.conv2': torch.Size([256, 256, 3, 3]),
    'backbone.layer3.0.downsample.0': torch.Size([256, 128, 1, 1]),
    'backbone.layer3.1.conv1': torch.Size([256, 256, 3, 3]),
    'backbone.layer3.1.conv2': torch.Size([256, 256, 3, 3]),
    'backbone.layer4.0.conv1': torch.Size([512, 256, 3, 3]),
    'backbone.layer4.0.conv2': torch.Size([512, 512, 3, 3]),
    'backbone.layer4.0.downsample.0': torch.Size([512, 256, 1, 1]),
    'backbone.layer4.1.conv1': torch.Size([512, 512, 3, 3]),
    'backbone.layer4.1.conv2': torch.Size([512, 512, 3, 3]),
    'backbone.layer4.2.conv1': torch.Size([512, 512, 3, 3]),
    'backbone.layer4.2.conv2': torch.Size([512, 512, 3, 3]),
    'aspp.aspp1.atrous_conv': torch.Size([256, 512, 1, 1]),
    'aspp.aspp2.atrous_conv': torch.Size([256, 512, 3, 3]),
    'aspp.aspp3.atrous_conv': torch.Size([256, 512, 3, 3]),
    'aspp.aspp4.atrous_conv': torch.Size([256, 512, 3, 3]),
    'aspp.global_avg_pool.1': torch.Size([256, 512, 1, 1]),
    'aspp.conv1': torch.Size([256, 1280, 1, 1]),
    'decoder.conv1': torch.Size([48, 64, 1, 1]),
    'decoder.last_conv.0': torch.Size([256, 304, 3, 3]),
    'decoder.last_conv.4': torch.Size([256, 256, 3, 3]),
    'decoder.last_conv.8': torch.Size([1, 256, 1, 1])
})

deeplab_param_decoder = collections.OrderedDict({
    'aspp.aspp1.atrous_conv': torch.Size([256, 512, 1, 1]),
    'aspp.aspp2.atrous_conv': torch.Size([256, 512, 3, 3]),
    'aspp.aspp3.atrous_conv': torch.Size([256, 512, 3, 3]),
    'aspp.aspp4.atrous_conv': torch.Size([256, 512, 3, 3]),
    'aspp.global_avg_pool.1': torch.Size([256, 512, 1, 1]),
    'aspp.conv1': torch.Size([256, 1280, 1, 1]),
    'decoder.conv1': torch.Size([48, 64, 1, 1]),
    'decoder.last_conv.0': torch.Size([256, 304, 3, 3]),
    'decoder.last_conv.4': torch.Size([256, 256, 3, 3]),
    'decoder.last_conv.8': torch.Size([1, 256, 1, 1])
})