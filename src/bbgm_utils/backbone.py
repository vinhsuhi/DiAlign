import torch.nn as nn
from torchvision import models

'''
checked!
'''

class VGG16_base(nn.Module):
    def __init__(self, batch_norm=True):
        super(VGG16_base, self).__init__()
        self.node_layers, self.edge_layers, self.final_layers = self.get_backbone(batch_norm)
        

    def forward(self, *input):
        raise NotImplementedError

    @staticmethod
    def get_backbone(batch_norm):
        """
        Get pretrained VGG16 models for feature extraction.
        :return: feature sequence
        """
        if batch_norm:
            model = models.vgg16_bn(pretrained=True)
        else:
            model = models.vgg16(pretrained=True)
            
        for param in model.parameters():
            param.requires_grad = False

        conv_layers = nn.Sequential(*list(model.features.children()))

        conv_list = node_list = edge_list = []

        # get the output of relu4_2(node features) and relu5_1(edge features)
        cnt_m, cnt_r = 1, 0
        for layer, module in enumerate(conv_layers):
            if isinstance(module, nn.Conv2d):
                cnt_r += 1
            if isinstance(module, nn.MaxPool2d):
                cnt_r = 0
                cnt_m += 1
            conv_list += [module]

            if cnt_m == 4 and cnt_r == 2 and isinstance(module, nn.ReLU):
                node_list = conv_list
                conv_list = []
            elif cnt_m == 5 and cnt_r == 1 and isinstance(module, nn.ReLU):
                edge_list = conv_list
                conv_list = []

        assert len(node_list) > 0 and len(edge_list) > 0

        # Set the layers as a nn.Sequential module
        node_layers = nn.Sequential(*node_list)
        edge_layers = nn.Sequential(*edge_list)
        final_layers = nn.Sequential(*conv_list, nn.AdaptiveMaxPool2d(1, 1))

        return node_layers, edge_layers, final_layers


class VGG16_bn(VGG16_base):
    def __init__(self):
        super(VGG16_bn, self).__init__(True)
