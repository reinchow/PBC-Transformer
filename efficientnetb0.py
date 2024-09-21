# 使用efficientNetb0
import torch
import torch.nn as nn
import torchvision.models as models

MAX_LENGTH = 10

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self, encoded_image_size=14):
        super(EncoderCNN, self).__init__()
        good_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        modules = list(good_model.children())[:-2]
        self.good_model = nn.Sequential(*modules)



        self.fine_tune()

    def forward(self, images):
        #print(self.good_model)
        # image.shape = torch.Size([8, 3, 256, 256])
        out = self.good_model(images)
        # out,shape = torch.Size([8, 2048, 8, 8])
        out = out.permute(0, 2, 3, 1)
        return out

    def fine_tune(self, fine_tune=True):
        for child in self.good_model.children():
            for param in child.parameters():
                param.requires_grad = fine_tune
    # def fine_tune(self, fine_tune=True):
    #     for p in self.good_model.parameters():
    #         p.requires_grad = False
    #     num_children = len(list(self.good_model.children()))
    #     print("Number of direct children in self.good_model:", num_children)
    #     for c in list(self.good_model.children())[5:]:
    #         for p in c.parameters():
    #             p.requires_grad = fine_tune


