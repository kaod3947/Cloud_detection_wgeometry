import torch
import torch.nn as nn
import torchvision.models as models


def build_model(num_classes=4):

    model = models.segmentation.deeplabv3_resnet101(
        pretrained=False,
        progress=True,
        num_classes=num_classes
    )

    # 🔥 4채널 입력으로 수정
    model.backbone.conv1 = nn.Conv2d(
        4, 64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    return model
