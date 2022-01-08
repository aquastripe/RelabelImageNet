import json
from collections import namedtuple
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch import Tensor
from torchvision.models.resnet import ResNet, Bottleneck, resnet50

from functional import class_activation_map


class ResNetWFeat(ResNet):

    def _forward_impl(self, x: Tensor) -> tuple[Any, Any]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        features = self.layer4(x)

        x = self.avgpool(features)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, features


def plot_label_map(label_maps, filename):
    fig, ax = plt.subplots(ncols=len(label_maps), figsize=(7.68, 2.66), tight_layout=True)
    for col, label_map in enumerate(label_maps):
        ax[col].imshow(label_map.image)
        if col != 0:
            ax[col].set_title(f'{label_map.label}/{label_map.score:.4f}')

        ax[col].axis('off')

    fig.savefig(filename)
    plt.close(fig)


def load_resnet50():
    model_r50_feat = ResNetWFeat(Bottleneck, [3, 4, 6, 3])
    model_r50 = resnet50(pretrained=True)
    model_r50_feat.load_state_dict(model_r50.state_dict())
    model_r50_feat.eval()
    return model_r50_feat


def load_image_files():
    image_folder = Path('examples')
    image_files = [file for file in image_folder.glob('*')]
    return image_files


def load_labels():
    with open('imagenet-labels.json') as f:
        labels = json.load(f)
    return labels


def preprocess(IMAGE_SIZE, image_files):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    transforms = T.Compose([
        T.ToTensor(),
        normalize,
    ])
    images_original = []
    images_input = []
    for image_file in image_files:
        with Image.open(image_file).convert('RGB') as image:
            origin = T.Resize((IMAGE_SIZE, IMAGE_SIZE))(image)
            images_original.append(origin)

            image_input = transforms(origin)
            images_input.append(image_input)

    images_input = torch.stack(images_input)
    return images_original, images_input


def plot_label_map_from(IMAGE_SIZE, K, image_files, labels, model_r50_feat):
    images_original, images_input = preprocess(IMAGE_SIZE, image_files)
    scores, features = model_r50_feat(images_input)
    outputs = [score.topk(K) for score in scores]
    scores = torch.stack([output.values for output in outputs])
    indices = torch.stack([output.indices for output in outputs])
    cam = class_activation_map(features, model_r50_feat.fc.weight, indices)
    cam_up = F.interpolate(cam, size=IMAGE_SIZE)
    LabelMap = namedtuple('LabelMap', 'image, label, score')
    for i, image_original in enumerate(images_original):
        label_maps = [LabelMap(image=image_original, label='', score=1.)]
        for k in range(K):
            label_maps.append(LabelMap(image=cam_up[i, k], label=labels[indices[i, k]], score=scores[i, k]))

        plot_label_map(label_maps, f'example_{i}.png')


@torch.no_grad()
def main():
    IMAGE_SIZE = 224
    K = 2

    labels = load_labels()
    image_files = load_image_files()
    model_r50_feat = load_resnet50()
    plot_label_map_from(IMAGE_SIZE, K, image_files, labels, model_r50_feat)


if __name__ == '__main__':
    main()
