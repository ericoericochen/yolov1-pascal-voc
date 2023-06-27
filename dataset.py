import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.datasets import VOCDetection
from torchvision import transforms


class PascalVOC(Dataset):
    """
    PASCAL VOC Dataset with images resized to 448 x 448 and annotations transformed to YOLOv1 targets
    """

    def __init__(self, pascal_voc: VOCDetection):
        print(f"TRANSFORMING PASCAL VOC")

        self.S = 7  # grid size
        self.C = 20  # num categories
        self.IMAGE_SIZE = 448

        self.dataset = pascal_voc  # original dataset

        # resize to 448 x 448 image and normalize
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((self.IMAGE_SIZE, self.IMAGE_SIZE)),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _target_from_annotation(self, annotation):
        pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, annotation = self.dataset.__getitem__(index)

        # transform image and annotation
        image = self.transform(image)

        return image, annotation
