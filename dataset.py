import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
from torchvision.datasets import VOCDetection
from torchvision import transforms


class PascalVOC(Dataset):
    """
    PASCAL VOC Dataset with images resized to 448 x 448 and annotations transformed to YOLOv1 targets
    """

    categories = {
        "aeroplane": 0,
        "bicycle": 1,
        "bird": 2,
        "boat": 3,
        "bottle": 4,
        "bus": 5,
        "car": 6,
        "cat": 7,
        "chair": 8,
        "cow": 9,
        "diningtable": 10,
        "dog": 11,
        "horse": 12,
        "motorbike": 13,
        "person": 14,
        "pottedplant": 15,
        "sheep": 16,
        "sofa": 17,
        "train": 18,
        "tvmonitor": 19,
    }
    
    categories_list = ["aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",]

    def __init__(self, pascal_voc: VOCDetection):
        self.S = 7  # grid size
        self.C = 20  # num categories
        self.IMAGE_SIZE = 224
        self.GRID_SIZE = self.IMAGE_SIZE / self.S

        self.dataset = pascal_voc  # original dataset

        # resize to 448 x 448 image and normalize
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((self.IMAGE_SIZE, self.IMAGE_SIZE), antialias=True),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _one_hot(self, name: str) -> torch.Tensor:
        i = PascalVOC.categories[name]
        one_hot = F.one_hot(torch.tensor(i), num_classes=self.C)

        return one_hot

    def _target_from_annotation(self, annotation):
        target = torch.zeros((self.S, self.S, 5 + self.C))

        # get original dimensions of image
        size = annotation["annotation"]["size"]
        width, height = float(size["width"]), float(size["height"])

        # scales to multiply by to resize dimensions to 448 x 448
        scale_x = self.IMAGE_SIZE / width
        scale_y = self.IMAGE_SIZE / height

        # get annotated objects in image
        objects = annotation["annotation"]["object"]

        # loop over objects' bounding boxes and create target tensor (p, x, y, w, h, ...C...)
        for obj in objects:
            name = obj["name"]
            box = obj["bndbox"]
            xmin, xmax, ymin, ymax = (
                float(box["xmin"]),
                float(box["xmax"]),
                float(box["ymin"]),
                float(box["ymax"]),
            )

            # scale dimensions to 448 x 448
            xmin *= scale_x
            xmax *= scale_x
            ymin *= scale_y
            ymax *= scale_y

            # normalized width and height
            width = (xmax - xmin) / self.IMAGE_SIZE
            height = (ymax - ymin) / self.IMAGE_SIZE

            # get center of bounding box
            x = (xmin + xmax) / 2
            y = (ymin + ymax) / 2

            # get top left coordinate of grid cell
            grid_i = int(x // self.GRID_SIZE)
            grid_j = int(y // self.GRID_SIZE)

            grid_x = grid_i * self.GRID_SIZE
            grid_y = grid_j * self.GRID_SIZE

            # get normalized offsets
            x = (x - grid_x) / self.GRID_SIZE
            y = (y - grid_y) / self.GRID_SIZE

            # construct target tensor
            box_tensor = torch.tensor([1, x, y, width, height])  # (p, x, y, w, h)
            
            # one-hot encoding for classification
            classification_tensor = self._one_hot(name)

            target_tensor = torch.cat((box_tensor, classification_tensor), dim=0)

            # get current tensor at grid location
            grid_cell = target[grid_i][grid_j]

            # add target tensor to target
            target[grid_j][grid_i] = target_tensor

        return target

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, annotation = self.dataset.__getitem__(index)

        # transform image and annotation
        image = self.transform(image)
        target = self._target_from_annotation(annotation)

        return image, target
