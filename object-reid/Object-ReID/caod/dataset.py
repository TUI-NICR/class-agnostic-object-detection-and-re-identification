import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from typing import Tuple, Union, List, Any
from numpy.typing import NDArray


class ReIDDataset(Dataset):
    """
    Basic Dataset. WARNING! Accessing items modifies them inplace, changing the underlying data!
    """
    def __init__(
        self,
        data: List[Tuple[Union[Union[Image.Image, NDArray], List[Union[Image.Image, NDArray]]], Tuple[Any, ...]]],
        transform: Compose =None
    ):
        self.data = data
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[List[Union[Image.Image, NDArray]], Tuple[Any, ...]]:
        x = self.data[index]
        if not isinstance(x[0], (list, tuple)):
            x[0] = [x[0]]
        if self.transform:
            for i, y in enumerate(x[0]):
                if isinstance(y, (np.ndarray, np.generic)):
                    y = Image.fromarray(y.astype(np.uint8))
                x[0][i] = self.transform(y)
        return x
    
    def __len__(self):
        return len(self.data)


def collate_fn(
    batch: List[Tuple[List[Union[Image.Image, NDArray]], Tuple[Any, ...]]]
) -> Tuple[List[Union[Image.Image, NDArray]], Tuple[List[Any], ...]]:
    imgs, others = zip(*batch)
    others = tuple(zip(*others))
    imgs = [img for ims in imgs for img in ims]
    return imgs, others
