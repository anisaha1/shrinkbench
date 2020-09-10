import pathlib
import glob

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from PIL import Image

class TinyImageNet(Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None, **kwargs):
        root = pathlib.Path(root)
        root /= 'train' if train else 'val'
        self.data = ImageFolder(root,
                                transform=transform,
                                target_transform=target_transform,
                                **kwargs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

# customFolder to load poisoned data from a root folder and a given target
# TODO think of better solution
class customFolder(Dataset):
	def __init__(self, path, target=None, transform=None, **kwargs):
		self.filelist = sorted(pathlib.Path(path).glob('*'))
		self.transform = transform
		self.target = target

	def __len__(self):
		'Denotes the total number of samples'
		return len(self.filelist)

	def __getitem__(self, index):
		img = Image.open(self.filelist[index]).convert("RGB")
		if self.transform is not None:
			img = self.transform(img)

		return img, self.target
