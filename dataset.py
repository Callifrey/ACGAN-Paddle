import paddle.vision.transforms as tran
from paddle.io import Dataset, DataLoader
from PIL import Image
import os
import os.path

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir, classes_idx=None):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    if classes_idx is not None:
        assert type(classes_idx) == tuple
        start, end = classes_idx
        classes = classes[start:end]
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        if target not in class_to_idx:
            continue
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class ImageFolder(Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=pil_loader, classes_idx=None):
        super().__init__()
        self.classes_idx = classes_idx
        classes, class_to_idx = find_classes(root, self.classes_idx)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    trans = tran.ToTensor()
    dataset = ImageFolder(root='/media/gallifrey/DJW/Dataset/Imagenet/train', transform=tran.Compose([
        tran.Resize(128),
        tran.CenterCrop(128),
        tran.ToTensor(),
        tran.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
    classes_idx=(10, 20))
    dataloader = DataLoader(dataset)
    print(len(dataloader))
    for data in dataloader:
        img, target = data
        print(img.shape)
        print(target.shape)
        break
