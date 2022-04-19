import numpy as np
import os
from PIL import Image
import copy
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch_geometric.transforms as transforms


class Pascal(Dataset):
    VOC_CATEGORY_NAMES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                          'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                          'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                          'sofa', 'train', 'tvmonitor']

    def __init__(self, root: str, image_set: str = "train", sal_type='supervised', transform=None, aug_transform=None):
        self.root = root
        assert image_set in ('train', 'val')
        self.split = image_set
        assert sal_type.lower() in ('supervised', 'unsupervised')

        _semseg_dir = os.path.join(self.root, 'SegmentationClass')
        _image_dir = os.path.join(self.root, 'images')
        _sal_dir = os.path.join(self.root, f'saliency_{sal_type}_model')

        self.transform = transform
        # self.sal_transform = self._get_saliency_transformations()
        self.aug_transform = aug_transform
        self.transforms_conversion = T.Compose([transforms.ToSLIC(n_segments=500, compactness=10, add_seg=True,
                                                                  enforce_connectivity=True),
                                                transforms.RadiusGraph(r=100, loop=True)])

        split_file = os.path.join(self.root, 'sets', self.split + '.txt')
        self.images = []
        self.semsegs = []
        self.saliencies = []

        with open(split_file, "r") as f:
            lines = f.read().splitlines()

        for ii, line in enumerate(lines):
            # Images and saliency
            _image = os.path.join(_image_dir, line + ".jpg")
            _sal = os.path.join(_sal_dir, line + '.png')
            _semseg = os.path.join(_semseg_dir, line + '.png')

            if os.path.isfile(_image) and os.path.isfile(_sal):
                self.images.append(_image)
                self.saliencies.append(_sal)

                # Semantic Segmentation
                assert os.path.isfile(_semseg)
                self.semsegs.append(_semseg)

        print('Number of dataset images: {:d}'.format(len(self.images)))

    def __len__(self):
        return len(self.images)

    def _get_saliency_transformations(self):
        sal_transform = []
        for t in self.transform.transforms:  # A bit of a trick here
            if 'Normalize' not in repr(t):
                sal_transform.append(t)

        return T.Compose(sal_transform)

    def __getitem__(self, index):
        # Load image
        img = np.array(Image.open(self.images[index]).convert('RGB'))
        saliency = self._read_saliency(index)
        # Load pixel-level annotations
        semseg = np.array(Image.open(self.semsegs[index]))[...,None]

        if self.transform is not None:
            transformed = self.transform(image=img, masks=[saliency, semseg])
            img = transformed['image']
            saliency = transformed['masks'][0]
            semseg = transformed['masks'][1]

        # sample['image'] = img
        data = self.transforms_conversion(img)
        data.sp_seg = data.seg
        del data.seg

        data_aug = data.clone()
        if self.aug_transform:
            data_aug = self.aug_transform(data_aug)

        # if _semseg.shape != _img.shape[:2]:
        #     _semseg = cv2.resize(_semseg, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

        return {'img': img, 'data': data, 'data_aug': data_aug, 'name': self.images[index], 'semseg': semseg,
                'sal': saliency}

    def _read_saliency(self, index):
        saliency = np.array(Image.open(self.saliencies[index])) / 255.
        saliency = saliency  # To add a channel before
        return saliency[...,None]


if __name__ == '__main__':
    from libs.data.transforms import EdgePerturbation, NodeDropping
    from libs.data.utils import visualize_superpixels
    import albumentations as A
    from libs.data.transforms import ToTensor

    path = '../../data/VOCSegmentation/'

    aug_t = T.Compose([NodeDropping(), EdgePerturbation()])
    t = A.Compose([
        A.Resize(height=224, width=224), A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), ToTensor()
    ])
    image_dataset = Pascal(root=path, image_set="train", transform=t, aug_transform=aug_t)
    a = image_dataset[0]
    print(a['sal'].shape)
    # visualize(a['img'], a['data'])
