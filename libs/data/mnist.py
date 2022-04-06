import torch_geometric.transforms as transforms
import torchvision.transforms as T
from torchvision.datasets import MNIST
from libs.data.transforms import EdgePerturbation, NodeDropping
import matplotlib.pyplot as plt


class MNISTSuperpixel(MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, aug_transform=None, download=False):
        '''

        :param root:
        :param train:
        :param transform: To apply to the image
        :param target_transform:
        :param aug_transform: To apply to the graph
        :param download:
        '''
        super(MNISTSuperpixel, self).__init__(root, train, transform, target_transform, download)
        self.aug_transform = aug_transform
        self.transforms_conversion = T.Compose([transforms.ToSLIC(n_segments=75, compactness=0.25, add_seg=True,
                                                                  enforce_connectivity=True),
                                                transforms.RadiusGraph(r=8, loop=True)])
        self.data_names = list(range(len(self.data)))

    def __getitem__(self, index):
        img, target = self.data[index].unsqueeze(0).float(), int(self.targets[index])

        if self.transform is not None:
            img = self.transform(img)

        data = self.transforms_conversion(img)
        data.sp_seg = data.seg
        del data.seg

        data_aug = data.clone()
        if self.aug_transform:
            data_aug = self.aug_transform(data_aug)

        semseg = (img > 0.5).int()

        return {'img': img, 'data': data, 'data_aug': data_aug, 'name': self.data_names[index], 'semseg': semseg}


if __name__ == '__main__':
    # transform = T.Compose([T.ToTensor()])
    # d = MNISTLabelSuperpixel('../../data/', transform=transform)
    # a = d[120]
    # visualize(a['img'], a['data'])
    # dataset = torchvision.datasets.MNIST('../../data/', train=True, transform=transform, download=True)

    path = '../../data/'

    # t =  T.Compose([T.ToTensor(), transforms.ToSLIC(n_segments=75, add_img=True, enforce_connectivity=False), transforms.KNNGraph(k=8, loop=True)])
    # t = T.Compose([T.ToTensor()])
    aug_t = T.Compose([NodeDropping(), EdgePerturbation()])
    image_dataset = MNISTSuperpixel(root=path, download=True, aug_transform=aug_t)
    a = image_dataset[0]
    # visualize(a['img'], a['data'])
    # plt.imshow(a['data'].sp_seg[0])
