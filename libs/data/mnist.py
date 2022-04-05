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


def visualize(image, data):
    import pandas as pd
    import numpy as np
    import networkx as nx

    plt.figure(figsize=(17, 8))

    # plot the mnist image
    plt.subplot(1, 2, 1)
    plt.title("MNIST")
    if len(image.shape):
        image = image.permute(1, 2, 0)
    np_image = np.array(image)
    plt.imshow(np_image)

    # plot the super-pixel graph
    plt.subplot(1, 2, 2)
    x, edge_index = data.x, data.edge_index

    # construct networkx graph
    df = pd.DataFrame({'from': edge_index[0], 'to': edge_index[1]})
    G = nx.from_pandas_edgelist(df, 'from', 'to')

    # flip over the axis of pos, this is because the default axis direction of networkx is different
    pos = {i: np.array([data.pos[i][0], 27 - data.pos[i][1]]) for i in range(data.num_nodes)}

    # get the current node index of G
    idx = list(G.nodes())

    # set the node sizes using node features
    size = x[idx] * 500 + 200

    # set the node colors using node features
    color = []
    for i in idx:
        grey = x[i]
        if grey == 0:
            color.append('skyblue')
        else:
            color.append('red')

    nx.draw(G, with_labels=True, node_size=size, node_color=color, pos=pos)
    plt.title("MNIST Superpixel")


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
