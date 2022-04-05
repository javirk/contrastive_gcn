#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import cv2
import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import torch
import torch.nn.functional as F
from libs.utils import get_cam_segmentation, one_hot

MAX_ITER = 10
POS_W = 3
POS_XY_STD = 1
Bi_W = 4
Bi_XY_STD = 67
Bi_RGB_STD = 3
# BGR_MEAN = np.array([104.008, 116.669, 122.675])


def dense_crf(image: torch.FloatTensor, output_logits: torch.FloatTensor, n_classes=None):
    channels = image.shape[0]
    H, W = image.shape[1:]
    image = np.ascontiguousarray(image).astype(np.uint8)

    if not torch.is_tensor(output_logits):
        output_logits = torch.from_numpy(output_logits)

    n_classes = n_classes if n_classes is not None else (output_logits.max() + 1).long().item()

    if output_logits.ndim == 2:
        output_logits = one_hot(output_logits.unsqueeze(0).long(), n_classes, dtype=torch.float)

    output_logits = F.interpolate(output_logits, size=(H, W), mode="bilinear", align_corners=False)
    output_probs = F.softmax(output_logits, dim=1).squeeze().cpu().numpy()

    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    if channels == 1:
        pairwise_energy = utils.create_pairwise_bilateral(sdims=(3, 3), schan=(0.01,), img=image, chdim=0)
        d.addPairwiseEnergy(pairwise_energy, compat=10)
    else:
        d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=image, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    return Q

if __name__ == '__main__':
    from libs.data.mnist import MNISTSuperpixel
    import matplotlib.pyplot as plt

    dataset = MNISTSuperpixel('../data/', train=True, aug_transform=None, download=True)

    img = dataset[0]['img']
    coarse_seg = get_cam_segmentation(img[None])
    q = dense_crf(img, coarse_seg)

    plt.imshow(img[0], cmap='gray')
    plt.imshow(q[1], alpha=0.5)