"""
Utility functions for cnn.py
"""
from math import floor
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

def get_MNIST(name, batchsize = 64):
    """
    Return decentered MNIST images.

    Args:
        name: 'top_left' for images translated toward the top-left corner,
              or 'bottom_right' for images translated toward the bottom-right.
        batchsize: Number of examples per batch

    Returns:
        train,test: lists of batches of training and test examples

    Shape:
        (batchsize, 1, 30, 30) for each batch
    """
    trn_dat = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
    tst_dat = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())
    trn = DataLoader(trn_dat, batch_size=batchsize) # X.shape: N, C, H, W; Y.shape: N
    tst = DataLoader(tst_dat, batch_size=batchsize)

    def decenter(loader, pad=2):
        ret = []
        for X,y in loader:
            s = X.shape
            new_X = torch.zeros((s[0], s[1], s[2]+abs(pad), s[3]+abs(pad)))
            assert new_X.shape[2] == new_X.shape[3] == 28+abs(pad)
            if pad>0:
                new_X[:,:,pad:s[2]+pad,pad:s[3]+pad] = X
            else:
                new_X[:,:,0:s[2],0:s[3]] = X
            ret.append((new_X,y))
        return ret

    if name == 'top_left':
        pad = -2
    elif name == 'bottom_right':
        pad = 2
    else:
        raise ValueError("Only valid names are ['top_left', 'bottom_right'], not '%s'" % name)

    return decenter(trn, pad), decenter(tst,pad)


def show_examples(left_examples, right_examples, fname='examples.png'):
    """
    Create a file containing side-by-side examples of the difference
    between top-left and bottom-right decentered digits.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(1,5))
        for digit in range(10):
            X1,y1 = next(iter(left_examples))
            X2,y2 = next(iter(right_examples))
            i = (y1==digit).type(torch.int).argmax(0)
            plt.subplot(10, 2, 2*digit+1)
            frame1 = plt.gca()
            frame1.axes.get_xaxis().set_visible(False)
            frame1.axes.get_yaxis().set_visible(False)

            plt.imshow(X1[i,0], cmap='gray')

            plt.subplot(10, 2, 2*digit+2)
            frame1 = plt.gca()
            frame1.axes.get_xaxis().set_visible(False)
            frame1.axes.get_yaxis().set_visible(False)

            plt.imshow(X2[i,0], cmap='gray')
        plt.savefig(fname)
    except ImportError:
        print('Skipping examples.png creation; could not import pyplot')

def hout(hin, kernel_size_y, stride_y):
    """
    Return the output height dimension of a `Conv2d` layer that has input
    height of `hin`, a kernel with height direction size of `kernel_size_y`,
    and height direction stride of `stride_y`.
    """
    return floor((hin - (kernel_size_y - 1) -1)/stride_y + 1)