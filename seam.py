from imageio import imread, imwrite
from numba import jit
import numpy as np
from scipy.ndimage.filters import convolve
from argparse import ArgumentParser
from warnings import filterwarnings
import matplotlib.pyplot as plt
import cv2

filterwarnings('ignore')
RED = np.array([255, 0, 0])


def convolve_img(image, kernel):
    """
    Convolves the provided kernel with the provided image and returns the results.

    Args:
        `img`:     A 2-dimensional ndarray input image.
        `kernel`:  A 2-dimensional kernel to convolve with the image.

    Returns:
        `ndarray`: The result of convolving the provided kernel with the image at location i, j.
    """
    if kernel.ndim == image.ndim:
        if image.shape[-1] == kernel.shape[-1]:
            return np.dstack([convolve(image[..., c], kernel[..., c]) for c in range(kernel.shape[-1])])
        elif image.ndim == 2:
            return convolve(image, kernel)
        else:
            raise RuntimeError("Invalid kernel shape. Kernel: %s Image: %s" % (
                kernel.shape, image.shape))
    elif kernel.ndim == image.ndim - 1:
        return np.dstack([convolve(image[..., c], kernel) for c in range(image.shape[-1])])
    else:
        raise RuntimeError("Invalid kernel shape. Kernel: %s Image: %s" % (
            kernel.shape, image.shape))


def backward_energy(img):
    """
    Creates an energy map using the image's gradients and the E1 energy function.

    Args:
        `img`:     A 2-dimensional ndarray input image.

    Returns:
        `ndarray`: An energy map of the given image.
    """
    sobel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ]) * (1/8)
    # Computes the gradients of img using the sobel operator
    Ix = convolve_img(img, sobel)
    Iy = convolve_img(img, sobel.T)
    # Returns the result of E1 = |Ix| + |Iy|
    return np.sum(np.abs(Ix) + np.abs(Iy), axis=img.ndim-1)


@jit
def forward_energy(img):
    """
    Creates an energy map using the forward energy method from the extension paper.

    Args:
        `img`:     A 2-dimensional ndarray input image.

    Returns:
        `ndarray`: An energy map of the given image.
    """
    # Creates a gray-scale version of the image
    im = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    h, w = im.shape
    cL = np.zeros(im.shape)
    cU = np.zeros(im.shape)
    cR = np.zeros(im.shape)
    E = np.zeros(im.shape)
    for i in range(0, h):
        for j in range(0, w):
            # In case j is 0, we use j=0 so that j-1 is not -1
            jj = j if j == 0 else j - 1
            # In case i is 0, we use i=0 so that i-1 is not -1
            ii = i if i == 0 else i - 1
            # Compute cU, cL, cR as described in the paper
            cU[i, j] = np.abs(im[i, j+1] - im[i, jj])
            cL[i, j] = cU[i, j] + np.abs(im[ii, j] - im[i, jj])
            cR[i, j] = cU[i, j] + np.abs(im[ii, j] - im[i, j+1])
            # The energy is the minimum between cU, cL, cR
            E[i, j] = min(cU[i, j], cL[i, j], cR[i, j])
    return E


@jit
def find_min_seam(E):
    """
    Finds the minimum seam in the given energy map.

    Args:
        `E`:       A 2-dimensional ndarray energy map.

    Returns:
        `ndarray`: The minimum seam.
        `ndarray`: The stored choices along the seam to backtrack with.
    """
    M = E.copy()
    back = np.zeros(E.shape)
    h, w = E.shape
    for i in range(0, h):
        for j in range(0, w):
            # In case j is 0, we only find the min between M(i-1,j) and M(i-1,j+1)
            jj = j if j == 0 else j - 1
            # In case i is 0, we use i=0 so that i-1 is not -1
            ii = i if i == 0 else i - 1
            # Find index of min between M(i-1,j-1), M(i-1,j), and M(i-1,j+1)
            min_j = np.argmin(M[ii, jj:j+2]) + jj
            # Computes the minimal aggregate cost
            M[i, j] = E[i, j] + M[ii, min_j]
            # Stores the index for backtracking
            back[i, j] = min_j
    # Returns the last index of M which is the minimum seam
    return M[-1], back.astype(np.int)


def remove_seam(img, seam, back, vertical=False, show_live=False):
    """
    Removes the given seam from the image.

    Args:
        `img`:     A 2-dimensional ndarray image.
        `seam`:    A 1-dimensional ndarray seam to remove.
        `back`:    A 2-dimensional ndarray to backtrack with.

    Returns:
        `ndarray`: The image with the seam removed.
    """
    live = img.copy()
    h, w, l = img.shape
    j = np.argmin(seam)

    # Goes through the seam, setting the seam in the image to -1
    for i in reversed(range(h)):
        img[i, j] = -1
        live[i, j] = RED
        j = back[i, j]

    # Shows live seam carving
    if show_live:
        live = np.transpose(live, (1, 0, 2)) if vertical else live
        live = cv2.cvtColor(live, cv2.COLOR_BGR2RGB)
        cv2.imshow("Live Seam Carving", live)
        cv2.waitKey(1)

    # Removes the values with -1 (the seam) and resizes the image
    return img[img != -1].reshape((h, w-1, l))


def seam_carving(img, scale, energy_func, vertical=False, show_live=False):
    """
    Resizes the given image with the given scale using seam carving.

    Args:
        `img`:      A 2-dimensional ndarray image.
        `scale`:    A float to scale the image by.
        `vertical`: Optional boolean used for vertical resizing.

    Returns:
        `ndarray`:  The resized image.
    """
    # Transposes the image for vertical resizing
    img = np.transpose(img, (1, 0, 2)) if vertical else img
    # Computes the number of rows/columns to remove from the image
    num_remove = int(img.shape[1] * (1 - scale))
    for _ in range(num_remove):
        E = energy_func(img)
        seam, back = find_min_seam(E)
        img = remove_seam(img, seam, back, vertical, show_live)
    # Transposes the image back for vertical resizing
    img = np.transpose(img, (1, 0, 2)) if vertical else img
    return img


def main():
    parser = ArgumentParser(
        description="Scales image by removing/inserting seams.")
    parser.add_argument(
        "image", type=str, help="The image you want to scale.")
    parser.add_argument(
        "scale", type=float, help="The number you want to scale by.")
    parser.add_argument(
        "--vertical", help="Scales horizontally instead of vertically.", action="store_true")
    parser.add_argument(
        "--forward", help="Useds forward energy instead of backward.", action="store_true")
    parser.add_argument(
        "--live", help="Shows live seam carving", action="store_true")
    args = parser.parse_args()

    img = imread(args.image)[:, :, :3].astype(np.float32) / 255.0
    s = args.scale
    dim = "vertically" if args.vertical else "horizontally"
    energy = "forward" if args.forward else "backward"
    func = forward_energy if args.forward else backward_energy

    if s > 0 and s < 1:
        print("Resizing {} with scale {} using {} energy...".format(dim, s, energy))
        out = seam_carving(img, s, func, args.vertical, args.live)
        imwrite(args.image.split(".")[0] + "_out.jpg", out)
    else:
        print("Scale must be between 0 and 1.")


if __name__ == "__main__":
    main()
