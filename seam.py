import numpy as np
import numba
import argparse
from imageio import imread, imwrite
from scipy.ndimage.filters import convolve
import warnings
warnings.filterwarnings('ignore')


@numba.jit
def convolve_img(image, kernel):
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


def e1(img):
    sobel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ])
    Ix = convolve_img(img, sobel)
    Iy = convolve_img(img, sobel.T)
    return np.sum(np.abs(Ix) + np.abs(Iy), axis=img.ndim-1)


@numba.jit
def find_min_seam(img, E):
    M = E.copy()
    back = np.zeros(E.shape)
    w, h = E.shape
    for i in range(0, w):
        for j in range(0, h):
            jj = j if j == 0 else j - 1
            k = np.argmin(M[i-1, jj:j+2]) + jj
            M[i, j] = E[i, j] + M[i-1, k]
            back[i, j] = k
    return M[-1], back.astype(np.int)


@numba.jit
def remove_seam(img, seam, back):
    h, w, l = img.shape
    j = np.argmin(seam)
    for i in reversed(range(h)):
        img[i, j] = -1
        j = back[i, j]
    return img[img != -1].reshape((h, w-1, l))


@numba.jit
def seam_carving(img, scale, vertical=False):
    img = np.transpose(img, (1, 0, 2)) if vertical else img
    num_remove = img.shape[1] - int(img.shape[1] * scale)
    for _ in range(num_remove):
        E = e1(img)
        seam, back = find_min_seam(img, E)
        img = remove_seam(img, seam, back)
    img = np.transpose(img, (1, 0, 2)) if vertical else img
    return img


def main():
    parser = argparse.ArgumentParser(
        description="Scales image by removing/inserting seams.")
    parser.add_argument(
        "image", type=str, help="The image you want to scale.")
    parser.add_argument(
        "scale", type=float, help="The number you want to scale by.")
    parser.add_argument(
        "--vertical", help="Scales horizontally instead of vertically.", action="store_true")
    args = parser.parse_args()

    img = imread(args.image)[:, :, :3].astype(np.float32) / 255.0
    s = args.scale
    dim = "vertically" if args.vertical else "horizontally"

    if s > 0 and s < 1:
        print("Resizing {} with scale {}...".format(dim, s))
        out = seam_carving(img, s, args.vertical)
        imwrite(args.image.split(".")[0] + "_out.jpg", out)
    else:
        print("Scale must be between 0 and 1.")


if __name__ == "__main__":
    main()
