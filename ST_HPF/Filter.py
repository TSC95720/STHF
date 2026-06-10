from matplotlib import pyplot as plt
import torch.fft as fft
import torch
import os


def hpf_st(fre, cutoff1, cutoff2):
    t, h, w = fre.size(0), fre.size(1), fre.size(2)
    t_crop, h_crop, w_crop = int(t / 2), int(h / 2), int(w / 2)
    mask = torch.zeros_like(fre, dtype=torch.uint8)
    mask[t_crop - cutoff1:t_crop + cutoff1, h_crop - cutoff2:h_crop + cutoff2, w_crop - cutoff2:w_crop + cutoff2] = 1
    mask = 1 - mask
    high_fre = fre * mask

    return high_fre


def hpf_t(fre, cutoff):
    t, h, w = fre.size(0), fre.size(1), fre.size(2)
    t_crop = int(t / 2)
    mask = torch.zeros_like(fre, dtype=torch.uint8)
    mask[t_crop - cutoff:t_crop + cutoff, ...] = 1
    mask = 1 - mask
    fre_high = fre * mask

    return fre_high


def hpf_s(fre, cutoff):
    t, h, w = fre.size(0), fre.size(1), fre.size(2)
    h_crop, w_crop = int(h / 2), int(w / 2)
    mask = torch.zeros((fre.size(0), h, w), dtype=torch.uint8)
    mask[:, h_crop - cutoff:h_crop + cutoff, w_crop - cutoff:w_crop + cutoff] = 1
    mask = 1 - mask
    fre_high = fre * mask

    return fre_high


def fre_aug(seq, cutoff1, cutoff2):
    seq_fre = fft.fftn(seq, dim=[0, 1, 2], norm='ortho')
    seq_fre = fft.fftshift(seq_fre)
    seq_fre_high = hpf_st(seq_fre, cutoff1, cutoff2)
    seq_fre_high = fft.ifftshift(seq_fre_high)
    new_seq = fft.ifftn(seq_fre_high, dim=[0, 1, 2], norm='ortho')
    new_seq = torch.real(new_seq)

    return new_seq


def save_images(seq, save_dir, save_name):
    """
    Save a sequence of images to specified directory.

    Parameters:
    seq (list of numpy.ndarray): Sequence of images to save.
    save_dir (str): Directory to save the images.
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx, image in enumerate(seq):
        save_path = os.path.join(save_dir, save_name[idx])
        plt.imsave(save_path, image, cmap='gray')
        print(save_path)