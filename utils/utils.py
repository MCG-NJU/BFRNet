import os
import librosa
import torch
import cv2
import numpy as np
import json
import shutil
import matplotlib.pyplot as plt
plt.switch_backend('Agg') 
plt.ioff()


def warpgrid(bs, HO, WO, warp=True):
    # meshgrid
    x = np.linspace(-1, 1, WO)
    y = np.linspace(-1, 1, HO)
    xv, yv = np.meshgrid(x, y)
    grid = np.zeros((bs, HO, WO, 2))
    grid_x = xv
    if warp:
        grid_y = (np.power(21, (yv+1)/2) - 11) / 10
    else:
        grid_y = np.log(yv * 10 + 11) / np.log(21) * 2 - 1
    grid[:, :, :, 0] = grid_x
    grid[:, :, :, 1] = grid_y
    grid = grid.astype(np.float32)
    return grid

def magnitude2heatmap(mag, log=True, scale=200.):
    if log:
        mag = np.log10(mag + 1.)
    mag *= scale
    mag[mag > 255] = 255
    mag = mag.astype(np.uint8)
    mag_color = cv2.applyColorMap(mag, cv2.COLORMAP_JET)
    mag_color = mag_color[:, :, ::-1]
    return mag_color

def visualizeSpectrogram(spectrogram, save_path):
    fig,ax = plt.subplots(1,1)
    plt.axis('off')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.pcolormesh(librosa.amplitude_to_db(spectrogram))
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def visualizePhase(spectrogram, save_path):
    fig,ax = plt.subplots(1,1)
    plt.axis('off')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.pcolormesh(spectrogram  + np.pi)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def istft_reconstruction(mag, phase, hop_length=160, win_length=400, length=65535):
    spec = mag.astype(np.complex) * np.exp(1j*phase)
    wav = librosa.istft(spec, hop_length=hop_length, win_length=win_length, length=length)
    return np.clip(wav, -1., 1.)

def istft_reconstruction_from_complex(real, imag, hop_length=160, win_length=400, length=65535):
    spec = real + 1j*imag
    wav = librosa.istft(spec, hop_length=hop_length, win_length=win_length, length=length)
    return np.clip(wav, -1., 1.)


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
    nets (network list)   -- a list of networks
    requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def load_json( json_fp ):
    with open( json_fp, 'r' ) as f:
        json_content = json.load(f)
    return json_content

def read_video(filename):
    cap = cv2.VideoCapture(filename)
    while(cap.isOpened()):
        ret, frame = cap.read() # BGR
        if ret:
            yield frame
        else:
            break
    cap.release()

def save2npz(filename, data=None):
    assert data is not None, "data is {}".format(data)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    np.savez_compressed(filename, data=data)

def mkdirs(path, remove=False):
    if os.path.isdir(path):
        if remove:
            shutil.rmtree(path)
        else:
            return
    os.makedirs(path)

def read_txt_lines(filepath):
    assert os.path.isfile( filepath ), "Error when trying to read txt file, path does not exist: {}".format(filepath)
    with open( filepath ) as myfile:
        content = myfile.read().splitlines()
    return content

#define customized collate to combine useful objects across video pairs
error_msg_fmt = "batch must contain tensors, numbers, dicts or lists; found {}"
numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}

def collate_fn(dataBatch):
    """
    Collate function definition used in Dataloaders.
    """
    keys = dataBatch[0].keys()
    outBatch = {}
    for key in keys:
        out = []
        for data in dataBatch:
            out.append(data[key])
        outBatch[key] = torch.cat(out)
    return outBatch
