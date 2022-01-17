import skvideo
import torch
from torchvision import transforms
from PIL import Image
import math
from model import ResNet50
import scipy.io
from scipy.io import wavfile
from scipy import signal
import numpy as np
import logging

skvideo.setFFmpegPath('/home/yzj/miniconda3/envs/world_on_rails/bin')
print('Set FFmpeg Successfully\n')

import skvideo.io

_PI = np.pi


def get_video_features(dis_video_data, ref_video_data, position, sal_index, patchSize, device='cuda'):
    """feature extraction"""
    extractor = ResNet50().to(device)
    dis_output = torch.Tensor().to(device)
    ref_output = torch.Tensor().to(device)
    extractor.eval()

    ipatch = 0
    with torch.no_grad():
        for iframe in range(0, 192, 2):
            sal_row = int(iframe / 2)
            # initialize
            dis_output1 = torch.Tensor().to(device)
            dis_output2 = torch.Tensor().to(device)
            ref_output1 = torch.Tensor().to(device)
            ref_output2 = torch.Tensor().to(device)

            for idx in range(25):
                patch_idx = sal_index[sal_row, idx]
                dis_batch = dis_video_data[iframe:iframe + 1, 0:3,
                            position[0][patch_idx]:position[0][patch_idx] + patchSize,
                            position[1][patch_idx]:position[1][patch_idx] + patchSize].to(device)
                ref_batch = ref_video_data[iframe:iframe + 1, 0:3,
                            position[0][patch_idx]:position[0][patch_idx] + patchSize,
                            position[1][patch_idx]:position[1][patch_idx] + patchSize].to(device)

                dis_features_mean, dis_features_std = extractor(dis_batch)
                dis_output1 = torch.cat((dis_output1, dis_features_mean), 0)
                dis_output2 = torch.cat((dis_output2, dis_features_std), 0)

                ref_features_mean, ref_features_std = extractor(ref_batch)
                ref_output1 = torch.cat((ref_output1, ref_features_mean), 0)
                ref_output2 = torch.cat((ref_output2, ref_features_std), 0)
                ipatch = ipatch + 1
                # print('\r Extracting Feature: iframe: {} ipatch: {} '.format(iframe, ipatch), end=' ')

            dis_output = torch.cat((dis_output, torch.cat(
                (dis_output1.mean(axis=0, keepdim=True), dis_output2.mean(axis=0, keepdim=True)), 1)), 0)
            ref_output = torch.cat(
                (ref_output,
                 torch.cat((ref_output1.mean(axis=0, keepdim=True), ref_output2.mean(axis=0, keepdim=True)), 1)), 0)

            ipatch = 0

        dis_output = dis_output.squeeze()
        ref_output = ref_output.squeeze()
    return dis_output, ref_output


def get_audio_features(audios_data, audio_tStamp, frameRate, video_length, device='cuda'):
    """audio feature extraction"""
    extractor = ResNet50().to(device)
    output1 = torch.Tensor().to(device)
    output2 = torch.Tensor().to(device)
    extractor.eval()

    patchSize = 224
    frameSkip = 2
    with torch.no_grad():
        for iFrame in range(1, video_length, frameSkip):
            tCenter = np.argmin(abs(audio_tStamp - iFrame / frameRate))
            tStart = tCenter - patchSize / 2 + 1
            tEnd = tCenter + patchSize / 2
            if tStart < 1:
                tStart = 1
                tEnd = patchSize
            else:
                if tEnd > audios_data.shape[2]:
                    tStart = audios_data.shape[2] - patchSize + 1
                    tEnd = audios_data.shape[2]

            specRef_patch = audios_data[:, :, int(tStart - 1): int(tEnd)]
            refRGB = torch.cat((specRef_patch, specRef_patch, specRef_patch), 0)
            last_batch = refRGB.view(1, 3, specRef_patch.shape[1], specRef_patch.shape[2]).float().to(device)
            features_mean, features_std = extractor(last_batch)
            output1 = torch.cat((output1, features_mean), 0)
            output2 = torch.cat((output2, features_std), 0)

        output = torch.cat((output1, output2), 1).squeeze()

    return output


def extract_video_features(dis_video_path, ref_video_path, video_height, video_width, device):
    ref_video_data = skvideo.io.vread(ref_video_path, video_height, video_width,
                                      inputdict={'-pix_fmt': 'yuvj420p'})
    dis_video_data = skvideo.io.vread(dis_video_path, video_height, video_width,
                                      inputdict={'-pix_fmt': 'yuvj420p'})

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    video_length = 192  # dis_video_data.shape[0]
    video_channel = dis_video_data.shape[3]
    video_height = dis_video_data.shape[1]
    video_width = dis_video_data.shape[2]
    transformed_dis_video = torch.zeros([video_length, video_channel, video_height, video_width])
    transformed_ref_video = torch.zeros([video_length, video_channel, video_height, video_width])

    for frame_idx in range(192):
        dis_frame = dis_video_data[frame_idx]
        dis_frame = Image.fromarray(dis_frame)
        dis_frame = transform(dis_frame)
        transformed_dis_video[frame_idx] = dis_frame

        ref_frame = ref_video_data[frame_idx]
        ref_frame = Image.fromarray(ref_frame)
        ref_frame = transform(ref_frame)
        transformed_ref_video[frame_idx] = ref_frame
    dis_patch = math.ceil(video_height / 1000) * 100
    # print('Extract Video length: {}'.format(transformed_dis_video.shape[0]))

    # Crop image patches
    patchSize = 224
    position_width = []
    position_height = []
    for h in range(0, video_height, dis_patch):
        if h < video_height - patchSize + 1:
            for w in range(0, video_width, dis_patch):
                if w < video_width - patchSize:
                    position_height.append(h)
                    position_width.append(w)
                else:
                    position_height.append(h)
                    position_width.append(video_width - patchSize)
                    break
        else:
            for w in range(0, video_width, dis_patch):
                if w < video_width - patchSize:
                    position_height.append(video_height - patchSize)
                    position_width.append(w)
                else:
                    position_height.append(video_height - patchSize)
                    position_width.append(video_width - patchSize)
                    break
            break

    # Video feature extraction
    position = [position_height, position_width]

    # Using saliency detection results from sal_position.m
    distorted_video_name = dis_video_path.split('/')[2]
    distorted_video_name = distorted_video_name.split('.')[0]
    SDdatainfo = './SD/' + distorted_video_name + '.mat'
    # print(SDdatainfo)
    SDInfo = scipy.io.loadmat(SDdatainfo)
    sal_index = SDInfo['sort_frame'] - 1

    dis_video_features, ref_video_features = get_video_features(transformed_dis_video, transformed_ref_video, position,
                                                                sal_index, patchSize, device)

    return dis_video_features, ref_video_features


def gga_freq_abs(x, sample_rate, freq):
    lx = len(x)
    pik_term = 2 * _PI * freq / sample_rate
    cos_pik_term = np.cos(pik_term)
    cos_pik_term2 = 2 * np.cos(pik_term)

    # number of iterations is (by one) less than the length of signal
    # Pipeline the first two iterations.
    s1 = x[0]
    s0 = x[1] + cos_pik_term2 * s1
    s2 = s1
    s1 = s0
    for ind in range(2, lx - 1):
        s0 = x[ind] + cos_pik_term2 * s1 - s2
        s2 = s1
        s1 = s0

    s0 = x[lx - 1] + cos_pik_term2 * s1 - s2

    # | s0 - s1 exp(-ip) |
    # | s0 - s1 cos(p) + i s1 sin(p)) |
    # sqrt((s0 - s1 cos(p))^2 + (s1 sin(p))^2)
    y = np.sqrt((s0 - s1 * cos_pik_term) ** 2 + (s1 * np.sin(pik_term)) ** 2)
    # y = np.sqrt(s0**2 + s1**2 - s0*s1*cos_pik_term2)
    return y


def spectrogram(x, window, window_overlap, bfs, fs):
    num_blocks = int((len(x) - window_overlap) / (len(window) - window_overlap))
    S = np.empty((len(bfs), num_blocks), dtype=np.float64)
    T = np.empty((num_blocks), dtype=np.float64)

    for i in range(num_blocks):
        block = window * x[i * (len(window) - window_overlap): i * (len(window) - window_overlap) + len(window)]
        S[:, i] = gga_freq_abs(block, fs, bfs)
        T[i] = (i * (len(window) - window_overlap) + len(window) / 2) / fs

    return S, T


def calcSpectrogram(audiofile):
    fs, audio = wavfile.read(audiofile)
    _, ref_audio = wavfile.read(audiofile)
    audio = (audio + 0.5) / 32767.5
    audio = audio[:, 0]

    windowsize = round(fs * 0.02)  # 20ms
    overlap = 0.75  # 75% overlap: a 20ms window every 5ms

    window_overlap = int(windowsize * overlap)
    window = signal.get_window('hamming', windowsize, fftbins=True)

    dim = 224
    bfs = [i for i in np.arange(30, 3820 + (3820 - 30) / (dim - 1), (3820 - 30) / (dim - 1))]
    bfs = np.array(bfs, dtype=float)
    bfs = 700 * (pow(10, (bfs / 2595)) - 1)
    S, t_sp = spectrogram(audio, window, window_overlap, bfs, fs)

    S = abs(np.array(S))  # remove complex component

    S[(S == 0)] = pow(2, -52)  # no -infs in power dB

    spec_bf = np.zeros((S.shape[0], S.shape[1]))
    for i in range(len(S)):
        for j in range(len(S[i])):
            spec_bf[i][j] = math.log(S[i][j])
    return spec_bf, t_sp


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_logger(filename, name=None):
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.INFO)
    fh = logging.FileHandler(filename)
    formatter = logging.Formatter('%(asctime)s - %(filename)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger
