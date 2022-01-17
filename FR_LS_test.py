# Author: Yuqin Cao
# Email: caoyuqin@sjtu.edu.cn
# Date: 2021/11/8
from torchvision import transforms
from torch.utils.data import Dataset
# import ipdb
import os

from argparse import ArgumentParser
import math
import torch
import numpy as np
import random
from model import ANNAVQA, ResNet50
from utils import extract_video_features, calcSpectrogram, get_audio_features

_PI = np.pi


if __name__ == "__main__":
    parser = ArgumentParser(description='"Test Demo of ANNAVQA')
    parser.add_argument("--seed", type=int, default=19990524)
    parser.add_argument('--model_path', default='./models/FR_model', type=str,
                        help='model path (default: ./models/FR)')
    parser.add_argument('--ref_video_path', default='./Videos/BigGreenRabbit.yuv', type=str,
                        help='video path (default: ./ref_test.yuv)')
    parser.add_argument('--dis_video_path', default='./Videos/BigGreenRabbit_QP16.yuv', type=str,
                        help='video path (default: ./dis_test.yuv)')
    parser.add_argument('--dis_audio_path', default='./Audios/BigGreenRabbit_128kbps.wav', type=str,
                        help='video path (default: ./dis_test.wav)')
    parser.add_argument('--ref_audio_path', default='./Audios/BigGreenRabbit.wav', type=str,
                        help='video path (default: ./ref_test.wav)')
    parser.add_argument('--frame_rate', default=29.97, type=float,
                        help='Frame Rate')
    parser.add_argument('--video_width', type=int, default=1920,
                        help='video width')
    parser.add_argument('--video_height', type=int, default=1080,
                        help='video height')
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.utils.backcompat.broadcast_warning.enabled = True
    np.random.seed(args.seed)
    random.seed(args.seed)

    video_length = 192
    video_features_dir = './video_features/'
    distorted_video_name = args.dis_video_path.split('/')[2]
    distorted_video_name = distorted_video_name.split('.')[0]
    ref_video_name = args.ref_video_path.split('/')[2]
    ref_video_name = ref_video_name.split('.')[0]

    if args.debug:
        print('Distorted Video:', distorted_video_name)
        print('Reference Video:', ref_video_name)

    if not os.path.isdir(video_features_dir):
        os.makedirs(video_features_dir)

    dis_video_features_file = video_features_dir + distorted_video_name + '.pt'
    ref_video_features_file = video_features_dir + distorted_video_name + '_ref.pt'
    if not os.path.isfile(dis_video_features_file) or not os.path.isfile(ref_video_features_file):
        dis_video_features, ref_video_features = extract_video_features(args.dis_video_path, args.ref_video_path,
                                                                        args.video_height, args.video_width,
                                                                        device)

        torch.save(dis_video_features.to('cpu'), dis_video_features_file)
        torch.save(ref_video_features.to('cpu'), ref_video_features_file)
    else:
        dis_video_features = torch.load(dis_video_features_file).to(device)
        ref_video_features = torch.load(ref_video_features_file).to(device)

    print('Video features are saved...\n')
    # print(dis_video_features.to('cpu').numpy()-np.load('../CNN_features_SJTU/skip2_SD_BigGreenRabbit_QP35.yuv_res5.npy')[96:,:4096])

    # Audio data preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Extract Audio Features
    audio_features_dir = './audio_features/'
    distorted_audio_name = args.dis_audio_path.split('/')[2]
    distorted_audio_name = distorted_audio_name.split('.')[0]
    ref_audio_name = args.ref_audio_path.split('/')[2]
    ref_audio_name = ref_audio_name.split('.')[0]

    if args.debug:
        print('Distorted Audio:', distorted_audio_name)
        print('Reference Audio:', ref_audio_name)

    dis_audio_features_file = audio_features_dir + distorted_audio_name + '.pt'
    ref_audio_features_file = audio_features_dir + ref_audio_name + '.pt'

    if not os.path.isdir(audio_features_dir):
        os.makedirs(audio_features_dir)

    if not os.path.isfile(dis_audio_features_file) or not os.path.isfile(ref_audio_features_file):
        [dis_S, dis_T] = calcSpectrogram(args.dis_audio_path)
        [ref_S, ref_T] = calcSpectrogram(args.ref_audio_path)
        transforms_dis_audio = transform(dis_S)
        transforms_ref_audio = transform(ref_S)
        dis_audio_features = get_audio_features(transforms_dis_audio, dis_T, args.frame_rate, video_length, device)
        ref_audio_features = get_audio_features(transforms_ref_audio, ref_T, args.frame_rate, video_length, device)
        torch.save(dis_audio_features.to('cpu'), dis_audio_features_file)
        torch.save(ref_audio_features.to('cpu'), ref_audio_features_file)
    else:
        dis_audio_features = torch.load(dis_audio_features_file).to(device)
        ref_audio_features = torch.load(ref_audio_features_file).to(device)

    print('Audio features are saved...\n')

    # Quality prediction using ANNAVQA
    seg_num = 4
    tmp_video_length = 96
    min_audio_len = 96
    feat_dim = 4096
    seg_video_len = int(tmp_video_length / seg_num)
    seg_audio_len = int(min_audio_len / seg_num)
    length = np.zeros((1, 1))
    length[0] = seg_video_len + seg_audio_len
    length = torch.from_numpy(length).float()

    model = ANNAVQA()
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()
    y_pred = 0
    with torch.no_grad():
        for seg_index in range(seg_num):
            features = torch.zeros(1, seg_video_len + seg_audio_len, feat_dim, device=device)

            video_features = abs(
                dis_video_features[seg_video_len * seg_index:seg_video_len * (seg_index + 1), :feat_dim] -
                ref_video_features[seg_video_len * seg_index: seg_video_len * (seg_index + 1), :feat_dim])

            audio_features = abs(
                ref_audio_features[seg_audio_len * seg_index:seg_audio_len * (seg_index + 1), :feat_dim]
                - dis_audio_features[seg_audio_len * seg_index:seg_audio_len * (seg_index + 1), :feat_dim])

            Feature = torch.cat([video_features.float(), audio_features.float()], axis=0)
            features[0, :Feature.shape[0], :] = Feature
            y_pred = y_pred + model(features, length, seg_video_len).to('cpu').numpy()

        y_pred = y_pred / seg_num

        print("Predicted quality: {}".format(y_pred[0][0]))
