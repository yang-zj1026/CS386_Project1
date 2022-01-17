from torchvision import transforms
from utils import extract_video_features, calcSpectrogram, get_audio_features
import os
import torch
import time

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_names = ['FootMusic', 'Sparks', 'BigGreenRabbit', 'RedKayak', 'PowerDig']
    # video_names = ['Sparks']
    dist_types = ['QP16', 'QP16_S', 'QP35', 'QP35_S', 'QP42', 'QP42_S', 'QP50', 'QP50_S']
    audio_types = ['8kbps', '32kbps', '128kbps']
    frame_rate = {'FootMusic': 29.97, 'Sparks': 29.97, 'BigGreenRabbit': 24, 'RedKayak': 29.97, 'PowerDig': 29.97}

    video_dir = './Videos'
    audio_dir = './Audios'
    video_features_dir = './video_features/'
    audio_features_dir = './audio_features/'

    for video_name in video_names:
        # for dist_type in dist_types:
        for audio_type in audio_types:
            # dis_video_path = os.path.join(video_dir, video_name + '_' + dist_type + '.yuv')
            # ref_video_path = os.path.join(video_dir, video_name + '.yuv')
            dis_audio_path = os.path.join(audio_dir, video_name+'_'+audio_type+'.wav')
            ref_audio_path = os.path.join(audio_dir, video_name+'.wav')
            #
            # dis_video_features_file = video_features_dir + video_name + '_' + dist_type + '.pt'
            # ref_video_features_file = video_features_dir + video_name + '_' + dist_type + '_ref.pt'
            # if not os.path.isdir(video_features_dir):
            #     os.makedirs(video_features_dir)
            #
            # if not os.path.isfile(dis_video_features_file) or not os.path.isfile(ref_video_features_file):
            #     st = time.time()
            #     dis_video_features, ref_video_features = extract_video_features(dis_video_path,
            #                                                                     ref_video_path,
            #                                                                     1080, 1920,
            #                                                                     device)
            #
            #     torch.save(dis_video_features.to('cpu'), dis_video_features_file)
            #     torch.save(ref_video_features.to('cpu'), ref_video_features_file)
            #     et = time.time()
            #     print("Extraction of %s_%s is finished" % (video_name, dist_type))
            #     print('time:', et - st)

            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            dis_audio_features_file = audio_features_dir + video_name + '_' + audio_type + '.pt'
            ref_audio_features_file = audio_features_dir + video_name + '.pt'

            if not os.path.isdir(audio_features_dir):
                os.makedirs(audio_features_dir)

            if not os.path.isfile(dis_audio_features_file) or not os.path.isfile(ref_audio_features_file):
                [dis_S, dis_T] = calcSpectrogram(dis_audio_path)
                [ref_S, ref_T] = calcSpectrogram(ref_audio_path)
                transforms_dis_audio = transform(dis_S)
                transforms_ref_audio = transform(ref_S)
                dis_audio_features = get_audio_features(transforms_dis_audio, dis_T, frame_rate[video_name], 192,
                                                        device)
                ref_audio_features = get_audio_features(transforms_ref_audio, ref_T, frame_rate[video_name], 192,
                                                        device)
                torch.save(dis_audio_features.to('cpu'), dis_audio_features_file)
                torch.save(ref_audio_features.to('cpu'), ref_audio_features_file)
                print("Extraction of %s_%s is finished\n" % (video_name, audio_type))
