import torch
import os
import torch.utils.data as data


train_class = ['FootMusic', 'Sparks', 'BigGreenRabbit', 'RedKayak']
test_class = ['PowerDig']


def make_dataset(classes, seg_num):
    dist_types = ['QP16', 'QP16_S', 'QP35', 'QP35_S', 'QP42', 'QP42_S', 'QP50', 'QP50_S']
    audio_types = ['8kbps', '32kbps', '128kbps']
    data_list = []
    for class_name in classes:
        for dist_type in dist_types:
            for audio_type in audio_types:
                for seg_index in range(seg_num):
                    data_list.append([class_name, dist_type, audio_type, seg_index])

    return data_list


def get_features(dis_feature_path, ref_feature_path, seg_len, seg_index, feat_dim):
    dis_feat = torch.load(dis_feature_path)
    dis_feat = dis_feat[seg_len * seg_index: seg_len * (seg_index + 1), :feat_dim]
    ref_feat = torch.load(ref_feature_path)
    ref_feat = ref_feat[seg_len * seg_index: seg_len * (seg_index + 1), :feat_dim]

    return abs(dis_feat - ref_feat)


class ANNAVQA_Dataset(data.Dataset):
    def __init__(self, video_feature_dir, audio_feature_dir, classes, seg_num,
                 seg_video_len, seg_audio_len, mos_file, feat_dim):
        self.video_feat_dir = video_feature_dir
        self.audio_feat_dir = audio_feature_dir
        self.data = make_dataset(classes, seg_num)
        self.mos_file = mos_file
        self.seg_video_len = seg_video_len
        self.seg_audio_len = seg_audio_len
        self.feat_dim = feat_dim

    def __getitem__(self, index):
        video_name, dist_type, audio_type, seg_index = self.data[index]

        dis_video_feat_path = os.path.join(self.video_feat_dir, '{}_{}.pt'.format(video_name, dist_type))
        ref_video_feat_path = os.path.join(self.video_feat_dir, '{}_{}_ref.pt'.format(video_name, dist_type))
        video_features = get_features(dis_video_feat_path, ref_video_feat_path,
                                      self.seg_video_len, seg_index, self.feat_dim)

        dis_audio_feat_path = os.path.join(self.audio_feat_dir, '{}_{}.pt'.format(video_name, audio_type))
        ref_audio_feat_path = os.path.join(self.audio_feat_dir, '{}.pt'.format(video_name))
        audio_features = get_features(dis_audio_feat_path, ref_audio_feat_path,
                                      self.seg_audio_len, seg_index, self.feat_dim)

        features = torch.cat([video_features.float(), audio_features.float()], axis=0)

        mos_dict = torch.load(self.mos_file)
        key_name = video_name + '-' + dist_type + '-' + audio_type
        mos_score = mos_dict[key_name]

        return features, mos_score, key_name

    def __len__(self):
        return len(self.data)