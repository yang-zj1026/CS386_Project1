import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNet50(torch.nn.Module):
    """Modified ResNet50 for feature extraction"""

    def __init__(self):
        super(ResNet50, self).__init__()
        self.features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == 7:
                features_mean = nn.functional.adaptive_avg_pool2d(x, 1)
                features_std = global_std_pool2d(x)
                return features_mean, features_std


def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)


class ANN(nn.Module):
    def __init__(self, input_size=4096, reduced_size=128, n_ANNlayers=1, dropout_p=0.5):
        super(ANN, self).__init__()
        self.n_ANNlayers = n_ANNlayers
        self.fc0 = nn.Linear(input_size, reduced_size)
        self.dropout = nn.AlphaDropout(p=dropout_p)
        self.fc = nn.Linear(reduced_size, reduced_size)

    def forward(self, x):
        x = self.fc0(x)
        for i in range(self.n_ANNlayers - 1):
            x = self.fc(self.dropout(F.relu(x)))
        return x


class ANNAVQA(nn.Module):
    def __init__(self, input_size=4096, min_len=48, reduced_size=2048, hidden_size=1024):
        super(ANNAVQA, self).__init__()
        self.hidden_size = hidden_size
        self.min_len = min_len
        self.video_ann = ANN(input_size, reduced_size, 1)
        self.video_rnn = nn.GRU(reduced_size, hidden_size, batch_first=True)
        self.video_q1 = nn.Linear(hidden_size, 512)
        self.video_relu = nn.ReLU()
        self.video_dro = nn.Dropout()
        self.video_q2 = nn.Linear(512, 1)

        self.audio_ann = ANN(input_size, reduced_size, 1)
        self.audio_rnn = nn.GRU(reduced_size, hidden_size, batch_first=True)
        self.audio_q1 = nn.Linear(hidden_size, 512)
        self.audio_relu = nn.ReLU()
        self.audio_dro = nn.Dropout()
        self.audio_q2 = nn.Linear(512, 1)

        self.fc1 = nn.Linear(min_len, 24)
        self.relu1 = nn.ReLU()
        self.dro1 = nn.Dropout()
        self.fc2 = nn.Linear(24, 16)
        self.relu2 = nn.ReLU()
        self.dro2 = nn.Dropout()
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x, input_length, video_length):
        # ipdb.set_trace()
        video_input = self.video_ann(x[:, :video_length, :])
        audio_input = self.audio_ann(x[:, video_length:x.size(1), :])
        video_outputs, _ = self.video_rnn(video_input, self._get_initial_state(x.size(0), x.device))
        audio_outputs, _ = self.audio_rnn(audio_input, self._get_initial_state(x.size(0), x.device))
        video_q1 = self.video_q1(video_outputs)
        audio_q1 = self.audio_q1(audio_outputs)
        video_relu = self.video_relu(video_q1)
        audio_relu = self.audio_relu(audio_q1)
        video_dro = self.video_dro(video_relu)
        audio_dro = self.audio_dro(audio_relu)
        video_q = self.video_q2(video_dro)
        audio_q = self.audio_q2(audio_dro)
        score = torch.zeros(x.shape[0], device=x.device)
        for i in range(x.shape[0]):
            video_qi = video_q[i, :]
            audio_qi = audio_q[i, :]
            fc1 = self.fc1(torch.cat([video_qi.squeeze(), audio_qi.squeeze()]))
            relu1 = self.relu1(fc1)
            dro1 = self.dro1(relu1)
            fc2 = self.fc2(dro1)
            relu2 = self.relu2(fc2)
            dro2 = self.dro2(relu2)
            score[i] = self.fc3(dro2)
        return score

    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h0


class MyModel(nn.Module):
    def __init__(self, input_size=4096, min_len=24, reduced_size=2048, hidden_size=1024):
        super(MyModel, self).__init__()
        self.hidden_size = hidden_size
        self.min_len = min_len
        self.video_ann = ANN(input_size, reduced_size, 1)
        self.video_rnn = nn.GRU(reduced_size, hidden_size, batch_first=True)
        self.video_q1 = nn.Linear(hidden_size, 512)
        self.video_relu = nn.ReLU()
        self.video_dro = nn.Dropout()
        self.video_q2 = nn.Linear(512, 128)

        self.audio_ann = ANN(input_size, reduced_size, 1)
        self.audio_rnn = nn.GRU(reduced_size, hidden_size, batch_first=True)
        self.audio_q1 = nn.Linear(hidden_size, 512)
        self.audio_relu = nn.ReLU()
        self.audio_dro = nn.Dropout()
        self.audio_q2 = nn.Linear(512, 128)

        self.w_a_rgb = nn.Bilinear(128, 128, 1, bias=True)
        self.fc1 = nn.Linear(min_len, 16)
        self.relu1 = nn.ReLU()
        self.dro1 = nn.Dropout()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x, input_length, video_length):
        # ipdb.set_trace()
        video_input = self.video_ann(x[:, :video_length, :])
        audio_input = self.audio_ann(x[:, video_length:x.size(1), :])
        video_outputs, _ = self.video_rnn(video_input, self._get_initial_state(x.size(0), x.device))
        audio_outputs, _ = self.audio_rnn(audio_input, self._get_initial_state(x.size(0), x.device))
        video_q1 = self.video_q1(video_outputs)
        audio_q1 = self.audio_q1(audio_outputs)
        video_relu = self.video_relu(video_q1)
        audio_relu = self.audio_relu(audio_q1)
        video_dro = self.video_dro(video_relu)
        audio_dro = self.audio_dro(audio_relu)
        video_q2 = self.video_q2(video_dro)
        audio_q2 = self.audio_q2(audio_dro)
        fusion_feature = self.w_a_rgb(video_q2, audio_q2)
        score = torch.zeros(x.shape[0], device=x.device)
        for i in range(x.shape[0]):
            fc1 = self.fc1(fusion_feature[i, :].squeeze())
            relu1 = self.relu1(fc1)
            dro1 = self.dro1(relu1)
            # fc2 = self.fc2(dro1)
            # relu2 = self.relu2(fc2)
            # dro2 = self.dro2(relu2)
            score[i] = self.fc2(dro1)
        return score

    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h0