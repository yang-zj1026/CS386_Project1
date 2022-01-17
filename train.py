import torch
import argparse
import torch.utils.data as data
from model import ANNAVQA, MyModel
from dataset import ANNAVQA_Dataset
from utils import AverageMeter, get_logger
from test import test
import os


def train_epoch(epoch, model, optimizer, train_loader, args, logger):
    model.train()
    seg_video_len = int(args.tmp_video_length / args.seg_num)
    length = torch.zeros((args.batch_size, 1))
    train_loss = AverageMeter()
    for i, (data, gt_mos, _) in enumerate(train_loader):
        data = data.to(args.device)
        gt_mos = gt_mos.to(args.device)
        pred_mos = model(data, length, seg_video_len)
        loss = torch.sum(abs(gt_mos - pred_mos)) / gt_mos.shape[0]
        train_loss.update(loss, gt_mos.shape[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    logger.info('Epoch {:3d}\t Train Loss: {train_loss.val:.4f}'.format(
        epoch + 1,
        train_loss=train_loss))


train_class = ['FootMusic', 'Sparks', 'BigGreenRabbit']
test_class = ['RedKayak', 'PowerDig']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument(
        '--seg_num', default=4, type=int, help='Seg no of short videos')
    parser.add_argument(
        '--tmp_video_length', default=96, type=int)
    parser.add_argument(
        '--min_audio_len', default=96, type=int)
    parser.add_argument(
        '--feat_dim', default=4096, type=int)
    parser.add_argument(
        '--n_epochs', default=200, type=int)
    parser.add_argument(
        '--lr', default=0.0005, type=float)
    parser.add_argument(
        '--batch_size', default=64, type=int)
    parser.add_argument(
        '--device', default='cuda', type=str)
    parser.add_argument(
        '--result_dir', default='./experiments', type=str)
    parser.add_argument(
        '--exp_name', default='my_model', type=str)

    args = parser.parse_args()
    args.seg_num = 6
    args.exp_name = 'base_seg_6'

    seg_video_len = int(args.tmp_video_length / args.seg_num)
    seg_audio_len = int(args.min_audio_len / args.seg_num)
    length = torch.zeros((args.batch_size, 1))

    training_set = ANNAVQA_Dataset(
        video_feature_dir='./video_features', audio_feature_dir='./audio_features',
        classes=train_class, seg_num=args.seg_num,
        seg_video_len=seg_video_len, seg_audio_len=seg_audio_len,
        mos_file='mos.pt', feat_dim=args.feat_dim)

    testing_set = ANNAVQA_Dataset(
        video_feature_dir='./video_features', audio_feature_dir='./audio_features',
        classes=test_class, seg_num=args.seg_num,
        seg_video_len=seg_video_len, seg_audio_len=seg_audio_len,
        mos_file='mos.pt', feat_dim=args.feat_dim)

    train_loader = data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = data.DataLoader(testing_set, batch_size=args.batch_size, shuffle=True, num_workers=1)

    model = ANNAVQA(min_len=seg_video_len*2)
    model = model.to(args.device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if not os.path.isdir(args.result_dir):
        os.makedirs(args.result_dir)

    log_file = os.path.join(args.result_dir, args.exp_name + '.log')
    logger = get_logger(log_file)

    logger.info('----------- Train -----------')
    best_srcc, best_plcc = 0, 0
    for i in range(args.n_epochs):
        train_epoch(i, model, optimizer, train_loader, args, logger)
        if (i + 1) % 10 == 0:
            srcc, plcc = test(model, test_loader, args, logger)
            if (srcc > best_srcc or plcc > best_plcc) and (srcc + plcc) > (best_plcc + best_srcc):
                best_srcc, best_plcc = srcc, plcc
                torch.save(model.state_dict(), 'my_model.pt')

    logger.info("Best - SRCC {:.4f}\t PLCC {:.4f}\n".format(best_srcc, best_plcc))