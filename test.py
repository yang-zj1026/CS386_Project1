import torch
from scipy.stats import spearmanr, pearsonr
import argparse
from dataset import ANNAVQA_Dataset
from model import MyModel
import torch.utils.data as data
from utils import get_logger


def test(model, test_loader, args, logger=None):
    seg_video_len = int(args.tmp_video_length / args.seg_num)
    length = torch.zeros((args.batch_size, 1))
    model.eval()
    model.to(args.device)

    pred_mos_all = torch.Tensor().to(args.device)
    gt_mos_all = torch.Tensor().to(args.device)

    for i, (test_data, gt_mos, _) in enumerate(test_loader):
        test_data = test_data.to(args.device)
        gt_mos = gt_mos.to(args.device)
        pred_mos = model(test_data, length, seg_video_len)
        pred_mos_all = torch.cat((pred_mos_all, pred_mos), 0)
        gt_mos_all = torch.cat((gt_mos_all, gt_mos), 0)

    srcc = spearmanr(pred_mos_all.detach().cpu(), gt_mos_all.detach().cpu())[0]
    plcc = pearsonr(pred_mos_all.detach().cpu(), gt_mos_all.detach().cpu())[0]
    if logger:
        logger.info("Test: SRCC {:.4f}\t PLCC {:.4f}\n".format(srcc, plcc))
    else:
        print("Test: SRCC {:.4f}\t PLCC {:.4f}\n".format(srcc, plcc))
    return srcc, plcc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing parameters')
    parser.add_argument(
        '--seg_num', default=4, type=int, help='Seg no of short videos')
    parser.add_argument(
        '--tmp_video_length', default=96, type=int)
    parser.add_argument(
        '--min_audio_len', default=96, type=int)
    parser.add_argument(
        '--feat_dim', default=4096, type=int)
    parser.add_argument(
        '--model_path', default='my_model.pt', type=str,
        help='model path')

    args = parser.parse_args()

    seg_video_len = int(args.tmp_video_length / args.seg_num)
    seg_audio_len = int(args.min_audio_len / args.seg_num)
    length = torch.zeros((args.batch_size, 1))

    test_class = ['RedKayak', 'PowerDig']

    testing_set = ANNAVQA_Dataset(
        video_feature_dir='./video_features', audio_feature_dir='./audio_features',
        classes=test_class, seg_num=args.seg_num,
        seg_video_len=seg_video_len, seg_audio_len=seg_audio_len,
        mos_file='mos.pt', feat_dim=args.feat_dim)

    test_loader = data.DataLoader(testing_set, batch_size=args.batch_size, shuffle=True, num_workers=1)

    model = MyModel(min_len=seg_video_len)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    logger = get_logger('test.log')

    test(model, test_loader, args, logger)


