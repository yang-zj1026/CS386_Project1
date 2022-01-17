# README
## Description
code for CS386 Assignment 1.**AV-QA: Audio Visual Quality Assessment** @SJTU

### Requirement

- PyTorch 1.5.0
- Matlab

## Preparation
You should follow the steps before training or testing.
### Saliency Detection
Firstly run `sal_position.m` in Matlab to get `test_position.mat`.
```
sal_position ./dis_test.yuv 1080 1920
```
You need to specify the `distorted video`, `video height` and `video width`.

After getting the saliency maps, move them into `./SD`

### Set FFMPEG PATH

Before running the python code, set the ffmpeg path in `utils.py`

```python
skvideo.setFFmpegPath('path to your ffmpeg')
```

### Extract Visual and Audio Features

run `extract_features.py` to get the visual and audio features of the videos. Please be sure that videos are stored in `./Video` directory and audios are stored in `./Audios`. The extracted video and audio features are saved in `./video_features` and `./audio_features`.

## Training

I select videos of FootMusic class, Sparks class and BigGreenRabbit class for training set and videos of RedKayak class and PowerDig class for testing set. 
```shell
python train.py --n_epochs 200 --lr 0.0005 --batch_size 64 --device cuda 
--result-dir './experiment' --exp-name baseline
```
You may need to specify the `n_epochs`, initial learning rate `lr`, batch size ` batch_size`, `distorted audio path` and `frame rate`.

## Testing
Run `test.py` to evalute your model on the testing set. Remember to specify the `--model-path `to be the path to the saved model.

```shell
python test.py --model-path path-to-your-model
```

