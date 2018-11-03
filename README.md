# FloWaveNet

A Generative Flow for Raw Audio

# Requirements

PyTorch 0.4.1 & python 3.6 & Librosa

# Examples

#### Step 1. Download Dataset

LJSpeech : [https://keithito.com/LJ-Speech-Dataset/](https://keithito.com/LJ-Speech-Dataset/)

#### Step 2. Preprocessing (Preparing Mel Spectrogram)

`python preprocessing.py --in_dir ljspeech --out_dir DATASETS/ljspeech`

#### Step 3. Train

`python train.py --model_name flowavenet --batch_size 8 --n_block 8 --n_flow 6 --n_layer 2 --causal no`

#### Step 4. Synthesize

`python synthesize.py --model_name flowavenet --n_block 8 --n_flow 6 --n_layer 2 --causal no --load_step 804804`


# Reference

[https://github.com/r9y9/wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder)

[https://github.com/rosinality/glow-pytorch](https://github.com/rosinality/glow-pytorch)


