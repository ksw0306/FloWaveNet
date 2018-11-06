# FloWaveNet

A Generative Flow for Raw Audio

This is a PyTorch implementation of FloWaveNet.

For the purpose of parallel sampling, we propose FloWaveNet, a flow-based generative model for raw audio synthesis.
FloWaveNet can generate audio samples as fast as ClariNet and Parallel WaveNet, while its training procedure is really easy and stable. Our generated audio samples are available at [http://bit.ly/2zpsElV](http://bit.ly/2zpsElV)


<img src="png/model.png">



# Requirements

PyTorch 0.4.1 & python 3.6 & Librosa

# Examples

#### Step 1. Download Dataset

- LJSpeech : [https://keithito.com/LJ-Speech-Dataset/](https://keithito.com/LJ-Speech-Dataset/)

#### Step 2. Preprocessing (Preparing Mel Spectrogram)

`python preprocessing.py --in_dir ljspeech --out_dir DATASETS/ljspeech`

#### Step 3. Train

`python train.py --model_name flowavenet --batch_size 8 --n_block 8 --n_flow 6 --n_layer 2 --causal no`

#### Step 4. Synthesize

This step should be followed by Step 3 to load pre-trained model.

--load_step CHECKPOINT # of Pre-trained model

ex) `python synthesize.py --model_name flowavenet --n_block 8 --n_flow 6 --n_layer 2 --causal no --load_step 100000`


# Sample Link

Sample Link : [http://bit.ly/2zpsElV](http://bit.ly/2zpsElV)


- Results 1 : Model Comparisons (WaveNet (MoL, Gaussian), ClariNet and FloWaveNet)

- Results 2 : Temperature effect on Audio Quality Trade-off (Temperature T : 0.0 ~ 1.0, Model : Gaussian IAF and FloWaveNet)

- Results 3 : Analysis of ClariNet Loss Terms (Loss functions : 1. KLD + Frame Loss 2. Only KL 3. Only Frame)

- Results 4 : Context Block and Long term Dependency (FloWaveNet : 8 Context Blocks, FloWaveNet_small : 6 Context Blocks)

- Results 5 : Causality of WaveNet Dilated Convolutions (FloWaveNet : Non-causal WaveNet Affine Coupling Layers, FloWaveNet_causal : Causal WaveNet Affine Coupling Layers)


# Reference

- WaveNet vocoder : [https://github.com/r9y9/wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder)
- glow-pytorch : [https://github.com/rosinality/glow-pytorch](https://github.com/rosinality/glow-pytorch)
