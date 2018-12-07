import torch
from torch.utils.data import DataLoader
from data import LJspeechDataset, collate_fn_synthesize
from model import Flowavenet
from torch.distributions.normal import Normal
import numpy as np
import librosa
import os
import argparse
import time

torch.backends.cudnn.benchmark = True
np.set_printoptions(precision=4)
parser = argparse.ArgumentParser(description='Train FloWaveNet of LJSpeech',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default='./DATASETS/ljspeech/', help='Dataset Path')
parser.add_argument('--sample_path', type=str, default='./samples', help='Sample Path')
parser.add_argument('--model_name', type=str, default='flowavenet', help='Model Name')
parser.add_argument('--num_samples', type=int, default=10, help='# of audio samples')
parser.add_argument('--load_step', type=int, default=0, help='Load Step')
parser.add_argument('--temp', type=float, default=0.7, help='Temperature')
parser.add_argument('--load', '-l', type=str, default='./params', help='Checkpoint path to resume / test.')
parser.add_argument('--n_layer', type=int, default=2, help='Number of layers')
parser.add_argument('--n_flow', type=int, default=6, help='Number of layers')
parser.add_argument('--n_block', type=int, default=8, help='Number of layers')
parser.add_argument('--cin_channels', type=int, default=80, help='Cin Channels')
parser.add_argument('--causal', type=str, default='no', help='Casuality')
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers')

parser.add_argument('--log', type=str, default='./log', help='Log folder.')
args = parser.parse_args()

if not os.path.isdir(args.sample_path):
    os.makedirs(args.sample_path)
if not os.path.isdir(os.path.join(args.sample_path, args.model_name)):
    os.makedirs(os.path.join(args.sample_path, args.model_name))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# LOAD DATASETS
test_dataset = LJspeechDataset(args.data_path, False, 0.1)
synth_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn_synthesize,
                          num_workers=args.num_workers, pin_memory=True)


def build_model():
    causality = True if args.causal == 'yes' else False
    model = Flowavenet(in_channel=1,
                       cin_channel=args.cin_channels,
                       n_block=args.n_block,
                       n_flow=args.n_flow,
                       n_layer=args.n_layer,
                       affine=True,
                       causal=causality,
                       pretrained=True)
    return model


def synthesize(model):
    global global_step
    model.eval()
    for batch_idx, (x, c) in enumerate(synth_loader):
        if batch_idx < args.num_samples:
            x, c = x.to(device), c.to(device)

            q_0 = Normal(x.new_zeros(x.size()), x.new_ones(x.size()))
            z = q_0.sample() * args.temp
            torch.cuda.synchronize()
            start_time = time.time()

            with torch.no_grad():
                y_gen = model.reverse(z, c).squeeze()
            torch.cuda.synchronize()
            print('{} seconds'.format(time.time() - start_time))
            wav = y_gen.to(torch.device("cpu")).data.numpy()
            wav_name = '{}/{}/generate_{}_{}_{}.wav'.format(args.sample_path, args.model_name,
                                                            global_step, batch_idx, args.temp)
            librosa.output.write_wav(wav_name, wav, sr=22050)
            print('{} Saved!'.format(wav_name))


def load_checkpoint(step, model):
    checkpoint_path = os.path.join(args.load, args.model_name, "checkpoint_step{:09d}.pth".format(step))
    print("Load checkpoint from: {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    # generalized load procedure for both single-gpu and DataParallel models
    # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except RuntimeError:
        print("INFO: this model is trained with DataParallel. Creating new state_dict without module...")
        state_dict = checkpoint["state_dict"]
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    return model


if __name__ == "__main__":
    step = args.load_step
    global_step = step
    model = build_model()
    model = load_checkpoint(step, model)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        synthesize(model)
