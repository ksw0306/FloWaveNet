import torch
from torch.utils.data import DataLoader
from data import LJspeechDataset, collate_fn_synthesize
from model import Glowavenet
from torch.distributions.normal import Normal
import numpy as np
import librosa
import os
import argparse
import time

torch.backends.cudnn.benchmark = True
np.set_printoptions(precision=4)

parser = argparse.ArgumentParser(description='Train GloWaveNet of LJSpeech',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default='./DATASETS/ljspeech/', help='Dataset Path')
parser.add_argument('--sample_path', type=str, default='./samples', help='Sample Path')
parser.add_argument('--model_name', type=str, default='glowavenet_12', help='Model Name')
parser.add_argument('--load_step', type=int, default=0, help='Load Step')
parser.add_argument('--load', '-l', type=str, default='./params', help='Checkpoint path to resume / test.')
parser.add_argument('--n_layer', type=int, default=2, help='Number of layers')
parser.add_argument('--n_flow', type=int, default=6, help='Number of layers')
parser.add_argument('--n_block', type=int, default=8, help='Number of layers')
parser.add_argument('--cin_channels', type=int, default=80, help='Cin Channels')
parser.add_argument('--causal', type=str, default='yes', help='Casuality')
parser.add_argument('--gpu_index', '-g', type=str, default="0", help='GPU index')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')
# i/o
parser.add_argument('--log', type=str, default='./log', help='Log folder.')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# LOAD DATASETS
test_dataset = LJspeechDataset(args.data_path, False, 0.1)
synth_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn_synthesize,
                          num_workers=args.num_workers, pin_memory=True)


def build_model():
    causality = True if args.causal == 'yes' else False
    model = Glowavenet(in_channel=1,
                       cin_channel=args.cin_channels,
                       n_block=args.n_block,
                       n_flow=args.n_flow,
                       n_layer=args.n_layer,
                       affine=True,
                       causal=causality)
    return model


def synthesize(model):
    global global_step
    model.eval()
    for batch_idx, (x, c) in enumerate(synth_loader):
        if batch_idx < 100:
            x, c = x.to(device), c.to(device)

            q_0 = Normal(x.new_zeros(x.size()), x.new_ones(x.size()))
            z = q_0.sample()
            start_time = time.time()

            with torch.no_grad():
                y_gen = model.reverse(z, c).squeeze()
            wav = y_gen.to(torch.device("cpu")).data.numpy()
            wav_name = '{}/{}/generate_{}_{}.wav'.format(args.sample_path, args.model_name, global_step, batch_idx)
            print('{} seconds'.format(time.time() - start_time))
            librosa.output.write_wav(wav_name, wav, sr=22050)
            print('{} Saved!'.format(wav_name))


def load_checkpoint(step, model):
    checkpoint_path = os.path.join(args.load, args.model_name, "checkpoint_step{:09d}.pth".format(step))
    print("Load checkpoint from: {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    return model


step = args.load_step
global_step = step
model = build_model()
model = load_checkpoint(step, model)
model = model.to(device)
model.eval()

with torch.no_grad():
    synthesize(model)




