import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from data import LJspeechDataset, collate_fn, collate_fn_synthesize
from model import Flowavenet
from torch.distributions.normal import Normal
import numpy as np
import librosa
import os
import argparse
import time
import json
import gc

torch.backends.cudnn.benchmark = True
np.set_printoptions(precision=4)
torch.manual_seed(1111)

parser = argparse.ArgumentParser(description='Train FloWaveNet of LJSpeech',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default='./DATASETS/ljspeech/', help='Dataset Path')
parser.add_argument('--sample_path', type=str, default='./samples', help='Sample Path')
parser.add_argument('--save', '-s', type=str, default='./params', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='./params', help='Checkpoint path')
parser.add_argument('--log', type=str, default='./log', help='Log folder.')
parser.add_argument('--model_name', type=str, default='flowavenet', help='Model Name')
parser.add_argument('--load_step', type=int, default=0, help='Load Step')
parser.add_argument('--epochs', '-e', type=int, default=5000, help='Number of epochs to train.')
parser.add_argument('--batch_size', '-b', type=int, default=2, help='Batch size.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='The Learning Rate.')
parser.add_argument('--loss', type=str, default='./loss', help='Folder to save loss')
parser.add_argument('--n_layer', type=int, default=2, help='Number of layers')
parser.add_argument('--n_flow', type=int, default=6, help='Number of layers')
parser.add_argument('--n_block', type=int, default=8, help='Number of layers')
parser.add_argument('--cin_channels', type=int, default=80, help='Cin Channels')
parser.add_argument('--causal', type=str, default='no', help='Casuality')
parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')
parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPUs to use. >1 uses DataParallel')
args = parser.parse_args()

# Init logger
if not os.path.isdir(args.log):
    os.makedirs(args.log)

# Checkpoint dir
if not os.path.isdir(args.save):
    os.makedirs(args.save)
if not os.path.isdir(args.loss):
    os.makedirs(args.loss)
if not os.path.isdir(args.sample_path):
    os.makedirs(args.sample_path)
if not os.path.isdir(os.path.join(args.sample_path, args.model_name)):
    os.makedirs(os.path.join(args.sample_path, args.model_name))
if not os.path.isdir(os.path.join(args.save, args.model_name)):
    os.makedirs(os.path.join(args.save, args.model_name))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# LOAD DATASETS
train_dataset = LJspeechDataset(args.data_path, True, 0.1)
test_dataset = LJspeechDataset(args.data_path, False, 0.1)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                          num_workers=args.num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn,
                         num_workers=args.num_workers, pin_memory=True)
synth_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn_synthesize,
                          num_workers=args.num_workers, pin_memory=True)


def build_model():
    pretrained = True if args.load_step > 0 else False
    model = Flowavenet(in_channel=1,
                       cin_channel=args.cin_channels,
                       n_block=args.n_block,
                       n_flow=args.n_flow,
                       n_layer=args.n_layer,
                       affine=True,
                       pretrained=pretrained,
                       block_per_split=args.block_per_split)
    return model


def train(epoch, model, optimizer, scheduler):
    global global_step
    epoch_loss = 0.0
    running_loss = [0., 0., 0.]
    model.train()
    display_step = 100
    for batch_idx, (x, c) in enumerate(train_loader):
        scheduler.step()
        global_step += 1

        x, c = x.to(device), c.to(device)

        optimizer.zero_grad()
        log_p, logdet = model(x, c)
        log_p, logdet = torch.mean(log_p), torch.mean(logdet)

        loss = -(log_p + logdet)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

        running_loss[0] += loss.item() / display_step
        running_loss[1] += log_p.item() / display_step
        running_loss[2] += logdet.item() / display_step

        epoch_loss += loss.item()
        if (batch_idx + 1) % display_step == 0:
            print('Global Step : {}, [{}, {}] [Log pdf, Log p(z), Log Det] : {}'
                  .format(global_step, epoch, batch_idx + 1, np.array(running_loss)))
            running_loss = [0., 0., 0.]
        del x, c, log_p, logdet, loss
    del running_loss
    gc.collect()
    print('{} Epoch Training Loss : {:.4f}'.format(epoch, epoch_loss / (len(train_loader))))
    return epoch_loss / len(train_loader)


def evaluate(model):
    model.eval()
    running_loss = [0., 0., 0.]
    epoch_loss = 0.
    display_step = 100
    for batch_idx, (x, c) in enumerate(test_loader):
        x, c = x.to(device), c.to(device)
        log_p, logdet = model(x, c)
        log_p, logdet = torch.mean(log_p), torch.mean(logdet)
        loss = -(log_p + logdet)

        running_loss[0] += loss.item() / display_step
        running_loss[1] += log_p.item() / display_step
        running_loss[2] += logdet.item() / display_step
        epoch_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            print('Global Step : {}, [{}, {}] [Log pdf, Log p(z), Log Det] : {}'
                  .format(global_step, epoch, batch_idx + 1, np.array(running_loss)))
            running_loss = [0., 0., 0.]
        del x, c, log_p, logdet, loss
    del running_loss
    epoch_loss /= len(test_loader)
    print('Evaluation Loss : {:.4f}'.format(epoch_loss))
    return epoch_loss


def synthesize(model):
    global global_step
    model.eval()
    for batch_idx, (x, c) in enumerate(synth_loader):
        if batch_idx == 0:
            x, c = x.to(device), c.to(device)

            q_0 = Normal(x.new_zeros(x.size()), x.new_ones(x.size()))
            z = q_0.sample()

            start_time = time.time()
            with torch.no_grad():
                if args.num_gpu == 1:
                    y_gen = model.reverse(z, c).squeeze()
                else:
                    y_gen = model.module.reverse(z, c).squeeze()
            wav = y_gen.to(torch.device("cpu")).data.numpy()
            wav_name = '{}/{}/generate_{}_{}.wav'.format(args.sample_path, args.model_name, global_step, batch_idx)
            print('{} seconds'.format(time.time() - start_time))
            librosa.output.write_wav(wav_name, wav, sr=22050)
            print('{} Saved!'.format(wav_name))
            del x, c, z, q_0, y_gen, wav


def save_checkpoint(model, optimizer, scheduler, global_step, global_epoch):
    checkpoint_path = os.path.join(args.save, args.model_name, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()
    torch.save({"state_dict": model.state_dict(),
                "optimizer": optimizer_state,
                "scheduler": scheduler_state,
                "global_step": global_step,
                "global_epoch": global_epoch}, checkpoint_path)


def load_checkpoint(step, model, optimizer, scheduler):
    global global_step
    global global_epoch

    checkpoint_path = os.path.join(args.save, args.model_name, "checkpoint_step{:09d}.pth".format(step))
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

    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    return model, optimizer, scheduler


if __name__ == "__main__":
    model = build_model()
    model.to(device)

    pretrained = True if args.load_step > 0 else False
    if pretrained is False:
        # do ActNorm initialization first (if model.pretrained is True, this does nothing so no worries)
        x_seed, c_seed = next(iter(train_loader))
        x_seed, c_seed = x_seed.to(device), c_seed.to(device)
        with torch.no_grad():
            _, _ = model(x_seed, c_seed)
        del x_seed, c_seed, _
    # then convert the model to DataParallel later (since ActNorm init from the DataParallel is wacky)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200000, gamma=0.5)
    criterion_frame = nn.MSELoss()

    global_step = 0
    global_epoch = 0
    load_step = args.load_step

    log = open(os.path.join(args.log, '{}.txt'.format(args.model_name)), 'w')
    state = {k: v for k, v in args._get_kwargs()}

    if load_step == 0:
        list_train_loss, list_loss = [], []
        log.write(json.dumps(state) + '\n')
        test_loss = 100.0
    else:
        model, optimizer, scheduler = load_checkpoint(load_step, model, optimizer, scheduler)
        list_train_loss = np.load('{}/{}_train.npy'.format(args.loss, args.model_name)).tolist()
        list_loss = np.load('{}/{}.npy'.format(args.loss, args.model_name)).tolist()
        list_train_loss = list_train_loss[:global_epoch]
        list_loss = list_loss[:global_epoch]
        test_loss = np.min(list_loss)

    if args.num_gpu > 1:
        print("num_gpu > 1 detected. converting the model to DataParallel...")
        model = torch.nn.DataParallel(model)

    for epoch in range(global_epoch + 1, args.epochs + 1):
        training_epoch_loss = train(epoch, model, optimizer, scheduler)
        with torch.no_grad():
            test_epoch_loss = evaluate(model)

        state['training_loss'] = training_epoch_loss
        state['eval_loss'] = test_epoch_loss
        state['epoch'] = epoch
        list_train_loss.append(training_epoch_loss)
        list_loss.append(test_epoch_loss)

        if test_loss > test_epoch_loss:
            test_loss = test_epoch_loss
            save_checkpoint(model, optimizer, scheduler, global_step, epoch)
            print('Epoch {} Model Saved! Loss : {:.4f}'.format(epoch, test_loss))
            with torch.no_grad():
                synthesize(model)
        np.save('{}/{}_train.npy'.format(args.loss, args.model_name), list_train_loss)
        np.save('{}/{}.npy'.format(args.loss, args.model_name), list_loss)

        log.write('%s\n' % json.dumps(state))
        log.flush()
        print(state)
        gc.collect()

    log.close()
