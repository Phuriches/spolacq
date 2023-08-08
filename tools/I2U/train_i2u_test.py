import json
import os
import statistics
os.chdir('../../tools/I2U')
import torch
import yaml
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from datasets import CaptionDataset
from models_i2u import ImageToUnit


def train(device, loader, model, reconstruction_loss, optimizer):
    accuracies = []
    losses = []

    for imgs, units, seq_lens, padding_masks in loader:
        imgs = imgs.to(device)
        units = units.to(device)
        seq_lens = seq_lens.to(device)
        padding_masks = padding_masks.to(device)

        logits, kl_loss = model(imgs, units, seq_lens, padding_masks)

        logits, _, _, _ = pack_padded_sequence(
            logits, (seq_lens - 1).cpu().tolist(), batch_first=True, enforce_sorted=False
        )
        targets, _, _, _ = pack_padded_sequence(
            units[:, 1:], (seq_lens - 1).cpu().tolist(), batch_first=True, enforce_sorted=False
        )

        loss = reconstruction_loss(logits, targets) + kl_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = torch.sum(torch.argmax(logits, dim=1) == targets) / logits.size(0)
        accuracies.append(accuracy.item())
        losses.append(loss.item())

    return statistics.mean(accuracies), statistics.mean(losses) # here we can also use numpy to calculate mean value


@torch.inference_mode()
def validate(device, loader, model, reconstruction_loss):
    accuracies = []
    losses = []

    for imgs, units, seq_lens, padding_masks in loader:
        imgs = imgs.to(device)
        units = units.to(device)
        seq_lens = seq_lens.to(device)
        padding_masks = padding_masks.to(device)

        logits, kl_loss = model(imgs, units, seq_lens, padding_masks)

        logits, _, _, _ = pack_padded_sequence(
            logits, (seq_lens - 1).cpu().tolist(), batch_first=True, enforce_sorted=False
        )
        targets, _, _, _ = pack_padded_sequence(
            units[:, 1:], (seq_lens - 1).cpu().tolist(), batch_first=True, enforce_sorted=False
        )

        loss = reconstruction_loss(logits, targets) + kl_loss

        accuracy = torch.sum(torch.argmax(logits, dim=1) == targets) / logits.size(0)
        accuracies.append(accuracy.item())
        losses.append(loss.item())

    return statistics.mean(accuracies), statistics.mean(losses)


def main(config):
    import time
    import socket
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--word_map', default='default')
    parser.add_argument('--pe_max_length', default='102')

    args = parser.parse_args()
    word_map = args.word_map
    pe_max_length = int(args.pe_max_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ============================ Added ===============================
    assert word_map in ['default', 'word_map_20', 'word_map_100', 'word_map_5_hubert', 'word_map_10_hubert', 'word_map_20_hubert', 'word_map_100_hubert', 'word_map_spokencoco', 'word_map_rawfood_100']
    if word_map == 'default':
        data_folder = "data_folder"
        word_map = "word_map"
        data_name = "data_name"
        nickname = "default"
    elif word_map == 'word_map_20':
        data_folder = "data_folder_20"
        word_map = "word_map_20"
        data_name = "data_name_20"
        nickname = "resdavenet"
    elif word_map == 'word_map_100':
        data_folder = "data_folder_100"
        word_map = "word_map_100"
        data_name = "data_name_100"
        nickname = "resdavenet"
    elif word_map == 'word_map_5_hubert':
        data_folder = "data_folder_5_hubert"
        word_map = "word_map_5_hubert"
        data_name = "data_name_5"
        nickname = "hubert"
    elif word_map == 'word_map_10_hubert':
        data_folder = "data_folder_10_hubert"
        word_map = "word_map_10_hubert"
        data_name = "data_name_10"
        nickname = "hubert"
    elif word_map == 'word_map_20_hubert':
        data_folder = "data_folder_20_hubert"
        word_map = "word_map_20_hubert"
        data_name = "data_name_20"
        nickname = "hubert"
    elif word_map == 'word_map_100_hubert':
        data_folder = "data_folder_100_hubert"
        word_map = "word_map_100_hubert"
        data_name = "data_name_100"
        nickname = "hubert"
    elif word_map == 'word_map_spokencoco':
        data_folder = "data_folder_spokencoco"
        word_map = "word_map_spokencoco"
        data_name = "data_name_spokencoco"
        nickname = "SpokenCOCO_pretrained_hubert"
    elif word_map == 'word_map_rawfood_100':
        data_folder = "data_folder_rawfood_100"
        word_map = "word_map_rawfood_100"
        data_name  = "data_name_rawfood_100"
        nickname = "rawfood_100"

    print(f'----------- data path -----------')
    print(f'word_map: {word_map}')
    print(f'data_folder: {data_folder}')
    print(f'data_name: {data_name}')
    print(f'nickname: {nickname}')
    print(f'---------------------------------')
    # ====================================================================


    data_folder = os.path.join(os.path.dirname(__file__), "../..", config["I2U"][data_folder])
    word_map_path = os.path.join(os.path.dirname(__file__), "../..", config["I2U"][word_map])
    model_path = os.path.join(os.path.dirname(__file__), "../..", config["I2U"]["model_path"])
    # model_path_front = model_path.split("/")[:-1]
    model_path_end = model_path.split("/")[-1]
    lname = len(model_path_end)
    model_path_front = model_path[:-lname]
    print(f'model_path: {model_path}')
    print(f'model_path_front: {model_path_front}')
    print(f'model_path_end: {model_path_end}')

    train_ID = str(time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()) )[2:].replace(" ", "_") + f"_{socket.gethostname()}" + f"_{word_map}" + f"_{nickname}"

    # print(f'model_path: {model_path}')
    # print(f'train_ID: {train_ID}')
    model_path = model_path_front+train_ID+"_"+model_path_end
    print(f'save model path: {model_path}')
    # exit()
    train_loader = DataLoader(
        CaptionDataset(data_folder, config["I2U"][data_name], "TRAIN"),
        config["I2U"]["batch_size"],
        shuffle=True,
        num_workers=config["I2U"]["num_workers"],
        pin_memory=True,
    )
    print(f'len(train_loader): {len(train_loader)}')

    val_loader = DataLoader(
        CaptionDataset(data_folder, config["I2U"][data_name], "VAL"),
        config["I2U"]["batch_size"],
        num_workers=config["I2U"]["num_workers"],
        pin_memory=True,
    )
    print(f'len(val_loader): {len(val_loader)}')

    with open(word_map_path) as j:
        word_map = json.load(j)

    # pe_max_length = len(word_map) - 2
    print(f'pe_max_length: {pe_max_length}')
    model = ImageToUnit(word_map, max_len=pe_max_length).to(device) # normal max_len=102
    reconstruction_loss = nn.CrossEntropyLoss(ignore_index=word_map["<pad>"], reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["I2U"]["lr"])

    max_accuracy = 0
    for epoch in range(1, 1 + config["I2U"]["epoch"]):
        train_accuracy, train_loss = train(device, train_loader, model, reconstruction_loss, optimizer) # training according to the batch size
        val_accuracy, val_loss = validate(device, val_loader, model, reconstruction_loss)

        print(
            "epoch",
            epoch,
            "train_accuracy",
            train_accuracy,
            "train_loss",
            train_loss,
            "val_accuracy",
            val_accuracy,
            "val_loss",
            val_loss,
            flush=True,
        )

        if max_accuracy < val_accuracy:
            max_accuracy = val_accuracy
            torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    with open("../../conf/spolacq3.yaml") as y:
        config = yaml.safe_load(y)

    main(config)
