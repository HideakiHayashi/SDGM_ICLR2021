import argparse
import gc
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
import json
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm
from basenet import simpleCNN_SDGM
from torch_ard import ELBOLoss


def train():
    utils.makedirs(args.save_dir)
    with open(f"{args.save_dir}/params.txt", "w") as f:
        json.dump(args.__dict__, f)
    if args.print_to_log:
        sys.stdout = open(f"{args.save_dir}/log.txt", "w")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_dataset = datasets.MNIST(
        "../data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    model = simpleCNN_SDGM(10).to(device)
    model.train()
    def get_kl_weight(epoch, max_epoch): return min(
        1, 1e-9 * epoch / max_epoch)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = ELBOLoss(model, F.cross_entropy).to("cuda")
    with tqdm(range(args.n_epochs)) as pbar:
        for epoch in pbar:
            total_loss = 0
            kl_weight = get_kl_weight(epoch, args.n_epochs)
            for (inputs, labels) in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs, labels)
                loss = criterion(outputs, labels, 1, kl_weight)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            pbar.set_description(f"Loss: {total_loss}")
    torch.save(model.state_dict(), f"{args.save_dir}/model.pth")
    print("Saved models.")
    del model, train_dataset
    gc.collect()


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    test_dataset = datasets.MNIST(
        "../data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True)
    model = simpleCNN_SDGM(10).to(device)
    model.eval()
    model.load_state_dict(torch.load(f"{args.save_dir}/model.pth"))
    correct = 0
    total = 0
    with torch.no_grad():
        for (inputs, labels) in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100.0 * correct / total
    np.savetxt(f"{args.save_dir}/accuracy.txt", np.array([acc]), fmt="%.5f")
    print(f"Accuracy of the network on the 10000 test images: {acc} %")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Sparse Discriminative Gaussian Mixture")
    parser.add_argument("--data_root", type=str, default="../data")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default='./experiment')
    parser.add_argument("--print_to_log", action="store_true",
                        help="If true, directs std-out to log file")
    parser.add_argument("--mode", type=str, default=None,
                        choices=["train", "test", None])
    args = parser.parse_args()
    if args.mode == "train":
        train()
    elif args.mode == "test":
        test()
    else:
        pass
