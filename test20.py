from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import sys
import time
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as dst

from utils import AverageMeter, accuracy, transform_time
from utils import load_pretrained_model, save_checkpoint
from utils import create_exp_dir, count_parameters_in_MB
from network import define_tsnet

# 创建一个 ArgumentParser 对象
# ArgumentParser 对象包含了需要处理的命令行参数和选项的信息
# description 参数是一个简短的程序描述，它会在帮助信息的开始部分显示
parser = argparse.ArgumentParser(description="train base net")

# various path
parser.add_argument(
    "--save_root", type=str, default="./results", help="models and logs are saved here"
)
parser.add_argument(
    "--img_root", type=str, default="./datasets", help="path name of image dataset"
)

# training hyper parameters
parser.add_argument(
    "--print_freq",
    type=int,
    default=50,
    help="frequency of showing training results on console",
)
parser.add_argument(
    "--epochs", type=int, default=200, help="number of total epochs to run"
)
parser.add_argument("--batch_size", type=int, default=128, help="The size of batch")
parser.add_argument("--lr", type=float, default=0.1, help="initial learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
parser.add_argument("--num_class", type=int, default=100, help="number of classes")
parser.add_argument("--cuda", type=int, default=1)

# others
parser.add_argument("--seed", type=int, default=2, help="random seed")
parser.add_argument("--note", type=str, default="try", help="note for this run")

# net and dataset choosen
parser.add_argument(
    "--data_name", type=str, required=True, help="name of dataset"
)  # cifar10/cifar100
parser.add_argument(
    "--net_name", type=str, required=True, help="name of basenet"
)  # resnet20/resnet110


# 使用 argparse 库的 parse_known_args 方法解析命令行参数
# args 是一个命名空间，包含了所有的命令行参数
# unparsed 是一个列表，包含了所有未被解析的命令行参数
args, unparsed = parser.parse_known_args()

# 将 args.note 添加到 args.save_root 的路径中
# os.path.join 用于连接两个或更多的路径名组件
args.save_root = os.path.join(args.save_root, args.note)

# 调用 create_exp_dir 函数创建实验目录
# 如果目录已存在，该函数将不会做任何事情
create_exp_dir(args.save_root)

# 定义日志格式
log_format = "%(message)s"

# 设置基本的日志配置
# stream 设置为 sys.stdout，表示日志输出到标准输出
# level 设置为 logging.INFO，表示记录 INFO 级别以上的日志
# format 设置为 log_format，表示日志的格式
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)

# 创建一个 FileHandler，用于将日志记录到文件中
# 文件路径为 args.save_root 下的 "log.txt"
fh = logging.FileHandler(os.path.join(args.save_root, "log.txt"))

# 为 FileHandler 设置格式
fh.setFormatter(logging.Formatter(log_format))

# 获取 root logger，并添加 FileHandler
# 这样，日志既会输出到标准输出，也会记录到文件中
logging.getLogger().addHandler(fh)


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        cudnn.enabled = True
        cudnn.benchmark = True
    logging.info("args = %s", args)
    logging.info("unparsed_args = %s", unparsed)

    logging.info("----------- Network Initialization --------------")

    # 加载模型
    net = define_tsnet(name=args.net_name, num_class=args.num_class, cuda=args.cuda)
    checkpoint = torch.load(
        "/home/corner/Knowledge-Distillation-Zoo/results/base/test-c100-r20/initial_r20.pth.tar"
    )
    net.load_state_dict(checkpoint["net"])
    logging.info("%s", net)
    logging.info("param size = %fMB", count_parameters_in_MB(net))
    logging.info("-----------------------------------------------")

    # save initial parameters
    logging.info("Saving initial parameters......")
    save_path = os.path.join(
        args.save_root, "initial_r{}.pth.tar".format(args.net_name[6:])
    )
    torch.save(
        {
            "epoch": 0,
            "net": net.state_dict(),
            "prec@1": 0.0,
            "prec@5": 0.0,
        },
        save_path,
    )

    # define loss functions
    if args.cuda:
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # define transforms
    if args.data_name == "cifar100":
        dataset = dst.CIFAR100
        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2762)
    else:
        raise Exception("Invalid dataset name...")

    train_transform = transforms.Compose(
        [
            transforms.Pad(4, padding_mode="reflect"),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    # define data loader
    train_loader = torch.utils.data.DataLoader(
        dataset(
            root=args.img_root, transform=train_transform, train=True, download=True
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset(
            root=args.img_root, transform=test_transform, train=False, download=True
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    resetter = WeightResetter(net)

    for epoch in range(1, args.epochs + 1):
        net.load_state_dict(checkpoint["net"])
        resetter.reset_weight_at_index(index=epoch, reset_method="xavier_uniform_")
        # resetter.reset_weight_at_index(index=epoch, reset_method="normal_")

        # evaluate on testing set
        logging.info("Testing the models......")
        test_top1, test_top5 = test(test_loader, net, criterion)

        # save model
        logging.info(f"Saving models for epoch_{epoch}......")
        save_path = os.path.join(args.save_root, "epoch_{}.pth.tar".format(epoch))
        torch.save(
            {
                "epoch": epoch,
                "net": net.state_dict(),
                "prec@1": test_top1,
                "prec@5": test_top5,
            },
            save_path,
        )

class WeightResetter:
    def __init__(self, net):
        # 初始化时，接收一个神经网络模型，并获取其所有权重及路径
        self.net = net
        self.weights_with_path = self.get_weights_with_path(self.net)

    def get_attribute(self, module, attribute_path):
        # 该函数用于获取模型中的特定属性，支持嵌套属性和列表索引
        parts = attribute_path.split('.')
        for part in parts:
            if '[' in part and ']' in part:
                attr, index = part.split('[')[0], int(part.split('[')[1].split(']')[0])
                module = getattr(module, attr)[index]
            else:
                module = getattr(module, part)
        return module

    def reset_weight_at_index(self, index, reset_method='xavier_uniform_'):
        # 该函数用于重置指定索引位置的权重，支持不同的初始化方法
        if index >= len(self.weights_with_path):
            print("Index out of range")
            return

        # 获取指定索引位置处的权重张量及其路径
        weight_path, weight_tensor = self.weights_with_path[index]

        # 重新初始化权重张量
        if isinstance(self.net, nn.DataParallel):
            module = self.net.module
        else:
            module = self.net
        weight_to_reset = self.get_attribute(module, weight_path)

        if reset_method == 'normal_':
            # 使用正态分布进行初始化，均值为当前权重的均值，方差为当前权重的方差
            with torch.no_grad():
                weight_to_reset.weight.data.normal_(mean=weight_tensor.data.mean(), std=weight_tensor.data.std())
        else:
            # 使用原来的方法进行初始化
            reset_func = getattr(init, reset_method)
            weight_to_reset.weight.data = reset_func(weight_tensor.data)

    def get_weights_with_path(self, module):
        # 该函数用于获取模型中所有权重及其路径
        weights_list = []

        def recursive_get_weights(module, path):
            # 递归遍历模型的所有子模块
            for name, child_module in module.named_children():
                # 根据路径是否为 "res1"、"res2" 或 "res3"，生成新的路径
                new_path = f"{path}[{name}]" if path in ["res1", "res2", "res3"] else f"{path}.{name}" if path else name
                # 如果当前子模块是卷积层或全连接层，则将其路径和权重添加到列表中
                if isinstance(child_module, (nn.Conv2d, nn.Linear)):
                    weights_list.append((new_path, child_module.weight))
                else:
                    # 如果当前子模块不是卷积层或全连接层，则递归遍历其子模块
                    recursive_get_weights(child_module, new_path)

        # 如果模型是并行模型，则获取其实际模型，否则，直接使用模型
        actual_model = module.module if isinstance(module, nn.DataParallel) else module
        recursive_get_weights(actual_model, "")

        return weights_list


def test(test_loader, net, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.eval()

    for i, (img, target) in enumerate(test_loader, start=1):
        if args.cuda:
            img = img.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        with torch.no_grad():
            _, _, _, _, _, out = net(img)
            loss = criterion(out, target)

        prec1, prec5 = accuracy(out, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    f_l = [losses.avg, top1.avg, top5.avg]
    logging.info("Loss: {:.4f}, Prec@1: {:.2f}, Prec@5: {:.2f}".format(*f_l))

    return top1.avg, top5.avg


if __name__ == "__main__":
    main()
