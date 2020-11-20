""" Search cell """
# This code fork from https://github.com/khanrc/pt.darts
# modified by Hao Lu for Heart Rate Estimation

import os
import torch
import torch.nn as nn
import numpy as np
import MyDataset
from tensorboardX import SummaryWriter
from config import SearchConfig
import utils
from models.search_cnn import SearchCNNController
from architect import Architect
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as transF
from torchvision import transforms

config = SearchConfig()
device = torch.device("cuda")
# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)

def main():
    logger.info("Logger is set - training start")
    fileRoot = r'/home/hlu/Data/VIPL'
    saveRoot = r'/home/hlu/Data/VIPL_STMap' + str(config.fold_num) + str(config.fold_index)
    n_classes = 1
    input_channels = 3
    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    toTensor = transforms.ToTensor()
    resize = transforms.Resize(size=(64, 300))
    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True  # 网络加速

    if config.reData == 1:
        test_index, train_index = MyDataset.CrossValidation(fileRoot, fold_num=config.fold_num, fold_index=config.fold_index)
    train_data = MyDataset.Data_STMap(root_dir=(saveRoot + '_Train'), frames_num=300,
                                    transform=transforms.Compose([resize, toTensor, normalize]))
    net_crit = nn.L1Loss().to(device)
    model = SearchCNNController(input_channels, config.init_channels, n_classes, config.layers,
                                net_crit, device_ids=config.gpus)
    model._init_weight()
    model = model.to(device)
    # weights optimizer
    w_optim = torch.optim.SGD(model.weights(), config.w_lr, momentum=config.w_momentum,
                              weight_decay=config.w_weight_decay)
    # w_optim = torch.optim.Adam(model.weights(), config.w_lr)
    # alphas optimizer
    alpha_optim = torch.optim.Adam(model.alphas(), config.alpha_lr, betas=(0.5, 0.999),
                                   weight_decay=config.alpha_weight_decay)

    # split data to train/validation
    n_train = len(train_data)
    split = n_train // 2
    indices = list(range(n_train))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               sampler=train_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               sampler=valid_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, config.epochs, eta_min=config.w_lr_min)
    architect = Architect(model, config.w_momentum, config.w_weight_decay)
    # training loop
    best_losses = 100
    for epoch in range(config.epochs):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]
        model.print_alphas(logger)
        # training
        train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch)
        # validation
        cur_step = (epoch+1) * len(train_loader)
        losses = validate(valid_loader, model, epoch, cur_step)
        # log
        # genotype
        genotype = model.genotype()
        logger.info("genotype = {}".format(genotype))

        # save
        if losses < best_losses:
            best_losses = losses
            best_genotype = genotype
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(model, config.path, is_best)
        print("")

    logger.info("Best Genotype = {}".format(best_genotype))


def train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch):
    losses = utils.AverageMeter()

    cur_step = epoch*len(train_loader)
    writer.add_scalar('train/lr', lr, cur_step)

    model.train()

    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, valid_loader)):
        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
        val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(device, non_blocking=True)
        trn_y = torch.squeeze(trn_y)
        val_y = torch.squeeze(val_y)
        N = trn_X.size(0)
        # phase 2. architect step (alpha)
        alpha_optim.zero_grad()
        architect.unrolled_backward(trn_X, trn_y, val_X, val_y, lr, w_optim)
        alpha_optim.step()

        # phase 1. child network step (w)
        w_optim.zero_grad()
        logits = model(trn_X)
        logits = torch.squeeze(logits)
        loss = model.criterion(logits, trn_y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        w_optim.step()

        losses.update(loss.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} ".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        cur_step += 1

def train_model(train_loader, valid_loader, model, w_optim, lr, epoch):
    losses = utils.AverageMeter()
    cur_step = epoch*len(train_loader)
    writer.add_scalar('train/lr', lr, cur_step)
    model.train()
    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, valid_loader)):
        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
        trn_y = torch.squeeze(trn_y)
        N = trn_X.size(0)
        # phase 1. child network step (w)
        w_optim.zero_grad()
        logits = model(trn_X)
        logits = torch.squeeze(logits)
        loss = model.criterion(logits, trn_y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        w_optim.step()
        losses.update(loss.item(), N)
        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Pre_Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} ".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses))
        writer.add_scalar('train/loss', loss.item(), cur_step)
        cur_step += 1


def validate(valid_loader, model, epoch, cur_step):
    losses = utils.AverageMeter()
    model.eval()
    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits = model(X)
            logits = torch.squeeze(logits)
            loss = model.criterion(logits, y)

            losses.update(loss.item(), N)

            if step % config.print_freq == 0 or step == len(valid_loader)-1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f}".format(
                        epoch+1, config.epochs, step, len(valid_loader)-1, losses=losses))

    writer.add_scalar('val/loss', losses.avg, cur_step)

    # logger.info("Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))

    return losses.avg


if __name__ == "__main__":
    main()
