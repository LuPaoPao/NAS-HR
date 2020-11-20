""" Training augmented model """
# This code fork from https://github.com/khanrc/pt.darts
# modified by Hao Lu for Heart Rate Estimation

import os
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from config import AugmentConfig
import MyDataset
import torchvision.transforms.functional as transF
from torchvision import transforms
import utils
from models.augment_cnn import AugmentCNN
from torch.utils.data import Dataset, DataLoader
import scipy.io as io
from thop import profile
from thop import clever_format

config = AugmentConfig()

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
    input_size = np.array([64, 300])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    toTensor = transforms.ToTensor()
    resize = transforms.Resize(size=(64, 300))
    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])
    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    # net acc
    torch.backends.cudnn.benchmark = True
    # get data with meta info
    if config.reData == 1:
        test_index, train_index = MyDataset.CrossValidation(fileRoot, fold_num=config.fold_num,
                                                            fold_index=config.fold_index)
        Train_Indexa = MyDataset.getIndex(fileRoot, train_index, saveRoot + '_Train', 'STMap_YUV_Align_CSI_POS.png', 15, 300)
        Test_Indexa = MyDataset.getIndex(fileRoot, test_index, saveRoot + '_Test', 'STMap_YUV_Align_CSI_POS.png', 15, 300)
    train_data = MyDataset.Data_STMap(root_dir=(saveRoot + '_Train'), frames_num=300,
                                      transform=transforms.Compose([resize, toTensor, normalize]))
    valid_data = MyDataset.Data_STMap(root_dir=(saveRoot + '_Test'), frames_num=300,
                                      transform=transforms.Compose([resize, toTensor, normalize]))
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.workers,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=config.batch_size,
                                               shuffle=False,
                                               num_workers=config.workers,
                                               pin_memory=True)
    # loss
    criterion = nn.L1Loss().to(device)
    # net
    Model_name = config.name + 'fn' + str(config.fold_num) + 'fi' + str(config.fold_index)
    use_aux = config.aux_weight > 0.
    if config.reTrain == 1:
        model = torch.load(os.path.join(config.path, Model_name + 'best.pth.tar'), map_location=device)
        print('load ' + Model_name + ' right')
        model = nn.DataParallel(model, device_ids=config.gpus).to(device)
    else:
        model = AugmentCNN(input_size, input_channels, config.init_channels, n_classes, config.layers,
                           use_aux, config.genotype)
        model._init_weight()
        model = nn.DataParallel(model, device_ids=config.gpus).to(device)

    # model size
    mb_params = utils.param_size(model)
    logger.info("Model size = {:.3f} MB".format(mb_params))
    # weights optimizer
    optimizer = torch.optim.Adam(model.parameters(), config.lr)
    best_losses = 10

    # training loop
    for epoch in range(config.epochs):
        # training
        train(train_loader, model, optimizer, criterion, epoch)
        # validation
        cur_step = (epoch+1) * len(train_loader)
        best_losses = validate(valid_loader, model, criterion, epoch, cur_step, best_losses)
    logger.info("Final best Losses@1 = {:.4%}".format(best_losses))


def train(train_loader, model, optimizer, criterion, epoch):
    losses = utils.AverageMeter()
    cur_step = epoch*len(train_loader)
    cur_lr = optimizer.param_groups[0]['lr']
    logger.info("Epoch {} LR {}".format(epoch, cur_lr))
    writer.add_scalar('train/lr', cur_lr, cur_step)
    model.train()
    for step, (X, y) in enumerate(train_loader):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        N = X.size(0)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        # if config.aux_weight > 0.:
        #     loss += config.aux_weight * criterion(aux_logits, y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        losses.update(loss.item(), N)
        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} ".format(
                    epoch + 1, config.epochs, step, len(train_loader) - 1, losses=losses))
        writer.add_scalar('train/loss', loss.item(), cur_step)
        cur_step += 1


def validate(valid_loader, model, criterion, epoch, cur_step, best_losses):
    losses = utils.AverageMeter()
    model.eval()
    HR_pr_temp = []
    HR_rel_temp = []
    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)
            logits = model(X)
            loss = criterion(logits, y)
            losses.update(loss.item(), N)
            HR_pr_temp.extend(logits.data.cpu().numpy())
            HR_rel_temp.extend(y.data.cpu().numpy())
            if step % config.print_freq == 0 or step == len(valid_loader)-1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f}".format(
                        epoch + 1, config.epochs, step, len(valid_loader) - 1, losses=losses))
        utils.MyEval(HR_pr_temp, HR_rel_temp)
        io.savemat(os.path.join(config.path, str(config.fold_index) + 'HR_pr.mat'), {'HR_pr': HR_pr_temp})
        io.savemat(os.path.join(config.path, str(config.fold_index) + 'HR_rel.mat'), {'HR_rel': HR_rel_temp})
        # save
        if best_losses > losses.avg:
            best_losses = losses.avg
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(model, config.path, is_best)

    writer.add_scalar('val/loss', losses.avg, cur_step)

    return best_losses


if __name__ == "__main__":
    main()
