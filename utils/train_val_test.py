
import os
import torch
import logging
import numpy as np
import torch.nn as nn
from torch import optim
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import models
from datasets.Set_division import datasets


def cosine(x, y):
    num = torch.sum(x*y, dim=1)
    denom = x.norm(p=2, dim=1)*y.norm(p=2, dim=1)
    cos = torch.mean(num / denom)
    return 0.5*(1 - cos)


def train(args):
    device = torch.device(args.device)

    # model preparing
    model = getattr(models, args.model_name)()
    model = model.to(device)
    model.train()
    train_loader, num = datasets(args, 'train')

    # Define the optimizer way
    optimizer = torch.optim.Adam(model.parameters(), args.lr, amsgrad=True)
    steps = [int(step) for step in args.steps.split(',')]
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, steps, gamma=args.gamma)

    for epoch in range(args.epoch):
        loss_epoch = 0

        for i, (data, y) in enumerate(train_loader):   # batch operation
            data = data.to(device)
            data = data.unsqueeze(1)
            data = data.reshape(data.shape[0], -1, int(args.sample_length / 2))

            optimizer.zero_grad()
            loss = model(data, args, 'train')
            loss_epoch += loss.item()*len(y)
            loss.backward()
            optimizer.step()

        lr_scheduler.step()
        if (epoch + 1) % args.print_epoch == 0:
            loss_epoch = loss_epoch / num
            log = "Epoch [{}/{}], lr {} ".format(epoch + 1, args.epoch, optimizer.param_groups[0]['lr'])
            log += 'loss {:.4f}'.format(loss_epoch)
            print(log)

        if (epoch + 1) % args.epoch == 0:
            save_dir = os.path.join(
                './trained_models/snr_{}'.format(args.snr))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(model.state_dict(), os.path.join(
                '{}/{}.pth'.format(save_dir, args.model_name)))


def val(args):
    device = torch.device(args.device)

    model = getattr(models, args.model_name)()
    model.load_state_dict(torch.load(
        './trained_models/snr_{}/{}.pth'.format(args.snr, args.model_name)))
    model = model.to(device)
    model.eval()

    mse = nn.MSELoss()
    val_loader, num = datasets(args, 'val')
    loss_all = torch.empty(num)
    step = 0

    for i, (data, y) in enumerate(val_loader):
        data = data.to(device)
        data = data.unsqueeze(1)
        x = data.reshape(data.shape[0], -1, int(args.sample_length / 2))

        if args.model_name == 'HRCAE':
            out1, out2 = model(x, args, 'val')
            for j in range(x.shape[0]):
                loss1 = (mse(out1[j], x[j]) + args.lambd * cosine(out1[j], x[j]))
                loss2 = (mse(out2[j], x[j]) + args.lambd * cosine(out2[j], x[j]))
                loss = loss1 + loss2
                loss_all[step] = loss.detach().item()
                step += 1
        else:
            out = model(x, args, 'val')
            for j in range(x.shape[0]):
                loss = (mse(out[j], x[j]) + args.lambd * cosine(out[j], x[j]))
                loss_all[step] = loss.detach().item()
                step += 1

    loss_avg = loss_all.mean()
    logging.info('Val average Loss: %.6f' % (loss_avg))
    threshold = np.percentile(loss_all, args.percentage_threshold)
    return threshold


def test(args, threshold):
    device = torch.device(args.device)

    model = getattr(models, args.model_name)()
    model.load_state_dict(torch.load(
        './trained_models/snr_{}/{}.pth'.format(args.snr, args.model_name)))
    model = model.to(device)
    model.eval()

    mse = nn.MSELoss()
    test_loader, num = datasets(args, 'test')
    scores = np.zeros(shape=(num, 2))
    loss_all = torch.empty(num)
    step = 0

    for i, (data, y) in enumerate(test_loader):
        data = data.to(device)
        data = data.unsqueeze(1)
        x = data.reshape(data.shape[0], -1, int(args.sample_length / 2))

        if args.model_name == 'HRCAE':
            out1, out2 = model(x, args, 'test')
            for j in range(x.shape[0]):
                loss1 = (mse(out1[j], x[j]) + args.lambd * cosine(out1[j], x[j]))
                loss2 = (mse(out2[j], x[j]) + args.lambd * cosine(out2[j], x[j]))
                loss = loss1 + loss2
                scores[step] = [int(y[j]), int(loss > threshold)]
                loss_all[step] = loss
                step += 1
        else:
            out = model(x, args, 'test')
            for j in range(x.shape[0]):
                loss = mse(out[j], x[j]) + args.lambd * cosine(out[j], x[j])
                scores[step] = [int(y[j]), int(loss > threshold)]
                loss_all[step] = loss
                step += 1

    loss_avg = loss_all.mean()
    logging.info('threshold: {:.4f}'.format(threshold))
    logging.info('Test average Loss: {:.6f}'.format(loss_avg))
    accuracy = accuracy_score(scores[:, 0], scores[:, 1])
    precision, recall, fscore, support = precision_recall_fscore_support(scores[:, 0], scores[:, 1],
                                                                         average='binary')
    logging.info('Accuracy: {:.4f}  Precision: {:.4f}  Recall: {:.4f}  F-score: {:.4f}'.format(
        accuracy, precision, recall, fscore))
