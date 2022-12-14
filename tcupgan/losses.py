import torch
from torch import nn


def tversky(y_true, y_pred, beta, batch_mean=True):
    tp = torch.sum(y_true * y_pred, axis=(1, 2, 3, 4))
    fn = torch.sum((1. - y_pred) * y_true, axis=(1, 2, 3, 4))
    fp = torch.sum(y_pred * (1. - y_true), axis=(1, 2, 3, 4))
    # tversky = reduce_mean(tp)/(reduce_mean(tp) +
    #                           beta*reduce_mean(fn) +
    #                           (1. - beta)*reduce_mean(fp))
    tversky = tp /\
        (tp + beta * fn + (1. - beta) * fp)

    if batch_mean:
        return torch.mean((1. - tversky))
    else:
        return (1. - tversky)


def fc_tversky(y_true, y_pred, beta, gamma=0.5, batch_mean=True):
    smooth = 1
    tp = torch.sum(y_true * y_pred, axis=(1, 2, 3, 4))
    fn = torch.sum((1. - y_pred) * y_true, axis=(1, 2, 3, 4))
    fp = torch.sum(y_pred * (1. - y_true), axis=(1, 2, 3, 4))
    # tversky = reduce_mean(tp)/(reduce_mean(tp) +
    #                           beta*reduce_mean(fn) +
    #                           (1. - beta)*reduce_mean(fp))
    tversky = (tp + smooth) /\
        (tp + beta * fn + (1. - beta) * fp + smooth)

    focal_tversky_loss = 1 - tversky

    if batch_mean:
        return torch.pow(torch.mean(focal_tversky_loss), gamma)
    else:
        return torch.pow(focal_tversky_loss, gamma)


def kl_loss(mu, sig):
    kl = 0.5*torch.mean(-1 - sig + torch.square(mu) + torch.exp(sig), axis=-1)
    return torch.mean(kl)


# alias
generator_seg_loss = fc_tversky
generator_vae_loss = nn.L1Loss()
discriminator_loss = nn.BCELoss()
