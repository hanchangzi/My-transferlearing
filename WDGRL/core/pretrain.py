"""Execute domain adaption for ARDA."""

import torch
from torch import nn
from core.test import test
from misc import params
from misc.utils import (calc_gradient_penalty, get_inf_iterator, get_optimizer,
                        make_variable, save_model)


def train(classifier, generator, critic, src_data_loader, tgt_data_loader):
    """Train generator, classifier and critic jointly."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    classifier.train()
    generator.train()
    # set criterion for classifier and optimizers
    criterion = nn.CrossEntropyLoss()
    optimizer_c = get_optimizer(classifier, "Adam")

    # zip source and target data pair
    data_iter_src = get_inf_iterator(src_data_loader)

    # counter
    g_step = 0


    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs):
        ###########################
        # 2.1 train discriminator #
        ###########################
        # requires to compute gradients for D
        for p in critic.parameters():
            p.requires_grad = True

        # set steps for discriminator
        if g_step < 25 or g_step % 500 == 0:
            # this helps to start with the critic at optimum
            # even in the first iterations.
            critic_iters = 100
        else:
            critic_iters = params.d_steps
        critic_iters = 0
        # loop for optimizing discriminator
        #for d_step in range(critic_iters):
            # convert images into torch.Variable
        images_src, labels_src = next(data_iter_src)
        
        images_src = make_variable(images_src).cuda()
        labels_src = make_variable(labels_src.squeeze_()).cuda()
        # print(type(images_src))

        ########################
        # 2.2 train classifier #
        ########################

        # zero gradients for optimizer
        optimizer_c.zero_grad()

        # compute loss for critic
        preds_c = classifier(generator(images_src))
        c_loss = criterion(preds_c, labels_src)

        # optimize source classifier
        c_loss.backward()
        optimizer_c.step()
        g_step += 1

        ##################
        # 2.4 print info #
        ##################
        if ((epoch + 1) % 500 == 0):
            # print("Epoch [{}/{}]:"
            #       "c_loss={:.5f}"
            #       "D(x)={:.5f}"
            #       .format(epoch + 1,
            #               params.num_epochs,
            #               c_loss.item(),
            #               ))
            test(classifier, generator, src_data_loader, params.src_dataset)
        if ((epoch + 1) % 500 == 0):
            save_model(generator, "Mnist-generator-{}.pt".format(epoch + 1))
            save_model(classifier, "Mnist-classifer{}.pt".format(epoch + 1))