#!/usr/bin/env python3

# Copyright (c)  2020  Xiaomi Corporation (author: Liyong Guo)
# Apache 2.0

import logging
import math
import torch

from common import load_checkpoint, save_checkpoint
from model import TransformerModel, RNNModel

# references:
# https://github.com/Hiroshiba/pytorch-trainer/blob/master/pytorch_trainer/training/trainer.py
# https://github.com/espnet/espnet/blob/master/espnet/lm/pytorch_backend/lm.py
# https://github.com/Hiroshiba/pytorch-trainer/blob/master/pytorch_trainer/training/trainer.py
# https://www.jianshu.com/p/c88df856dbc8


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class Trainer(object):

    def __init__(self,
                 device,
                 model=None,
                 criterion=None,
                 optimizer=None,
                 train_data_loader=None,
                 dev_data_loader=None,
                 ntokens=None,
                 batch_size=1,
                 epoch=0,
                 num_epochs=10,
                 clip=0.25,
                 log_interval=100,
                 model_dir="exp-nnlm/models/",
                 writer=None):
        self.device = device
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.ntokens = ntokens
        self.batch_size = batch_size
        self.epoch = epoch
        self.num_epochs = num_epochs
        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader
        self.iterations = 0
        self.writer = writer
        self.log_interval = log_interval
        self.clip = clip
        self.model_dir = model_dir
        self.num_infinite_grad_norm = 0

    def run(self):
        for epoch in range(self.num_epochs):
            if self.train_data_loader is not None:
                self.train()

            if self.dev_data_loader is not None:
                self.eval()
            save_checkpoint("{}/epoch_{}.pt".format(self.model_dir, epoch),
                            self.model)

            self.epoch += 1

    def train(self):
        self.model.train()
        total_loss = 0
        num_total_batch = len(self.train_data_loader)
        for batch_idx, batch in enumerate(self.train_data_loader):
            # batch_input, batch_target: [max_seq_len, batch_size]
            # max_seq_len is the maximum lenght in current batch
            batch_input, batch_target = batch
            assert batch_input.shape[1] == self.batch_size
            assert batch_target.shape[1] == self.batch_size
            batch_input = batch_input.to(self.device)
            batch_target = batch_target.to(self.device)
            self.model.to(self.device)
            if isinstance(self.model, TransformerModel):
                batch_output = self.model(batch_input)

                prediction = batch_output.view(-1, self.ntokens)
            else:
                # reinitiate hidden for everch batch
                # as batches are independent on each other
                hidden = self.model.init_hidden(batch_input.shape[1])
                prediction, _ = self.model(batch_input, hidden)

            # target: [max_seq_len * batch_size]
            # example_1_token_1 example_2_token_1 example_3_token_1 .....
            target = batch_target.view(-1)
            loss = self.criterion(prediction, target)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       self.clip)
            if torch.isfinite(grad_norm):
                self.optimizer.step()
            else:
                self.num_infinite_grad_norm += 1
            self.optimizer.zero_grad()

            self.writer.add_scalar('train_loss', loss, self.iterations)

            self.iterations += 1
            total_loss += loss.item()
            if batch_idx % self.log_interval == 0 and batch_idx > 0:
                cur_loss = total_loss / self.log_interval
                log_str = 'TRAIN Batch {}/{} loss {:.6f} ppl {:.6f} at epoch {}'.format(
                    batch_idx, num_total_batch, cur_loss, math.exp(cur_loss),
                    self.epoch)
                logging.info(log_str)
                logging.info('infinite grad_norm detected {} times'.format(
                    self.num_infinite_grad_norm))
                total_loss = 0.0
            if batch_idx % 10000 == 0 and batch_idx > 0:
                save_checkpoint(
                    "{}/epoch_{}-batch_{}.pt".format(self.model_dir,
                                                     self.epoch, batch_idx),
                    self.model)

    @torch.no_grad()
    def eval(self):
        self.model.eval()
        total_loss = 0.0
        total_examples = 0
        for batch_idx, batch in enumerate(self.dev_data_loader):
            batch_input, batch_target = batch
            batch_input = batch_input.to(self.device)
            batch_target = batch_target.to(self.device)
            self.model.to(self.device)
            if isinstance(self.model, TransformerModel):
                batch_output = self.model(batch_input)

                prediction = batch_output.view(-1, self.ntokens)
            else:
                hidden = self.model.init_hidden(batch_input.shape[1])
                prediction, _ = self.model(batch_input, hidden)
            # target: [max_seq_len * batch_size]
            # example_1_token_1 example_2_token_1 example_3_token_1 .....
            target = batch_target.view(-1)
            loss = self.criterion(prediction, target)
            total_loss += loss * batch_input.shape[1]
            total_examples += batch_input.shape[1]

        loss = total_loss / total_examples
        ppl = math.exp(loss)
        self.writer.add_scalar('dev_ppl', ppl, self.epoch)
        log_str = 'dev loss is {:.6f} and ppl {:.6f} at epoch {}'.format(
            loss.item(), ppl, self.epoch)
        logging.info(log_str)
