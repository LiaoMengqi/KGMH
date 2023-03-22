import torch

from basemodel import BaseModel


class Trainer(object):
    def __init__(self, model: BaseModel, opt: torch.optim.Optimizer):
        self.model = model
        self.opt = opt
        self.loss_history = []

    def train(self, batch_size, epochs, eval_epoch, output=True):
        for epoch in range(epochs):
            loss = self.model.train_step(batch_size, self.opt)
            self.loss_history.append(loss)
            if output:
                print('epoch: ' + str(epoch + 1) + ' | loss: ' + str(loss))
            if (epoch + 1) % eval_epoch == 0:
                self.test(output=output)

    def test(self, data_set='valid', output=True):
        return NotImplementedError

    def hist_loss(self):
        return NotImplementedError
