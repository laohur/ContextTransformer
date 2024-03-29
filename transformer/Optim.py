import numpy as np
import math


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)
        # self.init_lr = 1

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()
        # lr = self.init_lr * wave(self.n_current_steps, cycle=100)

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

        if self.n_current_steps % 10000 == 0:
            print("self.n_current_steps init_lr lr", self.n_current_steps, self.init_lr, lr)


def wave(step, cycle=1000):
    sub = step % cycle
    sub /= cycle
    if sub < 0.1:  # 变大
        sub /= 0.1
    else:  # 变小
        sub = 1 - sub
    sub = math.sin(sub * sub)+0.000001
    return sub
