# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
from typing import Optional, Union, Iterable

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from solo.utils.checkpointer import Checkpointer as soloCheckpointer
from solo.methods.base import BaseMethod, BaseMomentumMethod


class WeightsSaver(Callback):
    def __init__(self, model_to_save, epochs_to_save: Iterable = []):
        super().__init__()
        self.epochs_to_save = epochs_to_save
        self.model_to_save = model_to_save
        

    def initial_setup(self, trainer: pl.Trainer):
        for callback in trainer.callbacks:
            if  isinstance(callback, soloCheckpointer):
                self.path = callback.path
                self.ckpt_placeholder = callback.ckpt_placeholder.replace(".ckpt", ".pth")
                print("Weights save path", self.path)

        # if isinstance(trainer.model, BaseMethod):
        #     self.model_to_save =  trainer.model.model.backbone
        # elif isinstance(trainer.model, BaseMomentumMethod):
        #     self.model_to_save =  trainer.model.model.momentum_backbone
        # else:
            # self.model_to_save =  trainer.model.model

    def save(self, trainer: pl.Trainer):
        """Saves current checkpoint.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """
        state_dict = self.model_to_save.state_dict()
        if trainer.is_global_zero:
            if trainer.sanity_checking:
                filepath = os.path.join(self.path, "sanity_checking_weights_saving.pth")
                torch.save(state_dict, filepath)
                os.remove(filepath)
            else:
                epoch = trainer.current_epoch  # type: ignore
                ckpt = self.path / self.ckpt_placeholder.format(epoch)
                torch.save(state_dict, ckpt)

    def on_train_start(self, trainer: pl.Trainer, _):
        """ Get path of the solo checkpointer (if exists).

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """
        self.initial_setup(trainer)


    def on_train_epoch_end(self, trainer: pl.Trainer, _):
        """Tries to save current checkpoint at the end of each train epoch.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        epoch = trainer.current_epoch  # type: ignore
        if epoch in self.epochs_to_save:
            self.save(trainer)
