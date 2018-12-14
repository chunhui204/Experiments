from fastai.conv_learner import *


class EarlyStop(Callback):
    def __init__(self, monitor='val_loss', min_delta=0, patience=0):
        self._monitor = monitor
        self._min_delta = min_delta
        self._patience = patience
        self._idx_epoch = 0
        self._best_score = 0
        self._pat_epoch = 0

    def on_epoch_end(self, metrics):
        val_loss = metrics[0][0]
        if self._idx_epoch > 0:
            if val_loss - self._best_score <= self._min_delta:
                self._pat_epoch = 0
                self._best_score = val_loss if val_loss < self._best_score else self._best_score
            else:
                self._pat_epoch += 1
                if (self._pat_epoch >= self._patience):
                    return True
        else:
            self._best_score = val_loss

        self._idx_epoch += 1


class SaveBestModel(LossRecorder):
    def __init__(self, model, lr, name='best_model', monitor='loss', min_delta=0, patience=0):
        """
        :param model:
        :param lr:
        :param name:
        :param monitor: 'loss' or 'acc'
        :param min_delta:
        :param patience:
        """
        super().__init__(model.get_layer_opt(lr, None))
        self.name = name
        self.model = model
        self.first_monitor = None
        self.second_monitor = None
        self.monitor = monitor

        self._min_delta = min_delta
        self._patience = patience
        self._pat_epoch = 0

    def on_epoch_end(self, metrics):
        super().on_epoch_end(metrics)
        if self.monitor == 'loss':
            fst, scd = metrics
            scd = -scd
        else:
            scd, fst = metrics
            fst = -fst
        if self.first_monitor == None or fst < self.first_monitor:
            self._pat_epoch = 0
            self.second_monitor = scd
            self.first_monitor = fst
            self.model.save(f'{self.name}')
        elif fst == self.first_monitor and scd < self.second_monitor:
            self._pat_epoch = 0
            self.first_monitor = fst
            self.second_monitor = scd
            self.model.save(f'{self.name}')
        elif fst - self.first_monitor > self._min_delta:
            self._pat_epoch += 1
            if self._pat_epoch >= self._patience:
                print('best model,', self.first_monitor, self.second_monitor)
                return True
            
            
    def on_train_end(self):
        print('best model,', self.first_monitor, self.second_monitor)

def plot_trainval(learn):
    fig, ax = plt.subplots(2, 1, figsize=(8, 12))
    iterations = len(learn.sched.losses)
    epochs = len(learn.sched.val_losses)
    ax[0].plot(np.arange(epochs) * iterations / epochs, learn.sched.val_losses, label='Validation loss')
    ax[0].plot(np.arange(iterations), learn.sched.losses, label='Training loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend(loc='upper right')
    ax[1].plot(np.arange(len(learn.sched.rec_metrics)), learn.sched.rec_metrics)
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('metric')