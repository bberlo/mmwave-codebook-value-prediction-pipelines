import tensorflow as tf
import math


# Callback to halt training when loss is negative or diverges (EarlyStopping doesn't account for this)
class HaltCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        if logs.get('loss') < 0.0 or logs.get('val_loss') < 0.0 or math.isnan(logs.get('loss')) or math.isnan(
                logs.get('val_loss')) or logs.get('loss') > 100000.0 or logs.get('val_loss') > 100000.0:
            self.model.stop_training = True
            logs['val_ntxent_sim_loss'] = 1000.0


# Callback to allow every pre-train model to finish an equal amount of epochs
class WeightRestoreCallback(tf.keras.callbacks.EarlyStopping):

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)

        if current is None:
            return

        if self.restore_best_weights and self.best_weights is None:
            self.best_weights = self.model.get_weights()

        if tf.__version__ < "2.5.0":
            if self.monitor_op(current - self.min_delta, self.best):
                self.best = current
                self.best_epoch = epoch

                if self.restore_best_weights:
                    self.best_weights = self.model.get_weights()

        else:
            if self._is_improvement(current, self.best):
                self.best = current
                self.best_epoch = epoch

                if self.restore_best_weights:
                    self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)
