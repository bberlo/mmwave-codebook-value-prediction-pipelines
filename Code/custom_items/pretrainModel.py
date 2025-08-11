from experiment_environment.custom_items.rowInterlace import RowInterlace
import tensorflow as tf


class PretrainModel(tf.keras.Model):

    def compile(self, optimizer='rmsprop', loss=None, sim_loss_fn=None, metrics=None, model_shape=None,
                loss_weights=None, tau=0.02, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        super(PretrainModel, self).compile(optimizer, loss, metrics, loss_weights, weighted_metrics,
                                       run_eagerly, steps_per_execution, **kwargs)

        self.tau = tau
        self.sim_loss_fn = sim_loss_fn
        self.interlace = RowInterlace()
        self.model_shape = model_shape

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)  # x_view1 = x[0], x_view2 = x[1]
        x0 = tf.reshape(x[0], shape=tf.concat([[tf.shape(x[0])[0]], self.model_shape], axis=0))
        x1 = tf.reshape(x[1], shape=tf.concat([[tf.shape(x[1])[0]], self.model_shape], axis=0))

        with tf.GradientTape() as unsup_tape:

            pi_pred = self(x0, training=True)
            pj_pred = self(x1, training=True)

            pij_pred = self.interlace([pi_pred, pj_pred])
            sim_loss = self.sim_loss_fn(pij_pred, self.tau)

            if self.losses:
                sim_loss += tf.add_n(self.losses)

        trainable_vars = self.trainable_variables
        gradients = unsup_tape.gradient(sim_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result

        return_metrics["loss"] = sim_loss

        return return_metrics

    def test_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)  # x_view1 = x[0], x_view2 = x[1]
        xt0 = tf.reshape(x[0], shape=tf.concat([[tf.shape(x[0])[0]], self.model_shape], axis=0))
        xt1 = tf.reshape(x[1], shape=tf.concat([[tf.shape(x[1])[0]], self.model_shape], axis=0))

        pi_pred = self(xt0, training=False)
        pj_pred = self(xt1, training=False)

        pij_pred = self.interlace([pi_pred, pj_pred])
        sim_loss = self.sim_loss_fn(pij_pred, self.tau)

        if self.losses:
            sim_loss += tf.add_n(self.losses)

        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result

        return_metrics["loss"] = sim_loss

        return return_metrics
