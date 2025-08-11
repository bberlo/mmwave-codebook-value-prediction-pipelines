import tensorflow as tf


class AnyLabelModel(tf.keras.models.Model):

    def compile(self, optimizer='rmsprop', loss=None, metrics=None, loss_weights=None, model_shape=None,
                weighted_metrics=None, run_eagerly=None, steps_per_execution=None, feature_importance=None, **kwargs):
        super(AnyLabelModel, self).compile(optimizer, loss, metrics, loss_weights, weighted_metrics,
                                           run_eagerly, steps_per_execution, **kwargs)

        # Averaging idea taken from Contrastive Learning of General-Purpose Audio Representations, Saeed et al.
        # https://ieeexplore.ieee.org/document/9413528
        self.average = tf.keras.layers.Average()
        self.feat_importance = feature_importance
        self.model_shape = model_shape

    def train_step(self, data):

        # These are the only transformations `Model.fit` applies to user-input
        # data when a `tf.data.Dataset` is provided.
        # expand_1d function has been removed

        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        xts0 = tf.reshape(x[0], shape=tf.concat([[tf.shape(x[0])[0]], self.model_shape], axis=0))
        xts1 = tf.reshape(x[1], shape=tf.concat([[tf.shape(x[1])[0]], self.model_shape], axis=0))
        y = tf.reduce_max(y, axis=-2)

        with tf.GradientTape() as tape:

            y_pred_1 = self(xts0, training=True)
            y_pred_2 = self(xts1, training=True)
            y_pred = self.average([y_pred_1, y_pred_2])

            loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):

        # expand_1d function has been removed

        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        y = tf.reduce_max(y, axis=-2)

        # Additional Shap value logic in metric updates
        if self.feat_importance:

            self.compiled_metrics._metrics[0].update_state(modelref=self, x_true=x)
            return {"shapley_importance": self.compiled_metrics._metrics[0].result()}

        else:
            xtst0 = tf.reshape(x[0], shape=tf.concat([[tf.shape(x[0])[0]], self.model_shape], axis=0))
            xtst1 = tf.reshape(x[1], shape=tf.concat([[tf.shape(x[1])[0]], self.model_shape], axis=0))

            y_pred_1 = self(xtst0, training=False)
            y_pred_2 = self(xtst1, training=False)
            y_pred = self.average([y_pred_1, y_pred_2])

            # Updates stateful loss metrics.
            self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
            self.compiled_metrics.update_state(y, y_pred, sample_weight)

            return {m.name: m.result() for m in self.metrics}
