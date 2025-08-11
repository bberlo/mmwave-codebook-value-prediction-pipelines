import tensorflow as tf


class AnyLabelModel(tf.keras.models.Model):

    def compile(self, optimizer='rmsprop', loss=None, metrics=None, loss_weights=None,
                weighted_metrics=None, run_eagerly=None, steps_per_execution=None,
                feature_importance=None, model_shape=None, **kwargs):
        super(AnyLabelModel, self).compile(optimizer, loss, metrics, loss_weights, weighted_metrics,
                                           run_eagerly, steps_per_execution, **kwargs)

        self.feat_importance = feature_importance
        self.model_shape = model_shape

    def train_step(self, data):

        # These are the only transformations `Model.fit` applies to user-input
        # data when a `tf.data.Dataset` is provided.
        # expand_1d function has been removed

        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        x = tf.reshape(x, shape=tf.concat([[tf.shape(x)[0]], self.model_shape], axis=0))
        y = tf.reduce_max(y, axis=-2)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
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
            x = tf.reshape(x, shape=tf.concat([[tf.shape(x)[0]], self.model_shape], axis=0))

            y_pred = self(x, training=False)
            # Updates stateful loss metrics.
            self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
            self.compiled_metrics.update_state(y, y_pred, sample_weight)

            return {m.name: m.result() for m in self.metrics}
