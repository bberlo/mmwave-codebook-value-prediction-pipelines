import tensorflow as tf
import math


# Custom TensorFlow implementation of baseline OMP algorithm as proposed in Mateos-Ramos et al., Model-Based End-to-End
# Learning for Multi-Target Integrated Sensing and Communication, URL: https://arxiv.org/abs/2307.04111
class TargetMapRecovery(tf.keras.layers.Layer):

    def __init__(self, rmin, rmax, theta_min, theta_max, rsamp, theta_samp, n_adc, n_rx):
        super(TargetMapRecovery, self).__init__()

        self.rmin = rmin  # Closest possible target range, 0, should be given in meters
        self.rmax = rmax  # Furthest target range, i.e., distance to furthest corner in rectangular experiment area
        self.theta_min = theta_min * (math.pi / 180)  # Should be given in degrees, -90, converted to radians
        self.theta_max = theta_max * (math.pi / 180)  # +90

        self.rsamp = rsamp  # Sample number
        self.theta_samp = theta_samp
        self.n_adc = n_adc
        self.n_rx = n_rx

        self.max_iterations = 100  # Variable necessary to set loop limit for graph stitching
        self.termination_threshold = 2

    # Input_shape: (Seq-len, RX, fast-time), i.e., algorithm works per element in seq-len (considered batch dimension).
    # Additional batch dimension in addition to seq-len not considered due to conditional branch in while loop graph.
    # Increased chance for ghost targets the more batch samples are considered in the way the algorithm is designed.
    def build(self, input_shape):

        self.angle_grid = self.add_weight("angles", shape=(self.n_rx, self.theta_samp), dtype=tf.complex128,
                                          initializer=self._get_angle_grid, trainable=False)
        self.delay_grid = self.add_weight("delays", shape=(self.n_adc, self.rsamp), dtype=tf.complex128,
                                          initializer=self._get_delay_grid, trainable=False)
        self.grid_map = self.add_weight("gridmap", shape=(input_shape[1], self.rsamp, self.theta_samp), dtype=tf.int32,
                                        initializer=tf.keras.initializers.Zeros(), trainable=False)
        self.grid_map_done = self.add_weight("gridmapdone", shape=(input_shape[1]), dtype=tf.int32,
                                             initializer=tf.keras.initializers.Zeros(), trainable=False)
        self.delay_angle_map_max = self.add_weight("dammax", shape=(input_shape[1]), dtype=tf.float64,
                                                   initializer=tf.keras.initializers.Zeros(), trainable=False)

    def call(self, inputs, **kwargs):

        # Reset persistent grid map storage for persistence across loop iterations only
        self.grid_map.assign(tf.broadcast_to([[[0]]], shape=self.grid_map.shape))
        self.grid_map_done.assign(tf.broadcast_to([0], shape=self.grid_map_done.shape))

        inputs = tf.cast(tf.squeeze(inputs), dtype=tf.complex128)

        # Input amplitude normalization
        rsum = tf.reduce_sum(tf.abs(inputs))
        s = tf.cast(1.0 / tf.math.sqrt(rsum), dtype=tf.complex128)
        inputs = s * inputs

        angle_grid_adjointed = tf.expand_dims(tf.transpose(a=self.angle_grid, conjugate=True), axis=0)
        delay_grid_conj = tf.expand_dims(tf.math.conj(self.delay_grid), axis=0)
        delay_angle_mask = tf.zeros(shape=(tf.shape(inputs)[0], self.theta_samp, self.rsamp), dtype=tf.int32)
        range_tens = tf.broadcast_to(tf.range(0,self.rsamp*self.theta_samp, delta=1, dtype=tf.int32)[tf.newaxis, ...],
                                     shape=(tf.shape(inputs)[0], self.rsamp*self.theta_samp))

        def cond(the_input, curr_residual, curr_mask, curr_angle_grid, curr_delay_grid, threshold, curr_i):
            return tf.less(curr_i, self.max_iterations)

        def body(the_input, curr_residual, curr_mask, curr_angle_grid, curr_delay_grid, threshold, curr_i):

            grid_map_snap = self.grid_map.value()

            def true_fn():

                delay_angle_interm = tf.matmul(a=curr_angle_grid, b=curr_residual)
                delay_angle_map = tf.math.square(tf.abs(tf.matmul(a=delay_angle_interm, b=curr_delay_grid)))

                dam_max = tf.reduce_max(delay_angle_map, axis=[1,2])
                # should the computation stop for this frame?
                eq_cond_stop = tf.greater(dam_max, self.delay_angle_map_max.value()*3.0)
                # prevent updates to grid_map.
                self.grid_map_done.assign_add(tf.where(eq_cond_stop,1,0))
                self.delay_angle_map_max.assign(dam_max)

                # Mask update (referred to as atom set update in paper)
                curr_max_indices = tf.argmax(tf.reshape(delay_angle_map, shape=(tf.shape(delay_angle_map)[0], -1)), axis=-1, output_type=tf.int32)
                curr_max_filter = tf.broadcast_to(curr_max_indices[...,tf.newaxis], shape=tf.shape(range_tens))
                curr_max_eq = tf.equal(curr_max_filter, range_tens)

                curr_mask_flat = tf.reshape(curr_mask, shape=(tf.shape(curr_mask)[0],-1))
                curr_mask_updated_flat = tf.logical_or(tf.greater(curr_mask_flat, tf.constant([[0]])), curr_max_eq)
                curr_mask_updated = tf.reshape(tf.where(curr_mask_updated_flat,tf.constant([[1]]),tf.constant([[0]])), shape=tf.shape(curr_mask))

                # Get neighboring update candidates for offset updates done to locations already marked in previous iterations
                curr_mask_image = tf.cast(curr_mask_updated[...,tf.newaxis], dtype=tf.float32)
                curr_mask_sobel = tf.image.sobel_edges(curr_mask_image)

                sobel_sum = tf.squeeze(tf.cast(curr_mask_sobel[:,:,:,:,0]+curr_mask_sobel[:,:,:,:,1],dtype=tf.int32))
                notcurr_max = 1 - curr_mask_updated

                curr_max_sobel = tf.cast(tf.math.argmax(tf.math.abs(tf.reshape(tf.multiply(sobel_sum, notcurr_max),
                                                                               shape=(tf.shape(curr_mask_updated)[0],-1))),
                                                        axis=1), dtype=tf.int32)
                curr_max_sobel_filter = tf.broadcast_to(curr_max_sobel[...,tf.newaxis], shape=tf.shape(range_tens))
                curr_max_sobel_eq = tf.equal(curr_max_sobel_filter, range_tens)

                curr_mask_sobel_flat = tf.reshape(curr_mask_updated, shape=(tf.shape(curr_mask)[0],-1))
                curr_mask_sobel_updated_flat = tf.logical_or(tf.greater(curr_mask_sobel_flat, tf.constant([[0]])), curr_max_sobel_eq)
                curr_mask_sobel_updated = tf.reshape(tf.where(curr_mask_sobel_updated_flat,tf.constant([[1]]),tf.constant([[0]])), shape=tf.shape(curr_mask))

                # Multiplex between updates and neighboring candidates depending on if location was already updated in prev. iterations
                eq_cond = tf.equal(tf.reshape(curr_i,shape=(1,)), tf.reduce_sum(curr_mask_updated, axis=[1,2]))[...,tf.newaxis]
                curr_select=tf.where(eq_cond,
                                     y=tf.reshape(curr_mask_sobel_updated, (tf.shape(curr_mask)[0],-1)),
                                     x=tf.reshape(curr_mask_updated,(tf.shape(curr_mask)[0],-1)))
                curr_final_shape= tf.reshape(curr_select, tf.shape(curr_mask))

                # Gain estimate / residual update
                curr_residual_updated = self._gain_optimization_update(the_input, curr_final_shape)

                # Write update to iteration persistent grid map based on stopping criterion check
                gmd_bool = tf.greater(self.grid_map_done.value(), [0])[..., tf.newaxis]
                grid_map_snap_new = tf.where(gmd_bool, x=tf.reshape(grid_map_snap, (tf.shape(curr_mask)[0], -1)),
                                             y=tf.reshape(curr_final_shape, (tf.shape(curr_mask)[0], -1)))
                self.grid_map.assign(tf.reshape(grid_map_snap_new, self.grid_map.shape))

                return the_input, curr_residual_updated, curr_final_shape, curr_angle_grid, curr_delay_grid, threshold, tf.add(curr_i, 1)

            def false_fn():
                return the_input, curr_residual, curr_mask, curr_angle_grid, curr_delay_grid, threshold, tf.add(curr_i, 1)

            return tf.cond(self.grid_map_done.shape[0]-tf.math.count_nonzero(self.grid_map_done.value())>threshold, true_fn=true_fn, false_fn=false_fn)

        inputs, end_residual, end_mask, angle_grid_adjointed, delay_grid_conj, threshold, end_i = tf.while_loop(
            cond, body, [inputs, inputs, delay_angle_mask, angle_grid_adjointed, delay_grid_conj,
                         tf.convert_to_tensor(self.termination_threshold, dtype=tf.int64), 1],
            parallel_iterations=1, swap_memory=False
        )

        grid_map_max = tf.reduce_max(tf.reduce_sum(self.grid_map.value(), axis=[1, 2]))
        tf.debugging.assert_less_equal(grid_map_max, self.max_iterations,
                                       message="The termination threshod was not reached in the alloted number of iterations. You might want to optimize these graph settings further.")

        end_mask = tf.expand_dims(self.grid_map.value(), axis=0)
        return tf.cast(end_mask, dtype=tf.float32)

    # Range dictionary item rho(tau): e^(-j 2pi t slope tau Ts), t fast time ADC index, Ts ADC sampling period
    # Complex rotation component taken from Aydogdu et al., "Radar Interference Mitigation for Automated Driving:
    # Exploring Proactive Strategies", https://doi.org/10.1109/MSP.2020.2969319
    def _get_delay_grid(self, shape=None, dtype=None):

        min_delay = tf.convert_to_tensor((2 * self.rmin) / 299792458, dtype=tf.float64)   # speed of light m/s vacuum
        max_delay = tf.convert_to_tensor((2 * self.rmax) / 299792458, dtype=tf.float64)
        delay_space = tf.linspace(min_delay, max_delay, num=self.rsamp, axis=0)

        slope = 29.982 * (10 ** 6)  # slope, conv. to Hz, per microsecond, doesn't match number of ADC samples
        delay_slope_space = delay_space * (-1.0 * slope)

        # Ramp end 40.03 microseconds, ADC start 6.0 microseconds in, sampling for 25.6 microseconds (256 * 0.1
        # microseconds based on 10000 ksps sampling rate), 8.43 microseconds idle until ramp deactivated.
        max_adc_time = 40.03 - 8.43

        rotation_indices_space = tf.linspace(start=tf.convert_to_tensor(6.0, dtype=tf.float64), stop=max_adc_time, num=self.n_adc, axis=0)

        M1, M2 = tf.meshgrid(rotation_indices_space, delay_slope_space)

        return tf.transpose(tf.math.exp(
            tf.multiply(
                tf.convert_to_tensor(2j * math.pi, dtype=tf.complex128),
                tf.cast(tf.multiply(M1, M2), dtype=tf.complex128)
            )
        ))

    # Angle dictionary item a(theta): e^(-j 2pi k d (sin(theta) /lambda) ),
    # where d = lambda/2 linear antenna spacing, k antenna index, theta angle in radians
    # k - (K-1)/2 as term not considered since there are no indications that TI radar beam offsets RX steering vector
    # TX antenna effect not considered since TI radar configured to transmit according to one antenna (static)
    def _get_angle_grid(self, shape=None, dtype=None):

        wavelength = 299792458 / (77 * (10 ** 9))
        spacing = wavelength / 2

        angle_space = tf.linspace(start=tf.convert_to_tensor(self.theta_min, dtype=tf.float64), stop=self.theta_max, num=self.theta_samp, axis=0)
        angle_space = tf.math.divide(tf.math.sin(angle_space), wavelength)

        antenna_indices = tf.range(start=0, limit=self.n_rx, delta=1, dtype=tf.float64)
        antenna_space = tf.subtract(antenna_indices, 0)  # Offset via term (self.n_rx - 1) / 2 not considered

        M1, M2 = tf.meshgrid(antenna_space, angle_space)

        return tf.transpose(tf.math.exp(
            tf.multiply(
                tf.multiply(tf.convert_to_tensor(-2j * math.pi, dtype=tf.complex128), tf.cast(M1, dtype=tf.complex128)),
                tf.cast(tf.multiply(spacing, M2), dtype=tf.complex128)
            )
        ))

    # Least square Frobenius norm difference matrix Y linear combination matrices X_t, coefficients alpha_t of interest
    # Partial derivative convex minimum solution alpha_t of interest: sum_{t=1}^T alpha_t tr(X_tX_t^H) = tr(X_t^HY)

    # Used in system of equations in form Ax=b, where A contains tr(X_tX_t^H) subj. to C^{(T, T)}, x alpha's subj. to
    # C^{(T, 1)}, and b tr(X_t^HY) subj. to C^{(T, 1)}, solved using LU decomposition for numerical stability accuracy

    # Seq-len is treated as batch dimension since optimization problem operates in fast-time, RX antenna space.

    # Note: tr(X_tX_t^H) is computed using vec(X_t) and taking inner product with conj(vec(X_t)).
    # See: https://math.stackexchange.com/questions/476802/how-do-you-prove-that-trbt-a-is-a-inner-product
    def _gain_optimization_update(self, curr_inp, a_mask):

        # a_mask: (seq-len, theta_samp, rsamp)
        # curr_inp: (seq-len, n_rx, n_adc)

        # Atom set slicing
        slice_indices_angle = tf.cast(tf.reduce_sum(a_mask, axis=2), dtype=tf.int32)
        atom_set_angle = tf.broadcast_to(tf.expand_dims(self.angle_grid, axis=-1), shape=tf.concat([tf.shape(self.angle_grid), [tf.shape(curr_inp)[0]]], axis=0))
        atom_set_angle = tf.transpose(atom_set_angle, perm=[2, 1, 0])
        atom_set_angle = tf.reshape(tf.repeat(tf.reshape(atom_set_angle, shape=(-1,self.n_rx)),
                                              tf.reshape(slice_indices_angle,shape=(-1,)),
                                              axis=0),
                                    shape=(tf.shape(curr_inp)[0],-1, self.n_rx))
        atom_set_angle = tf.reshape(atom_set_angle, shape=(tf.shape(curr_inp)[0], -1, self.n_rx))
        atom_set_angle = tf.expand_dims(atom_set_angle, axis=-1)

        slice_indices_delay = tf.cast(tf.reduce_sum(a_mask, axis=1), dtype=tf.int32)
        atom_set_delay = tf.broadcast_to(tf.expand_dims(self.delay_grid, axis=-1), shape=tf.concat([tf.shape(self.delay_grid), [tf.shape(curr_inp)[0]]], axis=0))
        atom_set_delay = tf.transpose(atom_set_delay, perm=[2, 1, 0])
        atom_set_delay = tf.reshape(tf.repeat(tf.reshape(atom_set_delay, shape=(-1,self.n_adc)),
                                              tf.reshape(slice_indices_delay, shape=(-1,)),
                                              axis=0),
                                    shape=(tf.shape(curr_inp)[0],-1, self.n_adc))
        atom_set_delay = tf.reshape(atom_set_delay, shape=(tf.shape(curr_inp)[0], -1, self.n_adc))
        atom_set_delay = tf.expand_dims(atom_set_delay, axis=-2)

        # Gain estimation, using LU decomposition for solving system of equations
        # A_p: see algorithm based on 'partial pivoting' PA = LU
        psi_product_matrices = tf.matmul(atom_set_angle, atom_set_delay)
        psi_product_matrices_reshaped = tf.reshape(psi_product_matrices, shape=tf.concat([tf.shape(psi_product_matrices)[:2], [-1]], axis=0))
        curr_inp_reshaped = tf.expand_dims(tf.reshape(curr_inp, shape=(tf.shape(curr_inp)[0], -1)), axis=1)

        A = tf.matmul(psi_product_matrices_reshaped, tf.transpose(tf.math.conj(psi_product_matrices_reshaped), perm=[0, 2, 1]))
        b = tf.expand_dims(tf.reduce_sum(tf.multiply(tf.math.conj(psi_product_matrices_reshaped), curr_inp_reshaped), axis=-1), axis=-1)

        # Matrix diagonal epsilon offset to prevent zero-values causing Lapack decomposition problems
        myeye = tf.cast(tf.eye(tf.shape(A)[1]) * tf.keras.backend.epsilon(), dtype=tf.complex128)
        myeye = tf.reshape(myeye, shape=tf.concat([tf.constant([1]), tf.shape(myeye)], axis=0))
        A = A + myeye

        A_lu, A_p = tf.linalg.lu(input=A, output_idx_type=tf.int32)
        alpha = tf.linalg.lu_solve(lower_upper=A_lu, perm=A_p, rhs=b)

        # Residual update
        shape_diff = tf.size(tf.shape(psi_product_matrices)) - tf.size(tf.shape(alpha))
        alpha = tf.reshape(alpha, shape=tf.concat([tf.shape(alpha), tf.ones(shape=(shape_diff,), dtype=tf.int32)], axis=0))
        return tf.subtract(curr_inp, tf.reduce_sum(tf.multiply(alpha, psi_product_matrices), axis=1))

    def get_config(self):

        config = super().get_config()
        config.update({
               "rmin": self.rmin,
               "rmax": self.rmax,
               "theta_min": self.theta_min,
               "theta_max": self.theta_max,
               "rsamp": self.rsamp,
               "theta_samp": self.theta_samp,
               "n_adc": self.n_adc,
               "n_rx": self.n_rx
            })
        return config


# yolo = TargetMapRecovery(n_rx=4, n_adc=256, rmin=0., rmax=5., theta_min=-60., theta_max=60., rsamp=40, theta_samp=60)
# curr_input_r1 = tf.random.normal(shape=(100, 4, 256), seed=42, dtype=tf.float64)
# curr_input_r2 = tf.random.normal(shape=(100, 4, 256), seed=84, dtype=tf.float64)
# curr_input_c = tf.complex(curr_input_r1, curr_input_r2)
#
# yolo.build(input_shape=None)
# print(yolo(curr_input_c).shape)
