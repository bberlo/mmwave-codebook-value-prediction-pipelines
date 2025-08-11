from experiment_environment.custom_items.targetMapRecovery import TargetMapRecovery
import tensorflow as tf
import math


# Function to generate multidimensional joint complex plane noise
# Re-implemented from https://github.com/josemateosramos/SSLISAC/
def joint_complex_noise(var, shape):

    noise_real = tf.random.normal(shape=shape, mean=0.0, stddev=1.0, dtype=tf.float32)
    noise_imag = tf.random.normal(shape=shape, mean=0.0, stddev=1.0, dtype=tf.float32)
    noise = tf.complex(real=noise_real, imag=noise_imag)

    return tf.complex(real=tf.sqrt(var / 2.0), imag=0.) * noise


# Function to generate M-PSK constellation, m in M denotes complex value for specific bit pattern
# Re-implemented from https://github.com/josemateosramos/SSLISAC/
def mpsk(M, rotation=0.0):

    symbol_range = tf.range(start=0.0, limit=M, delta=1.0, dtype=tf.float32)
    exponential = 2.0 * math.pi / M * symbol_range + rotation

    return tf.exp(tf.complex(real=0., imag=exponential))


batch_size = 16
n_adc = 256
n_rx = 4
num_messages = 1
psk_type = 4
snr_db = 25

recovery_obj = TargetMapRecovery(n_rx=n_rx, n_adc=n_adc, rmin=0., rmax=5., theta_min=-60., theta_max=60., rsamp=40, theta_samp=60)
recovery_obj.build(input_shape=None)
rhoMatrix = tf.cast(tf.transpose(recovery_obj.delay_grid.value())[..., tf.newaxis], dtype=tf.complex64)[10:26]
thetaMatrix = tf.cast(tf.transpose(recovery_obj.angle_grid.value())[..., tf.newaxis], dtype=tf.complex64)[24:40]

beamform_selection = tf.random.uniform(shape=(batch_size,), minval=0, maxval=batch_size, dtype=tf.int32)
beamformMatrix = tf.gather(params=thetaMatrix, indices=beamform_selection, axis=0)

message_indicators = tf.random.uniform(shape=(batch_size, n_adc, 1), minval=0, maxval=psk_type, dtype=tf.int32)
symbols = tf.gather(params=mpsk(M=4, rotation=25 * (math.pi / 180)), indices=tf.reshape(message_indicators, shape=(-1,)))
symbols = tf.reshape(symbols, shape=(batch_size, n_adc, 1))

gain = joint_complex_noise(var=tf.sqrt(10 ** (snr_db / 10) * 1.0 / n_rx), shape=(batch_size, 1, 1))
rx_noise = joint_complex_noise(var=1.0, shape=(batch_size, n_rx, n_adc))

reflection_a = tf.complex(real=tf.sqrt(1.0 / n_adc), imag=0.) * gain * thetaMatrix
# reflection_a = gain * thetaMatrix

reflection_a = tf.matmul(reflection_a, tf.transpose(thetaMatrix, perm=[0, 2, 1]))
reflection_a = tf.matmul(reflection_a, beamformMatrix)

reflection_b = tf.transpose(symbols * rhoMatrix, perm=[0, 2, 1])
yr = tf.matmul(reflection_a, reflection_b)  # + rx_noise
yrv, _, yrc = tf.unique_with_counts(tf.reshape(tf.abs(yr), shape=(-1,)), out_idx=tf.int32)

print(yrv)
print(yrc)
