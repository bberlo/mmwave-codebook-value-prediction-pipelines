from tensorflow.keras.preprocessing.sequence import pad_sequences
from matplotlib.pyplot import plot, scatter, show, subplots, legend
from scipy.signal import convolve, filtfilt, firwin
from scipy.fft import fft, ifft
import multiprocessing
import numpy as np
import itertools
import argparse
import h5py
import re
import os


# Low/high frequency static noise filtering. Details derived from: Singh, A.K. and Kim, Y.H.,
# Accurate measurement of drone's blade length and rotation rate using pattern analysis with W-band radar.
# https://doi.org/10.1049/el.2017.4494
def outer_bandpass_filter(s, d, Fs):

    fir_filt_coeff = firwin(numtaps=4, cutoff=[100, 4900000], window='hanning', pass_zero='bandpass', fs=Fs)
    return filtfilt(b=fir_filt_coeff, a=1, x=s, axis=d)


# Periodic filtering across slow-time dimension for higher SNR. Details derived from:
# Lin et al., "Periodic-Filtering Method for Low-SNR Vibration Radar Signal",
# https://www.mdpi.com/2072-4292/15/14/3461
def periodic_filter_nd(s, M, d, Nc, Tc):
    """
    Perform periodic filtering of a specific dimension of a numpy ndarray.

    Parameters:
    s (np.ndarray): The input numpy ndarray.
    M (int): The order number of adjacent cycles involved in filtering.
    d (int): The dimension to apply the filter to.

    Returns:
    sf (np.ndarray): The filtered numpy ndarray.
    """
    # Move the dimension to filter to the last axis
    s = np.moveaxis(s, d, -1)

    a_shape = [1]*len(s.shape)
    a_shape[-1] = s.shape[-1]
    window = np.hanning(s.shape[-1]).reshape(a_shape)

    # Perform Hann-windowed Fourier transform to obtain frequency spectrum
    # spectrum = np.abs(fft(s * window, axis=-1))
    spectrum = np.abs(fft(s, axis=-1))
    k = np.concatenate((np.arange(0, Nc // 2 + 1), np.arange(-Nc // 2 + 1, 0)))
    frequencies = k / (Nc * Tc)

    # Find the peak frequency
    fv = abs(frequencies[np.argmax(spectrum, axis=-1)]) + np.finfo(frequencies.dtype).eps

    T = 1 / fv  # vibration period
    g = np.zeros_like(s)  # initialize g(v)

    # construct g(v)
    for m in range(-M, M + 1):
        for i in np.ndindex(*s.shape[:-1]):
            g[i] += np.roll(s[i], int(m * T[i]), axis=-1)
    g /= (1 + 2 * M)

    # Normalize g
    g /= np.sum(g, axis=-1, keepdims=True)

    # perform convolution
    sf = np.zeros_like(s)
    for i in np.ndindex(*s.shape[:-1]):
        sf[i] = convolve(s[i], g[i], mode='same')

    # Move the filtered dimension back to its original position
    sf = np.moveaxis(sf, -1, d)

    return sf


# DC offset correction across slow-time dimension, M.Alizadeh et al.,
# “Remote monitoring of human vital signs using mm-wave fmcw radar,”
# https://ieeexplore.ieee.org/document/8695699.
def least_square_optimization(complex_array, axis):

    # Move the specified axis to the last dimension
    complex_array = np.moveaxis(complex_array, axis, -1)

    A = np.stack([np.ones(shape=complex_array.shape), -np.real(complex_array), -np.imag(complex_array)], axis=-1)
    b = -np.linalg.norm(A, axis=-1)**2
    A_reshaped = A.reshape(-1, A.shape[-2], A.shape[-1])
    b_reshaped = b.reshape(-1, b.shape[-1])

    # Solve the normal equation (A^TA)^-1A^Tb
    ata = np.linalg.inv(np.matmul(np.transpose(A_reshaped, (0, 2, 1)), A_reshaped))
    atb = np.matmul(np.transpose(A_reshaped, (0, 2, 1)), b_reshaped[:, :, np.newaxis])
    y = np.squeeze(np.matmul(ata, atb))

    # Extract the circle's origin and radius from the result
    circle_origin_real = y[:, 1]
    circle_origin_imag = y[:, 2]

    return circle_origin_real, circle_origin_imag


# Static clutter removal (DC removal) via signal average subtraction across slow-time dimension
# Veld, R. (2023) Human gait model individualized by low-cost radar measurements. UT Master's thesis.
def arith_mean_dims(ndarray, dimensions):
    # Initialize a tuple for the shape of the output ndarray
    output_shape = []

    # Iterate over the shape of the input ndarray
    for i in range(len(ndarray.shape)):
        # If the current dimension is in the list of dimensions to average over
        if i in dimensions:
            # Append 1 to the output shape
            output_shape.append(1)
        else:
            # Otherwise, append the size of the current dimension
            output_shape.append(ndarray.shape[i])

    # Compute the average across the specified dimensions
    avg_ndarray = np.mean(ndarray, axis=tuple(dimensions), keepdims=True)

    # Reshape the averaged ndarray to the output shape
    avg_ndarray = avg_ndarray.reshape(output_shape)

    return avg_ndarray


def generate_indices(length, reduction_factors):
    indices = [0]
    i = 0
    while indices[-1] < length:
        indices.append(indices[-1] + reduction_factors[i])
        i = (i + 1) % len(reduction_factors)
    return indices[:-1]  # Exclude the last index if it's greater than or equal to length


# "Mmwave Radar Device ADC Raw Data Capture", TI Application report,
# https://www.ti.com/lit/an/swra581b/swra581b.pdf
def read_bin_dca1000evm_xwr14xx(inPath, fileName, dset_type, scenario):

    numADCBits = 16  # number of ADC bits per sample
    numLanes = 4  # do not change. number of lanes is always 4 even if only 1 lane is used. unused lanes
    isReal = 0  # set to 1 if real only data, 0 if complex dataare populated with 0
    chirps = 255
    numTX = 1
    pad_width = 1  # fast-time offset such that 0 Hann window values do not interfere with reconstruction

    if scenario == 1 or scenario == 2:
        frames = 70
        samplesADC = 256
        red_factors = [8, 9]
        frame_periodicity = 0.15  # frame time in seconds
    elif scenario == 3:
        frames = 35
        samplesADC = 64
        red_factors = [4, 4, 4, 5]
        frame_periodicity = 0.3  # frame time in seconds
    else:
        raise ValueError("Unknown scenario value. Allowed values: 1, 2, 3.")

    chirp_duration = frame_periodicity / chirps  # Denotes time in-between chirp ramp start instances
    flat_size = samplesADC * chirps * frames

    # read file and convert to signed number
    # read .bin file
    with open(inPath + os.sep + fileName, 'rb') as fid:
        adcData = np.fromfile(fid, dtype=np.int16)

    # if 12 or 14 bits ADC per sample compensate for sign extension
    if numADCBits != 16:
        l_max = 2**(numADCBits-1)-1
        adcData[adcData > l_max] -= 2**numADCBits

    # organize data by LVDS lane (a.k.a. each physical RX)
    # for real only data
    if isReal:
        # reshape data based on one samples per LVDS lane
        adcData = np.reshape(adcData, (numLanes, -1)).astype(np.int32)

    # for complex data
    else:
        # reshape and combine real and imaginary parts of complex number
        adcData = np.reshape(adcData, (numLanes*2, -1))
        adcData = (adcData[:numLanes,:] + 1j*adcData[numLanes:,:]).astype(np.complex64)

    # Cookie-cut frame, loop, adc samples and reshape according
    # to TX transmission structure TX1, (TX2), (TX3), TX1, ...,
    # Pad samples that have size which doesn't allow correct dimension shaping (falls within region that is truncated)
    try:
        adcData = np.reshape(adcData, (-1, samplesADC, numTX, chirps // numTX, frames))
    except ValueError:
        to_pad = ((0, 0), (0, flat_size - adcData.shape[-1]))
        adcData = np.pad(adcData, pad_width=to_pad, mode='constant', constant_values=0.)
        adcData = np.reshape(adcData, (-1, samplesADC, numTX, chirps // numTX, frames))

    # Hann-windowed Fourier transform across fast-time dimension to get baseband radar cube into
    # RX, f_b, TX, slow-time, frame version. No additional FFTs because of single target in experiment scenarios
    pads = [(0, 0)] * len(adcData.shape)
    pads[1] = (pad_width, pad_width)
    adcData_padded = np.pad(adcData, pad_width=pads, mode='constant', constant_values=0.)
    a_shape = [1] * len(adcData_padded.shape)
    a_shape[1] = adcData_padded.shape[1]
    window_fast_time = np.hanning(adcData_padded.shape[1]).reshape(a_shape)
    adcData = fft(adcData_padded * window_fast_time, axis=1)
    samplesADC += (2 * pad_width)

    # Plot complex constellation over slow-time dimension for all fast-time indices for RX/TX/Frame sample
    # fig, ((ax1, ax2), (ax3, ax4)) = subplots(nrows=2, ncols=2)
    # for elem in adcData[0, :, 0, :, 0]:
    #     ax1.scatter(np.real(elem), np.imag(elem))
    # for elem in adcData[1, :, 0, :, 0]:
    #     ax2.scatter(np.real(elem), np.imag(elem))
    # for elem in adcData[2, :, 0, :, 0]:
    #     ax3.scatter(np.real(elem), np.imag(elem))
    # for elem in adcData[3, :, 0, :, 0]:
    #     ax4.scatter(np.real(elem), np.imag(elem))

    # Plot complex magnitude over fast-time dimension for Slow-time/RX/TX/Frame sample
    # fig, ((ax1), (ax2), (ax3), (ax4)) = subplots(nrows=4, ncols=1)
    # yolo = adcData[0, :, 0, 100, 15]
    # ax1.plot(np.abs(yolo), label='step 1')

    # DC offset, causing unwanted phase rotation, correction
    circle_real, circle_imag = least_square_optimization(adcData, axis=3)
    offset_params = circle_real + 1j * circle_imag
    offset_params = np.reshape(offset_params, newshape=(-1, samplesADC, numTX, 1, frames))
    adcData = np.subtract(adcData, offset_params)

    # Plot complex magnitude over fast-time dimension, after DC offset correction, for Slow-time/RX/TX/Frame sample
    # yolo2 = adcData[0, :, 0, 100, 15]
    # ax2.plot(np.abs(yolo2), label='step 2')

    # Phase unwrapping across slow-time/fast-time dimensions, M.Alizadeh et al.,
    # “Remote monitoring of human vital signs using mm-wave fmcw radar,”
    # https://ieeexplore.ieee.org/document/8695699.
    adcData_abs = np.abs(adcData)
    adcData_ang = np.angle(adcData)
    adcData_ang_unwrap = np.unwrap(adcData_ang, axis=1)
    adcData_ang_unwrap = np.unwrap(adcData_ang_unwrap, axis=3)
    adcData = adcData_abs * np.exp(1j * adcData_ang_unwrap)

    # DC removal via mean signal across slow-time subtraction
    adcData_avg = arith_mean_dims(adcData, [3])
    adcData = np.subtract(adcData, adcData_avg)

    # Plot complex magnitude over fast-time dimension, after DC removal, for Slow-time/RX/TX/Frame sample
    # yolo3 = adcData[0, :, 0, 100, 15]
    # ax3.plot(np.abs(yolo3), label='step 3')

    # Higher SNR via periodic smoothing filter across slow-time, validated on extracted phase vibration signal in prev. literature.
    # Not considered due to causing significant peak position alterations to complex magnitude over fast-time dimension
    # adcData = periodic_filter_nd(adcData, M=1, d=3, Nc=chirps, Tc=chirp_duration)

    # Higher SNR via close to DC/high frequency band filtering with bandpass filter, validated on blade rotation signal prev. literature.
    # Not considered due to causing significant peak magnitude reductions across entire bin dimension (not just unwanted bins)
    # adcData = outer_bandpass_filter(adcData, d=1, Fs=10000000)

    # Plot complex magnitude over fast-time dimension, after SNR increase, for Slow-time/RX/TX/Frame sample
    # yolo4 = adcData[0, :, 0, 100, 15]
    # ax4.plot(np.abs(yolo4), label='step 4')
    # legend()
    # show()

    # Plot complex constellation over slow-time dimension, after DC offset correction, for all fast-time indices for RX/TX/Frame sample
    # fig2, ((ax5, ax6), (ax7, ax8)) = subplots(nrows=2, ncols=2)
    # for elem in adcData[0, :, 0, :, 0]:
    #     ax5.scatter(np.real(elem), np.imag(elem))
    # for elem in adcData[1, :, 0, :, 0]:
    #     ax6.scatter(np.real(elem), np.imag(elem))
    # for elem in adcData[2, :, 0, :, 0]:
    #     ax7.scatter(np.real(elem), np.imag(elem))
    # for elem in adcData[3, :, 0, :, 0]:
    #     ax8.scatter(np.real(elem), np.imag(elem))
    # show()

    # Reverse hann-windowed Fourier transform across fast-time dimension by means of performing IFFT
    # Reverse steps taken from: https://dsp.stackexchange.com/questions/42283/inverse-the-hanning-window
    adcData = ifft(adcData, axis=1)
    adcData = adcData / window_fast_time  # Rough compensation for the Hanning window
    adcData = adcData[:, pad_width:-pad_width, :, :, :]  # Remove the padding (at padding indices contains NaN due to zero-divide)
    samplesADC -= (2 * pad_width)

    # Exclude elev. TX antenna, exclude all but every nth chirp transmission TX structure,
    # where n is dictated by reduction factor
    if 2 < numTX <= 3:
        adcData = adcData[:, :, :-1, :, :]
        newshape = (numLanes * (numTX - 1), samplesADC, -1)
    elif 0 < numTX <= 2:
        adcData = adcData
        newshape = (numLanes * numTX, samplesADC, -1)
    else:
        raise ValueError("numTX cannot be larger than 3 or smaller than 0")

    # End-truncate across flat slow-time axis (slow-time dimensions across frames stitched together)
    adcData = np.transpose(adcData, axes=[0, 2, 1, 3, 4])
    adcData = np.reshape(adcData, newshape=newshape)
    time_slice = int(adcData.shape[-1] - (0.5 / chirp_duration))  # 10.500 ms - 500 ms = 10.000 ms
    adcData = adcData[:, :, :time_slice]

    # Dynamic decimation across flat slow-time axis (slow-time dimensions across frames stitched together)
    # Please note: this op causes the data to become unuseful for velocity calculation via Doppler shift
    index_array = generate_indices(adcData.shape[2], reduction_factors=red_factors)
    adcData = adcData[:, :, index_array]

    # Only pads if there is an absolute necessity to do so
    adcData = np.reshape(adcData, newshape=(-1, adcData.shape[-1]))
    adcData = pad_sequences(adcData, maxlen=2000, dtype='complex64', padding='post', truncating='post', value=0.)
    adcData = np.reshape(adcData, newshape=(*newshape[:-1], 2000))

    if dset_type == 'complex':
        adcData = np.transpose(adcData, axes=[2, 1, 0])
        return adcData.astype(dtype=np.complex128)
    else:
        adcData_abs = np.abs(adcData)
        adcData_ang = np.angle(adcData)
        adcData = np.stack([adcData_abs, adcData_ang], axis=-1)
        adcData = np.transpose(adcData, axes=[2, 1, 0, 3])
        return adcData.astype(dtype=np.float32)


def parse_bin(inPath, fileName, dset_type):

    fileName_split = fileName.split(sep='.')[0]
    fileName_split = fileName_split.split(sep='_')
    regex = re.compile("[^\\d.]", re.IGNORECASE)
    fileName_split = [int(re.sub(regex, '', x)) for x in fileName_split]

    radar = fileName_split[0]
    scenario = fileName_split[3]
    activity = fileName_split[4]

    rel_task_label_path = 'Labels' + os.sep + 'labelData_sc_{}_exp_{}.txt'.format(scenario, activity)
    rel_domain_label_path = 'Radar{}-domain-labels'.format(radar) + os.sep + 'Scenario {}.txt'.format(scenario)
    rel_class_label_path = 'Classes' + os.sep + 'Scenario {}.txt'.format(scenario)

    # Task label parsing
    task_label = np.genfromtxt(fname=inPath.rsplit('\\', 1)[0] + os.sep + rel_task_label_path, dtype=np.int8, delimiter=',')

    # Domain label parsing
    domain_factors_enc = np.genfromtxt(fname=inPath.rsplit('\\', 1)[0] + os.sep + rel_domain_label_path, dtype=np.int32, delimiter=',')[activity - 1]
    empty_domain_label = np.zeros(shape=(8, 15, 2, 1, 1), dtype=np.int8)  # top-down anti-clockwise start orientation, marked start position, radar platform, room, subject
    empty_domain_label[domain_factors_enc[0]-1, domain_factors_enc[1]-1, domain_factors_enc[2]-1, 0, 0] = 1
    domain_label = empty_domain_label.flatten()

    # Class label parsing
    classes_enc = np.genfromtxt(fname=inPath.rsplit('\\', 1)[0] + os.sep + rel_class_label_path, dtype=np.int32, delimiter=1)
    class_factor = classes_enc[activity - 1]
    empty_class_label = np.zeros(shape=(5,), dtype=np.int8)
    empty_class_label[class_factor - 1] = 1
    class_label = empty_class_label

    # Input parsing
    input_tensor = read_bin_dca1000evm_xwr14xx(inPath, fileName, dset_type, scenario)
    assert not np.any(np.isnan(input_tensor)), "nan values found in input_tensor"

    if dset_type == 'complex':
        input_shape = (1, 2000, 256, 4)
        max_shape = (None, 2000, 256, 4)
        dtype = "complex128"
        fname = 'codebook-value-prediction-domain-leave-out-benchmark-complex.hdf5'
    else:
        input_shape = (1, 2000, 256, 4, 2)
        max_shape = (None, 2000, 256, 4, 2)
        dtype = "float32"
        fname = 'codebook-value-prediction-domain-leave-out-benchmark-float.hdf5'

    f_path = inPath.rsplit('\\', 1)[0] + os.sep + 'Output' + os.sep + fname
    trans_table = str.maketrans({"\\": r"\\"})

    lock.acquire()

    with h5py.File(f_path.translate(trans_table), 'a') as f:

        if 'task_labels' in f and 'domain_labels' in f and 'inputs' in f and 'class_labels' in f:

            dset_1 = f['inputs']
            dset_2 = f['task_labels']
            dset_3 = f['domain_labels']
            dset_4 = f['class_labels']

            dset_1.resize(dset_1.shape[0] + 1, axis=0)
            dset_2.resize(dset_2.shape[0] + 1, axis=0)
            dset_3.resize(dset_3.shape[0] + 1, axis=0)
            dset_4.resize(dset_4.shape[0] + 1, axis=0)

            dset_1[-1] = input_tensor
            dset_2[-1] = task_label
            dset_3[-1] = domain_label
            dset_4[-1] = class_label

        else:

            dset_1 = f.create_dataset("inputs", input_shape, dtype=dtype, maxshape=max_shape)
            dset_2 = f.create_dataset("task_labels", (1, 2000, 8), dtype="int8", maxshape=(None, 2000, 8))
            dset_3 = f.create_dataset("domain_labels", (1, 240), dtype="int8", maxshape=(None, 240))
            dset_4 = f.create_dataset("class_labels", (1, 5), dtype="int8", maxshape=(None, 5))

            dset_1[0] = input_tensor
            dset_2[0] = task_label
            dset_3[0] = domain_label
            dset_4[0] = class_label

    lock.release()


def exclude_from_parsing(fileName):

    if '.bin' not in fileName:
        return False

    fileName_split = fileName.split(sep='.')[0]
    fileName_split = fileName_split.split(sep='_')
    regex = re.compile("[^\\d.]", re.IGNORECASE)
    fileName_split = [re.sub(regex, '', x) for x in fileName_split]

    if int(fileName_split[3]) > 2:
        return False

    return True


def init(a_lock):
    global lock
    lock = a_lock


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Experiment automation setup script.")
    parser.add_argument('-d_t', '--dset_type', help='<Required> Dset type to be parsed: "complex", "" (defaults to stacked ampphase last dim)', required=True)
    args = parser.parse_args()

    main_lock = multiprocessing.Lock()

    with multiprocessing.Pool(processes=multiprocessing.cpu_count(), initializer=init, initargs=(main_lock,)) as p:

        for root, _, files in os.walk(r"C:\TUe-PhD\UT-RS-3 Domain-Invariant Beam Adaptivity\experiment_environment\pre-processing\Radar data", topdown=True):
            if len(files) == 0:
                continue

            files.sort()
            files = list(filter(lambda x: exclude_from_parsing(x), files))

            if len(files) != 0:
                files = iter(files)
                res = p.starmap(func=parse_bin, iterable=zip(itertools.repeat(root), files, itertools.repeat(args.dset_type)))
