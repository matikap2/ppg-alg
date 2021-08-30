import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy.signal import lfilter, firwin, find_peaks
import math as m

def load_data():
    data = []
    with open('data.csv', 'r') as fd:
        reader = csv.reader(fd)
        for row in reader:
            for col in row:
                data.append(int(col))
    return data

def balance_signal(x):
    balanced = x - np.mean(x)
    return balanced.astype(int)

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def get_fir_coeffs(sample_rate, cutoff_hz, coeffs_num):
    nyq_rate = sample_rate / 2.
    return firwin(coeffs_num, cutoff_hz/nyq_rate, window='hamming')

def filter_signal(coeffs, signal):
    return lfilter(coeffs, 1.0, signal)

def calc_meas_time(sample_num, sample_rate):
    return sample_num / sample_rate

def calc_hr(peaks_num, meas_time):
    return m.floor(peaks_num * (60.0 / meas_time))

def main():
    data = load_data()

    # remove ac offset, decrease values to much lower
    balanced_signal = balance_signal(data)

    # smoothing
    running_mean_signal = running_mean(balanced_signal, 24) #todo: tweak window size

    # fir? why not, little better smoothed
    # somewhere i read that ppg is in range of 0.5-4hz
    sample_rate = 100 # 1/s
    coeffs = get_fir_coeffs(sample_rate=sample_rate, cutoff_hz=4.0, coeffs_num=8)  #todo: tweak coeffs_num, vals between (8;16) 
                                                                                    #      seems more or less fine
    print(f'lowpass filter coeffs: {coeffs} len: {len(coeffs)}')
    filtered_signal = filter_signal(coeffs, running_mean_signal)

    # find peaks
    peaks, _ = find_peaks(filtered_signal, distance=24) #todo: tweak distance (or maybe even remove?)

    # display results
    plt.plot(running_mean_signal)
    plt.plot(filtered_signal)
    plt.plot(peaks, filtered_signal[peaks], 'x')
    peaks_num = len(peaks)
    time = calc_meas_time(len(filtered_signal), sample_rate)
    hr = calc_hr(peaks_num, time)
    print(f'peak number: {peaks_num}, meas time: {time}s, hr: {hr}')
    plt.show()



if __name__ == '__main__':
    main()
