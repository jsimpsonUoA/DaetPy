'''
Extra analysis functions and utilities for
DaetPy objects.

Jonathan Simpson, Physical Acoustics Lab,
University of Auckland, 2020.
'''

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import correlate, detrend, resample, decimate
from scipy.stats import linregress

def coda_wave_interferometry(reference_waveform, times, data, min_time=0, 
                        max_time=1e6, window_size=20, overlap=0, taper=0, pad=0,
                        parabolic_approx=False, dt_from_grad=False, dt_from_avs=True,
                        plot_examples=False, return_all=False):
    '''
    Perform coda wave coda wave interferometry on the
    waveform(s) in data, comparing to reference_waveform
    in sliding windows. The only signal processing done is
    a demeaning of both the reference waveform and data before
    correlation.

    Arguments:
        --reference_waveform: The reference waveform to compare the data to
        --times: The time array for the reference_waveform and the waveforms
            in data
        --data: The waveform data to calculate the cwi for. This can be a 
            1D or 2D array, where the last axis has the same length as times.
        --min_time: The minimum time to perform cwi from. This will be the 
            left edge of the window at the first window position.
        --max_time: The maximum time to perform the cwi to. The actual maximum
            will be the next lowest right edge so that the final position contains
            a full window.
        --window_size: The size of the window, in units the same as time
        --taper: A percentage betewen 0 and 100 specifying how much of the
            windowed data to apply a taper to.
        --pad: How many powers of 2 above the length of the window to pad
            each window before correlating.
        --overlap: The amount that windows overlap, in units of time. 
            An overlap of 0 means there is no overlap, while an overlap 
            half of window_size means adjacent windows have half their 
            times overlapping.
        --parabolic_approx: True to obtain sub-sample accuracy in 
            the time delays by fitting a parabola around the maximum 
            three points of the cross correlation, rather than using 
            the time at the maximum as the delay (see Cespedes, 1995, 
            Ultrasonic Imaging).
        --dt_from_grad: Calculate dt/t for each waveform by finding the graident
            of the best fit line in the graph of dt versus window centre.
        --dt_from_av: Calculate dt/t for each waveform by taking the average 
            of dt divided by the window centre time for each window. Default True
        --plot_examples: Plot every twentieth cwi gradient plot
        --return_all: True to return an array with all the lag times in each
            window for all traces and a 1D array of the window centres, along
            with dt_over_t.

    Returns:
        --dt_over_t: The average fraction by which the data waveforms(s) lag
            the reference waveforms, determined from the gradient of the
            best fit line of the cross correlation lags of each window.
            Positive == data lags the reference waveform
        --dt_std_devs: The standard deviations for each of the estimate
            of dt_over_t
        
        If return_all is True:
        --all_windows: A 2D array of shape (trace_number, window_lags)
        --window_centres: A 1D array of the window centres.
    '''

    min_time, max_time = max(times[0],min_time), min(times[-1],max_time)
    min_time_ind = np.argmin(np.abs(times-min_time))
    max_time_ind = np.argmin(np.abs(times-max_time))
    first_window_centre = np.argmin(np.abs(times-(min_time+window_size/2)))
    ind_win_size = (first_window_centre - min_time_ind) * 2 + 1
    sampling_rate  = 1 / np.average(np.diff(times))
    ind_overlap_size = int(overlap * sampling_rate)

    num_windows = (max_time_ind - min_time_ind) // (ind_win_size - ind_overlap_size)
    if num_windows < 1:
        raise ValueError("max_time is too small for given min_time and window size.")
    if overlap > ind_win_size:
        raise ValueError('Overlap is greater than window size.')
    if len(data.shape) < 2:
        data = data.reshape((1,data.shape[0]))

    dt_over_t = np.zeros(data.shape[0])
    dt_std_devs = np.zeros(data.shape[0])
    all_lags = np.zeros((data.shape[0],num_windows))
    for i in range(data.shape[0]):

        comp_waveform = data[i]
        lags = np.zeros(num_windows)
        window_centres = np.zeros(num_windows)

        for x in range(num_windows):
            left_edge = int(min_time_ind + (ind_win_size - ind_overlap_size) * x)
            window_centre = int(first_window_centre + (ind_win_size - ind_overlap_size) * x)
            reference_signal, _ = get_windowed_data(reference_waveform, sampling_rate,min_ind=left_edge, max_ind=left_edge+ind_win_size, apply_detrend=True,taper=taper,pad=pad)
            data_signal, win_times = get_windowed_data(comp_waveform, sampling_rate,min_ind=left_edge, max_ind=left_edge+ind_win_size, apply_detrend=True,taper=taper,pad=pad)
            
            corr = correlate(reference_signal,data_signal,mode='full')
            corr_times = np.concatenate((np.flip(-1. * win_times[1:],axis=0), win_times))

            max_lag_ind = np.argmax(corr)
            max_lag = corr_times[max_lag_ind]

            if parabolic_approx and max_lag_ind != 0 and max_lag_ind != len(corr)-1:
                y0, y1, y2 = corr[max_lag_ind-1], corr[max_lag_ind], corr[max_lag_ind+1]
                delta_hat = (y0-y2) / (2* (y0-2*y1+y2) ) * (1 / sampling_rate)
                max_lag = max_lag + delta_hat

            lags[x] = max_lag  * -1.
            window_centres[x] = times[window_centre]
        
        all_lags[i] = lags
        if dt_from_grad:
            best_fit = linregress(window_centres, lags)
            dt_over_t[i] = best_fit[0]
            std_dev = best_fit[-1] * np.sqrt(len(window_centres))
            dt_std_devs[i] = std_dev
        elif dt_from_avs:
            dt_over_t[i] = np.average(lags/window_centres)
            dt_std_devs[i] = np.std(lags/window_centres)

        if plot_examples:# and i%20 == 0:
            if dt_from_grad:
                plt.plot(window_centres, lags, 'rx')
                plt.plot(window_centres,(best_fit[0]*window_centres+best_fit[1]))
                plt.xlabel('Time ($\mu$s)'); plt.ylabel('Lag ($\mu$s)')
                plt.axhline(0.,ls='--',color='k')
                plt.show()
            else:
                plt.plot(window_centres, lags/window_centres*100., 'rx')
                plt.axhline(np.average(lags/window_centres*100.))
                plt.xlabel('Time ($\mu$s)'); plt.ylabel('dt/t (\%)')
                plt.axhline(0.,ls='--',color='k')
                plt.show()

            
                plt.plot(corr_times,corr)
                plt.show()

    if not return_all:
        return dt_over_t, dt_std_devs

    return dt_over_t, dt_std_devs, all_lags, window_centres
    

def get_windowed_data(waveform, sampling_rate, min_ind=0, max_ind=-1, 
                    apply_detrend=False, taper=10, pad=0):
    '''
    Function which returns a segment of
    data from the larger traces as sepcified by
    the window.

    Arguments:
        waveform: The data to return the window from
        sampling_rate: The sampling rate of the data
        min_ind: The index of the left edge of the window
        max_ind: The index of the right edge of the window
        apply_detrend: True to detrend the data, False otherwise
        taper: A value (percentage) between 0 and 100 that
            specifies how much of the data to taper at the 
            edges of the window. 0 applies no taper, while
            100 applies a Hanning window to the entire window.
        pad: A non-negative integer which specifies how much 
            padding to apply to the data. 0 applies no padding,
            while 1 adds zeros to the end of the window
            to increase the number of points to the next pwoer of
            2 (i). A value of 2 increases the number of points to the
            to the i+1th power of 2, and so on.

    Returns:
        windowed_data: The properly formatted window from the waveform
        window_times: An array of times for the windowed data starting at 0.
    '''

    windowed_data = waveform[min_ind:max_ind]

    if apply_detrend:
        windowed_data = detrend(windowed_data,type='constant')

    if taper > 0:
        total_taper_points = int(taper / 100 * len(windowed_data)) // 2 * 2
        taper = np.hanning(total_taper_points)
        total_untapered_points = len(windowed_data) - total_taper_points
        first_taper = taper[:int(total_taper_points/2)]
        last_taper = taper[int(total_taper_points/2):]
        mask = np.concatenate((first_taper,np.ones(total_untapered_points),last_taper))
        windowed_data = np.multiply(windowed_data,mask)

    if pad > 0:
        power_of_two = np.log2(len(windowed_data)) // 1 + pad
        number_pad_samples = 2**power_of_two - len(windowed_data)
        windowed_data = np.concatenate((windowed_data,np.zeros(int(number_pad_samples))))

    window_times = np.arange(len(windowed_data)) / sampling_rate

    return windowed_data, window_times


def dtw_from_placescan(scan, min_time=0, max_time=np.inf, bandpass=None, averaging=1, **kwargs):
    """
    Perform dtw picking on the waveforms from a PlaceScan object.

    Arguments:
        --min_time: The minimum time to calculate the dtw for
        --max_time: The maximum time to calculate the dtw for
        --bandpass: The bandpass filter to apply to the data
        --averaging: An integer. Average this many waveforms
            together before doing the dtw. If multiple records
            are recorded per update, use this to only calculate
            one dtw result per update.
        --kwargs: The keyword arguments for the dtw (see dynamic_time_warping)

    Returns:
        dtw_lags_full, dt_times, percent_diff, percent_diff_std (see dynamic_time_warping)
    """

    values, times = get_placescan_data(scan, min_time, max_time, bandpass, averaging)
    return dynamic_time_warping(values,times,**kwargs)


def dynamic_time_warping(values,times,reference='adjacent_below',downsample=1,
                        alpha=0.0,plot_every=0,taper=0):
    """
    Function to perform dynamic time warping to find the
    offest between waveforms.

    Arguments:
        --values: A 2D Numpy array containing at least two
            waveforms to compare
        --times: The times (x values) of the waveforms in values
        --reference: The reference waveform(s) to use in the dtw. This can
            be 'adjacent_below', in which case the waveform one index below
            each waveform is used as the reference, or 'adjacent_above' in
            which case the waveform one index above each waveform is used as
            the reference. Otherwise, reference can be an integer referring
            to the waveform in values to use as the reference for all other
            waveforms.
        --downsample: An integer value to downsample the traces by before
           dtw. 1 is no downsampling, 5 takes every fifth sample, etc.
        --alpha: The regularization term for the dtw. 0. is no regularization.
            Values of 0.0-0.2 usually work well.
        --plot_every: Plot the result of the dtw for every ith waveform. 0 
           corresponds to no plotting.
        --taper: A number between 0 and 100 specifying the percentage of the
            dtw waveforms to apply a taper to. Default is 0 (off).
        
    Returns:
        --dtw_lags_full: An array of the same shape as values containing
            the offset between each waveform and the reference
            at each time
        --dtw_times: The time array for the values
        --percent_diff: The average percentage difference in time between 
            the waveform and its reference for each waveform in values.
        --percent_diff_std: The standard deviations of the values in percent_diff
    """

    from placescan.dtw_main import do_dtw

    sampling_rate = 1 / (times[1]-times[0])
    _, dt_times = get_windowed_data(values[0],sampling_rate,taper=taper)
    dt_times = dt_times + times[0]
    dt_times = dt_times[::downsample].copy(order='C')#

    dtw_lags_full = np.zeros((values.shape[0],len(dt_times)))
    percent_diff, percent_diff_std = np.zeros(values.shape[0]), np.zeros(values.shape[0])
    for i in range(values.shape[0]):

        if reference == 'adjacent_below':
            if i == 0:
                continue
            reference_index = i-1
            query_index = i
        elif reference == 'adjacent_above':
            if i == 0:
                continue
            reference_index = values.shape[0]-i
            query_index = values.shape[0]-i-1
        else:
            reference_index = reference
            query_index = i
            
        ref_waveform, _ = get_windowed_data(values[reference_index][::downsample],sampling_rate,taper=taper) # For tapering only
        query_waveform, _ = get_windowed_data(values[query_index][::downsample],sampling_rate,taper=taper)    
        ref_waveform = ref_waveform.copy(order='C')
        query_waveform = query_waveform.copy(order='C')
        t_times, q_times = do_dtw(ref_waveform, query_waveform, dt_times, dt_times, plot=False, alpha=alpha)

        lag_times = []
        cuml_length = 0
        for i2 in range(len(dt_times)):
            indices = np.where(np.abs(t_times-dt_times[i2]) < downsample / sampling_rate / 100)
            lag_times.append(np.average(q_times[indices]))
            cuml_length += len(indices[0])
        
        time_offset = lag_times-dt_times
        dtw_lags_full[query_index] = time_offset
        percent_diff[query_index] = 100.*np.average(time_offset/dt_times)
        percent_diff_std[query_index] = 100.*np.std(time_offset/dt_times)
        
        if plot_every and i%plot_every == 0:
            fig=plt.figure(figsize=(9,5))
            norm_factor = (np.amax(query_waveform)-np.amin(query_waveform)) / (np.amax(time_offset)-np.amin(time_offset))
            query_waveform_plot = query_waveform/norm_factor
            ref_waveform_plot = ref_waveform/norm_factor
            for i in range(len(t_times)):
                ref_time, query_time = t_times[i], q_times[i]
                ref_amp = ref_waveform_plot[np.argmin(np.abs(ref_time-dt_times))]
                query_amp = query_waveform_plot[np.argmin(np.abs(query_time-dt_times))]
                plt.plot([ref_time,query_time],[ref_amp, query_amp],color='r',ls='-',alpha=0.5)
            plt.plot(dt_times,ref_waveform/norm_factor, label='Reference Waveform',alpha=0.5,color='g',ls='-')
            plt.plot(dt_times,query_waveform/norm_factor, label='Query Waveform',alpha=0.5,color='C1')
            plt.plot(dt_times,time_offset, label='Time offset',color='C0')
            plt.plot(dt_times,-100*(time_offset)/dt_times, label='Percentage offset',color='r',ls='--')
            plt.title("Trace Number: {}".format(i)); plt.legend(fontsize=8)
            plt.ylabel('Amplitude'); plt.xlabel('Time')
            plt.show()

    return dtw_lags_full, dt_times, percent_diff, percent_diff_std


def get_placescan_data(scan, min_time=0, max_time=np.inf, bandpass=None, averaging=1):
    '''
    Get PlaceScan data for plotting or picking
    '''

    plot_data, plot_times = scan._get_plot_data(tmax=max_time, tmin=min_time, normed=True,bandpass=bandpass,detrend=True)
    values = np.zeros((plot_data.shape[0]//averaging,plot_data.shape[1]))
    for i in range(plot_data.shape[0]//averaging):
        values[i] = np.average(plot_data[i*averaging:(i+1)*averaging],axis=0)

    return values, plot_times