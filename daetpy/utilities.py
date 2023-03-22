'''
Extra analysis functions and utilities for
DaetPy objects.

Jonathan Simpson, Physical Acoustics Lab,
University of Auckland, 2020.
'''

import numpy as np
import matplotlib.pyplot as plt
import warnings
import time

from scipy.signal import correlate, detrend, resample, decimate
from scipy.stats import linregress
from scipy.interpolate import interp1d
from scipy.signal import iirfilter, zpk2sos, sosfiltfilt, correlate, detrend

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

def get_windowed_data_2d(waveform, sampling_rate, min_ind=0, max_ind=-1, 
                    apply_detrend=False, taper=10, pad=0):
    '''
    Same as get_windowed_data, but for 2D arrays where each
    row is a waveform. See the docs for get_windowed_data
    '''

    windowed_data = waveform[:,min_ind:max_ind]

    if apply_detrend:
        windowed_data = detrend(windowed_data,type='constant')

    d_len = windowed_data.shape[-1]
    if taper > 0:
        total_taper_points = int(taper / 100 * d_len) // 2 * 2
        taper = np.hanning(total_taper_points)
        total_untapered_points = d_len - total_taper_points
        first_taper = taper[:int(total_taper_points/2)]
        last_taper = taper[int(total_taper_points/2):]
        mask = np.concatenate((first_taper,np.ones(total_untapered_points),last_taper))
        windowed_data = windowed_data * np.tile(mask, (windowed_data.shape[0],1))

    if pad > 0:
        power_of_two = np.log2(d_len) // 1 + pad
        number_pad_samples = 2**power_of_two - d_len
        windowed_data = np.concatenate((windowed_data,np.zeros(windowed_data.shape[0],int(number_pad_samples))))

    window_times = np.arange(d_len) / sampling_rate

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
    Get PlaceScan data for plotting or picking. Normed is True.
    '''

    plot_data, plot_times = scan._get_plot_data(tmax=max_time, tmin=min_time, normed=False,bandpass=bandpass,detrend=True)
    values = np.zeros((plot_data.shape[0]//averaging,plot_data.shape[1]))
    for i in range(plot_data.shape[0]//averaging):
        values[i] = np.average(plot_data[i*averaging:(i+1)*averaging],axis=0)

    return values, plot_times


def stretching_cwi(reference_waveform, times, data, min_time=0, max_time=1e6, use_optimization=True,
                        num_iterations=2 ,max_stretch=0.1, stretch_limits=None, num_windows=1, taper=0, pad=0, 
                        plot_examples=False, return_all=False, different_references=False,
                        num_factors=201):
    '''
    Perform a trace-stretching type coda wave interferometry to calculate
    the relative time difference between two time series. This algorithm
    comes from Sens-Schoenfelder and Wegler (2006), GRL. The implementation
    is based off the MATLAB code associated with the paper "Coda Wave 
    Interferometry for Accurate Simultaneous Monitoring of Velocity and 
    Acoustic Source Locations in Experimental Rock Physics" by Singh,
    Curtis, Zhao, Cartwright-Taylor, and Main. See:
    https://github.com/JonathanSingh/cwi_codes/blob/master/cwi_codes/cwi_stretch_vel.m

    The time difference is found by stretching the query waveform(s)
    and finding the stretching factor for which the correlation factor
    between the reference and stretched waveform(s) is maximum. It has the
    advantage of not averaging over a window. Unlike regular CWI, it is 
    perfectly valid to use only one (long) window, but calculating the
    stretching factor over multiple windows provides an estimate of the
    uncertainty.

    Arguments:
        --reference_waveform: The reference waveform(s) to compare the data to
        --times: The time array for the reference_waveform and the waveforms
            in data (must be the same for all waveforms)
        --data: The waveform data to calculate the cwi for. This can be a 
            1D or 2D array, where the last axis has the same length as times.
        --min_time: The minimum time to perform cwi from. This will be the 
            left edge of the window at the first window position.
        --max_time: The maximum time to perform the cwi to.
        --use_optimization: If True, an optimised algorithm is used to find
            the best stretching factor for each waveform. This will significantly
            speed up the computation time, but it assumes that the function
            of correlation coefficient versus stretching factor is an inverted 
            parabola and has a single global maximum within the given range 
            of stretching factors. If False, then the cross correlation is 
            calculated for every possible stretching factor, and then the 
            maximum correlation coefficient is found.
        --num_iterations: The number of times to perform the stretch factor
            search. Each time, the search increment decreases by a factor
            of 20 and is re-centred around the previous iteration's best
            stretch factor.
        --max_stretch: The absolute value of the maximum stretching factor
            to search for. I.e. stretch facotrs of 1 - max_stretch to
            1 + max_stretch will be searched. Specified as a decimal number,
            not a percentage. Note that this is overriden if
            stretch_limits is specified.
        --stretch_limits: A two-element tuple specifying the lower and upper
            stretching limits. I.e. stretch factors of 1+lower_bound to
            1+upper bound will be searched. Overrides max_stretch.        
        --num_windows: The number of windows to independently calculate the 
                stretching factor for
        --taper: A percentage betewen 0 and 100 specifying how much of the
            windowed data to apply a taper to.
        --pad: How many powers of 2 above the length of the window to pad
            each window before correlating.
        --plot_examples: Plot example result plots.
        --return_all: True to return an array with all the lag times in each
            window for all traces and a 1D array of the window centres, along
            with dt_over_t.
        --different_references: True if there is a different reference for each
            waveform. In this case, reference_waveform must be the same length
            as data.
        --num_factors: The number of stretching factors to use in each iteration

    Returns:
        --dt_over_t: The average fraction by which the data waveforms(s) lag
            the reference waveforms, determined from the gradient of the
            best fit line of the cross correlation lags of each window.
            Positive == data lags the reference waveform
        --dt_std_devs: The standard deviations for each of the estimate
            of dt_over_t
        
        If return_all is True:
        --all_windows: A 2D array of shape (trace_number, window_lags)

    '''

    min_time, max_time = max(times[0],min_time), min(times[-1],max_time)
    min_time_ind = np.argmin(np.abs(times-min_time))
    max_time_ind = np.argmin(np.abs(times-max_time))

    total_range_inds = max_time_ind - min_time_ind
    window_size_ind = total_range_inds // num_windows

    ref_times = np.arange(0.,len(times))  # A reference array, independent of sampling rate

    if len(data.shape) < 2:
        data = data.reshape((1,data.shape[0]))

    dt_over_t = np.zeros(data.shape[0])
    dt_std_devs = np.zeros(data.shape[0])
    all_lags = np.zeros((data.shape[0],num_windows))

    all_times = 0
    for i in range(data.shape[0]):
        if i%100 == 0:
            print(i)
            if i>0:
                print(it_number)
            #    plt.plot(s_factors,max_coeffs)
            #    plt.show()
        comp_waveform = data[i]
        lags = np.zeros(num_windows)
        stretch_anchor = 0.

        if different_references:
            ref = reference_waveform[i]
        else:
            ref = reference_waveform
        
        
        for it_number in range(num_iterations):

            if stretch_limits == None:
                lower_limit = abs(max_stretch) / max(20 * it_number,1) * -1
                upper_limit = lower_limit * -1
            else:
                lower_limit = stretch_limits[0] / max(20 * it_number,1)
                upper_limit = stretch_limits[1] / max(20 * it_number,1)
            s_factors = np.linspace(stretch_anchor+lower_limit,stretch_anchor+upper_limit,num_factors) # Originally was 201 points
            
            if not use_optimization:
                i_func = interp1d(ref_times,comp_waveform)
                stretched_timeso = np.tile(ref_times, (len(s_factors),1)) / (1+np.repeat(s_factors,len(ref_times)).reshape(len(s_factors),len(ref_times)))
                stretched_timeso = np.where(stretched_timeso < ref_times[-1], stretched_timeso , 0.)
                stretched_waveformo = i_func(stretched_timeso)

            for j in range(num_windows):

                left_edge = min_time_ind+j*window_size_ind
                right_edge = left_edge + window_size_ind
                
                start_time = time.time()
                if use_optimization:
                    check_inds, it_number = [0,len(s_factors)-1], 0
                    do_fi_corr, do_li_corr = True, True
                    while (np.abs(check_inds[0]-check_inds[1]) > 2) and it_number < len(s_factors):
                        fi, li = check_inds[0], check_inds[1]
                        if do_fi_corr:
                            first_est = do_cwi_correlation(ref,ref_times,comp_waveform,s_factors[fi],left_edge,right_edge,taper,pad)
                        if do_li_corr:
                            last_est = do_cwi_correlation(ref,ref_times,comp_waveform,s_factors[li],left_edge,right_edge,taper,pad)
                        if first_est >= last_est:
                            check_inds = [fi,fi+int((li-fi)/2)]
                            do_fi_corr, do_li_corr = False, True
                        else:
                            check_inds = [fi+int((li-fi)/2),li]
                            do_fi_corr, do_li_corr = True, False
                        it_number += 1
                    best_index = int((check_inds[1]+check_inds[0])/2)
                else:
                    ref_trace0, _ = get_windowed_data_2d(np.tile(ref,(stretched_timeso.shape[0],1)), 1.,min_ind=left_edge, max_ind=right_edge, apply_detrend=True,taper=taper,pad=pad)
                    stretch_trace0, _ = get_windowed_data_2d(stretched_waveformo, 1.,min_ind=left_edge, max_ind=right_edge, apply_detrend=True,taper=taper,pad=pad)
                    get_divisor = lambda r,s: np.sqrt(np.dot(r,r)*np.dot(s,s))
                    max_coeffs = np.zeros(len(s_factors))
                    for k, s_factor in enumerate(s_factors):
                        corr = correlate(ref_trace0[k], stretch_trace0[k], mode='full')
                        corr /= get_divisor(ref_trace0[k],stretch_trace0[k])
                        max_coeffs[k] = np.amax(corr)
                        best_index = np.argmax(max_coeffs)
                best_stretching_factor = s_factors[best_index]
                lags[j] = best_stretching_factor
                all_times += time.time()-start_time

            all_lags[i] = lags
            time_lag = np.mean(lags)
            time_lag_err = np.std(lags)
            dt_over_t[i] = time_lag
            dt_std_devs[i] = time_lag_err
            stretch_anchor = time_lag

        if plot_examples:
            stretched_times = ref_times / (1+time_lag)
            stretched_times = np.where(stretched_times < ref_times[-1], stretched_times , 0.)
            i_func = interp1d(ref_times,comp_waveform)
            stretched_waveform = i_func(stretched_times)

            plt.plot(times, ref, 'r-',label='Reference')
            plt.plot(times, stretched_waveform, 'b-',label='Stretched Query')
            plt.xlabel('Time ($\mu$s)'); plt.ylabel('Amplitude')
            plt.legend(); plt.title('Best stretching factor: {}%'.format(time_lag*100.))
            plt.show()

    dt_over_t, all_lags = -1.*dt_over_t, -1.*all_lags

    if not return_all:
        return dt_over_t, dt_std_devs

    return dt_over_t, dt_std_devs, all_lags

def deprecated_stretching_cwi():
    """
    The old stretching cwi for loop structure. I'm
    keeping this here just in case there is a backwards
    compatibility issue and the current optimised version
    doesn't recreate the original.
    

    for it_number in range(num_iterations):

        stretch_limit = abs(max_stretch) / max(20 * it_number,1) * -1
        s_factors = np.linspace(stretch_anchor-stretch_limit,stretch_anchor+stretch_limit,201)
        i_func = interp1d(ref_times,comp_waveform)

        for j in range(num_windows):

            left_edge = min_time_ind+j*window_size_ind
            right_edge = left_edge + window_size_ind
    
            max_coeffs = np.zeros(len(s_factors))
            for k, s_factor in enumerate(s_factors):

                stretched_times = ref_times / (1+s_factor)
                stretched_times = np.where(stretched_times < ref_times[-1], stretched_times , 0.)
                stretched_waveform = i_func(stretched_times)

                ref_trace, _ = get_windowed_data(ref, 1.,min_ind=left_edge, max_ind=right_edge, apply_detrend=True,taper=taper,pad=pad)
                stretch_trace, _ = get_windowed_data(stretched_waveform, 1.,min_ind=left_edge, max_ind=right_edge, apply_detrend=True,taper=taper,pad=pad)
                    
                corr = correlate(ref_trace, stretch_trace, mode='full')
                corr /= np.sqrt(np.dot(ref_trace, ref_trace) * np.dot(stretch_trace, stretch_trace))
                max_coeffs[k] = np.amax(corr)

            best_stretching_factor = s_factors[np.argmax(max_coeffs)]
            lags[j] = best_stretching_factor
    """
    pass

def do_cwi_correlation(ref_waveform,ref_times,comp_waveform,s_factor,
                        left_edge,right_edge,taper,pad):
    """
    Support function for stretching CWI to do a stretching
    CWI of one reference and trace and return the correlation
    coefficient.
    """

    stretched_times = ref_times / (1+s_factor)
    stretched_times = np.where(stretched_times < ref_times[-1], stretched_times , 0.)
    i_func = interp1d(ref_times,comp_waveform)
    stretched_waveform = i_func(stretched_times)

    ref_trace, _ = get_windowed_data(ref_waveform, 1.,min_ind=left_edge, max_ind=right_edge, apply_detrend=True,taper=taper,pad=pad)
    stretch_trace, _ = get_windowed_data(stretched_waveform, 1.,min_ind=left_edge, max_ind=right_edge, apply_detrend=True,taper=taper,pad=pad)
        
    corr = correlate(ref_trace, stretch_trace, mode='full')
    corr /= np.sqrt(np.dot(ref_trace, ref_trace) * np.dot(stretch_trace, stretch_trace))
    return np.amax(corr)


def compute_time_delays(reference_waveform, data, times, trace_bandpass=None,  normalise=False, 
                        tmin=0., tmax=50., smoothing_amount=0, parabolic_approx=True):
    '''
    Function which computes the time delays using cross correlation
    Positive lag menas the sample  probe waveform lags behind the reference.

    Arguments:
        --reference_waveform: A reference waveform to compare probe
            waveforms to. See generate_reference_waveform
        --data: The data to compare to cross-correlate with the reference. Must
            be an array of at least one array that has a length equal to referece_waveform
        --times: A time array for reference_waveform (in units of seconds)
        --trace_bandpass: A bandpass filter applied to the probe waveforms before 
            cross correlation
        --normalise: True to normalise the probe waveforms before comparison
        --tmin: The minimum time to use for the window in the 
            cross correlation (in units of microseconds)
        --tmax: The maximum time for the window in the cross
            correlation (in units of microseconds)
        --smoothing_amount: An integer (or tuple) which specifies a number
            of probe waveforms to average before doing the cross
            correlation at each posiiton. Only previous waveforms
            will be averaged with the current waveform. If a tuple is given,
            the first number is the smoothing amount as specified above, and
            the subsequent numbers are probe indices to reset the averaging from
            (e.g. where the pump stops and starts)
        --parabolic_approx: True to obtain sub-sample accuracy in 
            the time delays by fitting a parabola around the maximum 
            three points of the cross correlation, rather than using 
            the time at the maximum as the delay (see Cespedes, 1995, 
            Ultrasonic Imaging).

    Returns:
        --lags: The time delays, front-padded to match the shape of
            probe_times.
    '''

    sampling_rate = 1/(times[1]-times[0])
    probe_data = data.copy()
    if trace_bandpass:
        probe_data = bandpass_filter(probe_data, trace_bandpass[0], trace_bandpass[1], sampling_rate)
    
    probe_data = detrend(probe_data)

    min_comp_ind = np.where(times >= tmin / 1e6)[0][0]
    max_comp_ind = np.where(times <= tmax / 1e6)[0][-1]

    if smoothing_amount:
        if type(smoothing_amount) == type((1,)):
            break_indices = [0]+list(smoothing_amount[1:])+[probe_data.shape[0]]
            smoothing_amount = smoothing_amount[0]
            next_break_ind = 1
            for i in range(probe_data.shape[0]):
                probe_data[i] = np.average(probe_data[max(break_indices[next_break_ind-1],i+1-smoothing_amount):i+1],axis=0)
                if next_break_ind<len(break_indices) and i == break_indices[next_break_ind]-1:
                    next_break_ind += 1
        else:
            for i in range(probe_data.shape[0]):
                probe_data[i] = np.average(probe_data[max(0,i+1-smoothing_amount):i+1],axis=0)

    lags = np.zeros((probe_data.shape[-2]))
    for i in range(probe_data.shape[-2]):

        array = detrend(probe_data[i][min_comp_ind:max_comp_ind+1], type='constant')
        if normalise:
            array = array / np.amax(np.abs(array))
        reference = detrend(reference_waveform[min_comp_ind:max_comp_ind+1],type='constant')
        
        #plt.plot(times[min_comp_ind:max_comp_ind+1],reference,label='ref')
        #plt.plot(times[min_comp_ind:max_comp_ind+1],array,label='dat')
        #plt.legend()
        #plt.show()

        corr = correlate(reference,array,mode='full')
        corr_times = np.arange(max_comp_ind+1 - min_comp_ind) / sampling_rate
        corr_times = np.concatenate((np.flip(-1. * corr_times[1:],axis=0), corr_times))

        max_lag_ind = np.argmax(corr)
        max_lag = corr_times[max_lag_ind]

        if parabolic_approx:
            y0, y1, y2 = corr[max_lag_ind-1], corr[max_lag_ind], corr[max_lag_ind+1]
            delta_hat = (y0-y2) / (2* (y0-2*y1+y2) ) * (1 / sampling_rate)
            max_lag = max_lag + delta_hat

        lags[i] = max_lag

    lags = lags * -1.

    return lags    


def bandpass_filter(data, min_freq, max_freq, sampling_rate):
    '''
    Apply a bandpass filter to data. Borrowed from obspy.signal.filter
    
    Arguments:
        --data: the data to be filtered
        --min_freq: The lower corner frequency of the bandpass filter
        --max_freq: The upper corner frequency of the bandpass filter
        --sampling_rate: The sampling rate of the data
        
    Returns:
        --data: The filtered data
    '''
    
    fe = 0.5 * sampling_rate
    low = min_freq / fe
    high = max_freq / fe
    
    # Raise for some bad scenarios
    if high - 1.0 > -1e-6:
        msg = ("Selected high corner frequency ({}) of bandpass is at or "
            "above Nyquist ({}). No filter applied.").format(max_freq, fe)
        warnings.warn(msg)
        return data
    
    if low > 1:
        msg = "Selected low corner frequency is above Nyquist."
        raise ValueError(msg)
        
    z, p, k = iirfilter(4, [low, high], btype='band',
                        ftype='butter', output='zpk')
    sos = zpk2sos(z, p, k)
    filtered_dat = sosfiltfilt(sos, data)
    
    return filtered_dat


def harmonic_coeffs_decomp(signal, times, freq0, n=7, return_all=False):
    """
    Find the coefficients of the harmonics of a 
    DAET signal using the Gram-Schmidt method 
    described in Riviere et al., 2013, Pump and
    probe waves in dynamic acousto-elasticity.

    Arguments:
      --signal: The signal to decompose
      --times: The times corresponding to signal
      --n: The highest harmonic to decompose for. 
      --freq0: The fundamental frequency (in Hz)
      --return_all: True to return all the coefficients.

    Returns:
      --spectral_coeffs: The values of the spectral 
        coefficients from k=0 to k=n
      --projected_signal: The reconstructed signal made
        from the linear sum of the sine and cosine harmonics
    If return_all is True, then this function also return:
      --an: The sine harmonics amplitudes
      --bn: The cosine harmonic amplitudes
      --qn: The maximum amplitudes of the orthonormal sine
        components
      --rn: The maximum amplitudes of the orthonormal cosine
        components
      --sn: The orthonormal sine functions
      --cn: The orthonormal cosine functions
    """

    def get_subtract_vals(prev_fs, new_f):
        coeffs = np.sum(np.multiply(prev_fs,np.tile(new_f, (n,1))),axis=1)
        all_fs = np.expand_dims(coeffs,0).T * prev_fs
        return np.sum(all_fs, axis=0)

    omega = 2*np.pi*freq0
    t = times

    sn = np.zeros((n,len(signal)))
    cn = np.zeros((n,len(signal)))

    w = omega
    sn[0] = np.sin(w*t) / np.sqrt(np.sum(np.sin(w*t)**2))
    cn[0] = (np.cos(w*t) - np.sum(sn[0]*np.cos(w*t))*sn[0]) / np.sqrt(np.sum(np.cos(w*t)**2))

    for i in range(2,n+1):
        w = omega*i
        bs = np.sin(w*t)
        bc = np.cos(w*t)

        sn[i-1] = (bs - get_subtract_vals(sn,bs) - get_subtract_vals(cn,bs)) / np.sqrt(np.sum(bs**2))
        cn[i-1] = (bc - get_subtract_vals(sn,bc) - get_subtract_vals(cn,bc)) / np.sqrt(np.sum(bc**2))

    an = np.dot(sn, signal)
    bn = np.dot(cn, signal)
    qn = sn.max(axis=1)
    rn = cn.max(axis=1)

    # Add in the 0th (DC offset) component
    an = np.concatenate(([0],an))
    bn = np.concatenate(([np.mean(signal)],bn))
    qn = np.concatenate(([0],qn))
    rn = np.concatenate(([1],rn))
    sn = np.concatenate(([[0]*len(times)],sn))
    cn = np.concatenate(([[1]*len(times)],cn))

    spectral_coeffs = np.sqrt((an*qn)**2 + (bn*rn)**2)
    projected_signal = np.sum(np.tile(an,[len(times),1]).T*sn,axis=0)+np.sum(np.tile(bn,[len(times),1]).T*cn,axis=0)
    
    # plt.plot(times,signal,'rx')
    # plt.plot(times,projected_signal,'b-')
    # plt.show()

    # plt.plot(spectral_coeffs)
    # plt.show()

    if not return_all:
        return spectral_coeffs, projected_signal
    else:
        return spectral_coeffs, projected_signal, an, bn, qn, rn, sn, cn
