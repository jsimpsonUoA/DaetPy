'''
daet_py is a Python library designed for 
processing and analysis of Dynamic Acousto-Elastic
Testing data caquired primarily with the PLACE
laboraotry acquisition software.

Written by Jonathan Simpson,
Physical Acoustics Lab UoA
March 2020
'''

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz

import warnings
from scipy.signal import iirfilter, zpk2sos, sosfiltfilt, correlate, detrend

from daetpy.plotting import format_fig



class DaetPy():
    def __init__(self, directory, pump_freq, probe_pulse_per_cycle,pump_index=0, probe_index=1, 
            pump_channel='CHANNEL_A',probe_channel='CHANNEL_B',scan_name=None, data_filename=None,
            pump_sensor='PolytecOFV5000X', probe_sensor='PolytecOFV5000', reconstruct_exp=True, 
            pump_min_time=-1e6, pump_max_time=1e6, pump_bandpass=None,fit_pump_sinusoid=False,
            correct_pump_offset=True,use_pump_fitted_curve=False,reference_min_ind=0,reference_max_ind=10,
            pump_preamp_gain=1):
        '''
        DaetPy is a class designed for handling, processing, and analysing
        Dynamic-Acoustoelastic Testing experiments recorded using PLACE.
        Methods are provided to reconstruct DAET data from LUS probe experiments
        where the time between probe pulses is much greataer than a single low 
        frequency pump period.

        directory is the name of the PlaceScan, containing data.npy and config.json.
        The index of the pump data in the traces can be specified with pump_index,
        as can the probe_index. pump_sensor and probe_sensor refer to the names of the
        instruments used to record the respective signals, as stated in the config
        file.
        '''

        self.directory = directory
        self.pump_freq = pump_freq
        self.probe_pulses = probe_pulse_per_cycle
        self.reconstruct_exp = reconstruct_exp
        self.reference_t0 = 1.0
        self.reference_min_ind = reference_min_ind
        self.reference_max_ind = reference_max_ind
        self.pump_start_index = 0
        self.pump_stop_index = -1

        if not scan_name:
            self.scan_name = directory[directory[:-1].rfind('/')+1:-1]
        else:
            self.scan_name = scan_name

        with open(directory+'config.json', 'r') as file:
            self.config = json.load(file)
        
        if not data_filename:
            try:
                self.npy = np.load(directory+'scan_data.npy')
            except FileNotFoundError:
                self.npy = np.load(directory+'data.npy')
        else:
            self.npy = np.load(directory+data_filename)

        self.metadata = self.config['metadata']
        self.updates = self.config['updates']

        trace_field = self._get_name('trace')
        self.trace_data = self.npy[trace_field]
        try:
            self.sampling_rate = self.metadata['sample_rate']
        except KeyError:
            self.sampling_rate = self.metadata['sampling_rate']

        self.pump_index = pump_index
        self.probe_index = probe_index

        self.pump_sensor = pump_sensor
        self.probe_sensor = probe_sensor

        self.pump_data, self.pump_label, pump_dec = self._calibrate_amplitudes(self.trace_data, self.pump_index, self.pump_sensor, pump_channel)
        self.probe_data, self.probe_label, probe_dec = self._calibrate_amplitudes(self.trace_data, self.probe_index, self.probe_sensor, probe_channel)

        num_averages = 1
        new_dat = self.probe_data.copy()
        for i in range(len(new_dat)-num_averages):
            new_dat[i+num_averages-1] = np.sum(self.probe_data[i:i+num_averages],axis=0)/num_averages
        self.probe_data=new_dat
        
        self.pump_times = self._get_times(self.pump_data, decoder=pump_dec)
        self.probe_times = self._get_times(self.probe_data, decoder=probe_dec)

        self.pump_signal, self.pump_signal_times, self.virtual_sampling_rate = \
            self.construct_pump_signal(min_time=pump_min_time,max_time=pump_max_time,
                                bandpass=pump_bandpass,fit_pump_sinusoid=fit_pump_sinusoid,pump_preamp_gain=pump_preamp_gain,
                                correct_pump_offset=correct_pump_offset,use_pump_fitted_curve=use_pump_fitted_curve)


    def construct_pump_signal(self, min_time=-1e6, max_time=1e6, bandpass=None,
                                fit_pump_sinusoid=False,correct_pump_offset=True,
                                use_pump_fitted_curve=False,pump_preamp_gain=1):
        '''
        This method constructs the DAET pump signal by taking
        the average values of the pump_data at each probe shot
        and appending these into one array.
        The minimum and maximum times to average each record over
        may be specified. A bandpass filter
        may be applied to the data beforehand (bandpass is a
        two-element tuple of low and high corner freqs).
        If the pump is a sinusoidal function, fit_pump_sinusoid=True
        will fit a sinusoid with the pump frequency to the data in
        order to better approximate a noisy or offset pump signal.
        The resulting fit can either be used to just correct the 
        DC offset (correct_pump_offset=True) or can be substituted for
        the pump signal entirely (use_pump_fitted_curve=True).

        The function returns the reconstructed probe signal and
        the corresponding (possibly virtual) time array, as well
        as the virtual probe sampling rate
        '''

        dat = self.pump_data.copy()
        dat /= pump_preamp_gain
        if bandpass:
            dat = self._bandpass_filter(dat, bandpass[0], bandpass[1], self.sampling_rate)

        if fit_pump_sinusoid:
            f = lambda x,a,c,d:a*np.sin(2*np.pi*self.pump_freq*x+c)+d
            for i in range(len(dat)):
                try:
                    fit, _ = curve_fit(f, self.pump_times, dat[i], p0=(0,0,0))
                except RuntimeError:
                    print("Curve fit didn't find optimal parameters. Using (0,0,0)")
                    fit = [0,0,0]
                if i%100 == 0:
                    print(self.pump_times[300])
                    plt.plot(self.pump_times, dat[i])
                    plt.plot(self.pump_times,f(self.pump_times,fit[0],fit[1],fit[2]))
                    plt.plot(self.pump_times,np.gradient(f(self.pump_times,fit[0],fit[1],fit[2]),self.pump_times)/(2*np.pi*self.pump_freq)**2)
                    plt.title("Result of pump signal fitting")
                    plt.show()
                if correct_pump_offset:
                    dat[i] = dat[i] - fit[-1]
                elif use_pump_fitted_curve:
                    dat[i] = f(self.pump_times,fit[0],fit[1],fit[2]) 

        min_ind = np.where(self.pump_times >= min_time)[0][0]
        max_ind = np.where(self.pump_times <= max_time)[0][-1]
        dat = dat[:,min_ind:max_ind+1] 

        pump_signal = np.sum(dat,axis=-1) / dat.shape[-1]
        pump_signal = pump_signal - np.average(pump_signal[self.reference_min_ind:self.reference_max_ind+1])

        virtual_probe_period = 1 / self.pump_freq / self.probe_pulses
        end_time = virtual_probe_period * pump_signal.shape[-1] - (virtual_probe_period / 2)
        pump_signal_times = np.arange(0., end_time, virtual_probe_period)

        return pump_signal, pump_signal_times, 1./ virtual_probe_period

    def generate_reference_waveform(self, bandpass=None, reference_use_times=False,
                                     normalise=False):
        '''
        A function which returns a probe reference waveform

        The reference signal is constructed from an average of the 
        probe signals from index (probe pulse number) reference_min
        to reference_max. Instead of using the indices, the time 
        interval (of the virtual time array) over which
        to average the probe waveforms can be specified if 
        reference_use_times is True. Function can be normalised
        with normalise = True.
        '''

        probe_data = self.probe_data.copy()

        if bandpass:
            probe_data = self._bandpass_filter(probe_data, bandpass[0], bandpass[1], self.sampling_rate)

        if reference_use_times:
            min_ind = np.where(self.pump_signal_times >= self.reference_min_ind / 1e6)[0][0]
            max_ind = np.where(self.pump_signal_times <= self.reference_max_ind / 1e6)[0][-1]
        else:
            min_ind = self.reference_min_ind
            max_ind = self.reference_max_ind

        reference = np.average(probe_data[min_ind:max_ind+1],axis=0)

        if normalise:
            reference = reference / np.amax(np.abs(reference))

        return reference


    def compute_time_delays(self, reference_waveform, trace_bandpass=None,  normalise=False, min_probe_index=0,
                            min_probe_time=False, tmin=0., tmax=50., smoothing_amount=0, 
                            parabolic_approx=True,lags_bandpass=None):
        '''
        Function which computes the time delays from the
        DAET probe signals. Positive lag menas the sample
        probe waveform lags behind the reference.

        **All times are specified in microseconds.**

        Arguments:
            --reference_waveform: A reference waveform to compare probe
                waveforms to. See generate_reference_waveform
            --trace_bandpass: A bandpass filter applied to the probe waveforms before 
              cross correlation
            --normalise: True to normalise the probe waveforms before comparison
            --min_probe_index: The index of the probe waveforms to start
                comparing with the reference
            --min_probe_time: min_probe_index can be specified in time of
                the pump_signal_times if this is True
            --tmin: The minimum time to use for the window in the 
                cross correlation
            --tmax: The maximum time for the window in the cross
                correlation
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
            --lags_bandpass: A bandpass filter applied to the lags after all
              lags have been computed. The pump_signals_times is used to determine
              the sampling rate here.

        Returns:
            --lags: The time delays, front-padded to match the shape of
                probe_times.
        '''

        probe_data = self.probe_data.copy()
        if trace_bandpass:
            probe_data = self._bandpass_filter(probe_data, trace_bandpass[0], trace_bandpass[1], self.sampling_rate)
        if min_probe_time != False:
            min_probe_index = np.where(self.pump_signal_times >= tmin / 1e6)[0][0]
        
        probe_data = detrend(probe_data)

        min_comp_ind = np.where(self.probe_times >= tmin / 1e6)[0][0]
        max_comp_ind = np.where(self.probe_times <= tmax / 1e6)[0][-1]

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

        lags = np.zeros((probe_data[min_probe_index:].shape[-2]))
        for i in range(probe_data[min_probe_index:].shape[-2]):

            array = detrend(probe_data[min_probe_index:][i][min_comp_ind:max_comp_ind+1], type='constant')
            if normalise:
                array = array / np.amax(np.abs(array))
            reference = detrend(reference_waveform[min_comp_ind:max_comp_ind+1],type='constant')

            #reference, array = reference/np.abs(reference), array/np.abs(array) # Bit filtering

            corr = correlate(reference,array,mode='full')
            corr_times = np.arange(max_comp_ind+1 - min_comp_ind) / self.sampling_rate
            corr_times = np.concatenate((np.flip(-1. * corr_times[1:],axis=0), corr_times))

            max_lag_ind = np.argmax(corr)
            max_lag = corr_times[max_lag_ind]

            #if i%50 == 0:
            #    plt.plot(corr_times*1e6,corr)
            #    plt.axvline(max_lag*1e6,color='r')
            #    plt.show()

            if parabolic_approx:
                y0, y1, y2 = corr[max_lag_ind-1], corr[max_lag_ind], corr[max_lag_ind+1]
                delta_hat = (y0-y2) / (2* (y0-2*y1+y2) ) * (1 / self.sampling_rate)
                max_lag = max_lag + delta_hat

            lags[i] = max_lag

        lags = np.concatenate((np.array([0.]*min_probe_index),lags)) * -1.

        if lags_bandpass:
            lags = self._bandpass_filter(lags, lags_bandpass[0], lags_bandpass[1], 1/np.diff(self.pump_signal_times)[0])
            lags = lags - np.average(lags[self.reference_min_ind:self.reference_max_ind+1])


        return lags    

    def pick_pump_start_stop(self):
        """
        Function to plot the pump signal so that
        the times when the pump starts and stops
        can be picked
        """

        global pick_line1, pick_line2, active_line

        def key_press(self, event, fig):

            global pick_line1, pick_line2, active_line
            pick = event.xdata
            if pick != None:
                if active_line == 1:
                    self.pump_start_index = int(round(pick))
                    active_line = 2
                    if pick_line1 != None:
                        pick_line1.remove() 
                    pick_line1 = plt.axvline(x=pick,color='r',ls='--',lw=1.)
                elif active_line == 2:
                    self.pump_stop_index = int(round(pick))
                    active_line = 1
                    if pick_line2 != None:
                        pick_line2.remove() 
                    pick_line2 = plt.axvline(x=pick,color='b',ls='--',lw=1.)
                fig.canvas.draw()

        fig = plt.figure(figsize=(8,5))
        plt.plot(self.pump_signal)
        plt.axhline(0.,ls="--",color="gray")
        plt.title('Hover mouse over start and end of pump to pick and press any key. Close to save picks.')

        pick_line1, pick_line2, active_line = None, None, 1
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
        fig.canvas.mpl_connect('key_press_event', lambda event: key_press(self, event, fig))         
    
        fig, ax = format_fig(fig, plt.gca(),ylab='Amplitude',xlab='Index')

        plt.show()

        return self.pump_start_index, self.pump_stop_index

    def pump_signal_bandpass(self, bandpass, sampling_rate=1, modify_dp_pump=True, 
                            pump_signal=None, start_ind=0,stop_ind=-1):
        """
        Apply a bandpass filter to a pump signal after
        it has been constructed.

        Arguments:
          --bandpass: The bandpass filter as a tuple (low_freq,high_freq)
          --samping_rate: The sampling rate of the data. If modify_dp_pump
            is True, then the pump_signal_times will determine this.
          --modify_dp_pump: If True, apply the bandpass filter to
            the pump_signal that is already assigned to this DaetPy object.
            This reassigns the filtered pump signal to self.pump_signal
          --pump_signal: If modify_dp_pump is False, this is the pump signal
            to filter
          --start_ind: The starting index of the pump
          --stop_ind: The stop index of the pump.
        """

        if modify_dp_pump:
            dat = self.pump_signal.copy()
            samp = 1/np.diff(self.pump_signal_times)[0]
        else:
            dat = pump_signal
            samp = sampling_rate

        if bandpass:
            pump_on = dat[start_ind:stop_ind]

            #before_filter = self._bandpass_filter(dat[:start_ind],bandpass[0],bandpass[1],samp)
            #pump_filter = self._bandpass_filter(pump_on,bandpass[0],bandpass[1],samp)
            #after_filter = self._bandpass_filter(dat[stop_ind:],bandpass[0],bandpass[1],samp)
            #filtered_dat = np.concatenate((before_filter,pump_filter,after_filter))
            
            filtered_dat = self._bandpass_filter(dat,bandpass[0],bandpass[1],samp)
            filtered_dat = filtered_dat - np.average(filtered_dat[self.reference_min_ind:self.reference_max_ind+1])

            if modify_dp_pump:
                self.pump_signal = filtered_dat

            return filtered_dat
        return dat


    def compute_all_pumps(self,start_pump_ind, stop_pump_ind, sample_length,
                    min_cycle_length=5, original_pump_type='vel'):
        """
        Compute the different types of pump signals.
        This function computes and returns the pump
        signal representing particle velocity, particle
        displacement, strain, and strain rate.
        
        Arguments:
          --start_pump_ind: The index where the pump turns on
          --stop_pump_ind: The index where the pump turns off 
          --min_cycle_length: The minimum numbers of indices in
              one pump cycle
          --sample_length: The length of the sample (in mm)
          --original_pump_type: The type specifier for the raw
              pump. 'vel' if it represents particle velocity,
              and 'displ' if it represents particle displacement.

        Returns:
          --pump_displ: The particle displacement pump
          --pump_vel: The particle velocity pump
          --pump_strain: The strain pump (note this is phase-corrected)
          --pump_strain_rate: The strain rate pump
        """

        #av_pump, av_probes, av_cycle_length = self.average_probe_over_pump_cycle(
        #        start_pump_ind, stop_pump_ind,min_cycle_length=min_cycle_length,integrate_pump_wave=False)

        sliced_original = self.pump_signal.copy()[start_pump_ind:stop_pump_ind]
        so_len = len(sliced_original)
        cos_func = lambda x, a, w, p: a*np.cos(w*x+p)
        fit, cov = curve_fit(cos_func, range(so_len),sliced_original,p0=(np.amax(sliced_original),2*np.pi/50,0))
        plt.plot(sliced_original)
        plt.plot(range(so_len),cos_func(range(so_len),*fit))
        plt.plot(np.diff(sliced_original))
        plt.show()

        plt.plot(self.pump_data.copy()[120],'b-')
        plt.plot(np.diff(self.pump_data.copy())[120],'r-')
        plt.show()
        
        return

        if original_pump_type == 'vel':
            pump_vel = self.pump_signal.copy()
        elif original_pump_type == 'displ':
            pump_displ = self.pump_signal.copy()

    def average_probe_over_pump_cycle(self,min_index=0,max_index=-1, min_cycle_length=10,
                                        integrate_pump_wave=True):
        """
        This function is designed to improve the S/N of
        the probe waveforms before velocity analysis. It
        does this by splitting up the pump into individual
        cycles, and then averaging both the probe waveforms
        and the pump signal across those cycles. This
        assumes that the velocity variation is the same 
        across all pump cycles, and that all pump cycles
        are the same too.

        Arguments:
          --min_index: The minimum index in the pump signal to
              average from
          --max_index: The maximum index in the pump signal to
              average to
          --min_cycle_length: The minimum number of indices that
              one period of the pump should be.
          --integrate_pump_wave: If True, then the final averaged pump
              wave will be "integrated". This assumes that the 
              pump is recording velocity. Integration is achieved by
              a cumulative trapezoidal sum, and the resulting values
              are in mm (assuming velocity was mm/s)

        Return:
          --av_pump: Average pump cycle
          --av_probe: The averaged probe signals
          --av_cycle_length: The average number of indices
            in one pump cycle.

        """

        pump = self.pump_signal.copy()[min_index:max_index]

        cycle_start_inds = []
        current_sign = round(pump[0]/abs(pump[0]))
        for i in range(len(pump)):
            new_sign = round(pump[i]/abs(pump[i]))
            if new_sign == 1 and current_sign == -1:
                if len(cycle_start_inds) == 0 or (i-cycle_start_inds[-1])>min_cycle_length:
                    cycle_start_inds.append(i)
            current_sign = new_sign
            
        av_cycle_length = int(np.average(np.diff(cycle_start_inds)))

        av_pump = np.zeros(av_cycle_length)
        av_probes = np.zeros((av_cycle_length,self.probe_data.shape[-1]))
        for start_ind in cycle_start_inds[:-1]:
            if start_ind+av_cycle_length<len(pump):
                single_pump_cycle = pump[start_ind:start_ind+av_cycle_length]
                probe_set = self.probe_data[start_ind+min_index:start_ind+min_index+av_cycle_length,:]
                av_pump = av_pump + single_pump_cycle
                av_probes = av_probes + probe_set

                plt.plot(single_pump_cycle)
        plt.show()

        av_pump = av_pump / (len(cycle_start_inds)-1)
        av_probes = av_probes / (len(cycle_start_inds)-1)

        if integrate_pump_wave:
            av_pump_times = self.pump_signal_times.copy()[:len(av_pump)]
            int_signal = detrend(cumtrapz(av_pump,x=av_pump_times),type='constant')
            av_pump = np.concatenate(([(int_signal[0]+int_signal[-1])/2],int_signal))
            # Deprecated (and wrong!)
            #end_part = av_pump[int(0.75*len(av_pump)):]
            #av_pump = np.concatenate((end_part,av_pump[:-len(end_part)]))

        plt.plot(av_pump)
        plt.show()    

        return av_pump, av_probes, av_cycle_length

    def average_lags_over_pump_cycle(self,lags,min_index=0,max_index=-1, min_cycle_length=10,
                                        integrate_pump_wave=True,show_plots=True):
        """
        This function is designed to improve the S/N of
        the lags after velocity analysis. It
        does this by splitting up the pump into individual
        cycles, and then averaging both the lags
        and the pump signal across those cycles. This
        assumes that the velocity variation is the same 
        across all pump cycles, and that all pump cycles
        are the same too.

        Arguments:
          --lags: The unaveraged lags arrays. Same length as self.pump_signal
          --min_index: The minimum index in the pump signal to
              average from
          --max_index: The maximum index in the pump signal to
              average to
          --min_cycle_length: The minimum number of indices that
              one period of the pump should be.
          --integrate_pump_wave: If True, then the final averaged pump
              wave will be "integrated". This assumes that the 
              pump is recording velocity. Integration is achieved by
              a cumulative trapezoidal sum, and the resulting values
              are in mm (assuming velocity was mm/s)
          --show_plots: True to show monitoring plots.

        Return:
          --av_pump: Average pump cycle
          --av_lags: The averaged lags
          --av_cycle_length: The average number of indices
            in one pump cycle.

        """

        pump = self.pump_signal.copy()[min_index:max_index]

        cycle_start_inds = []
        current_sign = round(pump[0]/abs(pump[0]))
        for i in range(len(pump)):
            new_sign = round(pump[i]/abs(pump[i]))
            if new_sign == 1 and current_sign == -1:
                if len(cycle_start_inds) == 0 or (i-cycle_start_inds[-1])>min_cycle_length:
                    cycle_start_inds.append(i)
            current_sign = new_sign
            
        av_cycle_length = int(np.average(np.diff(cycle_start_inds)))

        av_pump = np.zeros(av_cycle_length)
        av_lags = np.zeros(av_cycle_length)
        for start_ind in cycle_start_inds[:-1]:
            if start_ind+av_cycle_length<len(pump):
                single_pump_cycle = pump[start_ind:start_ind+av_cycle_length]
                cycle_lags = lags[start_ind+min_index:start_ind+min_index+av_cycle_length]
                av_pump = av_pump + single_pump_cycle
                av_lags = av_lags + cycle_lags

                if show_plots:
                    plt.plot(single_pump_cycle)
        if show_plots:
            plt.show()

        av_pump = av_pump / (len(cycle_start_inds)-1)
        av_lags = av_lags / (len(cycle_start_inds)-1)

        if integrate_pump_wave:
            av_pump_times = self.pump_signal_times.copy()[:len(av_pump)]
            int_signal = detrend(cumtrapz(av_pump,x=av_pump_times),type='constant')
            av_pump = np.concatenate(([(int_signal[0]+int_signal[-1])/2],int_signal))
            # Deprecated (and wrong!)
            #end_part = av_pump[int(0.75*len(av_pump)):]
            #av_pump = np.concatenate((end_part,av_pump[:-len(end_part)]))

        if show_plots:
            plt.plot(av_pump)
            plt.show()    

        return av_pump, av_lags, av_cycle_length


    def pick_reference_t0(self, reference_waveform, tmin=0., tmax=30., bandpass=None):
        '''
        Function which plots the reference waveform and
        allows the user to pick the t0 time. Limits are in microseconds.

        Generate a reference waveform with generate_reference_waveform()
        '''

        global pick_line

        def key_press(self, event, fig):

            global pick_line
            pick = event.xdata
            if pick != None:
                self.reference_t0 = pick / 1e6
                if pick_line != None:
                    pick_line.remove() 
                pick_line = plt.axvline(x=pick,color='r',ls='--',lw=1.)
                fig.canvas.draw()

        fig = plt.figure(figsize=(8,5))
        plt.plot(self.probe_times*1e6, reference_waveform)
        plt.xlim((tmin,tmax))
        plt.axhline(0.,ls="--",color="gray")
        plt.title('Hover mouse over t0 pick and press any key. Close to save pick.')

        pick_line = None
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
        fig.canvas.mpl_connect('key_press_event', lambda event: key_press(self, event, fig))         
    
        fig, ax = format_fig(fig, plt.gca(),ylab='Amplitude',xlab='Time ($\mu$s)')

        plt.show()

        return self.reference_t0


    #####################################################################
    ##                         Private Methods                         ##
    #####################################################################

    def _get_name(self, key):
        '''
        Extract the name of the field containing the key
        '''
        dtypes = self.npy.dtype.names

        name = None
        for name in dtypes:
            if name.find(key) != -1:
                return name

        print("DAETPy: Could not find a valid data field for '{}'".format(key))


    def _calibrate_amplitudes(self, data, index, module_name, channel_key):
        '''
        Function to calibrate the true amplitudes of the 
        data, given the index of the data and the name
        of the sensor which recorded the data (module).

        This currently works for Polytec modules.
        '''

        try:
            module = self.config['plugins'][module_name]
            module_config = module['config']

            try:
                decoder = [item for item in module_config.items() if item[1] == True and item[0] != "plot" and item[0] != "autofocus_everytime"][0][0]
                _range_s = [item for item in module_config.items() if decoder+'_range' in item[0]][0][1]
                first_ndig = next((s for s in _range_s if s not in ['0','1','2','3','4','5','6','7','8','9','.']), _range_s[-1])
                _range = _range_s[:_range_s.find(first_ndig)]
                _amp_label = _range_s[_range_s.find(first_ndig):]
                _amp_label = _amp_label[:_amp_label.rfind(r'/')]
            except IndexError:
                print('DaetPy: Warning: Could not calibrate sensor amplitudes.')
                return data[:,index][0], '', None

            sensor_range = float(_range)

        except KeyError:
            print('DaetPy: Could not detect recording module.')
            sensor_range, _amp_label, decoder = 1., 'V', None
        except:
            raise

        

        # For ATS cards
        trace_field = self._get_name('trace')
        scope_bits = 1
        if trace_field.find('9440') != -1:
            scope_bits = 14
        elif trace_field.find('660') != -1 or trace_field.find('9462') != -1:
            scope_bits = 16

        scope_module = self.config['plugins'][trace_field[:trace_field.find('-')]]
        inputs = scope_module['config']['analog_inputs']
        channel_config = [dict_ for dict_ in inputs if channel_key in dict_.values()][0]
        input_range_s = channel_config['input_range'].split('_')
        input_range = float(input_range_s[3])
        if input_range_s[-1] == 'MV':
            input_range /= 1000

        cal_data = data[:,index][0].copy().astype(float)
        calibraetd_data = (cal_data - 2**(scope_bits-1)) / (2**(scope_bits-1)) * input_range * sensor_range

        return calibraetd_data, _amp_label, decoder

    def _get_times(self, data, decoder=None):
        '''
        Function to get the times and remove any time delay
        for the pump and probe signals.
        '''

        num_samples = data.shape[-1]
        
        times = np.arange(num_samples) / self.sampling_rate

        if decoder:
            key = decoder + '_time_delay'
            try:
                time_delay = [item for item in self.metadata.items() if item[0] == key][0][1]
            except IndexError:
                time_delay = 0.
            times = times - time_delay / 1e6

        return times

    def _bandpass_filter(self, data, min_freq, max_freq, sampling_rate):
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
