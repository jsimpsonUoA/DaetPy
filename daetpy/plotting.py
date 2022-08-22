'''
The plotting routines for the DaetPy Package

Jonathan Simpson, April 2020
'''

import numpy as np
import scipy.signal as ss

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.gridspec as gridspec

def format_fig(fig, ax, title=None, xlab=None, ylab=None, xlim=None, ylim=None, 
               save_dir=None, show=True, lab_font=20.0, title_font=20.0, 
               tick_fontsize=14.0, grid=False, legend=False, legend_loc='lower left',
               legend_fontsize=12.0, polar=False, polar_min=0, colorbar=True, 
               color_key=None, color_label='', color_data=None, color_data_labels=None,
               colorbar_trim=[0,1], colorbar_round=1, colorbar_horiz=False, lines=None, **kwargs):
    '''
    General function to format and save a figure. Inherited from PlaceScan

    Arguemnts:
        --fig: The figure instance
        --ax: The current axis
        --title: The title of the plot
        --xlab: The x label for the axis
        --ylab: The y label for the axis
        --xlim: The x limit tuple
        --ylim: The y limit tuple
        --save_dir: The directory (or list of dirs) to save the figure to.
        --show: Set to True to show the plot
        --lab_font: The axis label fontsize
        --title_font: The title font size
        --tick_fontsize: The axis tick fontsize
        --grid: Show a grid on the plot
        --polar: True if this is a polar plot
        --polar_min: The minimum theta axis label for a polar plot. Either 
                   0 or -180.
        --colorbar: True to plot a colorbar, if color_key is specified.
        --color_[key, label, data, data_labels, bar_trim, bar_round, bar_horiz]: 
            The arguments for creating a sequential color bar for the plotted data.
            See create_colorbar function for more details.
        --lines: The lines plotted on the axis
        --kwargs: Keyword arguments that are needed for other functions

    Returns:
        --fig: The figure isntance
        --ax: The axis instance
    '''

    if title:
        fig.suptitle(title, fontsize=title_font)
    
    try:
        ax.yaxis.major.formatter.set_powerlimits((-5,5))
        ax.xaxis.major.formatter.set_powerlimits((-5,5))
    except:
        pass
    
    if xlab:
        ax.set_xlabel(xlab, fontsize=lab_font)
    if ylab:
        ax.set_ylabel(ylab, fontsize=lab_font)
        
    if xlim or polar_min:
        if polar:
            
            labels = np.linspace(0.,360., 8, endpoint=False)
            if polar_min == -180:
                labels = np.where(labels>xlim[1]+1.,labels-360.,labels)
            elif polar_min:
                labels = np.where(labels>polar_min+360.,labels-360.,labels)
            if not plt.rcParams['text.usetex']:
                ax.set_xticklabels(['{}°'.format(s) for s in labels.astype(int)])
            else:
                ax.set_xticklabels(['${}^\\circ$'.format(s) for s in labels.astype(int)])
        else:
            ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim) 

    if tick_fontsize:
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax.tick_params(axis='both', which='minor', labelsize=tick_fontsize)

    if grid:
        if polar:
            ax.set_rlabel_position(340) 
            ax.yaxis.get_major_locator().base.set_params(nbins=6)
            ax.grid(which='major', linestyle=':')
        else:
            ax.grid()

    if legend:
        ax.legend(loc=legend_loc,fontsize=legend_fontsize)

    if polar:
        #ax.set_rlabel_position(275.0)  Commented on 10/12/19
        ax.set_rlabel_position(23.0)

    if color_key:
        fig, ax = create_colorbar(fig, ax, color_key, color_label,
             color_data, lab_font, tick_fontsize, color_data_labels,
             colorbar_trim, colorbar, colorbar_round, colorbar_horiz,
             lines)
        plt.sca(ax)

    if save_dir:
        print('saving')
        if isinstance(save_dir, list):
            for _dir in save_dir:
                fig.savefig(_dir, bbox_inches='tight')
        else:
            fig.savefig(save_dir, bbox_inches='tight')
    
    if show:
        plt.show()
        plt.close()

    return fig, ax

def create_colorbar(fig, ax, color_key, color_label, color_data,
                    lab_font, tick_fontsize, color_data_labels=None,
                    colorbar_trim=[.2,9], colorbar=True, colorbar_round=1,
                    colorbar_horiz=False, lines=None):
    '''
    Function which is called within format_fig
    to color-code plotted lines by a 3rd variable.
    This function also handles the formatting of
    the colorbar

    Arguments:
        --fig: The matplotlib figure
        --ax: The matplotlib axis containing the lines
        --color_key: The color of the shaded colorbar
            Accepted values are 'grey', 'purple',
            'green', 'blue', orange', or 'red', or
            any matplotlib colorbar key.
        --color_label: A label for the colorbar axis
        --color_data: The 3rd varialbe to color code the
            plotted lines by. This can be a tuple of two
            numbers representing the lower and upper extremes,
            assuming the lines between are evenly spaced in
            this range. Alternatively, this can be a list
            of the same length as the number of plotted
            lines, containing numbers which will be used
            to assign colors to the lines. A list of numbers
            represented as strings is also acceptable.
        --lab_font: The axis label fontsize
        --tick_fontsize: The axis tick fontsize
        --color_data_labels: A  list containing tick labels for the
            colorbar, if something different from the automatic 
            ticklabels based on color_data is required. These are evenly
            spaced along the colorbar
        --colorbar_trim: A color map usually spans the range 0.0-->1.0
            If a smaller range is desired for better visibility etc.,
            then this can be given here as a two element list specifying
            the lower and upper bounds of the colors, e.g. [0.1,0.9]
        --colorbar: True to plot the colorbar, False to hide it
        --colorbar_round: The number of decimal places to round the 
                          colorbar axis labels to
        --colorbar_horiz: True to have a horizontal colorbar
        --lines: The specific lines to apply the color scale to.

    Returns:
        --fig: The figure with the colorbar plotted
    '''

    if not lines:
        lines = ax.lines

    #Find the upper and lower values for the colormap
    if isinstance(color_data, tuple):
        color_data = np.linspace(color_data[0], color_data[1], len(lines))
    else:
        if isinstance(color_data[0], str):
            color_data = np.array([float(num) for num in color_data])
    max_n, min_n = max(color_data), min(color_data)
    data_range = max_n - min_n
    interval = colorbar_trim[1] - colorbar_trim[0]
    data_midpoint = data_range / 2 + min_n
    interval_midpoint = interval / 2 + colorbar_trim[0]
    vmax = data_range / interval
    shift = (interval_midpoint * vmax) - data_midpoint
    vmax = vmax - shift
    vmin = 0.0 - shift
    #Create the colormap
    if color_key in ['grey', 'purple','green', 'blue', 'orange','red']:
        cmap = get_cmap(color_key.capitalize()+'s')
    else:
        cmap = get_cmap(color_key)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    #Color-code the plot
    for i in range(len(color_data)):
        if i < len(lines):
            lines[i].set_color(cmap(int(255*norm(color_data[i]))))

    #Only add the colorbar if there is not already one on the plot.
    #Note: This conditional may not behave as expected if Axes objects
    #are manually added elsewhere.
    add_axes = np.sum([isinstance(i,mpl.axes.Axes) for i in fig.get_children()]) < 2
    if colorbar and add_axes:
        #Rearrange the axes
        ax_pos = list(ax.get_position().bounds)
        ax_dim_to_use = 2 + int(colorbar_horiz)
        ax_size =  ax_pos[ax_dim_to_use]         #The width or height
        ax_pos[ax_dim_to_use] = ax_size * 0.91   #Shrink width or height
        ax.set_position(ax_pos)                     

        #Setup and insert the colorbar axis
        cbar_width = ax_size * (0.03 * (1+colorbar_horiz))
        if colorbar_horiz:
            cbar_pos, orien = [ax_pos[0], ax_pos[1]+ax_size-cbar_width, ax_pos[2], cbar_width], 'horizontal' 
        else:
            cbar_pos, orien = [ax_pos[0]+ax_size-cbar_width, ax_pos[1], cbar_width, ax_pos[3]], 'vertical' 
        cbar_ax = plt.axes(cbar_pos)
    
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array([])
        cb = plt.colorbar(mappable=mappable,cax=cbar_ax,boundaries=np.arange(min_n,max_n,data_range/1000), orientation=orien)
        
        #Format the colorbar
        if not isinstance(color_data_labels, type(None)):
            ticks = mpl.ticker.FixedLocator(np.linspace(min_n,max_n-data_range*0.0015, len(color_data_labels)))
            cb.set_ticks(ticks) 
            cb.set_ticklabels(color_data_labels)
        else:
            ticks = mpl.ticker.FixedLocator(np.linspace(min_n,max_n-data_range*0.0015, 5))
            cb.set_ticks(ticks) 
            if colorbar_round:
                cb.set_ticklabels(np.round(np.linspace(min_n,max_n, 5),colorbar_round))
            else:
                cb.set_ticklabels(np.linspace(int(min_n),int(max_n), 5,dtype=int))   #Integers
        cb.ax.tick_params(labelsize=tick_fontsize)
        cb.set_label(color_label, fontsize=lab_font, rotation=int(not colorbar_horiz)*270.,verticalalignment='bottom',labelpad=colorbar_horiz*10) 
        if colorbar_horiz:
            cb.ax.xaxis.tick_top(), cb.ax.xaxis.set_label_position('top')

    return fig, ax


def pump_dv_plot(dp_object, velocity_changes, pump_bandpass=None, normed_pump=True, 
                milliseconds=False, fig=None, figsize=(10,7), show=True, 
                ylab='Velocity Change (\%)', plot_pump_strain=False, 
                freq=0., sample_length=1., vel_ylim=None, plot_times_on_x=True, **kwargs):
    '''
    Make a typical DAET plot showing the pump oscillation
    in the top panel and the change in velocity in the
    second panel.

    Arguments:
        --dp_object: The DaetPy object
        --velocity_changes: The velocity variations
        --pump_bandpass: A bandpass filter to apply to the pump
            signal, given as (min_freq, max_freq)
        --normed_pump: True to normalise the pump amplitudes
        --milliseconds: True to plot the time scale in ms (instead of s)
        --fig: A Figure instance to plot onto
        --figsize: The size of the Figure
        --show: True to show the plot
        --ylab: The y-axis label for the velocity change axis
        --plot_pump_strain: True to plot strain on the y axis
        --freq: The frequency of the pump signal.
        --sample_length: The length of the sample
        --vel_ylim: Y limits for the velocity axis
        --plot_times_on_x: True to plot the pump times on the x axis, 
                False to plot based on probe index
        --kwargs: The keyword arguments for plotting (see format_fig)

    Returns:
        -fig: The Figure instance
        --axes: A list of the Axis instances
    '''

    if not fig:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 1,hspace=0.4)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1:, :],sharex=ax1)
    else:
        axes = fig.axes
        ax1, ax2 = axes[0], axes[1]

    times = dp_object.pump_signal_times.copy()
    pump = dp_object.pump_signal.copy()

    if normed_pump:
        pump = pump/np.amax(np.abs(pump))
    if pump_bandpass:
        pump = dp_object._bandpass_filter(pump, pump_bandpass[0], pump_bandpass[1], dp_object.virtual_sampling_rate)

    if plot_pump_strain:
        pump = pump / (2*np.pi*freq) / sample_length * 1e6

    if plot_times_on_x:
        ax1.plot(times, pump, label='Pump', color='blue')
        ax2.plot(times, velocity_changes, label='Cross Correlation')
    else:
        ax1.plot(pump, label='Pump', color='blue')
        ax2.plot(velocity_changes, label='Cross Correlation')        

    x_lab = 'Time (s)'
    if milliseconds:
        x_lab = 'Time (ms)'

    ylim=None
    pump_ylab = 'Amplitude ({})'.format(dp_object.pump_label)
    if normed_pump:
        ylim=(-1.1,1.1)
        pump_ylab= 'Amplitude (a.u.)'
    elif plot_pump_strain:
        pump_ylab= 'Amplitude (microstrain)'

    fig,ax1 = format_fig(fig, ax1, xlab='', ylab=pump_ylab, show=False, ylim=ylim, **kwargs)
    fig,ax2 = format_fig(fig, ax2, xlab=x_lab, ylab=ylab, show=show, ylim=vel_ylim, **kwargs)

    return fig, [ax1, ax2]

def bowtie_plot(pump_wave, velocity_changes, plot_strain=True, freq=1.,
                sample_length=1, figsize=(8,8), integrate_pump_wave=True,
                **kwargs):
    """
    Function to make a bowite-type plot from a single
    cycle of a pump wave and the corresponding change
    in velocity
    """

    fig=plt.figure(figsize=figsize)

    xlab = "Displacement (nm)"
    if plot_strain:
        pump_wave = pump_wave / (2*np.pi*freq) / sample_length * 1e6
        xlab = "Microstrain"

    plt.plot(pump_wave, velocity_changes, 'ro', markersize=5)
    plt.grid(ls=":")
    plt.axhline(0,ls='--',color='k')
    plt.axvline(0,ls='--',color='k')
    
    format_fig(fig, plt.gca(), xlab=xlab, ylab="Velocity Change (\%)",**kwargs)


def animated_probe_waveforms(dp_object, reference_waveform=None, update_interval=0.2, 
                pump_bandpass=None, pump_milliseconds=True, xlim=None, repeat_delay=0., 
                figsize=(8,6), show=True, color='r', save_dir=None, **kwargs):
    '''
    Function to create an animation of the probe waveform
    plots with the reference waveform comparison. Time is 
    plotted in microseconds. The location on the pump waveform
    where the probe was recorded is shown for reference.
    
    Arguments:
        --dp_object: The DaetPy object
        --reference_waveform: A reference waveform to keep on the plot.
        --xlim: The (min_time,max_time) limits for the x axis
        --update_interval: The interval between animation updates (in seconds)
        --pump_bandpass: A (min_freq, max_freq) bandpass filter to apply to the pump
        --pump_milliseconds: True to plot the pump time scale in milliseconds.
        --repeat_delay: The time delay (in seconds) between repeats of the animation
        --show: True to display the plot
        --figsize: The size of hte figure
        --color: Color of the probe waveforms
        --save_dir: A directory to save the gif to.
        --kwargs: Keyword arguments for signal_processing and format_fig
        
    Returns:
        None    
    '''
    
    global current_probe, pump_marker

    def update(i, fig, ax1, ax2, probe_times, probe_data, pump_times, color):
        
        global current_probe, pump_marker
        if current_probe != None:
            current_probe.remove(), pump_marker.remove()
        pump_marker = ax1.axvline(pump_times[i],color='k',alpha=0.7,lw=1)
        current_probe = ax2.plot(probe_times, probe_data[i], color=color)[0]
        fig.canvas.draw()
        
    
    import matplotlib.animation as animation

    fig = plt.figure(figsize=figsize)    
    gs = fig.add_gridspec(3, 1,hspace=0.4)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1:, :])

    pump = dp_object.pump_signal.copy()
    pump = pump/np.amax(np.abs(pump))
    if pump_bandpass:
        pump = dp_object._bandpass_filter(pump, pump_bandpass[0], pump_bandpass[1], dp_object.virtual_sampling_rate)
    if pump_milliseconds:
        pump_times = dp_object.pump_signal_times.copy() * 1e3
        pump_xlab = 'Time (ms)'
    else:
        pump_times = dp_object.pump_signal_times.copy() * 1e6
        pump_xlab = 'Time ($\mu$s)'
    ax1.plot(pump_times, pump, label='LF Pump', color='blue')
    pump_marker = ax1.axvline(0.,color='k',alpha=0.7,lw=0.7)

    
    data = dp_object.probe_data.copy()
    orig_times = dp_object.probe_times.copy()*1e6
    data, times = signal_processing(dp_object, data, orig_times, time_limits=xlim, sampling_rate=dp_object.sampling_rate,**kwargs)
    reference_waveform, ref_times = signal_processing(dp_object, reference_waveform, orig_times, time_limits=xlim, \
                                                        sampling_rate=dp_object.sampling_rate, **kwargs)

    num_updates = dp_object.probe_data.shape[0]
    ax2.plot(ref_times, reference_waveform, color='gray', alpha=0.5, label='Probe Reference')
    current_probe=None

    anim = animation.FuncAnimation(fig, update, interval=update_interval*1000, save_count=num_updates,
                         repeat_delay=repeat_delay*1000, repeat=True, frames=range(num_updates),
                         fargs=(fig,ax1,ax2,times,data,pump_times,color))

    fig,ax1 = format_fig(fig, ax1, xlab=pump_xlab, ylab='Amplitude (a.u.)', show=False, xlim=(pump_times[0],pump_times[-1]), ylim=(-1.1,1.1),legend=True,legend_loc='upper right')
    fig, ax2 = format_fig(fig, ax2, show=False, xlab='Time ($\mu$s)', xlim=xlim, **kwargs)

    if save_dir:
        anim.save(save_dir,writer='ffmpeg',fps=1/update_interval,bitrate=200,dpi=80)

    if show:
        plt.show()

    return fig

def signal_processing(dp_object, data, times, sampling_rate=None, bandpass=None,
                    time_limits=None, normalise=False, detrend=False, demean=False, 
                    smoothing_amount=0, **kwargs):
    '''
    Routines for signal processing to prepare the data for 
    plotting

    Arguments:
        --dp_object: The DaetPy object where the data is from
        --data: The data to process
        --times: The time array corresponding to data.
        --sampling_rate: The sampling rate of the data. Needed for bandpass
        --bandpass: A bandpass filter 9min,max) for the probe waveforms
        --normalise: True to normalise the waveforms
        --detrend: True to detrend the probe waveform data
        --demean: True to demean the waveforms
        --time_limits: A tuple of (min_time,max_time) to trim the data
        --smoothing_amount: An integer which specifies a number
            of probe waveforms to average before doing the cross
            correlation at each posiiton. Only previous waveforms
            will be averaged with the current waveform.
        --kwargs: Placeholder for kwargs to be used in future routines

    Returns:
        --data: The processed data
        --times: The correct times for data
    '''
    
    if bandpass:
        data = dp_object._bandpass_filter(data,bandpass[0],bandpass[1],sampling_rate)
    if time_limits:
        min_ind = np.argmin(np.abs(times - time_limits[0]))
        max_ind = np.argmin(np.abs(times - time_limits[1]))
        data = data[...,min_ind:max_ind+1]
        times = times.copy()[min_ind:max_ind+1]
    if detrend:
        data = ss.detrend(data)
    if demean:
        data = ss.detrend(data,type='constant')
    if normalise:
        maxes = np.amax(np.abs(data),axis=-1)
        data = np.divide(data.transpose(),maxes).transpose()
    if smoothing_amount:
        if type(smoothing_amount) == type((1,)):
            break_indices = [0]+list(smoothing_amount[1:])+[data.shape[0]]
            smoothing_amount = smoothing_amount[0]
            next_break_ind = 1
            for i in range(data.shape[0]):
                data[i] = np.average(data[max(break_indices[next_break_ind-1],i+1-smoothing_amount):i+1],axis=0)
                if next_break_ind<len(break_indices) and i == break_indices[next_break_ind]-1:
                    next_break_ind += 1
        else:
            for i in range(data.shape[0]):
                data[i] = np.average(data[max(0,i+1-smoothing_amount):i+1],axis=0)

    #plt.plot(times,trace50/np.amax(trace50),'r-')
    #plt.plot(times, trace501/np.amax(trace501))
    #plt.show()

    return data, times


def toggle_latex(on=True):
    '''
    Change whether plots display with latex formatting
    or not. on=True for latex formatting, on=False otherwise.
    '''

    if on:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
    else:
        plt.rc('text', usetex=False)
        plt.rc('font', family='sans-serif')


def set_color_cycle(colors_=None,markers_=None, only_colors=True):
    '''
    Function to set the color cylcle
    '''
    from cycler import cycler

    if colors_ and markers_:
        mpl.rcParams['axes.prop_cycle'] = cycler(c=colors_,marker=markers_)
    elif colors_:
        mpl.rcParams['axes.prop_cycle'] = cycler(c=colors_,marker=markers)
    elif markers_:
        mpl.rcParams['axes.prop_cycle'] = cycler(c=colors,marker=markers_)
    elif not only_colors:
        mpl.rcParams['axes.prop_cycle'] = cycler(c=colors,marker=markers)
    elif only_colors:
        mpl.rcParams['axes.prop_cycle'] = cycler(c=colors,marker=None)

toggle_latex(False)
colors = ['#feaa38', '#2272b5','r', '#6a52a3','#248c45','#d94901']   #Another red: '#cc191d'  #['#6a52a3','#248c45']#
markers = ["None"]*6
set_color_cycle(colors, only_colors=True)