from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import signal
import numpy as np
import torch
import audio_functions as auf
import pywt
from matplotlib.colors import LogNorm

nominal_oct_central_freqs = [31.5, 63, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 160000]


def plot_signal(*vectors, xticks=None, yticks=None, title=None, file_name=False, grid=False, log=False, figsize=False, show=True, y_label="Amplitud", xlimits = False, ylimits = False, legend=False):
    """
    Plots multiple time signals over the same plot.
    Input:
        - vectors: Optional amount of values. For each vector: Dict type object. Must contain:
            - time vector: array or Torch.tensor type object. Time vector.
            - signal: array or Torch.tensor type object. Amplitudes vector.
            - label: str type object. 
            - color: string type object.
            - alpha: float type array. Color opacity

        - xticks: Optional. Int type object.
        - yticks: array type object. Optional
        - title: string type object. Optional
        - file_name: string type object. Optional. If true, saves the figure in graficos folder.
        - grid: boolean type object. Optional.
        - log: boolean type object. Optional.
        - figsize: tuple of ints type object. Optional. In order to use with Multiplot function, it must be false.
        - show: Bool type object. If true, shows the plot. In order to use with Multiplot function, it must be false.
        - xlimits: tuple type object.
        - ylimits: tuple type object.
        - legend: bool type object. False by default.
    Output:
        - Signal plot
        - If file_name is true, saves the figure and prints a message.
    """
    if figsize:
        plt.figure(figsize=figsize) 

    if type(xticks) != int and type(xticks) != type(None):            
            raise Exception("xtick value must be an int")
    
    
    if type(xticks) == int:
        if xticks == 1:
            plt.xticks(np.arange(0, xticks + 0.1, 0.1))
        else:
            plt.xticks(np.arange(0, xticks+1, 1))

    for vector in vectors:

        #check keys

        #time vector
        if not ("time vector" in vector.keys()):
            raise Exception("time vector key missing")
        else:
            #turn to numpy
            n = vector["time vector"]

            if type(n) == torch.Tensor:
                n = n.numpy().astype(np.float32)
            elif type(n) != np.ndarray:
                raise ValueError("Time vector must be an array or a Tensor")

        #signal vector
        if not ("signal" in vector.keys()):
            raise Exception("signal key missing")
        else:
            #turn to numpy
            signal = vector["signal"]
            if type(signal) == torch.Tensor:
                signal = signal.numpy().astype(np.float32)
            elif type(signal) != np.ndarray:
                raise ValueError("Audio signal must be an array or a Tensor")

        label = vector["label"] if "label" in vector.keys() else None
        color = vector["color"] if "color" in vector.keys() else None
        alpha = vector["alpha"] if "alpha" in vector.keys() else None

        #plot signal
        plt.plot(n, signal, label=label, color=color, alpha=alpha)
        plt.xlabel("Tiempo [s]", fontsize=13)

    if type(yticks) == np.ndarray:
        if type(yticks) != np.ndarray:            
            raise Exception("ytick value must be an array")
        
        if not(ylimits):            
            plt.ylim(np.min(yticks), np.max(yticks))

        plt.yticks(yticks)

    plt.grid(grid)
    
    if xlimits:
        if type(xlimits) != tuple:
            raise ValueError("Xlimits must be tuple type")
        plt.xlim(xlimits)

    if ylimits:
        if type(ylimits) != tuple:
            raise ValueError("Xlimits must be tuple type")
        plt.ylim(ylimits)

    if log:
        plt.yscale("log")

    plt.ylabel(f"{y_label}", fontsize=13)

    if title:
        plt.title(title, fontsize=15)

    #save file
    if file_name:
        plt.savefig(f"../graficos/{file_name}.png")
        #print(f"File saved in graficos/{file_name}.png")
    
    if legend:
        plt.legend()

    if show: 
        plt.show()
    else:
        plt.ioff()


def plot_spectrogram(*vectors, xticks=None, yticks=None, title=None, file_name=False, grid=False, figsize=False, show=True, 
                     y_label="Frecuencia [Hz]", x_label="Tiempo [s]", xlimits=False, ylimits=False, log=False, cmap="viridis", NFFT=512, noverlap=128):
    """
    Plots spectrograms for one or more time signals.
    Input:
        - vectors: Optional amount of values. For each vector: Dict type object. Must contain:
            - signal: array or Torch.tensor type object. Amplitudes vector.
            - fs: int type object. Sampling frequency.
            - label: str type object. Optional. 
        - xticks: Optional. Int type object.
        - yticks: array type object. Optional.
        - title: string type object. Optional.
        - file_name: string type object. Optional. If true, saves the figure in the 'graficos' folder.
        - grid: boolean type object. Optional.
        - figsize: tuple of ints type object. Optional. Figure size.
        - show: Bool type object. If true, shows the plot.
        - xlimits: tuple type object. Limits for the x-axis.
        - ylimits: tuple type object. Limits for the y-axis.
        - log: boolean type object. If true, uses a logarithmic scale for the frequency axis.
        - cmap: string type object. Colormap for the spectrogram.
    Output:
        - Spectrogram plot(s).
        - If file_name is true, saves the figure and prints a message.
    """
    if figsize:
        plt.figure(figsize=figsize)

    for vector in vectors:
        # Validate keys
        if not ("signal" in vector.keys()):
            raise Exception("Signal key missing")
        if not ("fs" in vector.keys()):
            raise Exception("Sampling frequency (fs) key missing")

        # Extract signal and sampling frequency
        signal = vector["signal"]
        fs = vector["fs"]

        if isinstance(signal, torch.Tensor):
            signal = signal.numpy().astype(np.float32)
        elif not isinstance(signal, np.ndarray):
            raise ValueError("Signal must be an array or a Tensor")

        if not isinstance(fs, int):
            raise ValueError("Sampling frequency (fs) must be an integer")

        label = vector.get("label", None)

        # Plot spectrogram
        plt.specgram(signal, Fs=fs, cmap=cmap, scale="dB", NFFT=NFFT, noverlap=noverlap)
        plt.colorbar(label="Amplitud [dB]")

        if xlimits:
            if not isinstance(xlimits, tuple):
                raise ValueError("Xlimits must be a tuple")
            plt.xlim(xlimits)

        if ylimits:
            if not isinstance(ylimits, tuple):
                raise ValueError("Ylimits must be a tuple")
            plt.ylim(ylimits)

        if log:
            plt.yscale("log")
            plt.ylabel(f"{y_label} (log scale)", fontsize=13)
        else:
            plt.ylabel(y_label, fontsize=13)

        plt.xlabel(x_label, fontsize=13)
        if label:
            plt.title(label)

    if title:
        plt.title(title, fontsize=15)

    if xticks:
        if not isinstance(xticks, int):
            raise ValueError("xticks must be an integer")
        plt.xticks(np.arange(0, xticks+1, 1))

    if yticks:
        if not isinstance(yticks, (list, np.ndarray)):
            raise ValueError("yticks must be a list or an array")
        plt.yticks(yticks)

    plt.grid(grid)

    if file_name:
        plt.savefig(f"../graficos/{file_name}.png")
        print(f"File saved in graficos/{file_name}.png")

    if show:
        plt.show()
    else:
        plt.ioff()


def plot_cwt_spectrogram(time_vector, coeffs, scales, wavelet, fs=48000, title=None, file_name=False, figsize=False, show=True,
                         y_label="Escalas", x_label="Tiempo [s]", xlimits=False, ylimits=False, cmap="viridis", colorbar_label="Amplitud", shading="auto"):
    """
    Plots wavelet spectrograms (scalograms) for one or more signals.
    Input:
        - time_vector.
        - coeffs: wavelets coeficients
        - scales: array-like type object. Scales for the wavelet transform. If None, calculated automatically.
        - wavelet: str type object. 
        - fs: float type object. Sampling period of the signal (inverse of fs).
        - title: string type object. Plot title.
        - file_name: string type object. Optional. If true, saves the figure in the 'graficos' folder.
        - figsize: tuple of ints type object. Optional. Figure size.
        - show: Bool type object. If true, shows the plot.
        - y_label: string type object. Label for the y-axis.
        - x_label: string type object. Label for the x-axis.
        - xlimits: tuple type object. Limits for the x-axis.
        - ylimits: tuple type object. Limits for the y-axis.
        - cmap: string type object. Colormap for the scalogram.
        - colorbar_label: string type object. Label for the colorbar.
    Output:
        - Wavelet spectrogram plot(s).
        - If file_name is true, saves the figure and prints a message.
    """
    if figsize:
        plt.figure(figsize=figsize)

    if not isinstance(fs, int):
        raise ValueError("Sampling frequency (fs) must be an integer")
    
    #if isinstance(coeffs, torch.Tensor):
    #    coeffs = coeffs.numpy().astype(np.float32)

    frequencies = pywt.scale2frequency(wavelet, scales) * fs

    # Plot scalogram
    plt.pcolormesh(time_vector, frequencies, torch.abs(coeffs), cmap=cmap, shading=shading)
    plt.yscale('log')  # Set Y-axis to logarithmic scale
    plt.colorbar(label=colorbar_label)
    plt.ylabel(y_label, fontsize=13)
    plt.xlabel(x_label, fontsize=13)

    if title:
        plt.title(title, fontsize=15)

    if xlimits:
        if not isinstance(xlimits, tuple):
            raise ValueError("Xlimits must be a tuple")
        plt.xlim(xlimits)

    if ylimits:
        if not isinstance(ylimits, tuple):
            raise ValueError("Ylimits must be a tuple")
        plt.ylim(ylimits)

    if file_name:
        plt.savefig(f"../graficos/{file_name}.png")
        print(f"File saved in graficos/{file_name}.png")

    if show:
        plt.show()
    else:
        plt.ioff()

def multiplot(*plots, figsize=(8, 5)):
    """
    Receive single plots as lambda functions and subplots them all in rows of 2 columns.
    Inputs:
        - plots: lambda function type object. Every plot must have Show and Figsize arguments set to False.
        - figsize: structured type object.
    """

    if isinstance(plots[0], dict):
        plots_dict = plots[0]
        plots = list(plots_dict.values())

    num_plots = len(plots)
    rows = (num_plots + 1)//2
    plt.figure(figsize=figsize)
    for i, figure in enumerate(plots):
        plt.subplot(rows,2, i + 1)
        figure()
    plt.tight_layout()
    plt.show()

def plot_fft_mag(*in_signals, fs=44100, N=1, title=False, legend=False, show=True, xlimits = False, normalize=True, ylimits=False, figsize=False, xticks=False, grid=False):

    """
    Plots the magnitude of the fast fourier transform of an arbitrary ammount of audio signals.
    Inputs:
        - in_signals : Optional amount of values. For each signal: Dict type object. Must contain:
            - audio signal: array or Torch.tensor type object.
            - label: string type object.
            - color: string type object.
    """

    for in_signal in in_signals:
        #calcular fft

        if not ("audio signal" in in_signal.keys()):
            raise Exception("Audio signal key missing")
        else:
            audio_signal = in_signal["audio signal"]
            if type(audio_signal) == torch.Tensor:
                audio_signal = audio_signal.numpy().astype(np.float32)
            elif type(audio_signal) != np.ndarray:
                raise ValueError("Audio signal must be an array or a Tensor")

        label = in_signal["label"] if "label" in in_signal.keys() else None
        color = in_signal["color"] if "color" in in_signal.keys() else None

        in_freqs, fft_mag_norm, _ = auf.get_fft(audio_signal, fs, normalize=normalize)
        eps = np.finfo(float).eps
        fft_mag_db = 20*np.log10(fft_mag_norm + eps)

            # Apply the moving average filter
        if N > 1:
            ir = np.ones(N) * 1 / N  # Moving average impulse response
            fft_mag_db = signal.fftconvolve(fft_mag_db, ir, mode='same')

    # Logarithmic scale for the x-axis
        plt.semilogx(in_freqs, fft_mag_db, label=label, color=color)


    # grafico fft
    if xticks:
        plt.xticks([t for t in xticks], [f'{t}' for t in xticks])
    else:
        ticks = [31, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        plt.xticks([t for t in ticks], [f'{t}' for t in ticks])
        plt.xlim(20, 22000)
    plt.ylim(-80, np.max(fft_mag_db) + 10)
    plt.xlabel("Frecuencia [Hz]", fontsize=13)
    plt.ylabel("Amplitud [dB]", fontsize=13)

    if xlimits:
        if type(xlimits) != tuple:
            raise ValueError("Xlimits must be tuple type")
        plt.xlim(xlimits)

    if ylimits:
        if type(ylimits) != tuple:
            raise ValueError("ylimits must be tuple type")
        plt.ylim(ylimits)

    if figsize:
        plt.figure(figsize=figsize)

    plt.grid(grid)

    if title:
        plt.title(title)

    if legend:
        plt.legend()

    if show: 
        plt.show()

def plot_fft_phase(*in_signals, fs=44100, N=1, title=False, legend=False, show=True, xlimits = False, ylimits=False, grid=False, figsize=False, xticks=False):

    """
    Plots the phase of the fast fourier transform of an arbitrary ammount of audio signals.
    Inputs:
        - in_signals : Optional amount of values. For each signal: Dict type object. Must contain:
            - audio signal: array or Torch.tensor type object.
            - label: string type object.
            - color: string type object.
    """

    for in_signal in in_signals:

        if not ("audio signal" in in_signal.keys()):
            raise Exception("Audio signal key missing")
        else:
            audio_signal = in_signal["audio signal"]
            if type(audio_signal) == torch.Tensor:
                audio_signal = audio_signal.numpy().astype(np.float32)
            elif type(audio_signal) != np.ndarray:
                raise ValueError("Audio signal must be an array or a Tensor")

        label = in_signal["label"] if "label" in in_signal.keys() else None
        color = in_signal["color"] if "color" in in_signal.keys() else None

        in_freqs, _, fft_phase = auf.get_fft(audio_signal, fs)

    # Logarithmic scale for the x-axis
        plt.semilogx(in_freqs, fft_phase, label=label, color=color)


    # grafico fft
    if xticks:
        plt.xticks([t for t in xticks], [f'{t}' for t in xticks])
    else:
        ticks = [31, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        plt.xticks([t for t in ticks], [f'{t}' for t in ticks])
        plt.xlim(20, 22000)
    #plt.ylim(-80, np.max(fft_mag_db) + 10)
    plt.xlabel("Frecuencia [Hz]", fontsize=13)
    plt.ylabel("Amplitud [dB]", fontsize=13)

    if xlimits:
        if type(xlimits) != tuple:
            raise ValueError("Xlimits must be tuple type")
        plt.xlim(xlimits)

    if ylimits:
        if type(ylimits) != tuple:
            raise ValueError("ylimits must be tuple type")
        plt.ylim(ylimits)

    if figsize:
        plt.figure(figsize=figsize)

    if title:
        plt.title(title)

    if legend:
        plt.legend()

    plt.grid(grid)

    if show: 
        plt.show()

def unit_plot(*vectors, xticks=None, yticks=None, title=None, file_name=False, grid=False, log=False, figsize=False, show=True, y_label="", xlimits = False, ylimits = False, legend=False):
    """
    Plots a unities vector.
    Input:
        - vectors: Optional amount of values. For each vector: Dict type object. Must contain:
            - array: array or Torch.tensor type object. Amplitudes vector.
            - label: str type object. 
            - color: string type object.

        - xticks: Optional. Int type object.
        - yticks: array type object. Optional
        - title: string type object. Optional
        - file_name: string type object. Optional. If true, saves the figure in graficos folder.
        - grid: boolean type object. Optional.
        - log: boolean type object. Optional.
        - figsize: tuple of ints type object. Optional. In order to use with Multiplot function, it must be false.
        - show: Bool type object. If true, shows the plot. In order to use with Multiplot function, it must be false.
        - xlimits: tuple type object.
        - ylimits: tuple type object.
        - legend: bool type object. False by default.
    Output:
        - Signal plot
        - If file_name is true, saves the figure and prints a message.
    """
    if figsize:
        plt.figure(figsize=figsize) 

    if type(xticks) != int and type(xticks) != type(None):            
            raise Exception("xtick value must be an int")
    
    if type(xticks) == int:
        if xticks == 1:
            plt.xticks(np.arange(0, xticks + 0.1, 0.1))
        else:
            plt.xticks(np.arange(0, xticks+1, 1))

    for vector in vectors:

        #check keys

        #signal vector
        if not ("array" in vector.keys()):
            raise Exception("array key missing")
        else:
            #turn to numpy
            arr = vector["array"]
            if type(arr) == torch.Tensor:
                arr = arr.numpy().astype(np.float32)
            elif type(arr) != np.ndarray:
                raise ValueError("Array must be an ndarray or a Tensor")

        label = vector["label"] if "label" in vector.keys() else None
        color = vector["color"] if "color" in vector.keys() else None

        n = np.arange(len(arr))

        #plot signal
        plt.plot(n, arr, label=label, color=color)
        plt.xlabel("Unit", fontsize=13)

    if type(yticks) == np.ndarray:
        if type(yticks) != np.ndarray:            
            raise Exception("ytick value must be an array")
        
        if not(ylimits):            
            plt.ylim(np.min(yticks), np.max(yticks))

        plt.yticks(yticks)

    plt.grid(grid)
    
    if xlimits:
        if type(xlimits) != tuple:
            raise ValueError("Xlimits must be tuple type")
        plt.xlim(xlimits)

    if ylimits:
        if type(ylimits) != tuple:
            raise ValueError("Xlimits must be tuple type")
        plt.ylim(ylimits)

    if log:
        plt.yscale("log")

    plt.ylabel(f"{y_label}", fontsize=13)

    if title:
        plt.title(title, fontsize=15)

    #save file
    if file_name:
        plt.savefig(f"../graficos/{file_name}.png")
        #print(f"File saved in graficos/{file_name}.png")
    
    if legend:
        plt.legend()

    if show: 
        plt.show()
    else:
        plt.ioff()


def plot_signal_energy(*vectors, fs=48000, xticks=None, yticks=None, title=None, file_name=False, grid=False, log=False, figsize=False, show=True, y_label="Amplitud", xlimits = False, ylimits = False, legend=False):
    """
    Plots multiple signals energy vector over the same plot.
    Input:
        - vectors: Optional amount of values. For each vector: Dict type object. Must contain:
            - signal: array or Torch.tensor type object. Amplitudes vector.
            - label: str type object. 
            - color: string type object.

        - xticks: Optional. Int type object.
        - yticks: array type object. Optional
        - title: string type object. Optional
        - file_name: string type object. Optional. If true, saves the figure in graficos folder.
        - grid: boolean type object. Optional.
        - log: boolean type object. Optional.
        - figsize: tuple of ints type object. Optional. In order to use with Multiplot function, it must be false.
        - show: Bool type object. If true, shows the plot. In order to use with Multiplot function, it must be false.
        - xlimits: tuple type object.
        - ylimits: tuple type object.
        - legend: bool type object. False by default.
    Output:
        - Signal plot
        - If file_name is true, saves the figure and prints a message.
    """
    if figsize:
        plt.figure(figsize=figsize) 

    if type(xticks) != int and type(xticks) != type(None):            
            raise Exception("xtick value must be an int")
    
    
    if type(xticks) == int:
        if xticks == 1:
            plt.xticks(np.arange(0, xticks + 0.1, 0.1))
        else:
            plt.xticks(np.arange(0, xticks+1, 1))

    for vector in vectors:

        #check keys


        #signal vector
        if not ("signal" in vector.keys()):
            raise Exception("signal key missing")
        else:
            #turn to numpy
            signal = vector["signal"]
            if type(signal) == torch.Tensor:
                signal = signal.numpy().astype(np.float32)
            elif type(signal) != np.ndarray:
                raise ValueError("Audio signal must be an array or a Tensor")

        #time vector
        n = auf.get_audio_time_array(signal, fs)

        energy = signal**2
        label = vector["label"] if "label" in vector.keys() else None
        color = vector["color"] if "color" in vector.keys() else None

        #plot signal
        plt.plot(n, energy, label=label, color=color)
        plt.xlabel("Tiempo [s]", fontsize=13)

    if type(yticks) == np.ndarray:
        if type(yticks) != np.ndarray:            
            raise Exception("ytick value must be an array")
        
        if not(ylimits):            
            plt.ylim(np.min(yticks), np.max(yticks))

        plt.yticks(yticks)

    plt.grid(grid)
    
    if xlimits:
        if type(xlimits) != tuple:
            raise ValueError("Xlimits must be tuple type")
        plt.xlim(xlimits)

    if ylimits:
        if type(ylimits) != tuple:
            raise ValueError("Xlimits must be tuple type")
        plt.ylim(ylimits)

    if log:
        plt.yscale("log")

    plt.ylabel(f"{y_label}", fontsize=13)

    if title:
        plt.title(title, fontsize=15)

    #save file
    if file_name:
        plt.savefig(f"../graficos/{file_name}.png")
        #print(f"File saved in graficos/{file_name}.png")
    
    if legend:
        plt.legend()

    if show: 
        plt.show()
    else:
        plt.ioff()



#terminar función:
def shots_plot_data(*files, figsize=(8, 5), in_format=None, fol_dir=""):
    if in_format == "list":
        files = files[0]

    plots = {}
    i = 0

    def create_plot_lambda(t, audio, title):
        return lambda: plot_signal(
            {"time vector": t, "signal": audio}, title=title, show=False, ylimits=(-0.5,0.5)
        )

    for f in files:
        path = fol_dir + "/" + f
        audio, fs = auf.load_audio(path)
        dur, t_0 = auf.get_audio_time_array(audio, fs)
        file_name = f.split("/")[-1]
        i += 1
        plots[f"plot_{i}"] = create_plot_lambda(t_0, audio, file_name.split(".")[0])

    first_shot = int(files[0].split("_")[0])

    return plots