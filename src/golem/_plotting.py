import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd

def plot_seismic_image(context, xlabel, ylabel, y_spacing, x_header, perc, key="data", xlim=None, ylim=None):
    data = context.get(key)
    df = context.get("geometry")

    if data is None or not hasattr(data, "shape"):
        print("❌ Error: 'data' not found or invalid in context.")
        return

    if not isinstance(y_spacing, (int, float)) or y_spacing <= 0:
        print(f"❌ Error: y_spacing must be a positive number. Got: {y_spacing}")
        return

    if not isinstance(perc, (int, float)) or perc <= 0:
        print(f"❌ Error: perc must be a positive number. Got: {perc}")
        return

    try:
        if data.ndim == 2:
            if df is None or not isinstance(df, pd.DataFrame):
                print("❌ Error: 'geometry' not found or invalid in context.")
                return

            if x_header not in df.columns:
                print(f"❌ Error: Column '{x_header}' not found in geometry.")
                return

            num_samples, num_traces = data.shape
            x_values = df[x_header].to_numpy()

            if len(x_values) != num_traces:
                print("❌ Error: Length of x_header values does not match number of traces in data.")
                return

            y_values = np.arange(num_samples) * y_spacing
            clip_value = np.percentile(data, perc)
            extent = [x_values[0], x_values[-1], y_values[-1], y_values[0]]

            plt.figure(figsize=(12, 6), dpi=150)
            plt.imshow(data, aspect='auto', cmap='gray_r', vmin=-clip_value, vmax=clip_value, extent=extent)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title("Seismic Image")
            plt.colorbar(label="Amplitude")

            if xlim:
                plt.xlim(xlim)
            if ylim:
                plt.ylim(ylim)

            plt.tight_layout()
            plt.show()

        elif data.ndim == 1:
            y = np.arange(len(data)) * y_spacing

            # Use a more compact vertical layout
            plt.figure(figsize=(4, 6), dpi=100)
            plt.plot(data, y, color='black', linewidth=1)
            plt.gca().invert_yaxis()  # Time increases downward
            plt.xlabel("Amplitude")
            plt.ylabel(ylabel)
            plt.title("Stacked Seismic Trace")
            plt.grid(True)

            # if xlim:
            #     plt.xlim(xlim)
            if ylim:
                plt.ylim(ylim[::-1])  # Reverse because time is vertical

            plt.tight_layout()
            plt.show()

        else:
            print(f"❌ Error: 'data' must be 1D or 2D. Got shape: {data.shape}")

    except Exception as e:
        print(f"❌ Failed to plot seismic data: {e}")



def plot_seismic_comparison_with_trace(context, key1, key2, xlabel, ylabel, y_spacing, x_header, perc, xlim=None, ylim=None):
    data1 = context.get(key1)
    data2 = context.get(key2)
    df = context.get("geometry")

    # Validate inputs
    for key, data in zip([key1, key2], [data1, data2]):
        if data is None or not isinstance(data, np.ndarray) or data.ndim != 2:
            print(f"❌ Error: '{key}' not found or not a valid 2D NumPy array in context.")
            return

    if df is None or x_header not in df.columns:
        print(f"❌ Error: Geometry or header '{x_header}' not found in context.")
        return

    if not isinstance(y_spacing, (int, float)) or y_spacing <= 0:
        print(f"❌ Error: y_spacing must be a positive number. Got: {y_spacing}")
        return

    try:
        num_samples, num_traces = data1.shape
        x_values = df[x_header].to_numpy()
        y_values = np.arange(num_samples) * y_spacing
        extent = [x_values[0], x_values[-1], y_values[-1], y_values[0]]

        center_trace = num_traces // 2
        trace1 = data1[:, center_trace]
        trace2 = data2[:, center_trace]

        combined = np.hstack([data1, data2])
        clip = np.percentile(np.abs(combined), perc)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={'width_ratios': [1.2, 1.2, 0.8]})

        # Image 1
        im1 = axes[0].imshow(data1, aspect='auto', cmap='gray_r',
                             vmin=-clip, vmax=clip, extent=extent)
        axes[0].set_title(f"{key1}")
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel(ylabel)
        if xlim: axes[0].set_xlim(xlim)
        if ylim: axes[0].set_ylim(ylim)

        # Image 2
        im2 = axes[1].imshow(data2, aspect='auto', cmap='gray_r',
                             vmin=-clip, vmax=clip, extent=extent)
        axes[1].set_title(f"{key2}")
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel(ylabel)
        if xlim: axes[1].set_xlim(xlim)
        if ylim: axes[1].set_ylim(ylim)

        # Trace comparison
        t = y_values
        axes[2].plot(trace1, t, label=key1, color='blue')
        axes[2].plot(trace2, t, label=key2, color='red', linestyle='--')
        axes[2].invert_yaxis()
        axes[2].set_xlabel("Amplitude")
        axes[2].set_title("Central Trace Comparison")
        axes[2].legend()
        axes[2].grid(True)
        if ylim: axes[2].set_ylim(ylim[::-1])

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"❌ Failed to plot comparison: {e}")

def plot_spectrum(context, key="data", dt=1.0):
    wavelet = context.get(key)
    dt=dt/1000.
    if wavelet is None or not isinstance(wavelet, np.ndarray):
        print(f"❌ Error: '{key}' not found or not a valid NumPy array in context.")
        return

    try:
        if wavelet.ndim == 1:
            t = np.arange(len(wavelet)) * dt
            spectrum = np.fft.rfft(wavelet)
            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum) * (180 / np.pi)
            freqs = np.fft.rfftfreq(len(wavelet), d=dt)
        elif wavelet.ndim == 2:
            n_samples, n_traces = wavelet.shape
            t = np.arange(n_samples) * dt
            spectrum = np.fft.rfft(wavelet, axis=0)  # FFT along vertical (time) axis
            magnitude = np.sum(np.abs(spectrum), axis=1)
            phase = np.angle(np.sum(spectrum, axis=1)) * (180 / np.pi)
            freqs = np.fft.rfftfreq(n_samples, d=dt)
        else:
            print(f"❌ Error: Unsupported array dimension: {wavelet.ndim}")
            return

        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])

        # Time domain plot
        ax0 = plt.subplot(gs[:, 0])
        if wavelet.ndim == 1:
            ax0.plot(t, wavelet, color='b', label="Wavelet")
        else:
            ax0.imshow(wavelet, aspect='auto', cmap='gray', extent=[0, wavelet.shape[1], t[-1], t[0]])
        ax0.set_xlabel('Time' if wavelet.ndim == 1 else 'Trace')
        ax0.set_ylabel('Amplitude' if wavelet.ndim == 1 else 'Time (ms)')
        ax0.set_title('Wavelet in Time Domain')
        ax0.grid()
        ax0.legend(["Wavelet"])

        # Magnitude spectrum
        ax1 = plt.subplot(gs[0, 1])
        ax1.plot(freqs, magnitude, label='Magnitude Spectrum')
        ax1.set_ylabel('Magnitude')
        ax1.set_title('Wavelet Spectrum')
        ax1.grid()
        ax1.legend()

        # Phase spectrum
        ax2 = plt.subplot(gs[1, 1])
        ax2.plot(freqs, phase, label='Phase Spectrum', color='r')
        ax2.set_xlabel('Frequency')
        ax2.set_ylabel('Phase (degrees)')
        ax2.set_title('Wavelet Phase')
        ax2.grid()
        ax2.legend()

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"❌ Failed to generate wavelet spectrum and phase plot: {e}")

