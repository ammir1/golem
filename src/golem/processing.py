from scipy.signal import resample_poly
import numpy as np
import pandas as pd
import os
import inspect
import re
import pylops

def create_header(context, header_name, expression):
    df = context.get("geometry")

    if df is None:
        print("❌ Error: 'geometry' not found in context.")
        return None

    if not isinstance(df, pd.DataFrame):
        print("❌ Error: 'geometry' in context is not a valid DataFrame.")
        return None

    # Extract variable names from the expression (basic regex for word-like tokens)
    tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', expression)
    # Remove known functions like 'abs', 'sin', etc. — we keep only those not in numpy/pandas built-ins
    builtins = {"abs", "log", "sqrt", "sin", "cos", "tan", "exp", "min", "max", "mean", "std", "sum"}
    columns_used = [t for t in tokens if t not in builtins]

    # Check if all used columns exist
    missing = [col for col in columns_used if col not in df.columns]
    if missing:
        print(f"❌ Error: Column(s) not found in geometry: {missing}")
        return None

    try:
        df[header_name] = df.eval(expression)
        return df
    except Exception as e:
        print(f"❌ Failed to create header '{header_name}': {e}")
        return None
def subset_geometry_by_condition(context, condition, key="data"):
    df = context.get("geometry")
    data = context.get(key)

    if df is None or not isinstance(df, pd.DataFrame):
        print("❌ Error: 'geometry' not found or invalid.")
        return None

    if data is None or not hasattr(data, "__getitem__"):
        print("❌ Error: 'data' not found or not indexable.")
        return None

    try:
        # Filter the geometry using the condition string
        filtered_df = df.query(condition)
        if filtered_df.empty:
            print(f"⚠️ Warning: No rows match the condition: {condition}")
            return None

        # Subset the data using the filtered indices
        filtered_data = data[:, filtered_df.index.to_numpy()]

        # Update both geometry and data in context
        context["geometry"] = filtered_df.reset_index(drop=True)
        context[key] = filtered_data

        return filtered_df
    except Exception as e:
        print(f"❌ Failed to apply condition '{condition}': {e}")
        return None
def stack_data_along_axis(context, axis, method="sum"):
    data = context.get("data")

    if data is None or not isinstance(data, np.ndarray):
        print("❌ Error: 'data' not found or is not a NumPy array.")
        return None

    if data.ndim != 2:
        print(f"❌ Error: 'data' must be 2D. Got shape: {data.shape}")
        return None

    if axis not in [0, 1]:
        print(f"❌ Error: 'axis' must be 0 (stack over time) or 1 (stack over traces). Got: {axis}")
        return None

    if method not in ["sum", "mean"]:
        print(f"❌ Error: method must be either 'sum' or 'mean'. Got: {method}")
        return None

    try:
        if method == "sum":
            stacked = np.sum(data, axis=axis)
        elif method == "mean":
            stacked = np.mean(data, axis=axis)

        context["data"] = stacked
        return stacked
    except Exception as e:
        print(f"❌ Failed to stack data along axis {axis} using method '{method}': {e}")
        return None
def mute_data(context, start_sample):
    data = context.get("data")

    if data is None or not isinstance(data, np.ndarray):
        print("❌ Error: 'data' not found or not a NumPy array.")
        return None

    if not isinstance(start_sample, int) or start_sample < 0:
        print(f"❌ Error: 'start_sample' must be a non-negative integer. Got: {start_sample}")
        return None

    try:
        muted = data.copy()

        if data.ndim == 2:
            if start_sample >= data.shape[0]:
                print(f"⚠️ Warning: 'start_sample' ({start_sample}) exceeds number of samples ({data.shape[0]}). No muting applied.")
                return data
            muted[start_sample:, :] = 0

        elif data.ndim == 1:
            if start_sample >= data.shape[0]:
                print(f"⚠️ Warning: 'start_sample' ({start_sample}) exceeds trace length ({data.shape[0]}). No muting applied.")
                return data
            muted[start_sample:] = 0

        else:
            print(f"❌ Error: Unsupported data shape: {data.shape}")
            return None

        context["data"] = muted
        return muted

    except Exception as e:
        print(f"❌ Failed to mute data: {e}")
        return None
def resample(context, dt_in, dt_out, key='data', method="polyphase"):
    data = context.get(key)
    # survey_time = (len(data) - 1) * dt_in

    if data is None or not isinstance(data, np.ndarray):
        print("❌ Error: 'data' not found or invalid in context.")
        return None

    if not isinstance(dt_in, (int, float)) or dt_in <= 0:
        print(f"❌ Error: 'dt_in' must be a positive number. Got: {dt_in}")
        return None

    if not isinstance(dt_out, (int, float)) or dt_out <= 0:
        print(f"❌ Error: 'dt_out' must be a positive number. Got: {dt_out}")
        return None

    if method != "polyphase":
        print(f"❌ Error: Unsupported resampling method '{method}'. Only 'polyphase' is implemented.")
        return None

    try:
        ratio = dt_in / dt_out

        if data.ndim == 1:
            up = int(round(len(data) * ratio))
            down = len(data)
            resampled = resample_poly(data, up, down)

        elif data.ndim == 2:
            num_samples = data.shape[0]
            up = int(round(num_samples * ratio))
            down = num_samples
            resampled = resample_poly(data, up, down, axis=0)

        else:
            print(f"❌ Error: Unsupported data shape: {data.shape}")
            return None

        # context["data"] = resampled
        return resampled

    except Exception as e:
        print(f"❌ Failed to resample data: {e}")
        return None
def trim_samples(context, target_samples):
    data = context.get("data")

    if data is None or not isinstance(data, np.ndarray):
        print("❌ Error: 'data' not found or is not a NumPy array.")
        return None

    if not isinstance(target_samples, int) or target_samples <= 0:
        print(f"❌ Error: 'target_samples' must be a positive integer. Got: {target_samples}")
        return None

    try:
        current_samples = data.shape[0]

        if target_samples > current_samples:
            print(f"⚠️ Warning: target_samples ({target_samples}) > current_samples ({current_samples}). No trimming applied.")
            return data

        trimmed = data[:target_samples] if data.ndim == 1 else data[:target_samples, :]
        context["data"] = trimmed

        print(f"✅ Trimmed data from {current_samples} to {target_samples} samples.")
        return trimmed

    except Exception as e:
        print(f"❌ Failed to trim data: {e}")
        return None
def zero_phase_wavelet(context, key="data", shift=True):
    data = context.get(key)

    if data is None or not isinstance(data, np.ndarray):
        print("❌ Error: 'data' not found or is not a NumPy array.")
        return None

    if data.ndim != 1:
        print(f"❌ Error: Zero-phase conversion expects 1D wavelet. Got shape: {data.shape}")
        return None

    try:
        wavelet = np.copy(data)
        W_f = np.fft.fft(wavelet)
        magnitude = np.abs(W_f)
        zero_phase_W_f = magnitude
        zero_phase = np.fft.ifft(zero_phase_W_f)

        if shift:
            zero_phase = np.fft.ifftshift(zero_phase)

        print("✅ Created zero-phase version of wavelet.")
        return np.real(zero_phase)

    except Exception as e:
        print(f"❌ Failed to compute zero-phase wavelet: {e}")
        return None
def find_wavelet_main_lobe_center(wavelet, threshold_ratio=0.002):
    if not isinstance(wavelet, np.ndarray) or wavelet.ndim != 1:
        raise ValueError("Input must be a 1D NumPy array (wavelet)")

    max_amp = np.max(np.abs(wavelet))
    threshold = max_amp * threshold_ratio

    # Boolean mask where amplitude exceeds threshold
    above_thresh = np.abs(wavelet) >= threshold
    indices = np.where(above_thresh)[0]

    if len(indices) == 0:
        print("❌ No part of the wavelet exceeds the threshold.")
        return None, None, None

    start = indices[0]
    end = indices[-1] + 1  # Make the window inclusive

    wavelet = wavelet[start:end]

    center = len(wavelet) // 2
    length = len(wavelet)

    # print(f"✅ Wavelet main lobe: start={start}, end={end}, center={center}, length={length}")
    return wavelet, center
def calculate_convolution_operator(context, key="data", threshold_ratio=0.002):
    wavelet = context.get(key)

    if wavelet is None or not isinstance(wavelet, np.ndarray) or wavelet.ndim != 1:
        print(f"❌ Error: '{key}' in context must be a valid 1D NumPy wavelet.")
        return None

    try:
        wavelet_cut, center = find_wavelet_main_lobe_center(wavelet, threshold_ratio=threshold_ratio)

        if wavelet_cut is None or center is None:
            print("❌ Error: Failed to extract wavelet main lobe.")
            return None

        Cop = pylops.signalprocessing.Convolve1D(len(wavelet), h=wavelet_cut, offset=center, dtype="float32")

        print(f"✅ Convolution operator created from context['{key}']")
        return Cop

    except Exception as e:
        print(f"❌ Failed to create convolution operator: {e}")
        return None
def apply_designature(context, wavelet_key="wavelet", data_key="data"):
    wavelet = context.get(wavelet_key)
    data = context.get(data_key)
    operator = context.get("operator")
    if wavelet is None or data is None:
        print("❌ Error: wavelet or data not found in context.")
        return None

    try:
        wavelet_cut, offset = find_wavelet_main_lobe_center(wavelet)

        if data.ndim == 1:
            Cop = pylops.signalprocessing.Convolve1D(len(data), h=wavelet_cut, offset=offset)
            reflectivity = operator / data
            modeled = Cop @ reflectivity
            print("✅ 1D designature applied using operator inversion.")
            return modeled

        elif data.ndim == 2:
            n_samples, n_traces = data.shape
            modeled = np.zeros_like(data)
            for i in range(n_traces):
                Cop = pylops.signalprocessing.Convolve1D(n_samples, h=wavelet_cut, offset=offset)
                reflectivity = operator / data[:, i]
                modeled[:, i] = Cop @ reflectivity
            print("✅ 2D designature applied trace-by-trace using operator inversion.")
            return modeled

        else:
            print(f"❌ Unsupported data dimension: {data.ndim}")
            return None

    except Exception as e:
        print(f"❌ Failed to apply designature: {e}")
        return None
def sort(context, header1, header2=None, key="data"):
    df = context.get("geometry")
    data = context.get(key)

    if df is None or not isinstance(df, pd.DataFrame):
        print("❌ Error: 'geometry' not found or invalid.")
        return None

    if data is None or not isinstance(data, np.ndarray) or data.ndim != 2:
        print("❌ Error: 'data' not found or not a valid 2D array.")
        return None

    if header1 not in df.columns or (header2 and header2 not in df.columns):
        print(f"❌ Error: One or both headers not found in geometry. Got: '{header1}', '{header2}'")
        return None

    try:
        # Sort geometry
        sort_cols = [header1] if header2 is None else [header1, header2]
        sorted_df = df.sort_values(by=sort_cols, ignore_index=True)
        sorted_indices = sorted_df.index.to_numpy()

        # Use original DataFrame to find old-to-new index mapping
        sorted_positions = df.sort_values(by=sort_cols).index.to_numpy()

        # Reorder seismic data columns to match new geometry
        sorted_data = data[:, sorted_positions]

        # Update context
        context["geometry"] = sorted_df
        context[key] = sorted_data

        print(f"✅ Geometry and data sorted by: {', '.join(sort_cols)}")
        return sorted_df

    except Exception as e:
        print(f"❌ Failed to sort geometry: {e}")
        return None
    