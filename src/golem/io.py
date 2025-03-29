import os
import numpy as np
import pandas as pd

def read_data(context, file_path, format="binary", shape=None):
    if not os.path.isfile(file_path):
        print(f"❌ Error: File not found — {file_path}")
        return None

    if format not in ["binary", "npy"]:
        print(f"❌ Error: Format must be 'binary' or 'npy'. Got: {format}")
        return None

    try:
        if format == "npy":
            data = np.load(file_path)
        elif format == "binary":
            if shape is None:
                print("❌ Error: 'shape' must be provided for binary files.")
                return None
            data = np.fromfile(file_path, dtype=np.float32).reshape(shape, order='F')
        else:
            return None

        print(f"✅ Data loaded: {file_path} — shape: {data.shape}")
        return data

    except Exception as e:
        print(f"❌ Failed to read data: {e}")
        return None
    

def write_data(context, file_folder, file_name, format="binary"):
    data = context.get("data")

    if data is None or not isinstance(data, np.ndarray):
        print("❌ Error: 'data' not found or is not a NumPy array.")
        return None

    if not os.path.isdir(file_folder):
        print(f"❌ Error: Directory does not exist — {file_folder}")
        return None

    if format not in ["binary", "npy"]:
        print(f"❌ Error: Format must be 'binary' or 'npy'. Got: {format}")
        return None

    try:
        # Determine shape string for filename
        if data.ndim == 1:
            shape_str = f"_{len(data)}Samples"
        elif data.ndim == 2:
            shape_str = f"_{data.shape[1]}x{data.shape[0]}Samples"
        else:
            print(f"❌ Error: Only 1D or 2D arrays are supported. Got shape: {data.shape}")
            return None

        # Determine full file path
        ext = ".bin" if format == "binary" else ".npy"
        full_name = f"{file_name}{shape_str}{ext}"
        file_path = os.path.join(file_folder, full_name)

        if format == "binary":
            if data.ndim > 1:
                data = data.T
            data = np.ascontiguousarray(data.astype(np.float32))
            data.tofile(file_path)
        else:
            np.save(file_path, data.astype(np.float32))

        print(f"✅ Data written: {file_path}")
        return file_path

    except Exception as e:
        print(f"❌ Failed to write data: {e}")
        return None
    

def import_npy_mmap(file_in, mode="r"):
    valid_modes = [None, "r", "r+", "w+", "c"]

    if mode not in valid_modes:
        print(f"❌ Error: Invalid mode '{mode}'. Valid modes are: {valid_modes}")
        return None

    if not os.path.isfile(file_in):
        print(f"❌ Error: File '{file_in}' does not exist.")
        return None

    try:
        return np.load(file_in, mmap_mode=mode)
    except Exception as e:
        print(f"❌ Failed to load file '{file_in}' with mode '{mode}': {e}")
        return None
    
def import_parquet_file(file_in):
    if not os.path.isfile(file_in):
        print(f"❌ Error: File '{file_in}' does not exist.")
        return None

    try:
        return pd.read_parquet(file_in)
    except Exception as e:
        print(f"❌ Failed to read parquet file '{file_in}': {e}")
        return None