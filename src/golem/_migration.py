import cv2
import numpy as np
import numba as nb
from scipy.interpolate import CubicHermiteSpline
import os
import ctypes
import pandas as pd
import cupy as cp
import nvidia.dali as dali
from numba.typed import Dict
from numba.types import Tuple, int64

# Resolve absolute path
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "lib", "libEikonal.so"))
if not os.path.exists(lib_path):
    raise FileNotFoundError(f"❌ Shared library not found at: {lib_path}")

# Load the shared library
fsm_lib = ctypes.CDLL(lib_path)

# Declare argument types
fsm_lib.fast_sweeping_method.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
    ctypes.c_float, ctypes.c_float,  # sx, sz
    ctypes.c_float, ctypes.c_float,  # dx, dz
    ctypes.c_int, ctypes.c_int       # nx, nz
]
fsm_lib.fast_sweeping_method.restype = ctypes.POINTER(ctypes.c_float)

# --------------------------
# Declare migrate_constant_velocity
# --------------------------
fsm_lib.migrate_constant_velocity.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # data pointer
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # cdp pointer (array of length ntraces)
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # offsets pointer (array of length ntraces)
    ctypes.c_float,  # v (velocity)
    ctypes.c_float,  # dt (time sampling interval)
    ctypes.c_float,  # dx (lateral sampling interval)
    ctypes.c_float,  # dz (depth sampling interval)
    ctypes.c_int,    # nsmp (number of time samples)
    ctypes.c_int,    # ntraces (number of traces)
    ctypes.c_int,    # nx (output image lateral dimension)
    ctypes.c_int,    # nz (output image depth dimension)
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")   # output pointer R (flattened output image)
]
fsm_lib.migrate_constant_velocity.restype = None

# --------------------------
# Declare migrate_variable_velocity
# --------------------------
fsm_lib.init_cuda_with_mapped_host()

# Define the argument types for the function.
fsm_lib.migrate_variable_velocity.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # const float* data
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # const float* cdp
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # const float* offsets
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # const float* eikonal positions
    np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),    # const int* segments
    ctypes.c_float,    # float v
    ctypes.c_float,    # float dt
    ctypes.c_float,    # float dx
    ctypes.c_float,    # float dz
    ctypes.c_int,      # int nsmp
    ctypes.c_int,      # int ntraces
    ctypes.c_int,      # int nx_coarse
    ctypes.c_int,      # int nz_coarse
    ctypes.c_int,      # int nx_fine
    ctypes.c_int,      # int nz_fine
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # float* R
    ctypes.c_char_p,   # const char* traveltime_filename
    ctypes.c_char_p,   # const char* gradient_filename
    ctypes.c_int       # int num_segments
]

# The function returns void.
fsm_lib.migrate_variable_velocity.restype = None

# --------------------------
# Declare free_eikonal function
# --------------------------
fsm_lib.free_eikonal.argtypes = [ctypes.POINTER(ctypes.c_float)]
fsm_lib.free_eikonal.restype = None


@nb.njit
def compute_trace_segments(src, rec):
    n = src.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)
    
    # Allocate an output array with maximum possible size (n)
    segs = np.empty(n, dtype=np.int64)
    seg_count = 0
    count = 1
    prev_src = src[0]
    prev_rec = rec[0]
    
    for i in range(1, n):
        # Check if both source and receiver are identical to previous.
        if src[i] == prev_src and rec[i] == prev_rec:
            count += 1
        else:
            segs[seg_count] = count
            seg_count += 1
            count = 1
            prev_src = src[i]
            prev_rec = rec[i]
    
    segs[seg_count] = count
    seg_count += 1
    
    # Return the array up to the number of segments found.
    return segs[:seg_count]


def compute_traveltime_field(Vp, sx, sz, dx, dz, nx, nz):

    Vp_trans = np.ascontiguousarray(Vp.T, dtype=np.float32)
    # Vp = np.asfortranarray(Vp, dtype=np.float32)
    result_ptr = fsm_lib.fast_sweeping_method(Vp_trans, sx, sz, dx, dz, nx, nz)
    result_view = np.ctypeslib.as_array(result_ptr, shape=(nz * nx,))
    traveltime = np.copy(result_view).reshape((nz, nx), order='F')  # or 'C' based on CUDA layout
    fsm_lib.free_eikonal(result_ptr)
    return traveltime

def free_gpu_memory(func):
    def wrapper_func(*args, **kwargs):
        retval = func(*args, **kwargs)
        cp._default_memory_pool.free_all_blocks()
        return retval
    return wrapper_func

# @free_gpu_memory
def compute_traveltime_with_derivative(Vp, sx, sz, dx, dz, nx, nz):

    traveltime = 1.0/compute_traveltime_field(Vp, sx, sz, dx, dz, nx, nz)
    return traveltime, np.gradient(traveltime, dx, axis=1)

def resample_lanczos(input_array, dx_old, dy_old, dx_new, dy_new):
    """
    Resample a 2D array using Lanczos interpolation, and check that physical dimensions match.
    
    Parameters:
    - input_array: 2D array (shape [Nz, Nx])
    - dx_old, dy_old: Original grid spacing
    - dx_new, dy_new: New grid spacing
    
    Returns:
    - Resampled array (with shape based on physical size)
    """
    Nz, Nx = input_array.shape

    # Physical dimensions (in meters)
    dim_x = (Nx - 1) * dx_old
    dim_z = (Nz - 1) * dy_old

    # Expected new sizes to preserve physical dimensions
    Nx_new = int(round(dim_x / dx_new)) + 1
    Nz_new = int(round(dim_z / dy_new)) + 1

    # Calculate actual physical dimensions from new grid
    dim_x_new = (Nx_new - 1) * dx_new
    dim_z_new = (Nz_new - 1) * dy_new

    # Check if they match the original physical dimensions
    tol = 1e-3  # Tolerance in meters
    if abs(dim_x_new - dim_x) > tol or abs(dim_z_new - dim_z) > tol:
        print(f"[⚠️ Warning] New grid does not match original physical dimensions.")
        print(f"  Original: ({dim_z:.2f} m, {dim_x:.2f} m)")
        print(f"  New:      ({dim_z_new:.2f} m, {dim_x_new:.2f} m)")

        # Suggest corrected dx_new and dy_new
        dx_suggest = dim_x / (Nx_new - 1)
        dy_suggest = dim_z / (Nz_new - 1)

        print(f"[💡 Suggestion] To match physical size exactly, use:")
        print(f"  dx_new = {dx_suggest:.6f}")
        print(f"  dy_new = {dy_suggest:.6f}")

    # Perform the resampling using OpenCV (Lanczos)
    output_array = cv2.resize(input_array, (Nx_new, Nz_new), interpolation=cv2.INTER_LANCZOS4)

    return output_array

def collect_geometry_near_eikonal_points(df, spacing=300, radius=200, columns=("SourceX", "GroupX")):
    """
    Return a DataFrame of all traces where SourceX or GroupX is within `radius`
    of any reference point spaced every `spacing` meters.
    
    Adds two columns:
    - 'EikonalPosition': the x_ref reference point assigned to the row
    - 'DistanceToEikonal': the absolute distance (in meters) from that point
    
    The returned DataFrame preserves the original index.
    """
    all_x = pd.concat([df[columns[0]], df[columns[1]]])
    x_min, x_max = all_x.min(), all_x.max()

    reference_points = np.arange(x_min, x_max + spacing, spacing)
    return reference_points
    # selected_rows = []

    # for x_ref in reference_points:
    #     mask = ((df[columns[0]] - x_ref).abs() <= radius) | \
    #            ((df[columns[1]] - x_ref).abs() <= radius)

    #     if not mask.any():
    #         continue

    #     subset = df[mask].copy()  # retains original indices

    #     source_dist = (subset[columns[0]] - x_ref).abs()
    #     group_dist = (subset[columns[1]] - x_ref).abs()
    #     closest_dist = np.minimum(source_dist, group_dist)

    #     subset["EikonalPosition"] = x_ref
    #     subset["DistanceToEikonal"] = closest_dist

    #     selected_rows.append(subset)

    # if selected_rows:
    #     result_df = pd.concat(selected_rows, axis=0)
    #     return result_df
    # else:
    #     return pd.DataFrame(columns=list(df.columns) + ["EikonalPosition", "DistanceToEikonal"])




@nb.njit(parallel=True, fastmath=True)
def migrate_constant_velocity_numba(data, cdp_x, offsets, v, dx, dz, dt, nx, nz, aperture):
    """
    Optimized Kirchhoff migration for pre-stack data sorted by (CDP_X, offset),
    with NaN protection and numerical stability.

    Parameters:
        data : np.ndarray
            2D seismic data of shape (nsamples, ntraces)
        cdp_x : np.ndarray
            CDP x-location per trace (shape: ntraces,)
        offsets : np.ndarray
            Offset per trace (shape: ntraces,)
        v : float
            Constant velocity (scalar)
        dx, dz : float
            Horizontal and vertical sampling in the output image
        dt : float
            Time sampling interval
        nx, nz : int
            Output image dimensions in x and z

    Returns:
        R : np.ndarray
            Migrated image of shape (nx, nz)
    """
    nsmp, ntraces = data.shape
    R = np.zeros((nz, nx), dtype=np.float32)
    epsilon = 1e-10  # small number to avoid divide-by-zero

    for itrace in range(ntraces):
        cdp = cdp_x[itrace]
        h = offsets[itrace] * 0.5

        init_x = int(np.floor((cdp-aperture)/dx))
        end_x = int(np.ceil((cdp+aperture)/dx))

        percentage = (itrace+1)/ntraces * 100.

        if percentage % 10 == 0:
            print(percentage,"'%' pronto")

        if init_x < 0:
            init_x=0
        
        if end_x >= nx:
            end_x = nx
        
        if h < aperture:
            for iz in nb.prange(1,nz):
                z = iz * dz
                
                for ix in range(init_x,end_x):
                    x = ix * dx

                    dxs = x - (cdp - h)
                    dxg = x - (cdp + h)

                    rs = np.sqrt(dxs * dxs + z * z)
                    rr = np.sqrt(dxg * dxg + z * z)

                    # Stability fix
                    rs = max(rs, epsilon)
                    rr = max(rr, epsilon)

                    t = (rs + rr) / v
                    it = int(t / dt)

                    if 0 <= it < nsmp:
                        sqrt_rs_rr = np.sqrt(rs / rr)
                        sqrt_rr_rs = 1.0 / sqrt_rs_rr
                        wco = (z / rs * sqrt_rs_rr + z / rr * sqrt_rr_rs) / v

                        # if not np.isnan(wco):
                        R[iz, ix] -= data[it, itrace] * wco * 0.3989422804  #  1/sqrt(2π)

    return R




def compute_and_store_traveltime_fields(Vp, shot_positions, depth_positions, dx, dz, nx, nz, output_folder,output_filename,output_filename2):
    """
    Compute the traveltime field for each shot (or CDP) on a coarse grid,
    and store the results in a single NPY file.
    
    Parameters:
      Vp              : 2D numpy array representing the velocity model.
      shot_positions  : 1D numpy array of shot (or CDP) x positions (e.g., every 300 m).
      depth_positions : 1D numpy array of corresponding shot depths.
      dx, dz          : spatial sampling intervals for the coarse grid.
      nx, nz          : dimensions of the coarse traveltime grid.
      output_folder   : string path to the folder where the file will be stored.
      output_filename : string file name for the saved file (e.g., "tt_fields.npy").

    Returns:
      filepath: The full path to the saved NPY file.
    """
    nshots = shot_positions.shape[0]
    # Allocate array for traveltime.
    # T_all = np.zeros((nshots, 2, nz, nx), dtype=np.float32)
    
    filepath = os.path.join(output_folder, output_filename)
    filepath2 = os.path.join(output_folder, output_filename2)
    # Open the file in binary write mode once.
    file_traveltime = open(filepath, 'wb')
    file_gradient = open(filepath2, 'wb')
    for i in range(nshots):
        sx = shot_positions[i]
        sz = depth_positions[i]
        # Compute traveltime field for this shot.
        # Traveltime = 1.0/compute_traveltime_field(Vp, sx, sz, dx, dz, nx, nz)
        traveltime, gradient = compute_traveltime_with_derivative(Vp, sx, sz, dx, dz, nx, nz)
        # Convert to a contiguous array of type float32.
        data = np.ascontiguousarray(traveltime.astype(np.float32))
        data2 = np.ascontiguousarray(gradient.astype(np.float32))
        # Write raw bytes of the array to the file.
        file_traveltime.write(data.tobytes())
        file_gradient.write(data2.tobytes())

    # Ensure output folder exists.
    # os.makedirs(output_folder, exist_ok=True)
    
    
    # Save the traveltime fields as a single NPY file.
    # np.save(filepath, T_all)
    
    return filepath, filepath2


def migrate_constant_velocity_cuda(data, cdp_x, offsets, v, dx, dz, dt, nx, nz):
    """
    Fully vectorized GPU Kirchhoff migration using CuPy with trace‐batching to limit memory usage.
    
    Instead of processing all traces at once (which can exceed available GPU memory), the code
    processes traces in batches so that the maximum allocated memory does not surpass 4GB.
    
    Parameters:
      data    : np.ndarray
                2D seismic data on CPU of shape (nsmp, ntraces) (float32).
      cdp_x   : np.ndarray
                CDP X locations per trace (ntraces,).
      offsets : np.ndarray
                Offset per trace (ntraces,).
      v       : float
                Constant velocity.
      dx, dz, dt : float
                Spatial (lateral and depth) and time sampling intervals.
      nx, nz  : int
                Output image dimensions (lateral and depth samples).
                
    Returns:
      R       : np.ndarray
                Migrated image on CPU (shape: (nx, nz)).
    """

    nsmp, ntraces = data.shape

    R = np.zeros((nx * nz), dtype=np.float32)  # flattened array

    # Ensure arrays are contiguous:
    data_contig = np.ascontiguousarray(data.T.astype(np.float32))
    cdp_x_contig = np.ascontiguousarray(cdp_x.astype(np.float32))
    offsets_contig = np.ascontiguousarray(offsets.astype(np.float32))
    R = np.ascontiguousarray(np.zeros((nx*nz), dtype=np.float32))

    # Create CUDA events.
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()

    # Record the start event.
    start_event.record()

    fsm_lib.migrate_constant_velocity(data_contig, cdp_x_contig, offsets_contig,
                                        np.float32(v), np.float32(dt),
                                        np.float32(dx), np.float32(dz),
                                        ctypes.c_int(nsmp),
                                        ctypes.c_int(ntraces),
                                        ctypes.c_int(nx),
                                        ctypes.c_int(nz),
                                        R)

    # Record the end event.
    end_event.record()

    # Wait for the events to be completed.
    end_event.synchronize()

    # Get the elapsed time in milliseconds.
    elapsed_time_ms = cp.cuda.get_elapsed_time(start_event, end_event)
    print("Kernel execution time (ms):", elapsed_time_ms*0.001)

    # Reshape R to (nx, nz) if needed.
    migrated_image = -R.reshape((nz, nx))

    return migrated_image

from nvidia.dali import pipeline_def, fn

@pipeline_def(batch_size=1, num_threads=1, device_id=0)
def pipe_traveltime(file_root, files, Nx, Nz):
    traveltime = fn.readers.numpy(device="gpu", file_root=file_root, files=files)

    traveltime_resized = dali.fn.resize(
        traveltime,
        size=[Nz, Nx],
        interp_type= dali.types.DALIInterpType.INTERP_LANCZOS3
    )

    return traveltime_resized


def migrate_variable_velocity_cuda(data, Geometry_Dataframe, segments, eikonal_positions, v, dx, dz, dt, nx_coarse, nz_coarse, 
                                   nx_fine, nz_fine, traveltime_path, gradient_path, key_cdp = "CDP_X",key_offset='offset'):
    """
    Fully vectorized GPU Kirchhoff migration using CuPy with trace‐batching to limit memory usage.
    
    Instead of processing all traces at once (which can exceed available GPU memory), the code
    processes traces in batches so that the maximum allocated memory does not surpass 4GB.
    
    Parameters:
      data    : np.ndarray
                2D seismic data on CPU of shape (nsmp, ntraces) (float32).
      cdp_x   : np.ndarray
                CDP X locations per trace (ntraces,).
      offsets : np.ndarray
                Offset per trace (ntraces,).
      v       : float
                Constant velocity.
      dx, dz, dt : float
                Spatial (lateral and depth) and time sampling intervals.
      nx, nz  : int
                Output image dimensions (lateral and depth samples).
                
    Returns:
      R       : np.ndarray
                Migrated image on CPU (shape: (nx, nz)).
    """
    
    # pipe = pipe_traveltime(npy_folder,npy_filenames, nx, nz)
    # pipe.build()
    
    nsmp, ntraces = data.shape

    cdp_x = Geometry_Dataframe[key_cdp].to_numpy()
    offsets = Geometry_Dataframe[key_offset].to_numpy()

    # Ensure arrays are contiguous:
    data_contig = np.ascontiguousarray(data.T.astype(np.float32))
    cdp_x_contig = np.ascontiguousarray(cdp_x.astype(np.float32))
    offsets_contig = np.ascontiguousarray(offsets.astype(np.float32))
    eikonal_positions_contig = np.ascontiguousarray(eikonal_positions.astype(np.float32))
    num_segments = len(segments)
    segments = np.ascontiguousarray(segments.astype(np.int32))
    R = np.ascontiguousarray(np.zeros((nx_fine*nz_fine), dtype=np.float32))

    # Convert to bytes for ctypes (c_char_p expects bytes)
    clean_path = traveltime_path.strip()
    # Convert to bytes for ctypes (c_char_p expects a null-terminated byte string)
    traveltime_path_bytes = clean_path.encode('utf-8')

    # Convert to bytes for ctypes (c_char_p expects bytes)
    clean_path = gradient_path.strip()
    # Convert to bytes for ctypes (c_char_p expects a null-terminated byte string)
    gradient_path_bytes = clean_path.encode('utf-8')

    # fsm_lib.init_cuda_with_mapped_host()

    fsm_lib.migrate_variable_velocity(data_contig, cdp_x_contig, offsets_contig, eikonal_positions_contig, segments,
                                        np.float32(v), np.float32(dt),
                                        np.float32(dx), np.float32(dz),
                                        ctypes.c_int(nsmp),
                                        ctypes.c_int(ntraces),
                                        ctypes.c_int(nx_coarse), ctypes.c_int(nz_coarse), 
                                        ctypes.c_int(nx_fine), ctypes.c_int(nz_fine),
                                        R, traveltime_path_bytes, gradient_path_bytes, num_segments)
    
    # # Reshape R to (nx, nz) if needed.
    migrated_image = -R.reshape((nz_fine, nx_fine))

    return migrated_image


@nb.njit(parallel=True, fastmath=True)
def _migrate_trace_local(data_trace, it_field, valid_mask, cdp, h, dx_out, dz_out, Vp, nx, nz):
    """
    Compute the contribution of a single trace using the provided fine-grid time sample index field.
    
    Parameters:
      data_trace : 1D array (nsmp,) of seismic trace amplitudes.
      it_field   : 2D int array (nz, nx) of time sample indices (note: here rows = depth, cols = lateral)
      valid_mask : 2D bool array (nz, nx) indicating which indices are valid.
      cdp        : float, effective common depth point (computed as (SourceX+GroupX)/2).
      h          : float, half-offset for this trace.
      dx_out, dz_out : float, output grid sampling intervals.
      Vp         : 2D array (nz, nx) of local velocities on the output grid.
      nx, nz     : ints, dimensions of the output image.
      
    Returns:
      R_trace    : 2D float array (nz, nx) representing the trace’s contribution.
    """
    R_trace = np.zeros((nz, nx), dtype=np.float32)
    sqrt2pi = 1.0 / np.sqrt(2.0 * np.pi)
    for iz in nb.prange(nz):
        z_val = iz * dz_out
        for ix in range(nx):
            if valid_mask[iz, ix]:
                x_val = ix * dx_out
                it = it_field[iz, ix]
                if it >= 0 and it < data_trace.shape[0]:
                    rs = np.sqrt((x_val - (cdp - h))**2 + z_val**2)
                    rr = np.sqrt((x_val - (cdp + h))**2 + z_val**2)
                    if rs < 1e-10:
                        rs = 1e-10
                    if rr < 1e-10:
                        rr = 1e-10
                    # local_v = Vp[iz, ix]
                    weight = ((z_val / rs) * np.sqrt(rs / rr) + (z_val / rr) * np.sqrt(rr / rs)) #/ local_v
                    weight *= sqrt2pi
                    R_trace[iz, ix] += data_trace[it] * weight
    return R_trace

def migrate_kirchhoff(data, geometry, Vp, image_dims,
                                                dx_model, dz_model, dx_output, dz_output, dt,
                                                unique_positions, traveltime_mmap):
    """
    Perform Kirchhoff migration using precomputed traveltime fields (and derivatives) for both source and receiver.
    
    The precomputed traveltime fields are stored in a single dictionary tt_eikonal_dict,
    computed at positions given by eikonal_positions. For each trace, the traveltime field
    for the source (based on SourceX) and for the receiver (based on GroupX) are retrieved
    directly (since every possible position was computed exactly). Their interpolated values
    are then summed to yield the total traveltime.
    
    Parameters:
      data          : 2D seismic data, shape (nsmp, ntraces)
      geometry      : structured array or DataFrame with 'SourceX', 'GroupX', 'CDP_X', and 'offset'
      Vp            : 2D velocity model on the output grid, shape (nz, nx) [rows=depth, cols=lateral]
      image_dims    : tuple (nx, nz) for the migrated image (fine grid)
      dx_model, dz_model : spacing for the coarse model grid (used in traveltime precomputation)
      dx_output, dz_output : spacing for the output (fine) grid
      dt            : time sampling interval
      tt_eikonal_dict: dict mapping positions (floats) to a tuple (T_coarse, dT_dx, dT_dz)
      eikonal_positions: 1D numpy array of positions at which traveltime fields were computed
      
    Returns:
      R             : Migrated image of shape (nz, nx)
    """
    # Vp is assumed to be defined on the fine grid with shape (nz, nx).
    nz, nx = Vp.shape
    nx_image, nz_image = image_dims  # (nx_image, nz_image) should match (nx, nz)
    nsmp, ntraces = data.shape
    R = np.zeros((nz_image, nx_image), dtype=np.float32)  # migrated image: (nz, nx)

    for itrace in range(ntraces):
        sx = geometry['SourceX'][itrace]
        gx = geometry['GroupX'][itrace]
        h = geometry['offset'][itrace] * 0.5
        cdp = 0.5 * (sx + gx)
        
        # Directly index the precomputed traveltime dictionary:
        index = np.argmin(np.abs(unique_positions - sx))
        T_coarse_src = traveltime_mmap[index, :, :]
        T_coarse_rec = traveltime_mmap[index, :, :]
        
        T_source_fine = resample_lanczos(T_coarse_src, dx_output, dz_output, dx_model, dz_model)
        T_receiver_fine = resample_lanczos(T_coarse_rec, dx_output, dz_output, dx_model, dz_model)
        
        # Total traveltime is the sum.
        tt_field = T_source_fine + T_receiver_fine
        
        # Convert total traveltime to time sample indices.
        it_field = np.floor(tt_field / dt).astype(np.int32)
        valid_mask = (it_field >= 0) & (it_field < nsmp)
        
        data_trace = data[:, itrace]
        
        # Compute migration contribution from this trace using the Numba-accelerated inner loop.
        R_trace = _migrate_trace_local(data_trace, it_field, valid_mask,
                                       cdp, h, dx_output, dz_output, Vp, nx_image, nz_image)
        R += R_trace

    return R