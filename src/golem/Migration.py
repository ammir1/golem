import cv2
import numpy as np
import numba as nb

import ctypes
# Load the library
fsm_lib = ctypes.CDLL('./libEikonal.so')  # Change to .dll or .dylib if needed

# Set argument types
fsm_lib.fast_sweeping_method.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # Vp
    ctypes.c_float, ctypes.c_float,  # sx, sz
    ctypes.c_float, ctypes.c_float,  # dx, dz
    ctypes.c_int, ctypes.c_int       # nx, nz
    ]

fsm_lib.fast_sweeping_method.restype = ctypes.POINTER(ctypes.c_float)

def resample_lanczos(input_array, dx_old, dx_new, dy_old, dy_new):
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

def group_traces_by_cdp(geometry):
    """
    Group trace indices by unique CDP.
    Returns:
        unique_cdps: 1D array of unique CDP positions.
        groups: a dictionary mapping each unique CDP to a list of trace indices.
    """
    cdp_positions = geometry['CDP_X']
    unique_cdps = np.unique(cdp_positions)
    groups = {cdp: np.where(cdp_positions == cdp)[0] for cdp in unique_cdps}
    return unique_cdps, groups


def compute_traveltime_field(Vp, sx, sz, dx, dz, nx, nz):
    # Ensure the input array is in the correct format for ctypes
    Vp = np.ascontiguousarray(Vp, dtype=np.float32)

    # Call the C++ CUDA function
    result_ptr = fsm_lib.fast_sweeping_method(Vp, sx, sz, dx, dz, nx, nz)

    # Wrap the pointer as a NumPy array (without copying yet)
    result_view = np.ctypeslib.as_array(result_ptr, shape=(nz * nx,))

    # Copy the data into a new NumPy array and reshape to original dimensions
    traveltime = np.copy(result_view).reshape((nz, nx), order='F')

    # Free the memory allocated in C++
    fsm_lib.free_eikonal(result_ptr)

    return traveltime

# def migrate_with_cached_traveltimes(data, geometry, velocity_model, image_dims, dx, dz, dt, v):
#     """
#     Perform Kirchhoff migration using precomputed traveltime fields.
    
#     Parameters:
#         data: 2D seismic data, shape (nsamples, ntraces)
#         geometry: DataFrame or structured array with 'SourceX' and 'GroupX'
#         velocity_model: 2D grid of velocities
#         image_dims: (nx, nz) dimensions for the migrated image
#         dx, dz, dt: spatial and time sampling intervals
#         v: constant velocity (for weighting)
        
#     Returns:
#         R: Migrated image (nx, nz)
#     """
#     nx, nz = image_dims
#     R = np.zeros(image_dims, dtype=np.float32)

#     unique_cdps, groups = group_traces_by_cdp(geometry)
    
#     # Loop over each unique CDP group
#     for cdp in unique_cdps:
#         # Get indices of traces with this CDP
#         trace_indices = groups[cdp]
#         tt_field = compute_traveltime_field(Vp_resampled, sx, sz, dx_new, dz_new, Nx, Nz)  # traveltime field for this CDP, shape (nx, nz)
        
#         # For each trace in this group, compute its contribution.
#         for itrace in trace_indices:
#             # Determine the time sample from the traveltime field for each image point.
#             it_field = np.floor(tt_field / dt).astype(int)
            
#             # Make sure indices are within the data time range.
#             nsmp = data.shape[0]
#             valid_mask = (it_field >= 0) & (it_field < nsmp)
            
#             # Sum contributions into R. In a real code, you'd also apply amplitude weighting.
#             for ix in range(nx):
#                 for iz in range(nz):
#                     if valid_mask[ix, iz]:
#                         R[ix, iz] += data[it_field[ix, iz], itrace]
                        
#     return R



@nb.njit(parallel=True, fastmath=True)
def migrate_sorted_cdp_offset(data, cdp_x, offsets, v, dx, dz, dt, nx, nz):
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

    for iz in nb.prange(nz):
        z = iz * dz
        for ix in range(nx):
            x = ix * dx
            for itrace in range(ntraces):
                cdp = cdp_x[itrace]
                h = offsets[itrace] * 0.5

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

                    if not np.isnan(wco):
                        R[iz, ix] += data[it, itrace] * wco * 0.3989422804  # 1/sqrt(2π)

    return R