### Functions for fitting and plotting encoding models.

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from scipy import interpolate
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, sem, ttest_rel
from statsmodels.stats.multitest import multipletests

def fit_linear_encoding_models(
    photometry_df: pd.DataFrame,
    annotations_df: pd.DataFrame,
    kernel_length_sec: float = 10.0,
    pre_onset_sec: float = 5.0,
    regularization: str = 'ridge',
    alpha: float | list = 1.0,
    l1_ratio: float = 0.5,
    days_to_include: list = None,
    session_type: str = None,
    target_fps: float = None,
    n_splines: int = 20,
    spline_degree: int = 3,
    offset: bool = False,
    n_folds: int = 3,
    verbose: bool = True
) -> tuple:
    """
    Fit linear encoding models to predict photometry signals from behavioral onsets and optionally offsets.
    Always fits by mouse (concatenates all sessions per mouse).
    Updated for session format: CSDS-Day1-A_1-Defeat

    Handles NaN-masked data from quality control:
    - Entire sessions masked with NaNs are included in the concatenation but excluded from fitting.
    - Partial NaN regions within sessions are preserved in structure but excluded from fitting.
    - NaN masking is applied consistently during cross-validation and final model fitting.

    Parameters:
    -----------
    alpha : float or list of float
        If a single float, fit all data using that alpha directly.
        If a list of floats, use cross-validation (with n_folds folds) to select
        the best alpha per (mouse, signal) combination, then refit on all data
        with that alpha.
    n_folds : int
        Number of cross-validation folds when alpha is a list. Sessions are split
        per mouse so that every fold contains data from every mouse.

    Returns:
    --------
    results_df : pd.DataFrame
        Kernel results. Now includes two additional columns:
            - 'cv_alphas'    : np.ndarray of the alpha values evaluated during CV
                               (same order as alpha_list; NaN-filled array when CV
                                was not performed).
            - 'cv_mse_scores' : np.ndarray of the mean cross-validated MSE for each
                               alpha in cv_alphas.  The chosen alpha corresponds to
                               the minimum value in this vector.
    reconstruction_data : dict
        Dictionary keyed by (mouse_id, signal_name) containing:
            - 'design_matrix': np.ndarray, the full design matrix X
            - 'original_signal': np.ndarray, the original photometry signal (with NaNs)
            - 'valid_indices': np.ndarray of bool, which timepoints were used for fitting
            - 'coefficients': np.ndarray, full model coefficient vector
            - 'intercept': float, model intercept
            - 'effective_fps': float
            - 'original_fps': float
            - 'group_id': str
            - 'signal_name': str
            - 'behavioral_features': list of str
            - 'kernel_length_frames': int
            - 'pre_onset_frames': int
            - 'total_kernel_frames': int
            - 'offset_included': bool
            - 'mse_score': float
            - 'n_sessions': int
            - 'session_ids': list of str
    reconstructed_photometry_df : pd.DataFrame
        DataFrame structured identically to photometry_df, but containing
        reconstructed photometry traces from the fit encoding model instead
        of the original traces. Non-photometry columns (e.g., 'fps') are
        preserved from the original photometry_df. Regions that were NaN in
        the original data remain NaN in the reconstruction.
    """

    # Determine if we need cross-validation
    if isinstance(alpha, (list, tuple, np.ndarray)):
        alpha_list = list(alpha)
        do_cv = True
    else:
        alpha_list = [alpha]
        do_cv = False

    def set_cell_value(df, row, col, value):
        """
        Safely assign a value (including numpy arrays) to a single cell in a DataFrame.
        Works around pandas' tendency to broadcast array-like values.
        """
        if isinstance(value, np.ndarray):
            if df[col].dtype != object:
                df[col] = df[col].astype(object)
            idx_pos = df.index.get_loc(row)
            col_pos = df.columns.get_loc(col)
            df.iat[idx_pos, col_pos] = value
        else:
            df.at[row, col] = value

    # def _safe_nan_mask(arr):
    #     """
    #     Return a boolean mask that is True where arr is NaN.
    #     Works for both float and integer dtypes.
    #     Integer arrays cannot contain NaN, so the mask is all-False for them.
    #     """
    #     try:
    #         arr_float = np.asarray(arr, dtype=float)
    #         return np.isnan(arr_float)
    #     except (ValueError, TypeError):
    #         return np.zeros(len(arr), dtype=bool)

    def _ensure_float(arr):
        """
        Ensure array is float dtype so it can hold NaN values.
        Returns a copy cast to float64 if needed.
        """
        arr = np.asarray(arr)
        if not np.issubdtype(arr.dtype, np.floating):
            return arr.astype(np.float64)
        return arr
    
    def extract_day(session_id):
        """Extract the day number from a session ID, or return infinity if not found."""
        parts = session_id.split('-')
        if len(parts) >= 2 and parts[1].startswith('Day'):
            try:
                day = int(parts[1][3:])
                return day
            except ValueError:
                return float('inf')
        return float('inf')

    def resample_signal(signal, original_fps, target_fps):
        """Resample a signal, handling NaN regions by interpolating only valid segments."""
        if target_fps > original_fps:
            raise ValueError(f"Target FPS ({target_fps}) cannot be higher than original FPS ({original_fps})")
        if target_fps == original_fps:
            return signal.copy(), original_fps

        signal = _ensure_float(signal)
        original_duration = len(signal) / original_fps
        original_time = np.linspace(0, original_duration, len(signal))
        new_length = int(original_duration * target_fps)
        target_time = np.linspace(0, original_duration, new_length)
        if new_length <= 0:
            return signal.copy(), original_fps

        nan_mask = np.isnan(signal)

        # Check if signal is entirely NaN
        if np.all(nan_mask):
            return np.full(new_length, np.nan), target_fps

        # If there are NaNs, interpolate only the valid portions
        if np.any(nan_mask):
            valid_mask = ~nan_mask
            if np.sum(valid_mask) < 2:
                return np.full(new_length, np.nan), target_fps
            interpolator = interpolate.interp1d(
                original_time[valid_mask], signal[valid_mask], kind='linear',
                bounds_error=False, fill_value=np.nan
            )
            resampled_signal = interpolator(target_time)

            # Also create a NaN mask at the new resolution to preserve NaN regions
            nan_float = nan_mask.astype(float)
            nan_interpolator = interpolate.interp1d(
                original_time, nan_float, kind='nearest',
                bounds_error=False, fill_value=1.0
            )
            resampled_nan_mask = nan_interpolator(target_time) > 0.5
            resampled_signal[resampled_nan_mask] = np.nan

            return resampled_signal, target_fps
        else:
            interpolator = interpolate.interp1d(original_time, signal, kind='linear',
                                                bounds_error=False, fill_value='extrapolate')
            resampled_signal = interpolator(target_time)
            return resampled_signal, target_fps

    def resample_binary_signal(signal, original_fps, target_fps):
        """Resample a binary (0/1) signal to a new FPS, preserving NaN regions."""
        if target_fps > original_fps:
            raise ValueError(f"Target FPS ({target_fps}) cannot be higher than original FPS ({original_fps})")
        if target_fps == original_fps:
            return _ensure_float(signal).copy(), original_fps

        signal = _ensure_float(signal)
        nan_mask = np.isnan(signal)

        original_duration = len(signal) / original_fps
        new_length = int(original_duration * target_fps)

        if np.all(nan_mask):
            return np.full(new_length, np.nan), target_fps

        # Work with a clean copy where NaNs are replaced with 0 for onset/offset detection
        clean_signal = signal.copy()
        clean_signal[nan_mask] = 0.0

        diff_signal = np.diff(clean_signal.astype(int))
        onsets = np.where(diff_signal == 1)[0] + 1
        offsets = np.where(diff_signal == -1)[0] + 1
        if clean_signal[0] == 1:
            onsets = np.insert(onsets, 0, 0)
        if clean_signal[-1] == 1:
            offsets = np.append(offsets, len(clean_signal))

        onset_times = onsets / original_fps
        offset_times = offsets / original_fps
        new_signal = np.zeros(new_length, dtype=np.float64)
        new_time = np.linspace(0, original_duration, new_length)
        for onset_time, offset_time in zip(onset_times, offset_times):
            onset_idx = np.argmin(np.abs(new_time - onset_time))
            offset_idx = np.argmin(np.abs(new_time - offset_time))
            new_signal[onset_idx:offset_idx] = 1.0

        # Propagate NaN mask to resampled signal
        if np.any(nan_mask):
            original_time = np.linspace(0, original_duration, len(signal))
            nan_float = nan_mask.astype(float)
            nan_interpolator = interpolate.interp1d(
                original_time, nan_float, kind='nearest',
                bounds_error=False, fill_value=1.0
            )
            resampled_nan_mask = nan_interpolator(new_time) > 0.5
            new_signal[resampled_nan_mask] = np.nan

        return new_signal, target_fps

    def upsample_signal(signal, effective_fps, original_fps):
        """Upsample a signal from effective_fps back to original_fps using linear interpolation.
        Preserves NaN regions."""
        if effective_fps == original_fps:
            return signal.copy()

        signal = _ensure_float(signal)
        duration = len(signal) / effective_fps
        effective_time = np.linspace(0, duration, len(signal))
        new_length = int(duration * original_fps)
        original_time = np.linspace(0, duration, new_length)

        # Check if entirely NaN
        if np.all(np.isnan(signal)):
            return np.full(new_length, np.nan)

        nan_mask = np.isnan(signal)
        if np.any(nan_mask):
            valid_mask = ~nan_mask
            if np.sum(valid_mask) < 2:
                return np.full(new_length, np.nan)
            interpolator = interpolate.interp1d(
                effective_time[valid_mask], signal[valid_mask], kind='linear',
                bounds_error=False, fill_value=np.nan
            )
            upsampled = interpolator(original_time)

            # Propagate NaN mask
            nan_float = nan_mask.astype(float)
            nan_interpolator = interpolate.interp1d(
                effective_time, nan_float, kind='nearest',
                bounds_error=False, fill_value=1.0
            )
            upsampled_nan_mask = nan_interpolator(original_time) > 0.5
            upsampled[upsampled_nan_mask] = np.nan
            return upsampled
        else:
            interpolator = interpolate.interp1d(effective_time, signal, kind='linear',
                                                bounds_error=False, fill_value='extrapolate')
            return interpolator(original_time)

    common_sessions = set(photometry_df.index).intersection(set(annotations_df.index))

    if session_type is not None:
        common_sessions = {s for s in common_sessions if session_type in s}
        if verbose:
            print(f"Filtering for sessions containing '{session_type}'")

    if days_to_include is not None:
        filtered_sessions = []
        for session in common_sessions:
            parts = session.split('-')
            if len(parts) >= 2 and parts[1].startswith('Day'):
                try:
                    day = int(parts[1][3:])
                    if day in days_to_include:
                        filtered_sessions.append(session)
                except ValueError:
                    continue
        session_filtered = filtered_sessions
    else:
        session_filtered = list(common_sessions)

    if not session_filtered:
        print("Error: No sessions found")
        return pd.DataFrame(), {}, pd.DataFrame()

    mouse_sessions = {}
    for session_id in session_filtered:
        parts = session_id.split('-')
        if len(parts) >= 3:
            mouse_id = parts[2]
        else:
            mouse_id = parts[0]
        if mouse_id not in mouse_sessions:
            mouse_sessions[mouse_id] = []
        mouse_sessions[mouse_id].append(session_id)

    if verbose:
        print(f"Found {len(session_filtered)} sessions from {len(mouse_sessions)} mice")
        print(f"Fitting strategy: mouse")
        for mouse_id, sessions in mouse_sessions.items():
            print(f"  {mouse_id}: {len(sessions)} sessions")

    photometry_signals = [col for col in photometry_df.columns if col.endswith('_dFF')]

    behavioral_features = []
    for session_id in session_filtered:
        behavioral_row = annotations_df.loc[session_id]
        for feature_name in behavioral_row.index:
            feature_data = behavioral_row[feature_name]
            if isinstance(feature_data, np.ndarray) and len(feature_data) > 1:
                # Cast to float to safely check for NaN, then check binary
                feature_float = _ensure_float(feature_data)
                nan_mask = np.isnan(feature_float)
                non_nan_data = feature_float[~nan_mask]
                if len(non_nan_data) > 0:
                    unique_values = np.unique(non_nan_data)
                    if np.all(np.isin(unique_values, [0, 1])):
                        if feature_name not in behavioral_features:
                            behavioral_features.append(feature_name)

    if verbose:
        print(f"Found {len(behavioral_features)} behavioral features: {behavioral_features}")
        print(f"Found {len(photometry_signals)} photometry signals")
        if target_fps is not None:
            print(f"Target FPS for downsampling: {target_fps}")
        if do_cv:
            print(f"Cross-validating over alphas {alpha_list} with {n_folds} folds (per mouse per signal)")
        else:
            print(f"Using {regularization} regularization with alpha={alpha_list[0]}")
        if regularization == 'elastic_net':
            print(f"L1 ratio: {l1_ratio} (0=Ridge, 1=Lasso)")
        print(f"Modeling {'onsets and offsets' if offset else 'onsets only'}")

    def make_model(a):
        if regularization == 'ridge':
            return Ridge(alpha=a, fit_intercept=True)
        elif regularization == 'lasso':
            return Lasso(alpha=a, fit_intercept=True, max_iter=2000)
        elif regularization == 'elastic_net':
            return ElasticNet(alpha=a, l1_ratio=l1_ratio, fit_intercept=True, max_iter=2000)
        else:
            raise ValueError("regularization must be 'ridge', 'lasso', or 'elastic_net'")

    def detect_onsets_and_offsets(binary_signal):
        """Detect onsets and offsets in a binary signal, handling NaN values.

        NaN regions are treated as 0 (no behavior). To avoid spurious onsets/offsets
        at NaN boundaries, we first zero out the NaN positions and then detect
        transitions only among reliable data.
        """
        signal_float = _ensure_float(binary_signal)
        clean_signal = signal_float.copy()
        nan_mask = np.isnan(clean_signal)
        clean_signal[nan_mask] = 0.0

        diff_signal = np.diff(clean_signal.astype(int))
        onsets = np.where(diff_signal == 1)[0] + 1
        offsets = np.where(diff_signal == -1)[0] + 1

        # Filter out spurious transitions at NaN boundaries:
        # An onset at index i is spurious if i or i-1 is NaN
        # An offset at index i is spurious if i or i-1 is NaN
        if np.any(nan_mask):
            valid_onsets = []
            for o in onsets:
                if o > 0 and not nan_mask[o] and not nan_mask[o - 1]:
                    valid_onsets.append(o)
                elif o == 0 and not nan_mask[o]:
                    valid_onsets.append(o)
            onsets = np.array(valid_onsets, dtype=int)

            valid_offsets = []
            for off in offsets:
                if off > 0 and not nan_mask[off - 1]:
                    # offset at 'off' means signal went from 1 to 0;
                    # diff is at off-1, check both off-1 and off
                    if off < len(nan_mask) and not nan_mask[off]:
                        valid_offsets.append(off)
                    elif off >= len(nan_mask):
                        valid_offsets.append(off)
                    # If off is NaN, it's at a boundary - still include if
                    # the preceding region was valid
                    elif nan_mask[off]:
                        valid_offsets.append(off)
            offsets = np.array(valid_offsets, dtype=int)

        if len(clean_signal) > 0 and clean_signal[0] == 1 and not nan_mask[0]:
            if len(onsets) == 0 or onsets[0] != 0:
                onsets = np.insert(onsets, 0, 0)
        if len(clean_signal) > 0 and clean_signal[-1] == 1 and not nan_mask[-1]:
            if len(offsets) == 0 or offsets[-1] != len(clean_signal):
                offsets = np.append(offsets, len(clean_signal))

        return onsets, offsets

    def create_design_matrix(behavioral_data_dict, signal_length, fps, kernel_length_frames, pre_onset_frames, include_offset=False):
        n_features = len(behavioral_features)
        n_timepoints = signal_length
        total_kernel_frames = pre_onset_frames + kernel_length_frames
        predictors_per_feature = total_kernel_frames * (2 if include_offset else 1)
        total_predictors = n_features * predictors_per_feature
        X = np.zeros((n_timepoints, total_predictors))
        feature_idx = 0
        for feature_name in behavioral_features:
            if feature_name in behavioral_data_dict:
                behavioral_signal = behavioral_data_dict[feature_name]
                onsets, offsets_det = detect_onsets_and_offsets(behavioral_signal)
                for onset in onsets:
                    for lag in range(-pre_onset_frames, kernel_length_frames):
                        time_idx = onset + lag
                        if 0 <= time_idx < n_timepoints:
                            kernel_idx = feature_idx * predictors_per_feature + (lag + pre_onset_frames)
                            X[time_idx, kernel_idx] += 1

                if include_offset:
                    for off in offsets_det:
                        for lag in range(-pre_onset_frames, kernel_length_frames):
                            time_idx = off + lag
                            if 0 <= time_idx < n_timepoints:
                                kernel_idx = (feature_idx * predictors_per_feature +
                                              total_kernel_frames + (lag + pre_onset_frames))
                                X[time_idx, kernel_idx] += 1
            feature_idx += 1

        if X.shape[0] != signal_length:
            raise ValueError(f"Design matrix has {X.shape[0]} timepoints but expected {signal_length}")
        return X

    # ── Spline-basis helpers (all capture n_splines / spline_degree via closure) ──

    def _make_spline_basis(total_frames: int) -> np.ndarray:
        """
        Build a B-spline basis matrix of shape (total_frames, ns) where
        ns = min(n_splines, total_frames).  Each column is one basis function
        evaluated at total_frames evenly-spaced points in [0, 1].
        """
        ns = min(n_splines, total_frames)
        ns = max(ns, spline_degree + 1)   # need at least degree+1 basis functions
        x  = np.linspace(0.0, 1.0, total_frames)

        n_interior = ns - spline_degree - 1
        interior   = (np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
                      if n_interior > 0 else np.array([]))
        knots = np.concatenate([
            np.zeros(spline_degree + 1),
            interior,
            np.ones(spline_degree + 1),
        ])

        B = np.zeros((total_frames, ns))
        for i in range(ns):
            c = np.zeros(ns)
            c[i] = 1.0
            B[:, i] = interpolate.BSpline(knots, c, spline_degree)(x)
        return B

    def _project_spline(X_lag: np.ndarray, B: np.ndarray,
                        n_feat: int, total_kf: int, inc_offset: bool) -> np.ndarray:
        """Project an (n_time, n_feat * total_kf * n_evt) lag matrix into
        the spline basis, returning (n_time, n_feat * ns * n_evt)."""
        ns    = B.shape[1]
        n_evt = 2 if inc_offset else 1
        X_spl = np.zeros((X_lag.shape[0], n_feat * ns * n_evt))
        for f in range(n_feat):
            for e in range(n_evt):
                lag_s = f * total_kf * n_evt + e * total_kf
                spl_s = f * ns * n_evt + e * ns
                X_spl[:, spl_s:spl_s + ns] = X_lag[:, lag_s:lag_s + total_kf] @ B
        return X_spl

    def _kernels_from_spline_coef(coef: np.ndarray, B: np.ndarray,
                                   n_feat: int, total_kf: int,
                                   inc_offset: bool) -> np.ndarray:
        """Reconstruct smooth per-feature kernels from spline coefficients.
        Returns array of shape (n_feat, total_kf * n_evt)."""
        ns    = B.shape[1]
        n_evt = 2 if inc_offset else 1
        out   = np.zeros((n_feat, total_kf * n_evt))
        for f in range(n_feat):
            for e in range(n_evt):
                spl_s = f * ns * n_evt + e * ns
                k_s   = e * total_kf
                out[f, k_s:k_s + total_kf] = B @ coef[spl_s:spl_s + ns]
        return out

    def _extract_signal_array(row, signal_name):
        """
        Safely extract a numpy array from a photometry DataFrame row.
        Handles object-dtype columns, scalar NaN values, and missing columns.
        Returns a float64 numpy array or None if the data is invalid.
        """
        if signal_name not in row.index:
            return None
        val = row[signal_name]
        # Handle scalar NaN (entire session masked at the DataFrame level)
        if val is None:
            return None
        if isinstance(val, (int, float)):
            if np.isnan(val):
                return None
            return None  # scalar non-NaN doesn't make sense as a signal
        if isinstance(val, np.ndarray):
            if val.ndim == 0:
                # 0-d array, e.g. np.array(np.nan)
                return None
            if len(val) == 0:
                return None
            return _ensure_float(val)
        # Try to convert other array-like types
        try:
            arr = np.asarray(val, dtype=np.float64)
            if arr.ndim == 0 or len(arr) == 0:
                return None
            return arr
        except (ValueError, TypeError):
            return None

    def process_session_data(session_id):
        """Process a single session's data, preserving NaN regions from QC masking."""
        try:
            photometry_row = photometry_df.loc[session_id]
            behavioral_row = annotations_df.loc[session_id]

            # Extract FPS safely
            if 'fps' in photometry_row.index:
                fps_val = photometry_row['fps']
                if isinstance(fps_val, (int, float, np.integer, np.floating)):
                    original_fps = float(fps_val)
                else:
                    if verbose:
                        print(f"Skipping {session_id}: fps is not a scalar ({type(fps_val)})")
                    return None
            else:
                if verbose:
                    print(f"Skipping {session_id}: no 'fps' column")
                return None

            if original_fps is None or original_fps <= 0 or np.isnan(original_fps):
                if verbose:
                    print(f"Skipping {session_id}: invalid FPS ({original_fps})")
                return None

            if target_fps is not None:
                if target_fps > original_fps:
                    if verbose:
                        print(f"Skipping {session_id}: target FPS ({target_fps}) > original FPS ({original_fps})")
                    return None
                effective_fps = target_fps
            else:
                effective_fps = original_fps

            # Process behavioral features
            behavioral_data_dict = {}
            behavioral_lengths = []
            for feature_name in behavioral_features:
                if feature_name in behavioral_row.index:
                    feature_data = behavioral_row[feature_name]
                    if isinstance(feature_data, np.ndarray) and len(feature_data) > 1:
                        feature_float = _ensure_float(feature_data)
                        if target_fps is not None:
                            resampled_data, _ = resample_binary_signal(feature_float, original_fps, target_fps)
                            behavioral_data_dict[feature_name] = resampled_data
                        else:
                            behavioral_data_dict[feature_name] = feature_float
                        behavioral_lengths.append(len(behavioral_data_dict[feature_name]))

            if not behavioral_data_dict:
                if verbose:
                    print(f"Skipping {session_id}: no valid behavioral data")
                return None

            # Process photometry signals
            processed_signals = {}
            photometry_lengths = []
            has_any_valid_signal = False
            for signal_name in photometry_signals:
                signal_data = _extract_signal_array(photometry_row, signal_name)
                if signal_data is None:
                    continue

                if target_fps is not None:
                    signal_data_processed, _ = resample_signal(signal_data, original_fps, target_fps)
                else:
                    signal_data_processed = signal_data.copy()

                processed_signals[signal_name] = signal_data_processed
                photometry_lengths.append(len(signal_data_processed))
                if not np.all(np.isnan(signal_data_processed)):
                    has_any_valid_signal = True

            if not processed_signals:
                if verbose:
                    print(f"Skipping {session_id}: no photometry signal arrays found")
                return None

            if not has_any_valid_signal and verbose:
                print(f"  Note: {session_id} has all-NaN photometry (QC masked) - "
                      f"included in structure but excluded from fitting")

            # Compute min_length across BOTH behavioral and photometry lengths
            all_lengths = behavioral_lengths + photometry_lengths
            min_length = min(all_lengths) if all_lengths else 0

            if min_length <= 0:
                if verbose:
                    print(f"Skipping {session_id}: computed min_length is {min_length}")
                return None

            return {
                'session_id': session_id,
                'behavioral_data': behavioral_data_dict,
                'photometry_signals': processed_signals,
                'original_fps': original_fps,
                'effective_fps': effective_fps,
                'min_length': min_length
            }
        except Exception as e:
            if verbose:
                print(f"Error processing {session_id}: {e}")
                import traceback
                traceback.print_exc()
            return None

    def concatenate_sessions_data(sessions_data, buffer_frames=0):
        """
        Concatenate session data with optional buffer padding between sessions.
        NaN regions in photometry signals are preserved for masking during fitting.
        Buffer regions between sessions are filled with zeros for behavioral data
        and NaN for photometry data, ensuring they are excluded from fitting.
        """
        all_behavioral_data = {}
        all_photometry_data = {}
        session_lengths = []
        for feature_name in behavioral_features:
            all_behavioral_data[feature_name] = []
        for signal_name in photometry_signals:
            all_photometry_data[signal_name] = []

        valid_sessions = [sd for sd in sessions_data if sd is not None]

        for sess_idx, session_data in enumerate(valid_sessions):
            session_length = session_data['min_length']
            session_lengths.append(session_length)

            for feature_name in behavioral_features:
                if feature_name in session_data['behavioral_data']:
                    feature_data = session_data['behavioral_data'][feature_name]
                    # Truncate or pad to session_length
                    if len(feature_data) >= session_length:
                        all_behavioral_data[feature_name].append(
                            _ensure_float(feature_data[:session_length])
                        )
                    else:
                        padded = np.zeros(session_length, dtype=np.float64)
                        padded[:len(feature_data)] = _ensure_float(feature_data)
                        all_behavioral_data[feature_name].append(padded)
                else:
                    all_behavioral_data[feature_name].append(
                        np.zeros(session_length, dtype=np.float64)
                    )

            for signal_name in photometry_signals:
                if signal_name in session_data['photometry_signals']:
                    signal_data = _ensure_float(
                        session_data['photometry_signals'][signal_name]
                    )
                    # Truncate or pad to session_length
                    if len(signal_data) >= session_length:
                        all_photometry_data[signal_name].append(
                            signal_data[:session_length]
                        )
                    else:
                        padded = np.full(session_length, np.nan)
                        padded[:len(signal_data)] = signal_data
                        all_photometry_data[signal_name].append(padded)
                else:
                    all_photometry_data[signal_name].append(
                        np.full(session_length, np.nan)
                    )

            if buffer_frames > 0 and sess_idx < len(valid_sessions) - 1:
                for feature_name in behavioral_features:
                    all_behavioral_data[feature_name].append(
                        np.zeros(buffer_frames, dtype=np.float64)
                    )
                for signal_name in photometry_signals:
                    all_photometry_data[signal_name].append(
                        np.full(buffer_frames, np.nan)
                    )

        for feature_name in behavioral_features:
            if all_behavioral_data[feature_name]:
                all_behavioral_data[feature_name] = np.concatenate(
                    all_behavioral_data[feature_name]
                )
            else:
                all_behavioral_data[feature_name] = np.array([], dtype=np.float64)

        for signal_name in photometry_signals:
            if all_photometry_data[signal_name]:
                all_photometry_data[signal_name] = np.concatenate(
                    all_photometry_data[signal_name]
                )
            else:
                all_photometry_data[signal_name] = np.array([], dtype=np.float64)

        return all_behavioral_data, all_photometry_data, session_lengths

    # ---- Cross-validation logic ----
    def create_stratified_folds(mouse_sessions_dict, n_folds_cv):
        folds = [[] for _ in range(n_folds_cv)]
        for mouse_id, sessions in mouse_sessions_dict.items():
            shuffled = list(sessions)
            np.random.shuffle(shuffled)
            for i, sid in enumerate(shuffled):
                folds[i % n_folds_cv].append(sid)
        return folds

    def cv_select_alpha_for_signal(sessions_data_dict, mouse_session_list, signal_name,
                               alpha_candidates, n_folds_cv, effective_fps,
                               kernel_length_frames, pre_onset_frames, total_kernel_frames):
        """
        Use cross-validation to select the best alpha for a specific (mouse, signal) combination.
        Uses MSE to select the best alpha. Handles NaN-masked data properly.

        Returns:
        --------
        best_alpha : float
        mean_scores : dict
            Maps each alpha candidate to its mean cross-validated MSE across folds.
        """
        valid_sids = [sid for sid in mouse_session_list
                    if sid in sessions_data_dict and sessions_data_dict[sid] is not None]

        # Filter to sessions that have non-NaN data for this signal
        sids_with_signal_data = []
        for sid in valid_sids:
            sd = sessions_data_dict[sid]
            if signal_name in sd['photometry_signals']:
                sig = _ensure_float(sd['photometry_signals'][signal_name])
                if not np.all(np.isnan(sig)):
                    sids_with_signal_data.append(sid)

        default_scores = {a: np.inf for a in alpha_candidates}

        if len(sids_with_signal_data) < 2:
            if verbose:
                print(f"    CV [{signal_name}]: Not enough sessions with valid data "
                      f"({len(sids_with_signal_data)}). Using first alpha: {alpha_candidates[0]}")
            return alpha_candidates[0], default_scores

        effective_n_folds = min(n_folds_cv, len(sids_with_signal_data))
        if effective_n_folds < 2:
            if verbose:
                print(f"    CV [{signal_name}]: Cannot form ≥2 folds. "
                    f"Using first alpha: {alpha_candidates[0]}")
            return alpha_candidates[0], default_scores

        if effective_n_folds < n_folds_cv and verbose:
            print(f"    CV [{signal_name}]: Reducing folds from {n_folds_cv} "
                f"to {effective_n_folds} (only {len(sids_with_signal_data)} sessions with valid data)")

        single_mouse_dict = {"_mouse_": sids_with_signal_data}
        folds = create_stratified_folds(single_mouse_dict, effective_n_folds)
        folds = [f for f in folds if len(f) > 0]

        if len(folds) < 2:
            if verbose:
                print(f"    CV [{signal_name}]: Not enough non-empty folds. "
                    f"Using first alpha: {alpha_candidates[0]}")
            return alpha_candidates[0], default_scores

        alpha_scores = {a: [] for a in alpha_candidates}

        for fold_idx in range(len(folds)):
            test_session_ids = set(folds[fold_idx])
            train_session_ids = [sid for j, fold in enumerate(folds)
                                if j != fold_idx for sid in fold]

            train_sessions_data = [sessions_data_dict[sid] for sid in train_session_ids
                                if sid in sessions_data_dict and sessions_data_dict[sid] is not None]
            test_sessions_data = [sessions_data_dict[sid] for sid in test_session_ids
                                if sid in sessions_data_dict and sessions_data_dict[sid] is not None]

            if not train_sessions_data or not test_sessions_data:
                continue

            train_behavioral, train_photometry, _ = concatenate_sessions_data(
                train_sessions_data, buffer_frames=total_kernel_frames
            )
            test_behavioral, test_photometry, _ = concatenate_sessions_data(
                test_sessions_data, buffer_frames=total_kernel_frames
            )

            if signal_name not in train_photometry or signal_name not in test_photometry:
                continue

            train_signal = _ensure_float(train_photometry[signal_name])
            test_signal = _ensure_float(test_photometry[signal_name])

            if (np.all(np.isnan(train_signal)) or len(train_signal) == 0 or
                    np.all(np.isnan(test_signal)) or len(test_signal) == 0):
                continue

            if (len(train_signal) < total_kernel_frames or
                    len(test_signal) < total_kernel_frames):
                continue

            B_cv   = _make_spline_basis(total_kernel_frames)
            X_train = _project_spline(
                create_design_matrix(train_behavioral, len(train_signal), effective_fps,
                                     kernel_length_frames, pre_onset_frames,
                                     include_offset=offset),
                B_cv, len(behavioral_features), total_kernel_frames, offset)
            X_test = _project_spline(
                create_design_matrix(test_behavioral, len(test_signal), effective_fps,
                                     kernel_length_frames, pre_onset_frames,
                                     include_offset=offset),
                B_cv, len(behavioral_features), total_kernel_frames, offset)

            # Mask out NaN regions
            train_valid = ~np.isnan(train_signal)
            X_train_clean = X_train[train_valid, :]
            y_train_clean = train_signal[train_valid]

            test_valid = ~np.isnan(test_signal)
            X_test_clean = X_test[test_valid, :]
            y_test_clean = test_signal[test_valid]

            if len(y_train_clean) == 0 or len(y_test_clean) == 0:
                continue

            if X_train_clean.sum() == 0 or X_test_clean.sum() == 0:
                continue

            for a in alpha_candidates:
                try:
                    model = make_model(a)
                    model.fit(X_train_clean, y_train_clean)
                    y_pred = model.predict(X_test_clean)
                    mse = np.mean((y_test_clean - y_pred) ** 2)
                    alpha_scores[a].append(mse)
                except Exception:
                    continue

        mean_scores = {}
        for a in alpha_candidates:
            mean_scores[a] = np.mean(alpha_scores[a]) if alpha_scores[a] else np.inf

        best_alpha = min(mean_scores, key=mean_scores.get)

        if verbose:
            print(f"    CV [{signal_name}] ({len(folds)} folds):")
            for a in alpha_candidates:
                score_str = (f"{mean_scores[a]:.4f}"
                            if mean_scores[a] < np.inf else "N/A")
                marker = " <-- best" if a == best_alpha else ""
                print(f"      alpha={a}: mean MSE = {score_str}{marker}")

        return best_alpha, mean_scores

    def fit_model_for_mouse(sessions_data, mouse_id, signal_alphas, signal_cv_scores):
        """
        Fit encoding models for all photometry signals for one mouse.
        Handles NaN-masked data by excluding NaN timepoints from fitting.
        """
        if not sessions_data or all(data is None for data in sessions_data):
            return [], {}, {}

        valid_sessions_data = [data for data in sessions_data if data is not None]
        if not valid_sessions_data:
            return [], {}, {}

        # Sort sessions by day before concatenation
        valid_sessions_data_sorted = sorted(valid_sessions_data, key=lambda x: extract_day(x['session_id']))

        effective_fps = valid_sessions_data_sorted[0]['effective_fps']
        original_fps = valid_sessions_data_sorted[0]['original_fps']
        kernel_length_frames = int(kernel_length_sec * effective_fps)
        pre_onset_frames = int(pre_onset_sec * effective_fps)
        total_kernel_frames = pre_onset_frames + kernel_length_frames

        buffer_frames = total_kernel_frames

        all_behavioral_data, all_photometry_data, session_lengths = concatenate_sessions_data(
            valid_sessions_data_sorted, buffer_frames=buffer_frames
        )
        if not all_behavioral_data or not all_photometry_data:
            return [], {}, {}

        results = []
        recon_data = {}
        per_session_reconstructions = {}

        group_session_ids = [sd['session_id'] for sd in valid_sessions_data_sorted]

        _nan_cv_scores = np.array([np.nan] * len(alpha_list))

        for signal_name in photometry_signals:
            if signal_name not in all_photometry_data:
                continue
            signal_data = _ensure_float(all_photometry_data[signal_name])
            if len(signal_data) == 0 or np.all(np.isnan(signal_data)):
                if verbose:
                    print(f"Skipping {mouse_id} {signal_name}: all data is NaN (fully QC-masked)")
                continue
            if len(signal_data) < total_kernel_frames:
                if verbose:
                    print(f"Skipping {mouse_id} {signal_name}: signal too short "
                        f"({len(signal_data)} < {total_kernel_frames})")
                continue

            chosen_alpha = signal_alphas.get(signal_name, alpha_list[0])
            cv_mse_arr = signal_cv_scores.get(signal_name, _nan_cv_scores)

            B_fit = _make_spline_basis(total_kernel_frames)
            X = _project_spline(
                create_design_matrix(all_behavioral_data, len(signal_data), effective_fps,
                                     kernel_length_frames, pre_onset_frames, include_offset=offset),
                B_fit, len(behavioral_features), total_kernel_frames, offset)

            # Mask out NaN regions
            valid_indices = ~np.isnan(signal_data)
            X_clean = X[valid_indices, :]
            y_clean = signal_data[valid_indices]

            if len(y_clean) == 0:
                if verbose:
                    print(f"Skipping {mouse_id} {signal_name}: no valid (non-NaN) timepoints for fitting")
                continue

            # Check for NaN in y_clean (should not happen after masking, but be safe)
            if np.any(np.isnan(y_clean)):
                if verbose:
                    print(f"Warning {mouse_id} {signal_name}: NaN found in y_clean after masking. "
                          f"Removing {np.sum(np.isnan(y_clean))} additional NaN timepoints.")
                secondary_valid = ~np.isnan(y_clean)
                X_clean = X_clean[secondary_valid, :]
                y_clean = y_clean[secondary_valid]
                if len(y_clean) == 0:
                    continue

            # Check for NaN in X_clean
            if np.any(np.isnan(X_clean)):
                if verbose:
                    print(f"Warning {mouse_id} {signal_name}: NaN found in design matrix. "
                          f"Replacing with 0.")
                X_clean = np.nan_to_num(X_clean, nan=0.0)

            if X_clean.sum() == 0:
                if verbose:
                    print(f"Skipping {mouse_id} {signal_name}: no behavioral events in valid regions")
                continue

            n_valid = int(np.sum(valid_indices))
            n_total = len(signal_data)
            n_nan = n_total - n_valid

            n_predictors = X_clean.shape[1]
            n_timepoints_fit = X_clean.shape[0]
            predictor_ratio = n_predictors / n_timepoints_fit if n_timepoints_fit > 0 else np.inf

            if verbose and predictor_ratio > 0.1:
                print(f"Warning {mouse_id} {signal_name}: High predictor ratio ({predictor_ratio:.3f})")

            if verbose and n_nan > 0:
                pct_masked = 100.0 * n_nan / n_total
                print(f"  {mouse_id} {signal_name}: {n_nan}/{n_total} timepoints masked "
                      f"({pct_masked:.1f}% NaN from QC + buffers)")

            model = make_model(chosen_alpha)
            try:
                model.fit(X_clean, y_clean)
                model_score = model.score(X_clean, y_clean)
            except Exception as fit_error:
                if verbose:
                    print(f"Model fitting failed for {mouse_id} {signal_name}: {fit_error}")
                continue

            coefficients    = model.coef_
            n_nonzero_coefs = np.sum(np.abs(coefficients) > 1e-8)
            sparsity_ratio  = (1 - (n_nonzero_coefs / len(coefficients))
                               if len(coefficients) > 0 else 0.0)
            # Reconstruct smooth kernels by multiplying spline coef by basis matrix
            kernels = _kernels_from_spline_coef(
                coefficients, B_fit,
                len(behavioral_features), total_kernel_frames, offset)

            # Reconstruct the full concatenated signal (including NaN regions)
            reconstructed_concat = X @ coefficients + model.intercept_
            # Re-apply NaN mask to reconstruction so NaN regions stay NaN
            reconstructed_concat[~valid_indices] = np.nan

            # Split reconstruction back into per-session segments
            cursor = 0
            for i, sd in enumerate(valid_sessions_data_sorted):
                session_id = sd['session_id']
                seg_len = session_lengths[i]

                if cursor + seg_len > len(reconstructed_concat):
                    if verbose:
                        print(f"Warning: cursor ({cursor}) + seg_len ({seg_len}) exceeds "
                              f"reconstructed_concat length ({len(reconstructed_concat)}) "
                              f"for session {session_id}")
                    seg_len = max(0, len(reconstructed_concat) - cursor)

                reconstructed_segment = reconstructed_concat[cursor:cursor + seg_len]

                cursor += seg_len
                if i < len(valid_sessions_data_sorted) - 1:
                    cursor += buffer_frames

                if len(reconstructed_segment) == 0:
                    continue

                # Get the original signal at original FPS for length matching and NaN masking
                original_photometry_row = photometry_df.loc[session_id]
                original_signal_data = _extract_signal_array(original_photometry_row, signal_name)

                if target_fps is not None and effective_fps != original_fps:
                    reconstructed_at_original_fps = upsample_signal(
                        reconstructed_segment, effective_fps, original_fps
                    )
                    if original_signal_data is not None:
                        orig_len = len(original_signal_data)
                        if len(reconstructed_at_original_fps) > orig_len:
                            reconstructed_at_original_fps = reconstructed_at_original_fps[:orig_len]
                        elif len(reconstructed_at_original_fps) < orig_len:
                            pad_len = orig_len - len(reconstructed_at_original_fps)
                            reconstructed_at_original_fps = np.pad(
                                reconstructed_at_original_fps,
                                (0, pad_len),
                                mode='constant',
                                constant_values=np.nan
                            )
                        # Re-apply NaN mask from original data at original FPS
                        original_nan_mask = np.isnan(original_signal_data)
                        reconstructed_at_original_fps[original_nan_mask] = np.nan
                else:
                    reconstructed_at_original_fps = reconstructed_segment.copy()
                    if original_signal_data is not None:
                        orig_len = len(original_signal_data)
                        recon_len = len(reconstructed_at_original_fps)
                        if recon_len > orig_len:
                            reconstructed_at_original_fps = reconstructed_at_original_fps[:orig_len]
                        elif recon_len < orig_len:
                            pad_len = orig_len - recon_len
                            reconstructed_at_original_fps = np.pad(
                                reconstructed_at_original_fps,
                                (0, pad_len),
                                mode='constant',
                                constant_values=np.nan
                            )
                        # Re-apply NaN mask from original
                        original_nan_mask = np.isnan(original_signal_data)
                        check_len = min(len(reconstructed_at_original_fps), len(original_nan_mask))
                        reconstructed_at_original_fps[:check_len][original_nan_mask[:check_len]] = np.nan

                if session_id not in per_session_reconstructions:
                    per_session_reconstructions[session_id] = {}
                per_session_reconstructions[session_id][signal_name] = reconstructed_at_original_fps

            recon_data[(mouse_id, signal_name)] = {
                'design_matrix': X,           # spline design matrix (n_time × n_feat*ns*n_evt)
                'original_signal': signal_data.copy(),
                'valid_indices': valid_indices.copy(),
                'coefficients': coefficients.copy(),   # spline coefficients
                'intercept': model.intercept_,
                'effective_fps': effective_fps,
                'original_fps': original_fps,
                'group_id': mouse_id,
                'signal_name': signal_name,
                'behavioral_features': list(behavioral_features),
                'behavioral_signals': {                # concatenated binary signals per feature
                    feat: all_behavioral_data[feat].copy()
                    for feat in behavioral_features
                },
                'spline_basis': B_fit.copy(),          # (total_kernel_frames, ns)
                'kernel_length_frames': kernel_length_frames,
                'pre_onset_frames': pre_onset_frames,
                'total_kernel_frames': total_kernel_frames,
                'offset_included': offset,
                'mse_score': model_score,
                'n_sessions': len(valid_sessions_data_sorted),
                'session_ids': group_session_ids,
                'buffer_frames': buffer_frames,
                'n_valid_timepoints': int(n_valid),
                'n_total_timepoints': int(n_total),
                'n_nan_timepoints': int(n_nan),
            }

            total_events_per_feature = {}
            for feature_name in behavioral_features:
                if feature_name in all_behavioral_data:
                    onsets, offsets_det = detect_onsets_and_offsets(all_behavioral_data[feature_name])
                    total_events_per_feature[feature_name] = {
                        'onsets': len(onsets), 'offsets': len(offsets_det)
                    }
                else:
                    total_events_per_feature[feature_name] = {'onsets': 0, 'offsets': 0}

            for feature_idx, feature_name in enumerate(behavioral_features):
                feature_kernels = kernels[feature_idx, :]
                onset_kernel = feature_kernels[:total_kernel_frames]
                offset_kernel = feature_kernels[total_kernel_frames:] if offset else None
                n_onsets = total_events_per_feature[feature_name]['onsets']
                n_offsets = total_events_per_feature[feature_name]['offsets']

                result_dict = {
                    'session_id': mouse_id,
                    'group_type': 'mouse',
                    'signal_name': signal_name,
                    'behavioral_feature': feature_name,
                    'event_type': 'onset',
                    'kernel': onset_kernel,
                    'kernel_length_sec': kernel_length_sec,
                    'pre_onset_sec': pre_onset_sec,
                    'total_kernel_sec': pre_onset_sec + kernel_length_sec,
                    'original_fps': original_fps,
                    'effective_fps': effective_fps,
                    'n_events': n_onsets,
                    'mse_score': model_score,
                    'intercept': model.intercept_,
                    'n_predictors': n_predictors,
                    'predictor_ratio': predictor_ratio,
                    'regularization': regularization,
                    'alpha': chosen_alpha,
                    'cv_alphas': np.array(alpha_list),
                    'cv_mse_scores': cv_mse_arr.copy(),
                    'offset_included': offset,
                    'n_sessions': len(valid_sessions_data_sorted),
                    'n_valid_timepoints': int(n_valid),
                    'n_nan_timepoints': int(n_nan),
                    'pct_valid': 100.0 * n_valid / n_total if n_total > 0 else 0.0,
                }
                if regularization == 'elastic_net':
                    result_dict.update({
                        'l1_ratio': l1_ratio,
                        'n_nonzero_coefs': n_nonzero_coefs,
                        'sparsity_ratio': sparsity_ratio
                    })
                elif regularization == 'lasso':
                    result_dict.update({
                        'n_nonzero_coefs': n_nonzero_coefs,
                        'sparsity_ratio': sparsity_ratio
                    })
                results.append(result_dict)

                if offset and offset_kernel is not None:
                    offset_result_dict = result_dict.copy()
                    offset_result_dict.update({
                        'event_type': 'offset',
                        'kernel': offset_kernel,
                        'n_events': n_offsets
                    })
                    results.append(offset_result_dict)

            if verbose:
                sparsity_info = (f", Sparsity: {sparsity_ratio:.3f}"
                                if regularization in ['elastic_net', 'lasso'] else "")
                offset_info = " (onsets+offsets)" if offset else ""
                pct_valid = 100.0 * n_valid / n_total if n_total > 0 else 0.0
                print(f"  Fitted {mouse_id} {signal_name}: R² = {model_score:.3f}, "
                    f"FPS: {original_fps}->{effective_fps}, Predictors: {n_predictors}, "
                    f"alpha: {chosen_alpha}{sparsity_info}{offset_info} "
                    f"({len(valid_sessions_data_sorted)} sessions, "
                    f"buffer: {buffer_frames} frames, "
                    f"valid: {pct_valid:.1f}%)")

        return results, recon_data, per_session_reconstructions

    # ---- Main fitting loop (always by mouse) ----
    results = []
    all_reconstruction_data = {}
    all_per_session_reconstructions = {}

    for mouse_id, mouse_session_list in mouse_sessions.items():
        if verbose:
            print(f"\n--- Mouse: {mouse_id} ---")

        mouse_sessions_data_dict = {}
        mouse_sessions_data_list = []
        for session_id in mouse_session_list:
            session_data = process_session_data(session_id)
            mouse_sessions_data_dict[session_id] = session_data
            mouse_sessions_data_list.append(session_data)

        valid_sessions_data = [sd for sd in mouse_sessions_data_list if sd is not None]
        valid_sessions_data_sorted = sorted(valid_sessions_data, key=lambda x: x['session_id'])
        if not valid_sessions_data_sorted:
            if verbose:
                print(f"  No valid sessions for mouse {mouse_id}, skipping.")
            continue

        effective_fps = valid_sessions_data_sorted[0]['effective_fps']
        kernel_length_frames = int(kernel_length_sec * effective_fps)
        pre_onset_frames = int(pre_onset_sec * effective_fps)
        total_kernel_frames = pre_onset_frames + kernel_length_frames

        # Determine alpha (and CV MSE scores) per signal
        signal_alphas = {}
        signal_cv_scores = {}

        for signal_name in photometry_signals:
            if do_cv:
                if verbose:
                    print(f"  Selecting alpha for signal: {signal_name}")
                chosen_alpha, cv_mean_scores = cv_select_alpha_for_signal(
                    mouse_sessions_data_dict,
                    mouse_session_list,
                    signal_name,
                    alpha_list,
                    n_folds,
                    effective_fps,
                    kernel_length_frames,
                    pre_onset_frames,
                    total_kernel_frames
                )
                signal_cv_scores[signal_name] = np.array(
                    [cv_mean_scores[a] for a in alpha_list]
                )
            else:
                chosen_alpha = alpha_list[0]
                signal_cv_scores[signal_name] = np.array([np.nan] * len(alpha_list))

            signal_alphas[signal_name] = chosen_alpha

        mouse_results, mouse_recon, mouse_per_recon = fit_model_for_mouse(
            mouse_sessions_data_list, mouse_id, signal_alphas, signal_cv_scores
        )
        results.extend(mouse_results)
        all_reconstruction_data.update(mouse_recon)
        for sid, sig_dict in mouse_per_recon.items():
            if sid not in all_per_session_reconstructions:
                all_per_session_reconstructions[sid] = {}
            all_per_session_reconstructions[sid].update(sig_dict)

    results_df = pd.DataFrame(results)

    # --- Build reconstructed_photometry_df mirroring photometry_df structure ---
    reconstructed_photometry_df = photometry_df.copy()

    for signal_name in photometry_signals:
        if signal_name in reconstructed_photometry_df.columns:
            reconstructed_photometry_df[signal_name] = reconstructed_photometry_df[signal_name].astype(object)

    # Initialize all filtered sessions' photometry signals to NaN arrays matching original lengths
    for session_id in session_filtered:
        if session_id in reconstructed_photometry_df.index:
            for signal_name in photometry_signals:
                original_val = photometry_df.loc[session_id, signal_name]
                if isinstance(original_val, np.ndarray):
                    set_cell_value(reconstructed_photometry_df, session_id, signal_name,
                                np.full(len(original_val), np.nan))
                else:
                    # Handle scalar or non-array values
                    reconstructed_photometry_df.at[session_id, signal_name] = np.nan

    # Fill in the reconstructed traces
    for session_id, sig_dict in all_per_session_reconstructions.items():
        if session_id in reconstructed_photometry_df.index:
            for signal_name, recon_array in sig_dict.items():
                set_cell_value(reconstructed_photometry_df, session_id, signal_name, recon_array)

    # Sessions not in session_filtered should have NaN signals
    for session_id in photometry_df.index:
        if session_id not in session_filtered:
            for signal_name in photometry_signals:
                original_val = photometry_df.loc[session_id, signal_name]
                if isinstance(original_val, np.ndarray):
                    set_cell_value(reconstructed_photometry_df, session_id, signal_name,
                                np.full(len(original_val), np.nan))
                else:
                    reconstructed_photometry_df.at[session_id, signal_name] = np.nan

    if len(results_df) > 0:
        n_mice = results_df['session_id'].nunique()
        print(f"\nCompleted fitting for {len(results_df)} combinations across {n_mice} mice")
        print(f"Mean MSE score: {results_df['mse_score'].mean():.3f}")
        print(f"Mean predictor ratio: {results_df['predictor_ratio'].mean():.3f}")
        if 'pct_valid' in results_df.columns:
            print(f"Mean valid data: {results_df['pct_valid'].mean():.1f}%")
        if 'sparsity_ratio' in results_df.columns:
            print(f"Mean sparsity ratio: {results_df['sparsity_ratio'].mean():.3f}")
        if target_fps is not None:
            print(f"Resampling: {results_df['original_fps'].mean():.1f} -> "
                f"{results_df['effective_fps'].mean():.1f} FPS")
        onset_data = results_df[results_df['event_type'] == 'onset']
        if len(onset_data) > 0:
            print(f"Mean onsets per mouse: {onset_data['n_events'].mean():.1f}")
        if offset:
            offset_data = results_df[results_df['event_type'] == 'offset']
            if len(offset_data) > 0:
                print(f"Mean offsets per mouse: {offset_data['n_events'].mean():.1f}")
        print(f"Reconstruction data stored for {len(all_reconstruction_data)} "
            f"(mouse, signal) combinations")
        n_reconstructed_sessions = len(all_per_session_reconstructions)
        n_reconstructed_signals = sum(len(v) for v in all_per_session_reconstructions.values())
        print(f"Reconstructed photometry DataFrame: {n_reconstructed_sessions} sessions, "
            f"{n_reconstructed_signals} signal traces reconstructed")
        if do_cv:
            unique_alphas = results_df['alpha'].unique()
            print(f"Alpha(s) selected via CV: {unique_alphas}")
    else:
        print(f"\nNo results generated")

    return results_df, all_reconstruction_data, reconstructed_photometry_df

def plot_encoding_kernels_overlaid(
    kernels_df: pd.DataFrame,
    behavioral_feature_name: str,
    window_size_sec: list = [2.0, 10.0],
    figsize: tuple = (15, 4),
    verbose: bool = True,
    days_to_include: list = None,
    mice_included: list = None,
    peak_arrows: bool = False,
    stats_correction_type: str = "fdr_bh",
    min_events_threshold: int = 1,
    offset: bool = False,
    show_title: bool = True,
    save_fig: bool = False
) -> None:
    """
    Plot encoding kernels for a specific behavioral feature, overlaying DA and 5-HT.
    Updated to handle mouse ID extraction from the 'session_id' column.
    
    Args:
        kernels_df (pd.DataFrame): DataFrame containing kernel data.
        behavioral_feature_name (str): Name of the feature to plot.
        window_size_sec (list): Time window [start, end] in seconds.
        figsize (tuple): Figure dimensions.
        verbose (bool): Print statistics.
        days_to_include (list): List of days (integers) to include.
        mice_included (list, optional): List of mouse IDs to include.
        peak_arrows (bool): Draw arrows at peak locations.
        stats_correction_type (str): Method for multiple comparison correction.
        min_events_threshold (int): Minimum events required to include a kernel.
        offset (bool): Whether to plot offset kernels.
        show_title (bool): Whether to display the title.
        save_fig (bool): Whether to save the figure to disk.
    """
    
    def round_to_nice_limits(data_min, data_max, num_ticks=5):
        """Calculate nice, round y-axis limits with minimal decimal places."""
        if data_min == data_max:
            if data_min == 0:
                return -1, 1
            else:
                margin = abs(data_min) * 0.1
                return data_min - margin, data_max + margin
        
        data_range = data_max - data_min
        margin = data_range * 0.05
        extended_min = data_min - margin
        extended_max = data_max + margin
        
        extended_range = extended_max - extended_min
        raw_step = extended_range / (num_ticks - 1)
        
        magnitude = 10 ** np.floor(np.log10(abs(raw_step)))
        normalized_step = raw_step / magnitude
        
        if normalized_step <= 1:
            nice_step = 1 * magnitude
        elif normalized_step <= 2:
            nice_step = 2 * magnitude
        elif normalized_step <= 5:
            nice_step = 5 * magnitude
        else:
            nice_step = 10 * magnitude
        
        nice_min = np.floor(extended_min / nice_step) * nice_step
        nice_max = np.ceil(extended_max / nice_step) * nice_step
        
        return nice_min, nice_max
    
    # Set style
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 10,
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'grid.alpha': 0.0,
        'legend.frameon': False
    })

    # Ensure window start is negative if positive was passed
    if window_size_sec[0] > 0:
        window_size_sec[0] = -window_size_sec[0]

    # Filter for the specific behavioral feature
    feature_data = kernels_df[kernels_df['behavioral_feature'] == behavioral_feature_name].copy()
    
    if len(feature_data) == 0:
        print(f"No data found for behavioral feature: {behavioral_feature_name}")
        return
    
    # --- UPDATED FILTERING LOGIC FOR MICE ---
    if mice_included is not None:
        # Strategy 1: Check if 'mouse_id' column exists directly
        if 'mouse_id' in feature_data.columns:
            feature_data = feature_data[feature_data['mouse_id'].isin(mice_included)]
        
        # Strategy 2: Extract from 'session_id' if 'mouse_id' column is missing or empty
        else:
            # Create a temporary column for filtering
            def extract_mouse_id(sid):
                sid_str = str(sid)
                # Check if the session_id matches one of the mice directly
                if sid_str in mice_included:
                    return sid_str
                
                # Check standard format: CSDS-Day1-A_1-Defeat (Mouse ID is index 2)
                parts = sid_str.split('-')
                if len(parts) > 2:
                    potential_id = parts[2]
                    if potential_id in mice_included:
                        return potential_id
                
                # Fallback: check if any mouse ID appears in the session ID
                # as a whole token (bounded by delimiters or start/end of string)
                for m_id in mice_included:
                    # Use regex with word boundaries adapted for IDs containing underscores
                    # Match m_id only when it's surrounded by '-', start, or end of string
                    pattern = r'(?:^|-)' + re.escape(m_id) + r'(?:-|$)'
                    if re.search(pattern, sid_str):
                        return m_id
                return None

            feature_data['temp_mouse_filter'] = feature_data['session_id'].apply(extract_mouse_id)
            feature_data = feature_data[feature_data['temp_mouse_filter'].notna()]
            feature_data = feature_data.drop(columns=['temp_mouse_filter'])
        
    # ------------------------------------
        
    # ------------------------------------
    
    # Check fitting strategy
    group_type = feature_data['group_type'].iloc[0] if 'group_type' in feature_data.columns else 'session'
    
    # Check if offsets were included in the model fitting
    offset_included = feature_data['offset_included'].iloc[0] if 'offset_included' in feature_data.columns else False
    
    # Handle offset plotting logic
    if offset and not offset_included:
        print(f"Error: Offset plotting requested but model was not fitted with offsets for {behavioral_feature_name}")
        return
    
    if not offset and offset_included:
        print(f"Warning: Model was fitted with onsets and offsets, but only plotting onsets for {behavioral_feature_name}")
        feature_data = feature_data[feature_data['event_type'] == 'onset']
    
    # Filter by days if specified (only for session-level fitting)
    if days_to_include is not None and group_type == "session":
        filtered_sessions = []
        for session in feature_data['session_id'].unique():
            parts = str(session).split('-')
            # Check index 1 for Day info (e.g., Day1)
            if len(parts) >= 2 and parts[1].startswith('Day'):
                try:
                    day = int(parts[1][3:])
                    if day in days_to_include:
                        filtered_sessions.append(session)
                except ValueError:
                    continue
        feature_data = feature_data[feature_data['session_id'].isin(filtered_sessions)]
    elif days_to_include is not None and group_type != "session":
        print(f"Warning: days_to_include parameter is ignored for {group_type}-level fitting")
    
    # Filter by minimum events
    feature_data = feature_data[feature_data['n_events'] >= min_events_threshold]
    
    if len(feature_data) == 0:
        print(f"No data found after filtering for {behavioral_feature_name}")
        return
    
    # Determine what event types we have
    event_types = feature_data['event_type'].unique() if 'event_type' in feature_data.columns else ['onset']
    plot_onsets = 'onset' in event_types
    plot_offsets = offset and 'offset' in event_types
    
    if verbose:
        print(f"Found {len(feature_data)} kernels for {behavioral_feature_name}")
        print(f"Fitting level: {group_type}")
        print(f"Event types: {list(event_types)}")
    
    # Get photometry signals
    da_signals = [s for s in feature_data['signal_name'].unique() if '560' in s]
    serotonin_signals = [s for s in feature_data['signal_name'].unique() if '470' in s]
    
    # Group by brain area
    areas = set()
    for signal in feature_data['signal_name'].unique():
        area = signal.split('_')[0]
        areas.add(area)
    
    area_order = ['NAc', 'BLA', 'mPFC']
    ordered_areas = [area for area in area_order if area in areas]
    
    # Determine subplot layout
    n_rows = 1
    if plot_onsets and plot_offsets:
        n_rows = 2
        figsize = (figsize[0], figsize[1] * 2)
    
    # Create figure
    fig, axes = plt.subplots(n_rows, len(ordered_areas), figsize=figsize, 
                            sharex=True, sharey=True)
    
    # Handle axis indexing properly
    if n_rows == 1 and len(ordered_areas) == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [list(axes)]
    elif len(ordered_areas) == 1:
        axes = [[ax] for ax in axes]
    
    # Create title
    title = f'Encoding Kernels for {behavioral_feature_name.replace("_", " ").title()}'
    title += f' ({group_type.title()}-level fitting)'
    if show_title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Determine time vector based on kernel parameters
    kernel_length_sec = feature_data['kernel_length_sec'].iloc[0]
    pre_onset_sec = feature_data['pre_onset_sec'].iloc[0]
    effective_fps = feature_data['effective_fps'].mean()
    
    kernel_length_frames = int(kernel_length_sec * effective_fps)
    pre_onset_frames = int(pre_onset_sec * effective_fps)
    total_kernel_frames = pre_onset_frames + kernel_length_frames
    
    time_vector = np.linspace(-pre_onset_sec, kernel_length_sec, total_kernel_frames)
    
    # Find window indices
    window_start_idx = np.argmin(np.abs(time_vector - window_size_sec[0]))
    window_end_idx = np.argmin(np.abs(time_vector - window_size_sec[1]))
    if window_end_idx >= len(time_vector):
        window_end_idx = len(time_vector) - 1
    
    time_vector_windowed = time_vector[window_start_idx:window_end_idx+1]
    
    def plot_event_kernels(event_type, row_idx):
        event_data = feature_data[feature_data['event_type'] == event_type] if 'event_type' in feature_data.columns else feature_data
        
        # Calculate global y-limits
        all_data_values = []
        for area in ordered_areas:
            da_signal = f"{area}_560_dFF"
            serotonin_signal = f"{area}_470_dFF"
            
            da_kernels = event_data[event_data['signal_name'] == da_signal]['kernel'].values
            serotonin_kernels = event_data[event_data['signal_name'] == serotonin_signal]['kernel'].values
            
            if len(da_kernels) > 0:
                da_array = np.array([kernel[window_start_idx:window_end_idx+1] for kernel in da_kernels])
                if group_type == "all":
                    all_data_values.extend(da_kernels[0][window_start_idx:window_end_idx+1])
                else:
                    mean_da = np.mean(da_array, axis=0)
                    all_data_values.extend(mean_da)
                    if len(da_kernels) > 1:
                        sem_da = sem(da_array, axis=0)
                        all_data_values.extend(mean_da - sem_da)
                        all_data_values.extend(mean_da + sem_da)
            
            if len(serotonin_kernels) > 0:
                serotonin_array = np.array([kernel[window_start_idx:window_end_idx+1] for kernel in serotonin_kernels])
                if group_type == "all":
                    all_data_values.extend(serotonin_kernels[0][window_start_idx:window_end_idx+1])
                else:
                    mean_5ht = np.mean(serotonin_array, axis=0)
                    all_data_values.extend(mean_5ht)
                    if len(serotonin_kernels) > 1:
                        sem_5ht = sem(serotonin_array, axis=0)
                        all_data_values.extend(mean_5ht - sem_5ht)
                        all_data_values.extend(mean_5ht + sem_5ht)

        if all_data_values:
            global_min = np.min(all_data_values)
            global_max = np.max(all_data_values)
            nice_ymin, nice_ymax = round_to_nice_limits(global_min, global_max)
        else:
            nice_ymin, nice_ymax = -1, 1

        stats_results = {}
        
        for i, area in enumerate(ordered_areas):
            ax = axes[row_idx][i]
            da_signal = f"{area}_560_dFF"
            serotonin_signal = f"{area}_470_dFF"
            
            da_kernels = event_data[event_data['signal_name'] == da_signal]['kernel'].values
            serotonin_kernels = event_data[event_data['signal_name'] == serotonin_signal]['kernel'].values
            
            mean_traces = {}
            
            # Plot DA
            if len(da_kernels) > 0:
                if group_type == "all":
                    da_trace = da_kernels[0][window_start_idx:window_end_idx+1]
                    mean_traces['DA'] = {'trace': da_trace, 'color': 'red'}
                    ax.plot(time_vector_windowed, da_trace, color='red', linewidth=2, label=f'DA')
                else:
                    da_array = np.array([kernel[window_start_idx:window_end_idx+1] for kernel in da_kernels])
                    mean_da = np.mean(da_array, axis=0)
                    sem_da = sem(da_array, axis=0)
                    mean_traces['DA'] = {'trace': mean_da, 'color': 'red'}
                    ax.plot(time_vector_windowed, mean_da, color='red', linewidth=2, label=f'DA')
                    if len(da_kernels) > 1:
                        ax.fill_between(time_vector_windowed, mean_da - sem_da, mean_da + sem_da, color='red', alpha=0.2)
            
            # Plot 5-HT
            if len(serotonin_kernels) > 0:
                if group_type == "all":
                    sht_trace = serotonin_kernels[0][window_start_idx:window_end_idx+1]
                    mean_traces['5-HT'] = {'trace': sht_trace, 'color': 'blue'}
                    ax.plot(time_vector_windowed, sht_trace, color='blue', linewidth=2, label=f'5-HT')
                else:
                    serotonin_array = np.array([kernel[window_start_idx:window_end_idx+1] for kernel in serotonin_kernels])
                    mean_5ht = np.mean(serotonin_array, axis=0)
                    sem_5ht = sem(serotonin_array, axis=0)
                    mean_traces['5-HT'] = {'trace': mean_5ht, 'color': 'blue'}
                    ax.plot(time_vector_windowed, mean_5ht, color='blue', linewidth=2, label=f'5-HT')
                    if len(serotonin_kernels) > 1:
                        ax.fill_between(time_vector_windowed, mean_5ht - sem_5ht, mean_5ht + sem_5ht, color='blue', alpha=0.2)
            
            # Stats
            if (stats_correction_type is not False and 
                len(da_kernels) > 1 and len(serotonin_kernels) > 1 and 
                group_type in ["session", "mouse"]):
                
                da_array = np.array([kernel[window_start_idx:window_end_idx+1] for kernel in da_kernels])
                serotonin_array = np.array([kernel[window_start_idx:window_end_idx+1] for kernel in serotonin_kernels])
                
                p_values = []
                for frame in range(len(time_vector_windowed)):
                    da_frame_data = da_array[:, frame]
                    sht_frame_data = serotonin_array[:, frame]
                    if len(da_frame_data) > 1 and len(sht_frame_data) > 1:
                        _, p_val = ttest_ind(da_frame_data, sht_frame_data)
                        p_values.append(p_val)
                    else:
                        p_values.append(1.0)
                
                if stats_correction_type is None:
                    rejected = [p < 0.05 for p in p_values]
                else:
                    if len(p_values) > 0:
                        rejected, _, _, _ = multipletests(p_values, alpha=0.05, method=stats_correction_type)
                    else:
                        rejected = []
                
                if len(p_values) > 0:
                    stats_results[i] = {'rejected': rejected, 'ax': ax}
            
            # Peak arrows
            if peak_arrows:
                current_ylim = ax.get_ylim()
                spine_y = current_ylim[1]
                for signal_name, signal_data in mean_traces.items():
                    trace = signal_data['trace']
                    color = signal_data['color']
                    if not np.all(np.isnan(trace)) and len(trace) > 0:
                        peak_idx = np.nanargmax(trace)
                        peak_time = time_vector_windowed[peak_idx]
                        arrow_height = 0.05 * (current_ylim[1] - current_ylim[0])
                        ax.annotate('', xy=(peak_time, spine_y), xytext=(peak_time, spine_y + arrow_height),
                                    arrowprops=dict(arrowstyle='->', color=color, lw=2), annotation_clip=False)
            
            # Formatting
            # ax.axvline(0, color='black', linestyle='--', alpha=1)
            if n_rows == 1:
                ax.set_title(f'{area}', fontweight='bold')
            else:
                if row_idx == 0:
                    ax.set_title(f'{area}', fontweight='bold')
                if i == 0:
                    ax.set_ylabel(f'{event_type.title()}\nKernel Weight', fontweight='bold')
            
            ax.grid(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_position(('outward', 10))
            ax.spines['bottom'].set_position(('outward', 10))
            
            if row_idx == n_rows - 1:
                ax.set_xlabel('Time from event (s)', fontweight='bold')
            if i == 0 and n_rows == 1:
                ax.set_ylabel('Kernel Weight', fontweight='bold')
            
            ax.legend(frameon=False)
            ax.set_xlim(window_size_sec[0], window_size_sec[1])
            ax.set_ylim(nice_ymin, nice_ymax)
    
        # Plot significance bars
        bar_height = nice_ymax
        if stats_results:
            for ax_idx, stats_data in stats_results.items():
                ax = stats_data['ax']
                rejected = stats_data['rejected']
                for frame, reject in enumerate(rejected):
                    if reject:
                        bar_bottom = bar_height
                        bar_top = bar_height + 0.02 * (nice_ymax - nice_ymin)
                        ax.plot([time_vector_windowed[frame], time_vector_windowed[frame]], 
                            [bar_bottom, bar_top], color='black', linewidth=2, clip_on=False)

        # Final spine adjustments
        for i, area in enumerate(ordered_areas):
            ax = axes[row_idx][i]
            ax.set_ylim(nice_ymin, nice_ymax)
            ax.spines['left'].set_bounds(nice_ymin, nice_ymax)
            ax.spines['bottom'].set_bounds(window_size_sec[0], window_size_sec[1])
    
    if plot_onsets:
        plot_event_kernels('onset', 0)
    if plot_offsets:
        plot_event_kernels('offset', 1)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_fig:
        plt.savefig(f"defeat_kernel_comparison_{behavioral_feature_name}", dpi=300, format="pdf")
    else:
        plt.show()

def plot_kernel_window_comparison(
    kernels_df: pd.DataFrame,
    behavioral_feature_name: str,
    pre_window: list = [-15, -5],
    event_window: list = [0, 10],
    post_window: list = [20, 30],
    figsize: tuple = (15, 6),
    verbose: bool = True,
    days_to_include: list = None,
    stats_correction_type: str = "fdr_bh",
    min_events_threshold: int = 0,
    offset: bool = False
) -> None:
    """
    Plot bar plots comparing mean kernel weights between DA and 5-HT across different time windows.
    Updated for session format: CSDS-Day1-A_1-Defeat
    """
    
    # Set style.
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 10,
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': False,
        'axes.spines.bottom': False,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'grid.alpha': 0.3,
        'legend.frameon': False
    })
    
    # Filter for the specific behavioral feature
    feature_data = kernels_df[kernels_df['behavioral_feature'] == behavioral_feature_name].copy()
    
    if len(feature_data) == 0:
        print(f"No data found for behavioral feature: {behavioral_feature_name}")
        return
    
    # Check fitting strategy
    group_type = feature_data['group_type'].iloc[0] if 'group_type' in feature_data.columns else 'session'
    
    # Check if offsets were included in the model fitting
    offset_included = feature_data['offset_included'].iloc[0] if 'offset_included' in feature_data.columns else False
    
    # Handle offset plotting logic
    if offset and not offset_included:
        print(f"Error: Offset plotting requested but model was not fitted with offsets for {behavioral_feature_name}")
        return
    
    if not offset and offset_included:
        print(f"Warning: Model was fitted with onsets and offsets, but only plotting onsets for {behavioral_feature_name}")
        # Filter to only onsets
        feature_data = feature_data[feature_data['event_type'] == 'onset']
    
    # Filter by days if specified (only for session-level fitting)
    # Format: CSDS-Day1-A_1-Defeat
    if days_to_include is not None and group_type == "session":
        filtered_sessions = []
        for session in feature_data['session_id'].unique():
            parts = session.split('-')
            # Check index 1 for Day info
            if len(parts) >= 2 and parts[1].startswith('Day'):
                try:
                    day = int(parts[1][3:])
                    if day in days_to_include:
                        filtered_sessions.append(session)
                except ValueError:
                    continue
        feature_data = feature_data[feature_data['session_id'].isin(filtered_sessions)]
    elif days_to_include is not None and group_type != "session":
        print(f"Warning: days_to_include parameter is ignored for {group_type}-level fitting")
    
    # Filter by minimum events
    feature_data = feature_data[feature_data['n_events'] >= min_events_threshold]
    
    if len(feature_data) == 0:
        print(f"No data found after filtering for {behavioral_feature_name}")
        return
    
    # Determine what event types we have
    event_types = feature_data['event_type'].unique() if 'event_type' in feature_data.columns else ['onset']
    plot_onsets = 'onset' in event_types
    plot_offsets = offset and 'offset' in event_types
    
    # Determine kernel parameters
    kernel_length_sec = feature_data['kernel_length_sec'].iloc[0]
    pre_onset_sec = feature_data['pre_onset_sec'].iloc[0]
    effective_fps = feature_data['effective_fps'].mean()
    
    kernel_length_frames = int(kernel_length_sec * effective_fps)
    pre_onset_frames = int(pre_onset_sec * effective_fps)
    total_kernel_frames = pre_onset_frames + kernel_length_frames
    
    # Create time vector (starts at -pre_onset_sec, goes to +kernel_length_sec)
    time_vector = np.linspace(-pre_onset_sec, kernel_length_sec, total_kernel_frames)
    
    # Define time windows and their indices
    windows = {
        'Pre': pre_window,
        'Event': event_window,
        'Post': post_window
    }
    
    # Find window indices
    window_indices = {}
    for window_name, window_range in windows.items():
        start_idx = np.argmin(np.abs(time_vector - window_range[0]))
        end_idx = np.argmin(np.abs(time_vector - window_range[1]))
        if end_idx >= len(time_vector):
            end_idx = len(time_vector) - 1
        window_indices[window_name] = (start_idx, end_idx)
    
    # Get brain areas
    areas = set()
    for signal in feature_data['signal_name'].unique():
        area = signal.split('_')[0]
        areas.add(area)
    
    area_order = ['NAc', 'BLA', 'mPFC']
    ordered_areas = [area for area in area_order if area in areas]
    
    # Determine subplot layout
    n_rows = 1
    if plot_onsets and plot_offsets:
        n_rows = 2
        # Adjust figure size for multiple rows
        figsize = (figsize[0], figsize[1] * 2)
    
    # Create figure - FIX 1: Set sharey=False to allow independent y-axes
    fig, axes = plt.subplots(n_rows, len(ordered_areas), figsize=figsize, 
                            sharex=True, sharey=True)
    
    # Handle axis indexing properly
    if n_rows == 1 and len(ordered_areas) == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [list(axes)]
    elif len(ordered_areas) == 1:
        axes = [[ax] for ax in axes]
    
    # Create title
    title = f'Kernel Window Analysis for {behavioral_feature_name.replace("_", " ").title()}'
    title += f' ({group_type.title()}-level fitting)'
    if 'regularization' in feature_data.columns:
        reg_type = feature_data['regularization'].iloc[0]
        if reg_type == 'elastic_net' and 'l1_ratio' in feature_data.columns:
            l1_ratio_val = feature_data['l1_ratio'].iloc[0]
            title += f' (Elastic Net, L1={l1_ratio_val})'
        else:
            title += f' ({reg_type.replace("_", " ").title()})'
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    def compute_window_means(kernels, window_indices):
        """Compute mean kernel weights for each window"""
        window_means = {}
        for window_name, (start_idx, end_idx) in window_indices.items():
            means = []
            for kernel in kernels:
                window_data = kernel[start_idx:end_idx+1]
                means.append(np.mean(window_data))
            window_means[window_name] = np.array(means)
        return window_means
    
    def plot_event_window_analysis(event_type, row_idx):
        # Filter data for this event type
        event_data = feature_data[feature_data['event_type'] == event_type] if 'event_type' in feature_data.columns else feature_data
        
        # Store all p-values for multiple comparison correction
        all_p_values = []
        test_results = []
        
        for i, area in enumerate(ordered_areas):
            ax = axes[row_idx][i]
            
            da_signal = f"{area}_560_dFF"
            serotonin_signal = f"{area}_470_dFF"
            
            # Get kernels for this area and event type
            da_kernels = event_data[event_data['signal_name'] == da_signal]['kernel'].values
            serotonin_kernels = event_data[event_data['signal_name'] == serotonin_signal]['kernel'].values
            
            if len(da_kernels) == 0 or len(serotonin_kernels) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                if n_rows == 1:
                    ax.set_title(f'{area}', fontweight='bold')
                elif row_idx == 0:
                    ax.set_title(f'{area}', fontweight='bold')
                continue
            
            # Compute window means
            da_window_means = compute_window_means(da_kernels, window_indices)
            sht_window_means = compute_window_means(serotonin_kernels, window_indices)
            
            # Prepare data for plotting
            window_names = list(windows.keys())
            x_positions = np.arange(len(window_names))
            width = 0.35
            
            # Compute means and SEMs
            da_means = [np.mean(da_window_means[w]) for w in window_names]
            da_sems = [np.std(da_window_means[w]) / np.sqrt(len(da_window_means[w])) for w in window_names]
            
            sht_means = [np.mean(sht_window_means[w]) for w in window_names]
            sht_sems = [np.std(sht_window_means[w]) / np.sqrt(len(sht_window_means[w])) for w in window_names]
            
            # Plot bars
            da_bars = ax.bar(x_positions - width/2, da_means, width, 
                           yerr=da_sems, capsize=5, color='red', alpha=0.7, 
                           label='DA', edgecolor='darkred')
            
            sht_bars = ax.bar(x_positions + width/2, sht_means, width, 
                            yerr=sht_sems, capsize=5, color='blue', alpha=0.7, 
                            label='5-HT', edgecolor='darkblue')
            
            # Perform statistical tests for each window
            window_p_values = []
            for j, window_name in enumerate(window_names):
                da_data = da_window_means[window_name]
                sht_data = sht_window_means[window_name]
                
                # Use unpaired t-test for comparing DA vs 5-HT signals
                if group_type in ["session", "mouse"] and len(da_data) > 1 and len(sht_data) > 1:
                    # Independent (unpaired) t-test
                    _, p_val = ttest_ind(da_data, sht_data)
                    test_type = "independent"
                else:
                    p_val = np.nan
                    test_type = "none"
                
                window_p_values.append(p_val)
                all_p_values.append(p_val)
                test_results.append({
                    'area': area,
                    'event_type': event_type,
                    'window': window_name,
                    'p_value': p_val,
                    'test_type': test_type,
                    'da_mean': da_means[j],
                    'sht_mean': sht_means[j],
                    'da_n': len(da_data),
                    'sht_n': len(sht_data),
                    'x_pos': x_positions[j],
                    'ax_idx': (row_idx, i)
                })
            
            # Formatting
            ax.set_xticks(x_positions)
            ax.set_xticklabels([f"{w}\n({windows[w][0]:.0f}s to {windows[w][1]:.0f}s)" 
                               for w in window_names])
            
            if n_rows == 1:
                ax.set_title(f'{area}', fontweight='bold')
            else:
                if row_idx == 0:
                    ax.set_title(f'{area}', fontweight='bold')
                if i == 0:
                    ax.set_ylabel(f'{event_type.title()}\nMean Kernel Weight', fontweight='bold')

            if i == 0 and n_rows == 1:
                ax.set_ylabel('Mean Kernel Weight', fontweight='bold')
            
            ax.grid(True, alpha=0.3, axis='y')
            ax.grid(False, axis='x')  # Explicitly disable vertical grid lines
            ax.legend(frameon=False)
            
            # Store for significance annotation
            ax._test_results = [(x_positions[j], window_p_values[j]) for j in range(len(window_names))]
        
        return test_results
    
    # Collect all test results
    all_test_results = []
    
    # Plot onsets
    if plot_onsets:
        onset_results = plot_event_window_analysis('onset', 0)
        all_test_results.extend(onset_results)
    
    # Plot offsets
    if plot_offsets:
        offset_results = plot_event_window_analysis('offset', 1)
        all_test_results.extend(offset_results)
    
    # Apply multiple comparison correction
    valid_p_values = [r['p_value'] for r in all_test_results if not np.isnan(r['p_value'])]
    
    if len(valid_p_values) > 0 and stats_correction_type is not None:
        rejected, corrected_p_values, _, _ = multipletests(valid_p_values, alpha=0.05, method=stats_correction_type)
        
        # Map corrected p-values back to results
        corrected_idx = 0
        for result in all_test_results:
            if not np.isnan(result['p_value']):
                result['corrected_p'] = corrected_p_values[corrected_idx]
                result['significant'] = rejected[corrected_idx]
                corrected_idx += 1
            else:
                result['corrected_p'] = np.nan
                result['significant'] = False
        
        # FIX 2: Add significance annotations at fixed positions relative to each subplot
        for result in all_test_results:
            if result['significant']:
                row_idx, col_idx = result['ax_idx']
                ax = axes[row_idx][col_idx]
                
                # Get current y-limits for this specific subplot
                y_min, y_max = ax.get_ylim()
                y_range = y_max - y_min
                
                # Position stars at a fixed relative position (90% of the way up)
                y_sig = y_min + 0.9 * y_range
                
                # Add asterisk at fixed position
                ax.text(result['x_pos'], y_sig, '*', ha='center', va='center', 
                       fontsize=16, fontweight='bold', color='black')
                
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    if verbose:
        print(f"\nWindow Analysis Results:")
        print(f"Behavioral feature: {behavioral_feature_name}")
        print(f"Fitting level: {group_type}")
        print(f"Time windows:")
        for window_name, window_range in windows.items():
            print(f"  {window_name}: {window_range[0]}s to {window_range[1]}s")
        
        print(f"Statistical correction: {stats_correction_type}")
        print(f"Minimum events threshold: {min_events_threshold}")
        
        # Print regularization details
        if 'regularization' in feature_data.columns:
            reg_type = feature_data['regularization'].iloc[0]
            alpha_val = feature_data['alpha'].iloc[0]
            print(f"Regularization: {reg_type} (alpha={alpha_val})")
        
        # Print statistical results
        print(f"\nStatistical Results:")
        print(f"{'Area':<6} {'Event':<7} {'Window':<8} {'DA Mean':<8} {'5-HT Mean':<10} {'p-value':<10} {'Corrected p':<12} {'Significant':<11}")
        print("-" * 80)
        
        for result in all_test_results:
            if not np.isnan(result['p_value']):
                corrected_p_str = f"{result.get('corrected_p', np.nan):.4f}" if 'corrected_p' in result else "N/A"
                significant_str = "Yes" if result.get('significant', False) else "No"
                print(f"{result['area']:<6} {result['event_type']:<7} {result['window']:<8} "
                      f"{result['da_mean']:<8.4f} {result['sht_mean']:<10.4f} "
                      f"{result['p_value']:<10.4f} {corrected_p_str:<12} {significant_str:<11}")
            else:
                print(f"{result['area']:<6} {result['event_type']:<7} {result['window']:<8} "
                      f"{result['da_mean']:<8.4f} {result['sht_mean']:<10.4f} "
                      f"{'N/A':<10} {'N/A':<12} {'N/A':<11}")
        
        # Print sample sizes
        print(f"\nSample Sizes:")
        for area in ordered_areas:
            for event_type in event_types:
                event_subset = feature_data[feature_data['event_type'] == event_type] if 'event_type' in feature_data.columns else feature_data
                
                da_signal = f"{area}_560_dFF"
                serotonin_signal = f"{area}_470_dFF"
                
                da_data = event_subset[event_subset['signal_name'] == da_signal]
                sht_data = event_subset[event_subset['signal_name'] == serotonin_signal]
                
                da_count = len(da_data)
                sht_count = len(sht_data)
                
                event_label = f" ({event_type})" if len(event_types) > 1 else ""
                group_label = {"session": "sessions", "mouse": "mice", "all": "datasets"}[group_type]
                print(f"{area:>6}{event_label:>8} | DA: {da_count:3d} {group_label} | 5-HT: {sht_count:3d} {group_label}")

