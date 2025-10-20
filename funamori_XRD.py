import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
from scipy.interpolate import interp1d
#from scipy.optimize import minimize
from lmfit import Parameters, minimize, fit_report
import corner

#### Functions -----------------------------------------------------

def Gaussian(x, x0, sigma):
    return np.exp(-0.5 * ((x - x0) / sigma) ** 2)

def compute_strain(hkl, intensity, a_val, wavelength, c11, c12, c44, sigma_11, sigma_22, sigma_33, phi_values, psi_values, symmetry):
    """
    Evaluates strain_33 component for given hkl reflection.
    
    Parameters
    ----------
    hkl : tuple
        Miller indices (h, k, l)
    a_val : float
        Lattice parameter
    wavelength : float
        X-ray wavelength
    c11, c12, c44 : float
        Elastic constants
    phi_values : np.array
        Array of phi values in radians
    psi_values : np.array or scalar
        Array of psi values in radians (or 0 to auto-calculate)
    symmetry : str
        Crystal symmetry
    sigma_11, sigma_22, sigma_33 : float
        Stress tensor components (default assumes uniaxial stress)
    intensity : float
        Arbitrary intensity for plotting

    Returns
    -------
    hkl_label : str
        String label of hkl
    df : pd.DataFrame
        DataFrame with columns:
            - strain_33
            - psi (degrees)
            - phi (degrees)
            - d strain
            - 2theta (deg)
            - intensity
    psi_list : list
    strain_33_list : list
    """

    h, k, l = hkl
    if h == 0: h = 0.00000001
    if k == 0: k = 0.00000001
    if l == 0: l = 0.00000001
    # Normalize
    H = h / a_val
    K = k / a_val
    L = l / a_val
    N = np.sqrt(K**2 + L**2)
    M = np.sqrt(H**2 + K**2 + L**2)

    # Elastic constants matrix
    elastic = np.array([
        [c11, c12, c12, 0, 0, 0],
        [c12, c11, c12, 0, 0, 0],
        [c12, c12, c11, 0, 0, 0],
        [0, 0, 0, c44, 0, 0],
        [0, 0, 0, 0, c44, 0],
        [0, 0, 0, 0, 0, c44]
    ])
    elastic_compliance = np.linalg.inv(elastic)
    
    sigma = np.array([
        [sigma_11, 0, 0],
        [0, sigma_22, 0],
        [0, 0, sigma_33]
    ])

    #Method avoids looping and implements numpy broadcasting for speed
    #Check if phi_values are given or if it must be calculated for XRD generation
    if isinstance(psi_values, int):
        if psi_values==0:
            if symmetry == "cubic":
                d0 = a_val / np.linalg.norm([h, k, l])
                sin_theta0 = wavelength / (2 * d0)
                theta0 = np.arcsin(sin_theta0)
                #Compute the psi_value assuming compression axis aligned with X-rays
                psi_values = np.asarray([np.pi/2 - theta0])
            else:
                st.write("Support not yet provided for {} symmetry".format(symmetry))
    else:
        # Assume phi_values and psi_values are 1D numpy arrays
        psi_values = np.asarray(psi_values)
    phi_values = np.asarray(phi_values)
    
    cos_phi = np.cos(phi_values)
    sin_phi = np.sin(phi_values)
    cos_psi = np.cos(psi_values)
    sin_psi = np.sin(psi_values)
    
    # Create meshgrids for broadcasting
    cos_phi, cos_psi = np.meshgrid(cos_phi, cos_psi, indexing='ij')
    sin_phi, sin_psi = np.meshgrid(sin_phi, sin_psi, indexing='ij')
    
    # Rotation matrix A (shape: [n_phi, n_psi, 3, 3])
    A = np.empty((cos_phi.shape[0], cos_phi.shape[1], 3, 3))
    A[..., 0, 0] = cos_phi * cos_psi
    A[..., 0, 1] = -sin_phi
    A[..., 0, 2] = cos_phi * sin_psi
    A[..., 1, 0] = sin_phi * cos_psi
    A[..., 1, 1] = cos_phi
    A[..., 1, 2] = sin_phi * sin_psi
    A[..., 2, 0] = -sin_psi
    A[..., 2, 1] = 0
    A[..., 2, 2] = cos_psi
    
    # Matrix B is constant
    B = np.array([
        [N/M, 0, H/M],
        [-H*K/(N*M), L/N, K/M],
        [-H*L/(N*M), -K/N, L/M]
    ])
    
    # Apply rotation: sigma' = A @ sigma @ A.T
    sigma_prime = A @ sigma @ np.transpose(A, (0, 1, 3, 2))
    
    # Apply B transform: sigma'' = B @ sigma' @ B.T
    sigma_double_prime = B @ sigma_prime @ B.T  # shape: [n_phi, n_psi, 3, 3]
    
    # Strain tensor ε
    ε = np.zeros_like(sigma_double_prime)
    
    ε[..., 0, 0] = elastic_compliance[0, 0] * sigma_double_prime[..., 0, 0] + elastic_compliance[0, 1] * (sigma_double_prime[..., 1, 1] + sigma_double_prime[..., 2, 2])
    ε[..., 1, 1] = elastic_compliance[0, 0] * sigma_double_prime[..., 1, 1] + elastic_compliance[0, 1] * (sigma_double_prime[..., 0, 0] + sigma_double_prime[..., 2, 2])
    ε[..., 2, 2] = elastic_compliance[0, 0] * sigma_double_prime[..., 2, 2] + elastic_compliance[0, 1] * (sigma_double_prime[..., 0, 0] + sigma_double_prime[..., 1, 1])
    ε[..., 0, 1] = ε[..., 1, 0] = 0.5 * elastic_compliance[3, 3] * sigma_double_prime[..., 0, 1]
    ε[..., 0, 2] = ε[..., 2, 0] = 0.5 * elastic_compliance[3, 3] * sigma_double_prime[..., 0, 2]
    ε[..., 1, 2] = ε[..., 2, 1] = 0.5 * elastic_compliance[3, 3] * sigma_double_prime[..., 1, 2]
    
    # ε'_33
    b13, b23, b33 = B[0, 2], B[1, 2], B[2, 2]
    strain_prime_33 = (
        b13**2 * ε[..., 0, 0] +
        b23**2 * ε[..., 1, 1] +
        b33**2 * ε[..., 2, 2] +
        2 * b13 * b23 * ε[..., 0, 1] +
        2 * b13 * b33 * ε[..., 0, 2] +
        2 * b23 * b33 * ε[..., 1, 2]
    )
    
    # Convert psi and phi grid to degrees for output
    psi_deg_grid = np.degrees(np.meshgrid(phi_values, psi_values, indexing='ij')[1])
    phi_deg_grid = np.degrees(np.meshgrid(phi_values, psi_values, indexing='ij')[0])
    psi_list = psi_deg_grid.ravel()
    phi_list = phi_deg_grid.ravel()
    strain_33_list = strain_prime_33.ravel()

    #Compute d0 and 2th
    if symmetry == 'cubic':
        d0 = a_val / np.linalg.norm([h, k, l])
        #Compute strains
        d_strain = d0*(1+strain_33_list)
        #Compute 2ths
        sin_th = wavelength / (2 * d_strain)
        two_th = 2 * np.degrees(np.arcsin(sin_th))
    else:
        st.write("No support for {} symmetries".format(symmetry))
        d_strain = 0
        two_th = 0

    hkl_label = f"{int(h)}{int(k)}{int(l)}"
    df = pd.DataFrame({
        "h": int(h),
        "k": int(k),
        "l": int(l),
        "strain_33": strain_33_list,
        "psi (degrees)": psi_list,
        "phi (degrees)": phi_list,
        "d strain": d_strain,
        "2th" : two_th,
        "intensity": intensity
    })
    return hkl_label, df, psi_list, strain_33_list

def Generate_XRD(selected_hkls, intensities, Gaussian_FWHM, strain_sim_params):
    results_dict = {}
    all_dfs = []  # Collect all dfs here

    for hkl, intensity in zip(selected_hkls, intensities):
        hkl_label, df, psi_list, strain_33_list = compute_strain(hkl, intensity, *strain_sim_params)
        results_dict[hkl_label] = df
        all_dfs.append(df)

    # Concatenate all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Define constants
    sigma_gauss = Gaussian_FWHM / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma
    
    # Define common 2-theta range for evaluation
    twotheta_min = combined_df["2th"].min() - 1
    twotheta_max = combined_df["2th"].max() + 1
    twotheta_grid = np.linspace(twotheta_min, twotheta_max, 500)
    
    # Container to store individual peak curves
    peak_curves = {}
    
    # Loop over unique (h, k, l)
    for (h, k, l), group in combined_df.groupby(["h", "k", "l"]):
        peak_intensity = group["intensity"].iloc[0]
        total_gauss = np.zeros_like(twotheta_grid)
    
        for _, row in group.iterrows():
            two_theta = row["2th"]
            gaussian_peak = peak_intensity * Gaussian(twotheta_grid, two_theta, sigma_gauss) 
            total_gauss += gaussian_peak
    
        avg_gauss = total_gauss / len(group)
        peak_curves[(h, k, l)] = avg_gauss
        
    # Combined total pattern
    total_pattern = sum(peak_curves.values())
    total_df = pd.DataFrame({
        "2th": twotheta_grid,
        "Total Intensity": total_pattern
    })
    
    return total_df

def plot_overlay(x_exp, y_exp, x_sim, y_sim, title="XRD Overlay"):
    residuals = y_exp - y_sim
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(x_exp, y_exp, label="Experimental", color='black', lw=0.5)
    ax1.plot(x_exp, y_sim, label="Simulated", linestyle='--', color='red', lw=0.5)
    ax1.set_ylabel("Intensity")
    ax1.legend()
    ax1.set_title(title)

    ax2.plot(x_exp, residuals, color='blue', lw=0.5)
    ax2.axhline(0, color='gray', lw=0.5)
    ax2.set_xlabel("2θ (degrees)")
    ax2.set_ylabel("Residuals")

    st.pyplot(fig)

def get_initial_parameters(defaults):
    """Returns editable parameter fields with memory between runs."""
    if "params" not in st.session_state:
        st.session_state.params = {
            "a_val": defaults["a_val"],
            "c44": defaults["c44"],
            "t": defaults["sigma_33"] - defaults["sigma_11"]
        }

    st.subheader("Initial Refinement Parameters")
    a_val = st.number_input("Lattice parameter a", value=st.session_state.params["a_val"], format="%.6f")
    c44 = st.number_input("c44", value=st.session_state.params["c44"], format="%.3f")
    t = st.number_input("t", value=st.session_state.params["t"], format="%.3f")

    st.session_state.params.update({"a_val": a_val, "c44": c44, "t": t})
    return a_val, c44, t

def select_parameters_to_refine():
    """Returns flags for parameters the user wants to refine."""
    st.subheader("Select Parameters to Refine")
    return {
        "a_val": st.checkbox("Refine a", value=True),
        "c44": st.checkbox("Refine c44", value=False),
        "t": st.checkbox("Refine t", value=False),
        "peak_intensity": st.checkbox("Refine peak intensities", value=False)
    }

def run_refinement(a_val, c44, t, param_flags, selected_hkls, selected_indices, intensities, Gaussian_FWHM, phi_values, psi_values, wavelength, c11, c12, symmetry, x_exp, y_exp):
    
    #New logic for lmfit -----------------------------
    params = Parameters()
    if param_flags["a_val"]:
        params.add("a_val", value=a_val, min=0.5 * a_val, max=1.5 * a_val)
    else:
        params.add("a_val", value=a_val, vary=False)
    
    if param_flags["c44"]:
        params.add("c44", value=c44, min=0, max=400)
    else:
        params.add("c44", value=c44, vary=False)
    
    if param_flags["t"]:
        params.add("t", value=t, min=-20, max=20)
    else:
        params.add("t", value=t, vary=False)
    
    if param_flags["peak_intensity"]:
        for i, inten in zip(selected_indices, intensities):
            params.add(f"intensity_{i}", value=inten, min=0, max=400)
    else:
        for i, inten in zip(selected_indices, intensities):
            params.add(f"intensity_{i}", value=inten, vary=False)

    # --- First pass of refinement to determine common 2th domain ---
    a_val_opt = params["a_val"].value
    c44_opt = params["c44"].value
    t_opt = params["t"].value
    sigma_11 = -t_opt / 3
    sigma_22 = -t_opt / 3
    sigma_33 = 2 * t_opt / 3
    intensities_opt = [params[f"intensity_{i}"].value for i in selected_indices]

    strain_sim_params = (
        a_val_opt, wavelength, c11, c12, c44_opt,
        sigma_11, sigma_22, sigma_33,
        phi_values, psi_values, symmetry
    )

    # Run Generate_XRD once to get the simulation range
    XRD_df = Generate_XRD(selected_hkls, intensities_opt, Gaussian_FWHM, strain_sim_params)
    twoth_sim = XRD_df["2th"].values

    # Use overlap between simulation and experiment to define interpolation range 
    #Fix this for subsequent refinement
    #We tweak the range here to be slightly less than that returned by the simulation 
    #to eliminate NaN values in evaluating the interpolated data
    x_min_sim = np.min(twoth_sim) + 0.5
    x_max_sim = np.max(twoth_sim) - 0.5
    mask = (x_exp >= x_min_sim) & (x_exp <= x_max_sim)
    x_exp_common = x_exp[mask]
    y_exp_common = y_exp[mask]

    #Here we also determine the x_indices definining the binning around each peak for residual weighting
    #First we need the 2th center positions of each hkl reflection included
    hkl_peak_centers = []
    for hkl in selected_hkls:
        h, k, l = hkl
        #Compute d0 and 2th
        if symmetry == 'cubic':
            d0 = a_val_opt / np.linalg.norm([h, k, l])
            #Compute 2ths
            sin_th = wavelength / (2 * d0)
            two_th = 2 * np.degrees(np.arcsin(sin_th))
        else:
            st.write("No support for {} symmetries".format(symmetry))
            two_th = 0
        hkl_peak_centers = np.append(hkl_peak_centers, two_th)

    #Get the residual bin indices using these centers
    bin_indices = compute_bin_indices(x_exp_common, hkl_peak_centers, Gaussian_FWHM)

    # --- Wrapped cost function that implements this fixed domain ---
    def wrapped_cost_function(params):
        return cost_function(
            params, param_flags, selected_hkls, selected_indices, Gaussian_FWHM,
            phi_values, psi_values, wavelength, c11, c12, symmetry,
            x_exp_common, y_exp_common, bin_indices
        )

    # Run optimization
    result = minimize(wrapped_cost_function, params, method="least_squares")
    #-------------------------------------------------

    return result

def cost_function(params, param_flags, selected_hkls, selected_indices, Gaussian_FWHM, phi_values, psi_values, wavelength, c11, c12, symmetry, x_exp_common, y_exp_common, bin_indices):

    a_val_opt = params["a_val"].value
    c44_opt = params["c44"].value
    t_opt = params["t"].value

    sigma_11_opt = -t_opt/3
    sigma_22_opt = -t_opt/3
    sigma_33_opt = 2*t_opt/3

    intensities_opt = [params[f"intensity_{i}"].value for i in selected_indices]

    strain_sim_params = (
        a_val_opt, wavelength, c11, c12, c44_opt,
        sigma_11_opt, sigma_22_opt, sigma_33_opt,
        phi_values, psi_values, symmetry
    )
    XRD_df = Generate_XRD(selected_hkls, intensities_opt, Gaussian_FWHM, strain_sim_params)
    twoth_sim = XRD_df["2th"]
    intensity_sim = XRD_df["Total Intensity"]

    interp_sim = interp1d(twoth_sim, intensity_sim, bounds_error=False, fill_value=np.nan)
    y_sim_common = interp_sim(x_exp_common)

    residuals = np.asarray(y_exp_common - y_sim_common)

    # Peak position binned normalization of residuals
    norm_residuals = []
    for idx_range in bin_indices:
        if len(idx_range) == 0:
            continue  # skip empty bins
        res_bin = residuals[idx_range]
        y_bin = y_exp_common[idx_range]

        norm = np.max(np.abs(y_bin)) if np.max(np.abs(y_bin)) != 0 else 1
        norm_residuals.append(res_bin / norm)

    #Combine bins into a single array of weighted residuals
    weighted_residuals = np.concatenate(norm_residuals)
    return weighted_residuals

def update_refined_intensities(refined_intensities, selected_indices):
    for val, i in zip(refined_intensities, selected_indices):
        key = f"intensity_{i}"
        st.session_state.intensities[key] = val

def compute_bin_indices(x_exp_common, hkl_peak_centers, window_width=0.2):
    """
    Compute index ranges (bins) around each peak center in x_exp_common.
    
    Parameters:
        x_exp_common (np.ndarray): Experimental 2θ values, common domain.
        peak_centers (List[float]): Estimated peak centers (from HKLs).
        window_width (float): Total width of the window (e.g., 0.2 for ±0.1).
        
    Returns:
        List of slice objects (or index arrays) to use for residual slicing.
    """
    
    hkl_peak_centers = np.sort(hkl_peak_centers)
    
    bin_indices = []
    for center in hkl_peak_centers:
        low = center - 6*window_width 
        high = center + 6*window_width 
        mask = (x_exp_common >= low) * (x_exp_common <= high)
        indices = np.where(mask)[0]
        if len(indices) > 0:
            bin_indices.append(indices)
    return bin_indices

def generate_posterior(steps, walkers, burn, thin, fit_result, param_flags, selected_hkls, selected_indices, intensities, Gaussian_FWHM, phi_values, psi_values, wavelength, c11, c12, symmetry, x_exp, y_exp):

     # --- First pass of refinement to determine common 2th domain ---
    a_val_opt = fit_result.params["a_val"].value
    c44_opt = fit_result.params["c44"].value
    t_opt = fit_result.params["t"].value
    sigma_11 = -t_opt / 3
    sigma_22 = -t_opt / 3
    sigma_33 = 2 * t_opt / 3
    intensities_opt = [fit_result.params[f"intensity_{i}"].value for i in selected_indices]

    strain_sim_params = (
        a_val_opt, wavelength, c11, c12, c44_opt,
        sigma_11, sigma_22, sigma_33,
        phi_values, psi_values, symmetry
    )

    # Run Generate_XRD once to get the simulation range
    XRD_df = Generate_XRD(selected_hkls, intensities_opt, Gaussian_FWHM, strain_sim_params)
    twoth_sim = XRD_df["2th"].values

    # Use overlap between simulation and experiment to define interpolation range 
    #Fix this for subsequent refinement
    #We tweak the range here to be slightly less than that returned by the simulation 
    #to eliminate NaN values in evaluating the interpolated data
    x_min_sim = np.min(twoth_sim) + 0.5
    x_max_sim = np.max(twoth_sim) - 0.5
    mask = (x_exp >= x_min_sim) & (x_exp <= x_max_sim)
    x_exp_common = x_exp[mask]
    y_exp_common = y_exp[mask]

    #Here we also determine the x_indices definining the binning around each peak for residual weighting
    #First we need the 2th center positions of each hkl reflection included
    hkl_peak_centers = []
    for hkl in selected_hkls:
        h, k, l = hkl
        #Compute d0 and 2th
        if symmetry == 'cubic':
            d0 = a_val_opt / np.linalg.norm([h, k, l])
            #Compute 2ths
            sin_th = wavelength / (2 * d0)
            two_th = 2 * np.degrees(np.arcsin(sin_th))
        else:
            st.write("No support for {} symmetries".format(symmetry))
            two_th = 0
        hkl_peak_centers = np.append(hkl_peak_centers, two_th)

    #Get the residual bin indices using these centers
    bin_indices = compute_bin_indices(x_exp_common, hkl_peak_centers, Gaussian_FWHM)
    
    def wrapped_cost_function(params):
        return cost_function(
            params, param_flags, selected_hkls, selected_indices, Gaussian_FWHM,
            phi_values, psi_values, wavelength, c11, c12, symmetry,
            x_exp_common, y_exp_common, bin_indices
        )
    posterior = minimize(wrapped_cost_function, 
                         method='emcee', nan_policy='omit', burn=burn, steps=steps, thin=thin, nwalkers=walkers,
                         params=fit_result.params, is_weighted=False, progress=False)
    return posterior

#### Main App logic -----------------------------------------------------
    
st.set_page_config(layout="wide")
st.title("XRD Strain Simulator (with Batch Mode)")

st.subheader("Upload hkl.csv Input File")
uploaded_file = st.file_uploader("Upload CSV file with elastic parameters and hkl reflections", type=["csv"])

if uploaded_file:
    content = uploaded_file.getvalue().decode("utf-8")
    lines = content.strip().splitlines()

    # Parse constants from first row
    constants_header = lines[0].split(',')
    raw_values = lines[1].split(',')
    # Dynamically convert values with proper types
    constants = {}
    for key, val in zip(constants_header, raw_values):
        try:
            constants[key] = float(val)
        except ValueError:
            constants[key] = val.strip()

    required_keys = {'a', 'wavelength', 'C11', 'C12', 'C44', 'sig11', 'sig22', 'sig33', 'symmetry'}
    if not required_keys.issubset(constants):
        st.error(f"CSV must contain: {', '.join(required_keys)}")
    else:
        col1,col2 = st.columns([4,6])
        with col1:
            st.subheader("Select Reflections and Edit Intensities")
        with col2:
            st.subheader("Material Constants")
            
        col1, col2, col3, col4 = st.columns([4,2,2,2])
        with col1:
            # Parse HKL section including intensity
            hkl_df = pd.read_csv(io.StringIO("\n".join(lines[2:])))
            if not {'h', 'k', 'l', 'intensity'}.issubset(hkl_df.columns):
                st.error("HKL section must have columns: h, k, l, intensity")
            else:
                # Ensure intensity is numeric
                hkl_df['intensity'] = pd.to_numeric(hkl_df['intensity'], errors='coerce').fillna(1.0)
            
                hkl_list = hkl_df[['h', 'k', 'l']].drop_duplicates().values.tolist()
            
                selected_hkls = []
                intensities = []
                selected_indices = []
                peak_intensity_default = {}
                intensity_boxes = {}
            
                for i, hkl in enumerate(hkl_list):
                    # Find matching row to get intensity
                    h_match = (hkl_df['h'] == hkl[0]) & (hkl_df['k'] == hkl[1]) & (hkl_df['l'] == hkl[2])
                    default_intensity = float(hkl_df[h_match]['intensity'].values[0]) if h_match.any() else 1.0
    
                    peak_intensity_default[f"intensity_{i}"] = default_intensity
    
                # Initialize state for peak intensity
                if "intensities" not in st.session_state:
                    st.session_state.intensities = peak_intensity_default
    
                for i, hkl in enumerate(hkl_list):
                    cols = st.columns([2, 2, 2])    
                    with cols[0]:
                        label = f"hkl = ({int(hkl[0])}, {int(hkl[1])}, {int(hkl[2])})"
                        selected = st.checkbox(label, value=True, key=f"chk_{i}")
                    with cols[1]:
                        intensity_boxes[f"intensity_{i}"] = st.number_input(
                            "Intensity",
                            min_value=0.0,
                            value=st.session_state.intensities[f"intensity_{i}"],
                            step=1.0,
                            key=f"intensity_{i}",
                            label_visibility="collapsed"
                        )
                
                    if selected:
                        selected_hkls.append(hkl)
                        selected_indices.append(i)  # Save which index was selected
                        intensities.append(st.session_state.intensities[f"intensity_{i}"])
    
                st.session_state.intensities.update(intensity_boxes)
        with col2:
            a_val = st.number_input("Lattice parameter a (Å)", value=constants['a'], step=0.01, format="%.4f")
            wavelength = st.number_input("Wavelength (Å)", value=constants['wavelength'], step=0.01, format="%.4f")
            symmetry = st.text_input("Symmetry", value=constants['symmetry'])
        with col3:
            c11 = st.number_input("C11", value=constants['C11'])
            c12 = st.number_input("C12", value=constants['C12'])
            c44 = st.number_input("C44", value=constants['C44'])
        with col3:
            sigma_11 = st.number_input("σ₁₁", value=constants['sig11'])
            sigma_22 = st.number_input("σ₂₂", value=constants['sig22'])
            sigma_33 = st.number_input("σ₃₃", value=constants['sig33'])
        
        st.subheader("Computation Settings")
        col1, col2, col3 = st.columns(3)
        with col1:
            total_points = st.number_input("Total number of points (φ × ψ)", value=20000, min_value=10, step=5000)
        with col2:
            Gaussian_FWHM = st.number_input("Gaussian FWHM", value=0.05, min_value=0.005, step=0.005, format="%.3f")
        with col3:
            selected_psi = st.number_input("Psi slice", value=54.7356, min_value=0.0, step=5.0, format="%.4f")
        
        # Determine grid sizes
        psi_steps = int(2 * np.sqrt(total_points))
        phi_steps = int(np.sqrt(total_points) / 2)

        results_dict = {}  # Store results per HKL reflection
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Compute Strains") and selected_hkls:
                fig, axs = plt.subplots(len(selected_hkls), 1, figsize=(8, 5 * len(selected_hkls)))
                if len(selected_hkls) == 1:
                    axs = [axs]

                phi_values = np.linspace(0, 2 * np.pi, phi_steps)
                psi_values = np.linspace(0, np.pi/2, psi_steps)

                for ax, hkl, intensity in zip(axs, selected_hkls, intensities):
                    hkl_label, df, psi_list, strain_33_list = compute_strain(hkl, intensity, a_val, wavelength, c11, c12, c44, sigma_11, sigma_22, sigma_33, phi_values, psi_values, symmetry)

                    #Insert a placeholder column for the average strain at each psi
                    df["Mean strain"] = np.nan
                    #Initialise a list of the mean strains
                    mean_strain_list = []
                    #Compute the average strains and append to df
                    for psi in np.unique(psi_list):
                        #Obtain all the strains at this particular psi
                        mask = psi_list == psi
                        strains = strain_33_list[mask]
                        mean_strain = np.mean(strains)
                        #Append to list
                        mean_strain_list.append(mean_strain)
                        #Update the mean strain column at the correct psi values
                        df.loc[df["psi (degrees)"] == psi, ["Mean strain"]] = mean_strain
                    results_dict[hkl_label] = df

                    scatter = ax.scatter(psi_list, strain_33_list, color="black", s=0.2, alpha=0.1)
                    #Plot the mean strain curve
                    unique_psi = np.unique(psi_list)
                    #Plot the mean curve
                    ax.plot(unique_psi, mean_strain_list, color="blue", lw=0.8, label="mean strain")
                    ax.set_xlabel("ψ (degrees)")
                    ax.set_ylabel("ε′₃₃")
                    ax.set_xlim(0,90)
                    ax.set_title(f"Strain ε′₃₃ for hkl = ({hkl_label})")
                    ax.legend()
                st.pyplot(fig)
        
                if results_dict != {}:
                    st.subheader("Download Computed Data")
                    output_buffer = io.BytesIO()
                    with pd.ExcelWriter(output_buffer, engine='xlsxwriter') as writer:
                        for hkl_label, df in results_dict.items():
                            sheet_name = f"hkl_{hkl_label}"
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                            # auto-width adjustment
                            worksheet = writer.sheets[sheet_name]
                            for i, col in enumerate(df.columns):
                                max_width = max(df[col].astype(str).map(len).max(), len(col)) + 2
                                worksheet.set_column(i, i, max_width)
                
                    output_buffer.seek(0)
                
                    st.download_button(
                        label="📥 Download Results as Excel (.xlsx)",
                        data=output_buffer,
                        file_name="strain_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        with col2:
            if st.button("Generate XRD") and selected_hkls:
                phi_values = np.linspace(0, 2 * np.pi, 72)
                psi_values = 0
                strain_sim_params = (a_val, wavelength, c11, c12, c44, sigma_11, sigma_22, sigma_33, phi_values, psi_values, symmetry)

                XRD_df = Generate_XRD(selected_hkls, intensities, Gaussian_FWHM, strain_sim_params)
                twotheta_grid = XRD_df["2th"]
                total_pattern = XRD_df["Total Intensity"]

                # Plotting the total pattern
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(twotheta_grid, total_pattern, label="Simulated XRD", lw=0.5, color="black")
                ax.set_xlabel("2θ (deg)")
                ax.set_ylabel("Intensity (a.u.)")
                ax.set_title("Simulated XRD Pattern")
                ax.legend()
            
                st.pyplot(fig)

            batch_upload = st.file_uploader("Upload batch XRD parameters", type=["csv"])
            if batch_upload:
                batch_upload.seek(0)  # reset pointer
                # Read everything into a DataFrame
                df = pd.read_csv(batch_upload)
            
                # Convert numerical columns where possible
                for col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except:
                        pass
            
                # Required columns check
                required_columns = {'a', 'wavelength', 'C11', 'C12', 'C44', 'sig11', 'sig22', 'sig33', 'symmetry'}
                if not required_columns.issubset(df.columns):
                    st.error(f"CSV must contain columns: {', '.join(required_columns)}")
                else:
                    st.success("CSV loaded successfully!")

                    # Store parameters in one DataFrame
                    parameters_df = df[list(required_columns)].copy()
                    # Store results side-by-side
                    results_blocks = []

                    phi_values = np.linspace(0, 2*np.pi, 72)
                    psi_values = 0

                    for idx, row in df.iterrows():
                        # Extract row parameters for strain_sim_params
                        strain_sim_params = (
                            row['a'], row['wavelength'], row['C11'], row['C12'], row['C44'],
                            row['sig11'], row['sig22'], row['sig33'],
                            phi_values, psi_values, row['symmetry']
                        )
            
                        # Run Generate_XRD for this row
                        xrd_df = Generate_XRD(selected_hkls, intensities, Gaussian_FWHM, strain_sim_params) 
    
                        # Rename columns so each block is unique
                        xrd_df = xrd_df.rename(columns={
                            "2th": f"2th_iter{idx+1}",
                            "Total Intensity": f"Intensity_iter{idx+1}"
                        }).reset_index(drop=True)
            
                        results_blocks.append(xrd_df)
                
                    # Align all result blocks by index and combine
                    results_df = pd.concat(results_blocks, axis=1)

                    #Plot up the data
                    fig, ax = plt.subplots(figsize=(10, 6))

                    #Get the first y dataset to compute the offset
                    y_initial = results_df["Intensity_iter1"]
                    y_offset = 0
                    offset_step = np.max(y_initial)*0.5
                    
                    for idx in range(len(results_blocks)):
                        x_col = f"2th_iter{idx+1}"
                        y_col = f"Intensity_iter{idx+1}"
                        
                        x = results_df[x_col]
                        y = results_df[y_col]
                        
                        ax.plot(x, y + y_offset, color="black", lw=1, label=f"Iteration {idx+1}")
                        #Increase the offset
                        y_offset = y_offset+offset_step
                    
                    ax.set_xlabel("2θ (degrees)")
                    ax.set_ylabel("Intensity (a.u.)")
                    ax.set_title("Batch XRD")
                    plt.tight_layout()
                    #Display the plot
                    st.pyplot(fig)
                    
                    # Now you have two parts: parameters_df and results_df
                    # Export format: parameters first, then results
                    st.subheader("Download Computed Data")
                    output_buffer = io.BytesIO()
                    with pd.ExcelWriter(output_buffer, engine='xlsxwriter') as writer:
                        parameters_df.to_excel(writer, sheet_name="Parameters", index=False)
                        results_df.to_excel(writer, sheet_name="Results", index=False)

                        # Auto-width adjustment for Parameters sheet
                        worksheet_params = writer.sheets["Parameters"]
                        for i, col in enumerate(parameters_df.columns):
                            max_width = max(parameters_df[col].astype(str).map(len).max(), len(str(col))) + 2
                            worksheet_params.set_column(i, i, max_width)

                        # Auto-width adjustment for "Results" sheet
                        worksheet = writer.sheets["Results"]
                        for i, col in enumerate(results_df.columns):
                            max_width = max(results_df[col].astype(str).map(len).max(), len(str(col))) + 2
                            worksheet.set_column(i, i, max_width)

                    output_buffer.seek(0)
                
                    st.download_button(
                        label="📥 Download Batch XRD as Excel (.xlsx)",
                        data=output_buffer,
                        file_name="XRD_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
                    #st.write("Parameters", parameters_df)
                    #st.write("Results", results_df)

                
        with col3:
            if st.button("Plot axial cake") and selected_hkls:
                fig, axs = plt.subplots(len(selected_hkls), 1, figsize=(8, 5 * len(selected_hkls)))
                if len(selected_hkls) == 1:
                    axs = [axs]

                for ax, hkl, intensity in zip(axs, selected_hkls, intensities):
                    phi_values = np.linspace(0, 2 * np.pi, phi_steps)
                    selected_psi_rads = np.radians(selected_psi)
                    psi_values = [selected_psi_rads]
                    #Get the aximuth and strain values for the selected psi
                    hkl_label, df, psi_list, strain_33_list = compute_strain(hkl, intensity, a_val, wavelength, c11, c12, c44, sigma_11, sigma_22, sigma_33, phi_values, psi_values, symmetry)
                    phi_list = df["phi (degrees)"]
                    scatter = ax.scatter(phi_list, strain_33_list, color="black", s=2)
                    ax.set_xlabel("phi (degrees)")
                    ax.set_ylabel("ε′₃₃")
                    #ax.set_xlim(-180,180)
                    ax.set_title(f"Strain ε′₃₃ for hkl = ({hkl_label}) at psi = ({selected_psi})")
                    plt.tight_layout()
                st.pyplot(fig)
    ### XRD Refinement ----------------------------------------------------------------
    st.subheader("Refine XRD")

    uploaded_XRD = st.file_uploader("Upload .xy experimental XRD file", type=[".xy"])

    if uploaded_XRD is not None:
        raw_lines = uploaded_XRD.read().decode("utf-8").splitlines()
        data_lines = [line for line in raw_lines if not line.strip().startswith("#") and line.strip()]
        data = pd.read_csv(io.StringIO("\n".join(data_lines)), delim_whitespace=True, header=None, names=['2th', 'intensity'])
        x_exp = data['2th'].values
        y_exp = data['intensity'].values
        #Normalise exp data
        y_exp = y_exp/ np.max(y_exp)*100

        col1, col2 = st.columns([2, 2])
        with col1:
            if st.button("Overlay XRD"):
                phi_values = np.linspace(0, 2 * np.pi, 72)
                psi_values = 0
                t = sigma_33 - sigma_11
                strain_sim_params = (
                    a_val, wavelength, c11, c12, c44,
                    sigma_11, sigma_22, sigma_33,
                    phi_values, psi_values, symmetry
                )
                XRD_df = Generate_XRD(selected_hkls, intensities, Gaussian_FWHM, strain_sim_params)
                twoth_sim = XRD_df["2th"]
                intensity_sim = XRD_df["Total Intensity"]
                
                #Determine common data and interpolate
                x_min_sim = np.min(twoth_sim)
                x_max_sim = np.max(twoth_sim)
                mask = (x_exp >= x_min_sim) & (x_exp <= x_max_sim)
                x_exp_common = x_exp[mask]
                y_exp_common = y_exp[mask]
                interp_sim = interp1d(twoth_sim, intensity_sim, bounds_error=False, fill_value=np.nan)
                y_sim_common = interp_sim(x_exp_common)
        
                plot_overlay(x_exp_common, y_exp_common, x_exp_common, y_sim_common)

        defaults = {
            "a_val": a_val,
            "c44": c44,
            "sigma_11": sigma_11,
            "sigma_33": sigma_33
            }
        col1, col2, col3 = st.columns(3)
        with col1:
            a_val_refine, c44_refine, t_refine = get_initial_parameters(defaults)
        with col2:
            param_flags = select_parameters_to_refine()

        if st.button("Refine XRD"):
            phi_values = np.linspace(0, 2 * np.pi, 72)
            psi_values = 0
            result = run_refinement(
                a_val_refine, c44_refine, t_refine, param_flags, selected_hkls, selected_indices, intensities, Gaussian_FWHM,
                phi_values, psi_values, wavelength, c11, c12, symmetry, x_exp, y_exp
                )
            st.session_state["refinement_result"] = result
        
            if result.success:
                st.success("Refinement successful!")
                #Updated logic for lmfit ----------------------------------------
                # Extract refined values from result.params
                a_refined = result.params["a_val"].value
                c44_refined = result.params["c44"].value
                t_refined = result.params["t"].value

                # Update session state
                st.session_state.params["a_val"] = a_refined
                st.session_state.params["c44"] = c44_refined
                st.session_state.params["t"] = t_refined

                # Handle refined intensities
                if param_flags["peak_intensity"]:
                    intensities_refined = [result.params[f"intensity_{i}"].value for i in selected_indices]
                else:
                    intensities_refined = intensities
        
                update_refined_intensities(intensities_refined, selected_indices)

                #------------------------------------------------------

                # Generate the fit report string
                report_str = fit_report(result)
                
                # Display in Streamlit
                st.markdown("### Fit Report")
                st.code(report_str)
            
                # Final simulation and plot
                a_val_opt = st.session_state.params["a_val"]
                c44_opt = st.session_state.params["c44"]
                t_opt = st.session_state.params["t"]
                sigma_11_opt = -t_opt/3
                sigma_22_opt = -t_opt/3
                sigma_33_opt = 2*t_opt/3
                
                strain_sim_params = (
                    a_val_opt, wavelength, c11, c12, c44_opt,
                    sigma_11_opt, sigma_22_opt, sigma_33_opt,
                    phi_values, psi_values, symmetry
                )
                XRD_df = Generate_XRD(selected_hkls, intensities_refined, Gaussian_FWHM, strain_sim_params)
                twoth_sim = XRD_df["2th"]
                intensity_sim = XRD_df["Total Intensity"]
                x_min_sim = np.min(twoth_sim)
                x_max_sim = np.max(twoth_sim)
                mask = (x_exp >= x_min_sim) & (x_exp <= x_max_sim)
                x_exp_common = x_exp[mask]
                y_exp_common = y_exp[mask]
                interp_sim = interp1d(twoth_sim, intensity_sim, bounds_error=False, fill_value=np.nan)
                y_sim_common = interp_sim(x_exp_common)
    
                plot_overlay(x_exp_common, y_exp_common, x_exp_common, y_sim_common, title="Refined Fit")

            else:
                st.error("Refinement failed.")

        #Next display a button to compute the posterior probability distribution
        st.subheader("Probe fit surface")
        col1, col2, col3 = st.columns(3)
        with col1:
            steps = st.number_input("Total steps", value=200, min_value=10, step=10)
            walkers = st.number_input("Total walkers", value=50, min_value=10, step=10)
            burn = st.number_input("Burn points", value=20, min_value=5, step=5)
            thin = st.number_input("Thinning", value=1, min_value=1, step=1)
        if st.button("Compute Posterior probability distribution"):
            phi_values = np.linspace(0, 2 * np.pi, 36)
            psi_values = 0

            if "refinement_result" in st.session_state:
                result = st.session_state["refinement_result"]
                if result.success:
                    plt.close("all")
                    posterior = generate_posterior(steps, walkers, burn, thin, result, param_flags, selected_hkls, selected_indices, intensities, Gaussian_FWHM, phi_values, psi_values, wavelength, c11, c12, symmetry, x_exp, y_exp)
                    # Match the sampled parameters only
                    truths = [
                        posterior.params["a_val"].value,
                        posterior.params["intensity_0"].value,
                        posterior.params["__lnsigma"].value,
                    ]
                    emcee_plot = corner.corner(
                        posterior.flatchain,
                        labels=posterior.var_names,
                        truths=truths
                    )
                    st.pyplot(emcee_plot)
