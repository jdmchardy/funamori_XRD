import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io

def Gaussian(x, x0, sigma):
    return np.exp(-0.5 * ((x - x0) / sigma) ** 2)

def compute_strain(hkl, a_val, wavelength, c11, c12, c44, phi_values, psi_values, symmetry):
    """Evaluates strain_33 component for given hkl reflection
    
    Returns:
    --------
    hkl_label, dataframe
    dataframe contains: 
        strain_33, 
        phi_list, 
        psi_list,
        d_strain
        two_th,
        intensity
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
    if psi_values==0:
        d0 = a_val / np.linalg.norm([h, k, l])
        sin_theta0 = wavelength / (2 * d0)
        theta0 = np.arcsin(sin_theta0)
        #Compute the psi_value assuming compression axis aligned with X-rays
        psi_values = np.asarray([np.pi/2 - theta0])
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
        "strain_33": strain_33_list,
        "psi (degrees)": psi_list,
        "phi (degrees)": phi_list,
        "d strain": d_strain,
        "2th" : two_th,
        "intensity": intensity
    })
    return hkl_label, df, psi_list, strain_33_list

st.set_page_config(layout="wide")
st.title("Funamori Strain (Batch Mode: ε′₃₃ vs ψ)")

st.subheader("Upload CSV Input File")
uploaded_file = st.file_uploader("Upload CSV file with parameters and multiple hkl reflections", type=["csv"])

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
        st.subheader("Material Constants")

        col1, col2, col3 = st.columns(3)
        with col1:
            a_val = st.number_input("Lattice parameter a (Å)", value=constants['a'], step=0.01, format="%.4f")
        with col2:
            wavelength = st.number_input("Wavelength (Å)", value=constants['wavelength'], step=0.01, format="%.4f")
        with col3:
            symmetry = st.text_input("Symmetry", value=constants['symmetry'])

        col1, col2, col3 = st.columns(3)
        with col1:
            c11 = st.number_input("C11", value=constants['C11'])
            sigma_11 = st.number_input("σ₁₁", value=constants['sig11'])
        with col2:
            c12 = st.number_input("C12", value=constants['C12'])
            sigma_22 = st.number_input("σ₂₂", value=constants['sig22'])
        with col3:
            c44 = st.number_input("C44", value=constants['C44'])
            sigma_33 = st.number_input("σ₃₃", value=constants['sig33'])

        # Parse HKL section including intensity
        hkl_df = pd.read_csv(io.StringIO("\n".join(lines[2:])))
        if not {'h', 'k', 'l', 'intensity'}.issubset(hkl_df.columns):
            st.error("HKL section must have columns: h, k, l, intensity")
        else:
            # Ensure intensity is numeric
            hkl_df['intensity'] = pd.to_numeric(hkl_df['intensity'], errors='coerce').fillna(1.0)
        
            hkl_list = hkl_df[['h', 'k', 'l']].drop_duplicates().values.tolist()
        
            st.subheader("Select Reflections and Edit Intensities")
            selected_hkls = []
            intensities = []
        
            for i, hkl in enumerate(hkl_list):
                # Find matching row to get intensity
                h_match = (hkl_df['h'] == hkl[0]) & (hkl_df['k'] == hkl[1]) & (hkl_df['l'] == hkl[2])
                default_intensity = float(hkl_df[h_match]['intensity'].values[0]) if h_match.any() else 1.0
        
                # Horizontal layout: checkbox left, intensity right
                cols = st.columns([2, 2, 8])  # Wider for checkbox label, narrower for intensity
                with cols[0]:
                    label = f"hkl = ({int(hkl[0])}, {int(hkl[1])}, {int(hkl[2])})"
                    selected = st.checkbox(label, value=True, key=f"chk_{i}")
                with cols[1]:
                    intensity = st.number_input(
                        "Intensity", min_value=0.0, value=default_intensity, step=0.1, key=f"intensity_{i}", label_visibility="collapsed"
                    )
        
                if selected:
                    selected_hkls.append(hkl)
                    intensities.append(intensity)

            st.subheader("Computation Settings")
            col1, col2, col3 = st.columns(3)
            with col1:
                total_points = st.number_input("Total number of points (φ × ψ)", value=20000, min_value=10, step=1000)

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
    
                    for ax, hkl in zip(axs, selected_hkls):
                        hkl_label, df, psi_list, strain_33_list = compute_strain(hkl, a_val, wavelength, c11, c12, c44, phi_values, psi_values, symmetry)
                        results_dict[hkl_label] = df
    
                        scatter = ax.scatter(psi_list, strain_33_list, color="black", s=0.2, alpha=0.1)
                        ax.set_xlabel("ψ (degrees)")
                        ax.set_ylabel("ε′₃₃")
                        ax.set_xlim(0,90)
                        ax.set_title(f"Strain ε′₃₃ for hkl = ({hkl_label})")
    
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
                    phi_values = np.linspace(0, 2 * np.pi, 100)
                    psi_values = 0
                    results_dict = {}
                    for hkl in selected_hkls:
                        hkl_label, df, psi_list, strain_33_list = compute_strain(hkl, a_val, wavelength, c11, c12, c44, phi_values, psi_values, symmetry)
                        results_dict[hkl_label] = df

                    # Define constants
                    fwhm = 0.06  # degrees
                    sigma_gauss = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma
                    
                    # Define common 2-theta range for evaluation
                    theta_min = df["2theta (deg)"].min() - 1
                    theta_max = df["2theta (deg)"].max() + 1
                    theta_grid = np.linspace(theta_min, theta_max, 500)
                    
                    # Container to store individual peak curves
                    peak_curves = {}
                    
                    # Loop over unique (h, k, l)
                    for (h, k, l), group in df.groupby(["h", "k", "l"]):
                        peak_intensity = group["intensity"].iloc[0]
                        total_gauss = np.zeros_like(theta_grid)
                    
                        for _, row in group.iterrows():
                            two_theta = row["2theta (deg)"]
                            gaussian_peak = peak_intensity * Gaussian(theta_grid, two_theta, sigma_gauss) 
                            total_gauss += gauss
                    
                        avg_gauss = total_gauss / len(group)
                        peak_curves[(h, k, l)] = avg_gauss
                    
                        # Save to file
                        peak_df = pd.DataFrame({
                            "2theta (deg)": theta_grid,
                            "Intensity": avg_gauss
                        })
                        filename = f"peak_h{h:.1f}_k{k:.1f}_l{l:.1f}.csv".replace('.', 'p')
                        #peak_df.to_csv(filename, index=False)
                        
                    # Save combined total pattern
                    total_pattern = sum(peak_curves.values())
                    total_df = pd.DataFrame({
                        "2theta (deg)": theta_grid,
                        "Total Intensity": total_pattern
                    })

                    # Plotting the total pattern
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(theta_grid, total_pattern, label="Simulated XRD", lw=1, color="black")
                    ax.set_xlabel("2θ (deg)")
                    ax.set_ylabel("Intensity (a.u.)")
                    ax.set_title("Simulated XRD Pattern")
                    ax.legend()
                    ax.grid(True)
                
                    st.pyplot(fig)
