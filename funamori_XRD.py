import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io

st.set_page_config(layout="wide")
st.title("Funamori Strain (Batch Mode: ε′₃₃ vs ψ)")

st.subheader("Upload CSV Input File")
uploaded_file = st.file_uploader("Upload CSV file with parameters and multiple hkl reflections", type=["csv"])

if uploaded_file:
    content = uploaded_file.getvalue().decode("utf-8")
    lines = content.strip().splitlines()

    # Parse constants from first row
    constants_header = lines[0].split(',')
    constants_values = list(map(float, lines[1].split(',')))
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
            col1, col2 = st.columns(2)
            with col1:
                total_points = st.number_input("Total number of points (φ × ψ)", value=20000, min_value=10, step=1000)

            # Determine grid sizes
            psi_steps = int(2 * np.sqrt(total_points))
            phi_steps = int(np.sqrt(total_points) / 2)

            results_dict = {}  # Store results per HKL reflection

            if st.button("Compute Strains") and selected_hkls:
                fig, axs = plt.subplots(len(selected_hkls), 1, figsize=(8, 5 * len(selected_hkls)))
                if len(selected_hkls) == 1:
                    axs = [axs]

                for ax, hkl in zip(axs, selected_hkls):
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

                    phi_values = np.linspace(0, 2 * np.pi, phi_steps)
                    psi_values = np.linspace(0, np.pi/2, psi_steps)

                    #Method avoids looping and implements numpy broadcasting for speed
                    # Assume phi_values and psi_values are 1D numpy arrays
                    phi_values = np.asarray(phi_values)
                    psi_values = np.asarray(psi_values)
                    
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
                    
                    # Convert psi grid to degrees for output
                    psi_deg_grid = np.degrees(np.meshgrid(phi_values, psi_values, indexing='ij')[1])
                    psi_list = psi_deg_grid.ravel()
                    strain_list = strain_prime_33.ravel()

                    hkl_label = f"{int(h)}{int(k)}{int(l)}"
                    df = pd.DataFrame({
                        "psi (degrees)": psi_list,
                        "ε′₃₃": strain_list,
                        "intensity": intensity
                    })
                    results_dict[hkl_label] = df

                    scatter = ax.scatter(psi_list, strain_list, color="black", s=0.2, alpha=0.1)
                    ax.set_xlabel("ψ (degrees)")
                    ax.set_ylabel("ε′₃₃")
                    ax.set_xlim(0,90)
                    ax.set_title(f"Strain ε′₃₃ for hkl = ({int(h)}, {int(k)}, {int(l)})")

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
