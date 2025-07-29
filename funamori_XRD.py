import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io

st.set_page_config(layout="wide")
st.title("Funamori Strain (Batch Mode: ε′₃₃ vs ψ & φ)")

st.subheader("Upload CSV Input File")
uploaded_file = st.file_uploader("Upload CSV file with parameters and multiple hkl reflections", type=["csv"])

if uploaded_file:
    content = uploaded_file.getvalue().decode("utf-8")
    lines = content.strip().splitlines()

    # Parse constants from first row
    constants_header = lines[0].split(',')
    constants_values = list(map(float, lines[1].split(',')))
    constants = dict(zip(constants_header, constants_values))

    required_keys = {'a', 'C11', 'C12', 'C44', 'sig11', 'sig22', 'sig33'}
    if not required_keys.issubset(constants):
        st.error(f"CSV must contain: {', '.join(required_keys)}")
    else:
        a_val = constants['a']
        c11 = constants['C11']
        c12 = constants['C12']
        c44 = constants['C44']
        sigma_11 = constants['sig11']
        sigma_22 = constants['sig22']
        sigma_33 = constants['sig33']

        # Parse HKL section
        hkl_df = pd.read_csv(io.StringIO("\n".join(lines[2:])))
        if not {'h', 'k', 'l'}.issubset(hkl_df.columns):
            st.error("HKL section must have columns: h, k, l")
        else:
            hkl_list = hkl_df[['h', 'k', 'l']].drop_duplicates().values.tolist()

            st.subheader("Select Reflections to Compute")
            selected_hkls = []
            for i, hkl in enumerate(hkl_list):
                label = f"hkl = ({int(hkl[0])}, {int(hkl[1])}, {int(hkl[2])})"
                if st.checkbox(label, value=True, key=f"chk_{i}"):
                    selected_hkls.append(hkl)

            st.subheader("Computation Settings")
            col1, col2 = st.columns(2)
            with col1:
                total_points = st.number_input("Total number of points (φ × ψ)", value=20000, min_value=10, step=1000)

            # Determine grid sizes
            psi_steps = int(2 * np.sqrt(total_points))
            phi_steps = int(np.sqrt(total_points) / 2)

            if st.button("Run Calculation") and selected_hkls:
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

                    phi = np.linspace(0, 2 * np.pi, phi_steps)
                    psi = np.linspace(0, np.pi/2, psi_steps)
                    PHI, PSI = np.meshgrid(phi, psi)

                    e33_prime = np.zeros_like(PHI)

                    for i in range(PSI.shape[0]):
                        for j in range(PSI.shape[1]):
                            cos_psi = np.cos(PSI[i,j])
                            sin_psi = np.sin(PSI[i,j])
                            cos_phi = np.cos(PHI[i,j])
                            sin_phi = np.sin(PHI[i,j])
                
                            # A (rotation wrt lab frame)
                            A = np.array([
                                [cos_phi * cos_psi, -sin_phi, cos_phi * sin_psi],
                                [sin_phi * cos_psi, cos_phi, sin_phi * sin_psi],
                                [-sin_psi, 0, cos_psi]
                            ])
                
                            # B (crystal orientation)
                            B = np.array([
                                [N/M, 0, H/M],
                                [-H*K/(N*M), L/N, K/M],
                                [-H*L/(N*M), -K/N, L/M]
                            ])
                
                            sigma_prime = A @ sigma @ A.T
                            sigma_double_prime = B @ sigma_prime @ B.T
                
                            # Strain tensor
                            ε = np.zeros((3, 3))
                            ε[0, 0] = elastic_compliance[0, 0] * sigma_double_prime[0, 0] + elastic_compliance[0, 1] * (sigma_double_prime[1, 1] + sigma_double_prime[2, 2])
                            ε[1, 1] = elastic_compliance[0, 0] * sigma_double_prime[1, 1] + elastic_compliance[0, 1] * (sigma_double_prime[0, 0] + sigma_double_prime[2, 2])
                            ε[2, 2] = elastic_compliance[0, 0] * sigma_double_prime[2, 2] + elastic_compliance[0, 1] * (sigma_double_prime[0, 0] + sigma_double_prime[1, 1])
                            ε[0, 1] = ε[1, 0] = 0.5 * elastic_compliance[3, 3] * sigma_double_prime[0, 1]
                            ε[0, 2] = ε[2, 0] = 0.5 * elastic_compliance[3, 3] * sigma_double_prime[0, 2]
                            ε[1, 2] = ε[2, 1] = 0.5 * elastic_compliance[3, 3] * sigma_double_prime[1, 2]
                
                            # ε'_33
                            b13 = B[0, 2]
                            b23 = B[1, 2]
                            b33 = B[2, 2]
                
                            strain_prime_33 = (
                                b13**2 * ε[0, 0] +
                                b23**2 * ε[1, 1] +
                                b33**2 * ε[2, 2] + 
                                2 * b13 * b23 * ε[0, 1] +
                                2 * b13 * b33 * ε[0, 2] +
                                2 * b23 * b33 * ε[1, 2]
                            )
                            
                            e33_prime[i,j] = strain_prime_33

                    scatter = ax.scatter(np.degrees(PHI), e33_prime, color="black", s=0.2, alpha=0.1)
                    ax.set_xlabel("ψ (degrees)")
                    ax.set_ylabel("ε′₃₃")
                    ax.set_xlim(0,90)
                    ax.set_title("Strain ε′₃₃ vs ψ")

                    #c = ax.pcolormesh(PHI*180/np.pi, PSI*180/np.pi, e33_prime, shading='auto', cmap='viridis')
                    #ax.set_title(f"ε′₃₃ for hkl = ({int(h)}, {int(k)}, {int(l)})")
                    #ax.set_xlabel("φ (deg)")
                    #ax.set_ylabel("ψ (deg)")
                    #fig.colorbar(c, ax=ax, label="Strain ε′₃₃")

                st.pyplot(fig)
