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

    required_keys = {'a', 'C11', 'C12', 'C44', 'sigma1', 'sigma2', 'sigma3'}
    if not required_keys.issubset(constants):
        st.error(f"CSV must contain: {', '.join(required_keys)}")
    else:
        a_val = constants['a']
        c11 = constants['C11']
        c12 = constants['C12']
        c44 = constants['C44']
        sigma_11 = constants['sigma1']
        sigma_22 = constants['sigma2']
        sigma_33 = constants['sigma3']

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

            # Derived from original logic
            psi_steps = int(2 * np.sqrt(total_points))
            phi_steps = int(np.sqrt(total_points) / 2)

            if st.button("Run Calculation") and selected_hkls:
                fig, axs = plt.subplots(len(selected_hkls), 1, figsize=(8, 5 * len(selected_hkls)))
                if len(selected_hkls) == 1:
                    axs = [axs]

                for ax, hkl in zip(axs, selected_hkls):
                    h, k, l = hkl
                    if h == 0: h = 0.0
                    if k == 0: k = 0.0
                    if l == 0: l = 0.0

                    # Elastic matrix
                    elastic = np.array([
                        [c11, c12, c12, 0, 0, 0],
                        [c12, c11, c12, 0, 0, 0],
                        [c12, c12, c11, 0, 0, 0],
                        [0, 0, 0, c44, 0, 0],
                        [0, 0, 0, 0, c44, 0],
                        [0, 0, 0, 0, 0, c44]
                    ])
                    S = np.linalg.inv(elastic)
                    sigma = np.array([
                        [sigma_11, 0, 0],
                        [0, sigma_22, 0],
                        [0, 0, sigma_33]
                    ])

                    phi = np.linspace(0, 2 * np.pi, phi_steps)
                    psi = np.linspace(0, np.pi/2, psi_steps)
                    PHI, PSI = np.meshgrid(phi, psi)

                    # Unit diffraction vector
                    g_hkl = np.array([h, k, l])
                    g_hkl = g_hkl / np.linalg.norm(g_hkl)

                    e33_prime = np.zeros_like(PHI)

                    for i in range(PSI.shape[0]):
                        for j in range(PSI.shape[1]):
                            # Rotation matrix
                            R = np.array([
                                [np.cos(PHI[i,j]) * np.sin(PSI[i,j]), np.sin(PHI[i,j]) * np.sin(PSI[i,j]), np.cos(PSI[i,j])]
                            ])
                            n = R[0,:]

                            m = np.outer(n, n)
                            strain = np.tensordot(S, sigma, axes=([1],[0]))
                            strain_tensor = np.dot(strain, m)
                            e33_prime[i,j] = np.sum(n * np.dot(strain_tensor, n))

                    c = ax.pcolormesh(PHI*180/np.pi, PSI*180/np.pi, e33_prime, shading='auto', cmap='viridis')
                    ax.set_title(f"ε′₃₃ for hkl = ({int(h)}, {int(k)}, {int(l)})")
                    ax.set_xlabel("φ (deg)")
                    ax.set_ylabel("ψ (deg)")
                    fig.colorbar(c, ax=ax, label="Strain ε′₃₃")

                st.pyplot(fig)