import numpy as np
import pandas as pd
from tmm_function import tmm_multilayer

# --- Load refractive index data ---
def load_refractive_index(file):
    data = np.loadtxt(file)
    wavelengths = data[:, 0]  # µm
    n_complex = data[:, 1] + 1j * data[:, 2]
    return wavelengths, n_complex

# --- Interpolation ---
def interpolate_index(wl, wl_data, n_data):
    return np.interp(wl, wl_data, n_data.real) + 1j * np.interp(wl, wl_data, n_data.imag)

# --- Load materials ---
wl_ag, n_ag_data = load_refractive_index('Ag.txt')
wl_crys, n_crys_data = load_refractive_index('Crystalline_Sb2S3.txt')
wl_amorph, n_amorph_data = load_refractive_index('Amorphous_Sb2S3.txt')

# --- Define ranges ---
wavelengths = np.linspace(0.4, 1.5, 150)  # µm
angles_deg = np.linspace(0, 70, 10)
angles_rad = np.radians(angles_deg)

sio2_range = np.arange(70, 102, 2)       # nm
sb2s3_range = np.arange(120, 232, 2)     # nm
phases = [0, 1]  # 0: amorphous, 1: crystalline

dataset = []

print("Generating dataset...")

for d_sio2 in sio2_range:
    for d_sb2s3 in sb2s3_range:
        for phase in phases:
            for angle_rad, angle_deg in zip(angles_rad, angles_deg):
                row = [d_sio2, d_sb2s3, phase, angle_deg]
                T_spectrum = []

                for wl in wavelengths:
                    # Fixed values
                    n_air = 1.0
                    n_sio2 = 1.45
                    n_ag_interp = interpolate_index(wl, wl_ag, n_ag_data)

                    if phase == 0:
                        n_sb2s3_interp = interpolate_index(wl, wl_amorph, n_amorph_data)
                    else:
                        n_sb2s3_interp = interpolate_index(wl, wl_crys, n_crys_data)

                    # Build stack
                    n_stack = [n_air, n_sio2, n_ag_interp, n_sb2s3_interp, n_ag_interp, n_air]
                    d_stack = [d_sio2 / 1000, 0.015, d_sb2s3 / 1000, 0.015]  # µm

                    r_s, r_p = tmm_multilayer(n_stack, d_stack, wl, angle_rad)

                    # Transmission = 1 - Reflectance (neglecting absorption for now)
                    R_s = np.abs(r_s)**2
                    R_p = np.abs(r_p)**2
                    T_avg = 1.0 - 0.5 * (R_s + R_p)

                    T_spectrum.append(T_avg)

                dataset.append(row + T_spectrum)

# --- Save as CSV ---
columns = ['d_sio2 (nm)', 'd_sb2s3 (nm)', 'phase', 'angle (deg)'] + [f'T_{w:.3f}um' for w in wavelengths]
df = pd.DataFrame(dataset, columns=columns)
df.to_csv('transmission_dataset.csv', index=False)
print("Dataset saved as transmission_dataset.csv")
