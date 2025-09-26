import numpy as np

def tmm_multilayer(n_list, d_list, wavelength, theta_inc):
    """
    Calculate reflection coefficients for multilayer stack at given wavelength and incidence angle.
    Returns reflection coefficients for s- and p-polarization.
    """
    k0 = 2 * np.pi / wavelength
    theta = [theta_inc]
    
    # Snell's law to get angles
    for i in range(1, len(n_list)):
        theta.append(np.arcsin(n_list[0] * np.sin(theta_inc) / n_list[i]))

    # Admittance
    Y_s = [n * np.cos(t) for n, t in zip(n_list, theta)]
    Y_p = [n / np.cos(t) for n, t in zip(n_list, theta)]

    r_s, r_p = 0 + 0j, 0 + 0j

    for i in reversed(range(len(d_list))):
        delta = k0 * n_list[i+1] * d_list[i] * np.cos(theta[i+1])
        phase = np.exp(2j * delta)

        rs_i = (Y_s[i] - Y_s[i+1]) / (Y_s[i] + Y_s[i+1])
        rp_i = (Y_p[i] - Y_p[i+1]) / (Y_p[i] + Y_p[i+1])

        r_s = (rs_i + r_s * phase) / (1 + rs_i * r_s * phase)
        r_p = (rp_i + r_p * phase) / (1 + rp_i * r_p * phase)

    return r_s, r_p
