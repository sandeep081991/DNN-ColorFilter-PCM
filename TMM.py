import numpy as np

def tmm_multilayer(n_list, d_list, wavelength, theta_inc):
    k0 = 2 * np.pi / wavelength
    theta = [theta_inc]
    
    # Snell's law to calculate angle in each layer
    for i in range(1, len(n_list)):
        theta_i = np.arcsin(n_list[0] * np.sin(theta_inc) / n_list[i])
        theta.append(theta_i)
    
    Y_s = [n * np.cos(t) for n, t in zip(n_list, theta)]
    Y_p = [n / np.cos(t) for n, t in zip(n_list, theta)]

    r_s, r_p = 0, 0
    t_s, t_p = 1, 1

    for i in reversed(range(len(d_list))):
        delta = k0 * n_list[i+1] * d_list[i] * np.cos(theta[i+1])
        phase = np.exp(-2j * delta)

        rs_num = Y_s[i] - Y_s[i+1]
        rs_den = Y_s[i] + Y_s[i+1]
        rp_num = Y_p[i] - Y_p[i+1]
        rp_den = Y_p[i] + Y_p[i+1]

        r_s_i = rs_num / rs_den
        r_p_i = rp_num / rp_den

        r_s = (r_s_i + r_s * phase) / (1 + r_s_i * r_s * phase)
        r_p = (r_p_i + r_p * phase) / (1 + r_p_i * r_p * phase)

        t_s *= (1 + r_s_i) / (1 + r_s_i * r_s * phase)
        t_p *= (1 + r_p_i) / (1 + r_p_i * r_p * phase)

    T_s = (np.abs(t_s) ** 2).real
    T_p = (np.abs(t_p) ** 2).real

    return T_s, T_p
