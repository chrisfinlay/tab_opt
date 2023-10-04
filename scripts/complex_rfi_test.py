#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import datetime

print()
start = datetime.now()
print(f"Start Time : {start}")


# In[2]:


from tab_opt.mcmc import (
    inv_kernel_vmap,
    inv_kernel,
    log_normal,
    log_multinorm,
    log_multinorm_sum,
)
from tab_opt.transform import (
    affine_transform_full,
    affine_transform_full_inv,
    affine_transform_diag,
    affine_transform_diag_inv,
)
from tab_opt.data import extract_data
from tabascal.jax.coordinates import orbit
from tabascal.utils.jax import progress_bar_scan

from jax import random, jit, vmap, jacrev
import jax.numpy as jnp

from jax.flatten_util import ravel_pytree as flatten
from jax.lax import scan

from jax.tree_util import tree_map
import jax

import optax

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import h5py

import xarray as xr

plt.rcParams["font.size"] = 16


# In[3]:

N_ant = 64

data_dir = "/home/users/f/finlay/tabascal/tabascal/examples/data/"

f_name = f"{data_dir}target_obs_{N_ant:02.0f}A_450T-0440-1338_128I_001F-1.227e+09-1.227e+09_100AST_1SAT_0GRD/"

# f_name = "/Users/chrisfinlay/Documents/PhD/tabascal/tabascal/tabascal/examples/target_obs_08A_450T-0440-1338_128I_001F-1.227e+09-1.227e+09_100AST_1SAT_0GRD/"

# f_name = "/Users/chrisfinlay/Documents/PhD/tabascal/tabascal/examples/data/target_obs_04A_450T-0440-1338_128I_001F-1.227e+09-1.227e+09_100AST_1SAT_0GRD/"

# f_name = "/Users/chrisfinlay/Documents/PhD/tabascal/tabascal/examples/data/target_obs_08A_450T-0440-1338_128I_001F-1.227e+09-1.227e+09_100AST_1SAT_0GRD/"

# f_name = "/Users/chrisfinlay/Documents/PhD/tabascal/tabascal/examples/data/target_obs_16A_450T-0440-1338_128I_001F-1.227e+09-1.227e+09_100AST_1SAT_0GRD/"

# f_name = "/Users/chrisfinlay/Documents/PhD/tabascal/tabascal/examples/data/target_obs_32A_450T-0440-1338_128I_001F-1.227e+09-1.227e+09_100AST_1SAT_0GRD/"

# f_name = "/Users/chrisfinlay/Documents/PhD/tabascal/tabascal/examples/data/target_obs_64A_450T-0440-1338_128I_001F-1.227e+09-1.227e+09_100AST_1SAT_0GRD/"


# In[4]:

N_time = 450
sampling = 1

(
    N_int_samples,
    N_ant,
    N_bl,
    a1,
    a2,
    times,
    times_fine,
    bl_uvw,
    ants_uvw,
    ants_xyz,
    vis_ast,
    vis_rfi,
    vis_obs,
    noise,
    noise_data,
    int_time,
    freqs,
    gains_ants,
    rfi_A_app,
    rfi_orbit,
) = extract_data(f_name, sampling=sampling, N_time=N_time)

# In[5]:


from tabascal.jax.interferometry import ants_to_bl

gains_bl = ants_to_bl(gains_ants.reshape(N_time, -1, N_ant, 1).mean(axis=1), a1, a2)

v_ast = vis_ast.reshape(N_time, -1, N_bl, 1).mean(axis=1)

v_cal = vis_obs / gains_bl

flags = jnp.where(jnp.abs(v_ast - v_cal) > 3 * noise, True, False)


# In[6]:


rng = np.random.default_rng(123)
rfi_gains = (
    np.exp(1.0j * times_fine / 100.0)[:, None, None]
    * np.exp(2.0j * np.pi * rng.uniform(size=N_ant))[None, :, None]
)

v_rfi = rfi_gains[:, a1] * np.conj(rfi_gains[:, a2]) * vis_rfi

del vis_rfi


# In[7]:


v_obs = (gains_ants[:, a1] * np.conj(gains_ants[:, a2]) * (vis_ast + v_rfi)).reshape(
    N_time, N_int_samples, N_bl, 1
).mean(axis=1) + noise_data


# In[8]:


print(
    f"Mean RFI Amp. : {jnp.mean(jnp.abs(v_rfi)):.0f} Jy\nFlag Rate :     {100*flags.mean():.2f} %"
)


# In[9]:


print()
print(f"Number of Antennas: {N_ant}")
print()
print(f"Number of Time Steps: {N_time}")


# In[10]:


@jit
def kernel(x, x_, var, l, noise=1e-3):
    """
    x: array (n_points, n_dim)
    """
    x = x[:, None] if x.ndim == 1 else x
    x_ = x_[:, None] if x_.ndim == 1 else x_
    chi = jnp.linalg.norm(x[None, :, :] - x_[:, None, :], axis=-1) / l
    cov = jnp.abs(var) * jnp.exp(-0.5 * chi**2)
    if chi.shape[0] == chi.shape[1]:
        cov += noise * jnp.eye(x.shape[0])
    return cov


@jit
def resampling_kernel(x, x_, var, l, noise=1e-3):
    K_inv = jnp.linalg.inv(kernel(x, x, var, l, noise))
    K_s = kernel(x, x_, var, l)
    return K_s @ K_inv


# In[11]:


@jit
def gp_resample(y, x, x_, var, l, noise=1e-3):
    K_inv = jnp.linalg.inv(kernel(x, x, var, l, noise))
    K_s = kernel(x, x_, var, l)
    return K_s @ K_inv @ y


@jit
def resampled_cov(x, x_, var, l, noise_cov):
    K = kernel(x, x, var, l, 1e-3)
    K_inv = jnp.linalg.inv(K + noise_cov)
    K_s = kernel(x, x_, var, l)
    K_ss = kernel(x_, x_, var, l)
    return K_ss - K_s @ K_inv @ K_s.T


# In[12]:


def l_from_uv(uv, l0=7e2, a=6e-4):
    return l0 * jnp.exp(-a * uv)


def get_times(gp_l, times):
    int_time = times[1] - times[0]
    t_i = times[0] - int_time / 2
    t_f = times[-1] + int_time / 2
    n_vis_times = jnp.ceil(2.0 * ((t_f - t_i) / gp_l) + 1).astype(int)
    vis_times = jnp.linspace(t_i, t_f, n_vis_times)

    return vis_times


def fringe_phasor(w, freq):
    from tabascal.jax.interferometry import c

    phasor = jnp.exp(-2.0j * jnp.pi * freq / c * w)

    return phasor


def derotated_vis_phase(vis, w, freq):
    derotated_vis = vis * fringe_phasor(w, freq)
    phase = jnp.unwrap(jnp.angle(derotated_vis), discont=0.0)

    return phase


def derotated_vis(vis, w, freq):
    return vis * fringe_phasor(w, freq)


# In[13]:


# @jit
# def rfi_closures(q, params, ants):
#     # rfi_amp = flatten(q['rfi_amp'])[0].reshape(N_ant,N_rfi_time)[ants]
#     # rfi_amp = (params['resample_rfi']@rfi_amp.T).T

#     rfi_real = flatten(q["rfi_real"])[0].reshape(N_ant, N_rfi_time)[ants]
#     rfi_imag = flatten(q["rfi_imag"])[0].reshape(N_ant, N_rfi_time)[ants]
#     rfi_amp = (params["resample_rfi"] @ (rfi_real + 1.0j * rfi_imag).T).T

#     rfi_xyz = orbit(params["times_fine"], *q["rfi_orbit"])
#     distances = jnp.linalg.norm(
#         params["ants_xyz"][:, ants] - rfi_xyz[:, None, :], axis=2
#     )
#     c_distances = (distances - params["phase_corrections"][:, ants])[..., None]

#     return rfi_amp, c_distances

# @jit
# def rfi_vis(app_amplitude, c_distances, freqs):
#     """
#     Calculate visibilities from distances to rfi sources.

#     Parameters:
#     -----------
#     app_amplitude: jnp.array (n_time, 2, n_freq, n_src)
#         Apparent amplitude at the antennas.
#     c_distances: jnp.array (n_time, 2, n_src)
#         The phase corrected distances between the rfi sources and the antennas.
#     freqs: jnp.array (n_freq,)
#         Frequencies.

#     Returns:
#     --------
#     vis: jnp.array (n_time, 1, n_freq)
#         The visibilities.
#     """
#     n_time, n_ant, n_freq, n_src = app_amplitude.shape
#     c = 2.99792458e8

#     # Create array of shape (n_time, n_bl, n_freq, n_src) and then sum over n_src

#     minus_two_pi_over_lamda = (-2.0 * jnp.pi * freqs / c).reshape(1, 1, n_freq, 1)

#     c_distances = c_distances.reshape(n_time, n_ant, 1, n_src)

#     phase = minus_two_pi_over_lamda * (c_distances[:, 0] - c_distances[:, 1])
#     intensities_app = app_amplitude[:, 0] * jnp.conj(app_amplitude[:, 1])

#     vis = jnp.sum(intensities_app * jnp.exp(1.0j * phase), axis=-1)

#     return vis

@jit
def get_rfi_vis(rfi_amp, rfi_kernel):
    
    rfi_I = rfi_amp[0] * jnp.conjugate(rfi_amp[1])
    vis_rfi = rfi_kernel @ rfi_I
    
    return vis_rfi


@jit
def rfi_vis(q, params, ants, bl):
    
    rfi_real = flatten(q["rfi_real"])[0].reshape(N_ant, N_rfi_time)[ants]
    rfi_imag = flatten(q["rfi_imag"])[0].reshape(N_ant, N_rfi_time)[ants]
    rfi_amp = rfi_real + 1.0j * rfi_imag
    # rfi_amp = (params["resample_rfi"] @ (rfi_real + 1.0j * rfi_imag).T).T
    
    vis_rfi = get_rfi_vis(rfi_amp, params["rfi_kernel"][bl])
    
    return vis_rfi


@jit
def baseline_gains(q, params, ants):
    G_amp = flatten(q["g_amp"])[0].reshape(N_ant, N_g_time)[ants]
    G_phase = flatten(q["g_phase"])[0].reshape(N_ant - 1, N_g_time)
    G_phase = jnp.concatenate([G_phase, jnp.zeros((1, N_g_time))], axis=0)[ants]

    G_amp = (params["resample_g_amp"] @ G_amp.T).T
    G_phase = (params["resample_g_phase"] @ G_phase.T).T

    G = G_amp * jnp.exp(1.0j * G_phase)

    return G[0] * jnp.conjugate(G[1])


@jit
def pad_vis(vis):
    return jnp.pad(vis, (0, N_vis_time - vis.shape[0]))


@jit
def rfi_vis_model(q, params, ant1, ant2):
    # q = scale_parameters(q, inv_scalings)

    ants = jnp.array([ant1, ant2])

    # rfi_amp, c_distances = rfi_closures(q, params, ants)
    # V_rfi = rfi_vis(rfi_amp.T[:, :, None, None], c_distances, params["freqs"])[..., 0]
    
    V_rfi = rfi_vis(q, params, ants)

    return V_rfi


@jit
def ast_vis_model(q, params, bl):
    V_real = flatten(tree_map(pad_vis, q["v_real"]))[0].reshape(N_bl, N_vis_time)[bl]
    V_imag = flatten(tree_map(pad_vis, q["v_imag"]))[0].reshape(N_bl, N_vis_time)[bl]
    V_ast = V_real + 1.0j * V_imag

    V_ast = params["resample_vis"][bl] @ V_ast

    return V_ast


@jit
def model(q, params, ant1, ant2, bl):
    
    # q = transform_parameters(q, params)

    ants = jnp.array([ant1, ant2])

    V_real = flatten(tree_map(pad_vis, q["v_real"]))[0].reshape(N_bl, N_vis_time)[bl]
    V_imag = flatten(tree_map(pad_vis, q["v_imag"]))[0].reshape(N_bl, N_vis_time)[bl]
    V_ast = V_real + 1.0j * V_imag

    V_ast = params["resample_vis"][bl] @ V_ast

    #     Calculate the visibility contribution from the RFI
    # rfi_amp, c_distances = rfi_closures(q, params, ants)
    # V_rfi = rfi_vis(rfi_amp.T[:, :, None, None], c_distances, params["freqs"])[..., 0]
    # V_rfi = V_rfi.reshape(N_time, N_int_samples).mean(axis=1)
    
    V_rfi = rfi_vis(q, params, ants, bl)

    G_bl = baseline_gains(q, params, ants)

    model_vis = G_bl * (V_ast + V_rfi)

    return jnp.concatenate([model_vis.real.flatten(), model_vis.imag.flatten()])


@jit
def transform_parameters(q, transform_params):
    q_new = {
        # "rfi_orbit": affine_transform_full(
        #     q["rfi_orbit"],
        #     transform_params["L_RFI_orbit"],
        #     transform_params["mu_RFI_orbit"],
        # ),
        "rfi_real": tree_map(
            lambda x: affine_transform_diag(
                x, transform_params["std_RFI"], transform_params["mu_RFI"]
            ),
            q["rfi_real"],
        ),
        "rfi_imag": tree_map(
            lambda x: affine_transform_diag(
                x, transform_params["std_RFI"], transform_params["mu_RFI"]
            ),
            q["rfi_imag"],
        ),
        "g_amp": tree_map(
            lambda x, mu: affine_transform_full(x, transform_params["L_G_amp"], mu),
            q["g_amp"],
            transform_params["mu_G_amp"],
        ),
        "g_phase": tree_map(
            lambda x, mu: affine_transform_full(x, transform_params["L_G_phase"], mu),
            q["g_phase"],
            transform_params["mu_G_phase"],
        ),
        "v_real": tree_map(
            affine_transform_full,
            q["v_real"],
            transform_params["L_vis"],
            transform_params["mu_vis"],
        ),
        "v_imag": tree_map(
            affine_transform_full,
            q["v_imag"],
            transform_params["L_vis"],
            transform_params["mu_vis"],
        ),
    }
    return q_new


@jit
def inv_transform_parameters(q, transform_params):
    q_new = {
        # "rfi_orbit": affine_transform_full_inv(
        #     q["rfi_orbit"],
        #     jnp.linalg.inv(transform_params["L_RFI_orbit"]),
        #     transform_params["mu_RFI_orbit"],
        # ),
        "rfi_real": tree_map(
            lambda x: affine_transform_diag_inv(
                x, 1.0 / transform_params["std_RFI"], transform_params["mu_RFI"]
            ),
            q["rfi_real"],
        ),
        "rfi_imag": tree_map(
            lambda x: affine_transform_diag_inv(
                x, 1.0 / transform_params["std_RFI"], transform_params["mu_RFI"]
            ),
            q["rfi_imag"],
        ),
        "g_amp": tree_map(
            lambda x, mu: affine_transform_full_inv(
                x, jnp.linalg.inv(transform_params["L_G_amp"]), mu
            ),
            q["g_amp"],
            transform_params["mu_G_amp"],
        ),
        "g_phase": tree_map(
            lambda x, mu: affine_transform_full_inv(
                x, jnp.linalg.inv(transform_params["L_G_phase"]), mu
            ),
            q["g_phase"],
            transform_params["mu_G_phase"],
        ),
        "v_real": tree_map(
            affine_transform_full_inv,
            q["v_real"],
            transform_params["L_vis_inv"],
            transform_params["mu_vis"],
        ),
        "v_imag": tree_map(
            affine_transform_full_inv,
            q["v_imag"],
            transform_params["L_vis_inv"],
            transform_params["mu_vis"],
        ),
    }
    return q_new

# Calculate approximate True values


def l_from_uv(uv, l0=7e2, a=6e-4):
    return l0 * jnp.exp(-a * uv)


a1, a2 = jnp.triu_indices(N_ant, 1)
mag_uvw = jnp.linalg.norm(ants_uvw[0, a1] - ants_uvw[0, a2], axis=-1)
vis_var = (jnp.abs(vis_ast).max(axis=(0, 2)) + noise) ** 2
vis_l = l_from_uv(mag_uvw, l0=5e2)

g_amp_var = (0.01) ** 2
g_phase_var = (jnp.deg2rad(1.0)) ** 2
g_l = 10e3

rfi_var, rfi_l = 1e6, 15.0

T_time = times[-1] - times[0] + int_time
n_vis_times = jnp.ceil(2.0 * T_time / vis_l).astype(int)
get_times = partial(jnp.linspace, *(times[0] - int_time / 2, times[-1] + int_time / 2))
vis_times = tuple(map(get_times, n_vis_times))
N_vis_time = int(n_vis_times.max())

vis_idxs = [
    jnp.floor(jnp.linspace(0, len(times_fine) - 1, n_vis_times[i])).astype(int)
    for i in range(N_bl)
]

Nt_fine = len(times_fine)
#####
Nt_g = N_int_samples * 45
# Nt_g = N_int_samples * 150
g_idx = jnp.array(
    list(np.arange(0, Nt_fine, Nt_g))
    + [
        Nt_fine - 1,
    ]
)
times_g = times_fine[g_idx]
N_g_time = len(times_g)
G = gains_ants[g_idx, :, 0].T
#####


def get_times(gp_l, times):
    int_time = times[1] - times[0]
    t_i = times[0] - int_time / 2
    t_f = times[-1] + int_time / 2
    n_vis_times = jnp.ceil(2.0 * ((t_f - t_i) / gp_l) + 1).astype(int)
    vis_times = jnp.linspace(t_i, t_f, n_vis_times)

    return vis_times

times_rfi = get_times(rfi_l, times)
rfi_induce = vmap(jnp.interp, in_axes=(None, None, 1))(
    times_rfi, times_fine, (rfi_A_app * rfi_gains)[:, :, 0]
)

rfi_real = rfi_induce.real
rfi_imag = rfi_induce.imag

N_rfi_time = len(times_rfi)

rfi_signal = rfi_A_app * rfi_gains


# Nt_rfi = N_int_samples  # 2
# rfi_idx = jnp.array(
#     list(np.arange(0, Nt_fine, Nt_rfi))
#     + [
#         Nt_fine - 1,
#     ]
# )
# times_rfi = times_fine[rfi_idx]
# N_rfi_time = len(times_rfi)
# rfi_A = rfi_A_app[rfi_idx, :, 0].T

# rfi_signal = rfi_A_app * rfi_gains
# rfi_real = rfi_signal[rfi_idx, :, 0].T.real
# rfi_imag = rfi_signal[rfi_idx, :, 0].T.imag

######
true_values = {
    "g_amp": {i: x for i, x in enumerate(jnp.abs(G))},
    "g_phase": {i: x for i, x in enumerate(jnp.angle(G[:-1]))},
    "rfi_real": {i: x for i, x in enumerate(rfi_real)},
    "rfi_imag": {i: x for i, x in enumerate(rfi_imag)},
    # "rfi_orbit": rfi_orbit,
    "v_real": {i: vis_ast.real[vis_idxs[i], i, 0] for i in range(N_bl)},
    "v_imag": {i: vis_ast.imag[vis_idxs[i], i, 0] for i in range(N_bl)},
}

####################################################################

print()
print(f"Max and Min Vis samples : ({int(n_vis_times.max())}, {int(n_vis_times.min())})")
print()
print(f"Number of RFI inducing points: {N_rfi_time}")

print()
print("Creating RFI kernel ...")

from tab_opt.vis import averaging

rfi_kernel_fn = lambda x, y: averaging(x[:,None]*y, N_int_samples)

# rfi_I = rfi_signal[:,a1,0]*jnp.conjugate(rfi_signal[:,a2,0])

# rfi_kernel = vmap(rfi_kernel_fn, in_axes=(1, None))(
#     v_rfi[:, :, 0] / rfi_I, 
#     resampling_kernel(times_rfi, times_fine, rfi_var, rfi_l, 1e-6)
# )

rfi_phasor = v_rfi[:, :, 0] / (rfi_signal[:,a1,0]*jnp.conjugate(rfi_signal[:,a2,0]))

rfi_resample = resampling_kernel(times_rfi, times_fine, rfi_var, rfi_l, 1e-6)

rfi_kernel = np.array([rfi_kernel_fn(rfi_phasor[:,i], rfi_resample) for i in range(N_bl)])

print()
print("RFI Kernel Created.")



# In[15]:


# Set Constant Parameters
params = {
    "freqs": freqs,
    "times_fine": times_fine,
    "noise": noise if noise > 0 else 0.2,
    "ants_xyz": ants_xyz,
    "phase_corrections": ants_uvw[..., -1],
    # 'vis_obs': vis_obs[:,:,0],
    "vis_obs": v_obs[:, :, 0],
}


def sym(x):
    return (x + x.T) / 2.0


rfi_cov_fp = "../notebooks/data/RFIorbitCov5min.npy"

transform_params = {
    # "mu_RFI_orbit": rfi_orbit,
    # "L_RFI_orbit": jnp.linalg.cholesky(np.load(rfi_cov_fp)),
    "mu_RFI": 0.0,
    "std_RFI": 100.0,
    "mu_G_amp": {i: x for i, x in enumerate(jnp.abs(G))},
    "L_G_amp": jnp.linalg.cholesky(
        kernel(times_g[:, None], times_g[:, None], g_amp_var, g_l, 1e-8)
    ),
    "mu_G_phase": {i: x for i, x in enumerate(jnp.angle(G)[:-1])},
    "L_G_phase": jnp.linalg.cholesky(
        kernel(times_g[:, None], times_g[:, None], g_phase_var, g_l, 1e-8)
    ),
    "mu_vis": {i: jnp.zeros(len(vis_times[i])) for i in range(N_bl)},
    "L_vis": {
        i: jnp.linalg.cholesky(
            kernel(
                vis_times[i][:, None], vis_times[i][:, None], vis_var[i], vis_l[i], 1e-3
            )
        )
        for i in range(N_bl)
    },
    "L_vis_inv": {
        i: jnp.linalg.inv(
            jnp.linalg.cholesky(
                kernel(
                    vis_times[i][:, None],
                    vis_times[i][:, None],
                    vis_var[i],
                    vis_l[i],
                    1e-3,
                )
            )
        )
        for i in range(N_bl)
    },
}


true_values = inv_transform_parameters(true_values, transform_params)
# true_values = scale_parameters(true_values, scalings)
flat_true_values, unflatten = flatten(true_values)


sample_params = {
    "resample_vis": {
        i: resampling_kernel(vis_times[i], times, vis_var[i], vis_l[i], 1e-3)
        for i in range(N_bl)
    },
    "resample_g_amp": resampling_kernel(times_g, times, g_amp_var, g_l, 1e-8),
    "resample_g_phase": resampling_kernel(times_g, times, g_phase_var, g_l, 1e-8),
    "rfi_kernel": rfi_kernel, 
    # "resample_rfi": resampling_kernel(
    #     times_rfi, params["times_fine"], rfi_var, rfi_l, 1e-6
    # ),
}

sample_params["resample_vis"] = jnp.array(
    [
        jnp.pad(x, ((0, 0), (0, N_vis_time - x.shape[1])))
        for x in sample_params["resample_vis"].values()
    ]
)

params.update(transform_params)
params.update(sample_params)

flatten(true_values["v_real"])[0].shape[0], N_vis_time * N_bl, N_time * N_bl, flatten(
    true_values["v_real"]
)[0].shape[0] / (N_vis_time * N_bl), flatten(true_values["v_real"])[0].shape[0] / (
    N_time * N_bl
)


# In[16]:



@jit
def nlp(q, params):
    flat_q = flatten(q)[0]
    return -1.0 * jnp.sum(log_normal(flat_q, 0.0, 1.0))


@jit
def nll(q, params, ant1, ant2, bl):
    V_model = model(q, params, ant1, ant2, bl)

    V_obs = jnp.concatenate(
        [params["vis_obs"][:, bl].real, params["vis_obs"][:, bl].imag]
    )

    return -1.0 * log_normal(V_obs, V_model, params["noise"])


@jit
def U(q, params, a1, a2, bl):
    
    nl_prior = nlp(q, params)
    
    q = transform_parameters(q, params)
    
    nl_like = vmap(nll, in_axes=(None, None, 0, 0, 0))(q, params, a1, a2, bl).sum()

    nl_post = nl_prior + nl_like

    return nl_post

@jit
def chi2(q, params):
    a1, a2 = jnp.triu_indices(N_ant, 1)
    bl = jnp.arange(len(a1))
    q = transform_parameters(q, params)
    
    nl_like = vmap(nll, in_axes=(None, None, 0, 0, 0))(q, params, a1, a2, bl).sum()

    return nl_like/params['vis_obs'].size

delU = jit(jacrev(U, 0))

bl = jnp.arange(len(a1))

print()
print(f'Number of data points: {2*params["vis_obs"].size}')
print()
print(f"Number of parameters: {len(flatten(true_values)[0])}")
print()
print(f'Energy per data point @ true values: {U(true_values, params, a1, a2, bl)/params["vis_obs"].size}')
print()

#################################################

flat_q, unflatten = flatten(true_values)

q_dev = unflatten(random.normal(random.PRNGKey(101), (len(flat_q),)))

qi = tree_map(lambda x, y: x + 1e0*y, true_values, q_dev)

print(f'Energy per data point @ qi: {U(qi, params, a1, a2, bl)/params["vis_obs"].size}')
print()

##################################

# q_i = qi
q_i = true_values

# opt = optax.adam(1e-1)
opt = optax.adabelief(1e-3)
state = opt.init(q_i)

grads = delU(q_i, params, a1, a2, bl)
updates, opt_state = opt.update(grads, state)
q_new = optax.apply_updates(q_i, updates)
U_new = U(q_new, params, a1, a2, bl) / vis_obs.size

history = {
    **qi,
    "U": jnp.array([U(qi, params, a1, a2, bl)/vis_obs.size])
    }
history = tree_map(lambda x, y: jnp.concatenate([x[None,:], y[None,:]], axis=0), history, {**q_new, "U": jnp.array([U_new])})

pbar = tqdm(range(200))
pbar.set_description(
    f"NL Posterior = {round(float(U_new), 3)}"
)
for _ in pbar:
    grads = delU(q_new, params, a1, a2, bl)
    updates, opt_state = opt.update(grads, opt_state)
    q_new = optax.apply_updates(q_new, updates)
    U_new = U(q_new, params, a1, a2, bl) / vis_obs.size
    pbar.set_description(
        f"NL Posterior = {round(float(U_new), 3)}"
    )
    history = tree_map(lambda x, y: jnp.concatenate([x, y[None,:]], axis=0), history, {**q_new, "U": jnp.array([U_new])})



print()
print(f"End Time : {datetime.now()}")
print()
print(f"Time Taken: {datetime.now()-start}")
