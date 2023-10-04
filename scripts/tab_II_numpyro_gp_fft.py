from datetime import datetime

print()
start = datetime.now()
print(f"Start Time : {start}")

import os

import jax.numpy as jnp
from jax import random, vmap, jit
from jax.tree_util import tree_map

from jax.flatten_util import ravel_pytree as flatten

from numpyro.infer import MCMC, NUTS, Predictive

from tabascal.jax.interferometry import ants_to_bl
from tab_opt.data import extract_data
from tab_opt.opt import run_svi, svi_predict
from tab_opt.gp import (
    get_times,
    kernel,
    resampling_kernel,
)
from tab_opt.plot import plot_predictions
from tab_opt.vis import averaging
from tab_opt.models import fixed_orbit_real_rfi_compressed_fft


import matplotlib.pyplot as plt

init_plot = 1
true_plot = 1
prior_plot = 1
mcmc = 0
opt = 1
max_iter = 1_000
# guide = "multinormal"
# guide = "normal"
guide = "laplace"
# guide = "map"
epsilon = 1e-1
max_iter = 5_000

N_ant = 4
N_time = 450
sampling = 1


### Define Model
model = fixed_orbit_real_rfi_compressed_fft
model_name = "fixed_orbit_real_rfi_compressed_fft"

print(f"Model : {model_name}")


sim_dir = "/Users/chrisfinlay/Documents/PhD/tabascal/tabascal/examples/data"
f_name = f"target_obs_{N_ant:02}A_450T-0440-1338_128I_001F-1.227e+09-1.227e+09_100AST_1SAT_0GRD/"
sim_path = os.path.join(sim_dir, f_name)

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
) = extract_data(sim_path, sampling=sampling, N_time=N_time)

gains_true = vmap(jnp.interp, in_axes=(None, None, 1))(
    times, times_fine, gains_ants[:, :, 0]
).T
vis_ast_true = vis_ast.reshape(N_time, N_int_samples, N_bl).mean(axis=1)
vis_rfi_true = vis_rfi.reshape(N_time, N_int_samples, N_bl).mean(axis=1)
gains_bl = ants_to_bl(gains_true[:, :, None], a1, a2)
vis_cal = (vis_obs / gains_bl)[:, :, 0]
flags = jnp.where(jnp.abs(vis_ast_true - vis_cal) > 3 * noise, True, False)

vis_obs = (
    (ants_to_bl(gains_ants, a1, a2) * (vis_ast + vis_rfi))
    .reshape(N_time, N_int_samples, N_bl, 1)
    .mean(axis=1)
) + noise_data

bl = jnp.arange(N_bl)

print()
print(
    f"Mean RFI Amp. : {jnp.mean(jnp.abs(vis_rfi_true)):.0f} Jy\nFlag Rate :     {100*flags.mean():.2f} %"
)
print()
print(f"Number of Antennas   : {N_ant: 4}")
print(f"Number of Time Steps : {N_time: 4}")

### Vis AST Fourier modes
ast_k = jnp.fft.fft(vis_ast_true, axis=0).T
k_ast = jnp.fft.fftfreq(N_time, int_time)

ast_k_mean = jnp.fft.fft(
    vis_ast_true.mean(axis=0)[None, :] * jnp.ones((N_time, 1)), axis=0
).T


# def f(k, P0=2e3, k0=8e-4, gamma=1.1):
#     return P0 / ((1.0 + (k / k0) ** 2) ** (gamma / 2))


def f(k, P0=2e3, k0=8e-4, gamma=1.1):
    return P0 / ((1.0 + (k / k0) ** 2) ** (gamma / 2))


### GP Parameters

g_amp_var = (1e-2) ** 2  # 1%
g_phase_var = jnp.deg2rad(1.0) ** 2  # 1 degree
g_l = 3 * 60.0 * 60.0  # 3 hours

# rfi_var, rfi_l = 1e3, 15.0
rfi_var, rfi_l = jnp.abs(vis_obs).max() / 2.0, 15.0


### Gain Sampling Times
g_times = get_times(times, g_l)
gains_induce = vmap(jnp.interp, in_axes=(None, None, 1))(
    g_times, times_fine, gains_ants[:, :, 0]
)
n_g_times = len(g_times)

### RFI Sampling Times
rfi_times = get_times(times, rfi_l)
rfi_induce = vmap(jnp.interp, in_axes=(None, None, 1))(
    rfi_times, times_fine, rfi_A_app[:, :, 0]
)
n_rfi_times = len(rfi_times)

print()
print("Number of parameters per antenna/baseline")
print(f"Gains : {n_g_times: 4}")
print(f"RFI   : {n_rfi_times: 4}")
print(f"AST   : {N_time: 4}")
print()
print(
    f"Number of parameters : {((2 * N_ant - 1) * n_g_times) + (2 * N_time * N_bl) + (2 * N_ant * n_rfi_times)}"
)
print(f"Number of data points: {2* N_bl * N_time}")


### Define RFI Kernel
resample_rfi = resampling_kernel(rfi_times, times_fine, rfi_var, rfi_l, 1e-8)

phase_error_std = 0e-3
traj_phase_error = jnp.exp(
    1.0j
    * phase_error_std
    * random.normal(random.PRNGKey(1), (N_ant, 1))
    * times_fine[None, :]
)

rfi_A_perturb = rfi_A_app[..., 0].T * traj_phase_error


@jit
def rfi_kernel_fn(v_rfi, rfi_I):
    return averaging((v_rfi / rfi_I)[:, None] * resample_rfi, N_int_samples)


rfi_kernel = jnp.array(
    [
        rfi_kernel_fn(v_rfi, rfi_I)
        for v_rfi, rfi_I in zip(
            vis_rfi[:, :, 0].T,
            (rfi_A_perturb[a1] * jnp.conjugate(rfi_A_perturb[a2])),
        )
    ]
)

### Define True Parameters
true_params = {
    **{f"g_amp_induce": jnp.abs(gains_induce)},
    **{f"g_phase_induce": jnp.angle(gains_induce[:-1])},
    **{f"rfi_r_induce": rfi_induce.real},
    **{f"rfi_i_induce": rfi_induce.imag},
    **{"ast_k_r": ast_k.real},
    **{"ast_k_i": ast_k.imag},
}

v_obs_ri = jnp.concatenate([vis_obs[:, :, 0].real, vis_obs[:, :, 0].imag], axis=0).T

# Set Constant Parameters
args = {
    "noise": noise if noise > 0 else 0.2,
    "vis_ast_true": vis_ast_true.T,
    "vis_rfi_true": vis_rfi_true.T,
    "gains_true": gains_true.T,
    "times": times,
    "g_times": g_times,
    "N_time": N_time,
    "N_ants": N_ant,
    "N_bl": N_bl,
    "a1": a1,
    "a2": a2,
    "bl": bl,
    "n_int": N_int_samples,
}

pow_spec_args = {"P0": 2e3, "k0": 8e-4, "gamma": 1.1}
pow_spec_args = {"P0": 2e3, "k0": 2e-3, "gamma": 1.1}
pow_spec_args = {"P0": 2e3, "k0": 2e-3, "gamma": 1.5}

### Define Prior Parameters
args.update(
    {
        "mu_G_amp": jnp.abs(gains_induce),
        "mu_G_phase": jnp.angle(gains_induce[:-1]),
        # "mu_rfi_r": true_params["rfi_r_induce"],
        # "mu_rfi_i": true_params["rfi_i_induce"],
        "mu_rfi_r": jnp.zeros((N_ant, n_rfi_times)),
        "mu_rfi_i": jnp.zeros((N_ant, n_rfi_times)),
        # "mu_ast_k_r": true_params["ast_k_r"],
        # "mu_ast_k_i": true_params["ast_k_i"],
        "mu_ast_k_r": ast_k_mean.real,
        "mu_ast_k_i": ast_k_mean.imag,
        # "mu_ast_k_r": jnp.zeros((N_bl, N_time)),
        # "mu_ast_k_i": jnp.zeros((N_bl, N_time)),
        "L_G_amp": jnp.linalg.cholesky(kernel(g_times, g_times, g_amp_var, g_l)),
        "L_G_phase": jnp.linalg.cholesky(kernel(g_times, g_times, g_phase_var, g_l)),
        "sigma_ast_k": f(k_ast, **pow_spec_args),
        "L_RFI": jnp.linalg.cholesky(kernel(rfi_times, rfi_times, rfi_var, rfi_l)),
        "resample_g_amp": resampling_kernel(g_times, times, g_amp_var, g_l, 1e-8),
        "resample_g_phase": resampling_kernel(g_times, times, g_phase_var, g_l, 1e-8),
        "rfi_kernel": rfi_kernel,
    }
)

rfi_r_induce_init = jnp.interp(
    rfi_times, times, jnp.sqrt(jnp.abs(vis_cal - vis_ast_true).max(axis=1))
)[None, :] * jnp.ones((N_ant, 1))

ast_k_init = jnp.fft.fft(
    vis_ast_true.mean(axis=0)[:, None] * jnp.ones((N_bl, N_time)), axis=1
)

n_sigma = 10.0
ast_k_init = jnp.fft.fft(vis_ast_true + n_sigma * noise_data[..., 0], axis=0).T

# ast_k_init = ast_k

init_params = {
    "g_amp_induce": true_params["g_amp_induce"],
    "g_phase_induce": true_params["g_phase_induce"],
    "ast_k_r": ast_k_init.real,
    "ast_k_i": ast_k_init.imag,
    "rfi_r_induce": rfi_r_induce_init,
    "rfi_i_induce": true_params["rfi_i_induce"],
}


print()
end_start = datetime.now()
print(f"End Time   : {end_start}")
print(f"Total Time : {end_start - start}")

guides = {
    "map": "AutoDelta",
    "normal": "AutoDiagonalNormal",
    "multinormal": "AutoMultivariateNormal",
    "laplace": "AutoLaplaceApproximation",
}


def reduced_chi2(pred, true, noise):
    rchi2 = ((jnp.abs(pred - true) / noise) ** 2).sum() / (2 * true.size)
    return rchi2


### Check and Plot Model at true parameters
pred = Predictive(
    model=model,
    posterior_samples=tree_map(lambda x: x[None, :], true_params),
    batch_ndims=1,
)
true_pred = pred(random.PRNGKey(2), args=args)
rchi2 = reduced_chi2(true_pred["vis_obs"][0], vis_obs[:, :, 0].T, noise)
print()
print(f"Reduced Chi^2 @ true: {rchi2}")
print()

pred = Predictive(
    model=model,
    posterior_samples=tree_map(lambda x: x[None, :], init_params),
    batch_ndims=1,
)
init_pred = pred(random.PRNGKey(2), args=args)
rchi2 = reduced_chi2(init_pred["vis_obs"][0], vis_obs[:, :, 0].T, noise)
print()
print(f"Reduced Chi^2 @ init: {rchi2}")
print()

if init_plot:
    plot_predictions(
        times=times,
        pred=init_pred,
        args=args,
        type="init",
        model_name=model_name,
        max_plots=10,
    )

if true_plot:
    plot_predictions(
        times=times,
        pred=true_pred,
        args=args,
        type="true",
        model_name=model_name,
        max_plots=10,
    )

print()
end_true = datetime.now()
print(f"End Time  : {end_true}")
print(f"Plot Time : {end_true - end_start}")

### Check and Plot Model at prior parameters
if prior_plot:
    pred = Predictive(model, num_samples=100)
    prior_pred = pred(random.PRNGKey(2), args=args)
    plot_predictions(
        times=times,
        pred=prior_pred,
        args=args,
        type="prior",
        model_name=model_name,
        max_plots=10,
    )


print()
end_prior = datetime.now()
print(f"End Time  : {end_prior}")
print(f"Plot Time : {end_prior - end_true}")

### Run Inference
if mcmc:
    num_warmup = 500
    num_samples = 1000

    nuts_kernel = NUTS(model, dense_mass=False)  # [('g_phase_0', 'g_phase_1')])
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples)
    rng_key = random.PRNGKey(0)
    mcmc.run(
        rng_key,
        args=args,
        v_obs=v_obs_ri,
        extra_fields=("potential_energy",),
        init_params=true_params,
    )

    pred = Predictive(model, posterior_samples=mcmc.get_samples())
    mcmc_pred = pred(random.PRNGKey(2), args=args)
    plot_predictions(
        times=times,
        pred=mcmc_pred,
        args=args,
        type="mcmc",
        model_name=model_name,
        max_plots=10,
    )

print()
end_mcmc = datetime.now()
print(f"End Time  : {end_mcmc}")
print(f"Plot Time : {end_mcmc - end_prior}")

if opt:
    guide_family = guides[guide]
    num_samples = 100
    vi_results, vi_guide = run_svi(
        model=model,
        args=args,
        obs=v_obs_ri,
        max_iter=max_iter,
        guide_family=guide_family,
        init_params={
            **{k + "_auto_loc": v for k, v in init_params.items()},
            # **{k + "_auto_scale": v for k, v in param_scale.items()},
        },
        epsilon=epsilon,
    )
    vi_pred = svi_predict(
        model=model,
        guide=vi_guide,
        vi_params=vi_results.params,
        args=args,
        num_samples=num_samples,
    )
    plot_predictions(
        times,
        pred=vi_pred,
        args=args,
        type=guide,
        model_name=model_name,
        max_plots=10,
    )

    rchi2 = reduced_chi2(vi_pred["vis_obs"][0], vis_obs[:, :, 0].T, noise)
    print()
    print(f"Reduced Chi^2 : {rchi2}")
    print()

    plt.semilogy(vi_results.losses)
    plt.show()

print()
end_opt = datetime.now()
print(f"End Time  : {end_opt}")
print(f"Plot Time : {end_opt - end_prior}")

print()
end_final = datetime.now()
print(f"End Time  : {end_final}")
print(f"Infer Time : {end_final - end_opt}")
