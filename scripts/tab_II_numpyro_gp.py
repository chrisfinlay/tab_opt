from datetime import datetime

print()
start = datetime.now()
print(f"Start Time : {start}")

import os

import jax.numpy as jnp
from jax import random, vmap
from jax.tree_util import tree_map
from numpyro.infer import MCMC, NUTS, Predictive
from tabascal.jax.interferometry import ants_to_bl

from tab_opt.data import extract_data
from tab_opt.opt import run_svi, svi_predict
from tab_opt.gp import (
    get_times,
    get_vis_gp_params,
    kernel,
    resampling_kernel,
)
from tab_opt.plot import plot_predictions
from tab_opt.vis import averaging, get_rfi_vis, rmse
from tab_opt.models import fixed_orbit_real_rfi


true_plot = False
prior_plot = False
mcmc = False
opt = True
guide = "multinormal"
# guide = "normal"


N_ant = 3
sim_dir = "/Users/chrisfinlay/Documents/PhD/tabascal/tabascal/tabascal/examples/"
f_name = f"target_obs_{N_ant:02}A_450T-0440-1338_128I_001F-1.227e+09-1.227e+09_100AST_1SAT_0GRD/"

sim_path = os.path.join(sim_dir, f_name)

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
) = extract_data(sim_path, sampling=sampling, N_time=N_time)

gains_true = vmap(jnp.interp, in_axes=(None, None, 1))(
    times, times_fine, gains_ants[:, :, 0]
).T
vis_ast_true = vis_ast.reshape(N_time, N_int_samples, N_bl).mean(axis=1)
vis_rfi_true = vis_rfi.reshape(N_time, N_int_samples, N_bl).mean(axis=1)
gains_bl = ants_to_bl(gains_true[:, :, None], a1, a2)
vis_cal = (vis_obs / gains_bl)[:, :, 0]
flags = jnp.where(jnp.abs(vis_ast_true - vis_cal) > 3 * noise, True, False)

print()
print(
    f"Mean RFI Amp. : {jnp.mean(jnp.abs(vis_rfi_true)):.0f} Jy\nFlag Rate :     {100*flags.mean():.2f} %"
)
print()
print(f"Number of Antennas   : {N_ant: 4}")
print(f"Number of Time Steps : {N_time: 4}")

### GP Parameters
vis_var, vis_l = get_vis_gp_params(ants_uvw, vis_ast, a1, a2, noise)

vis_var = vis_var.max()
vis_l = vis_l.min()

g_amp_var = (1e-2) ** 2
g_phase_var = jnp.deg2rad(1.0) ** 2
g_l = 10e3

rfi_var, rfi_l = 1e3, 15.0

### Visibility Sampling Times
# vis_ast_times = [get_times(times, l) for l in vis_l]
# vis_ast_induce = [
#     jnp.interp(vis_ast_times[i], times, vis_ast_true[:, i]) for i in range(N_bl)
# ]
# n_vis_times = jnp.array([len(vis_ast_times[i]) for i in range(N_bl)])

vis_ast_times = get_times(times, vis_l)
vis_ast_induce = vmap(jnp.interp, in_axes=(None, None, 1))(vis_ast_times, times_fine, vis_ast[:, :, 0])
n_vis_times = jnp.array([len(vis_ast_times) for _ in range(N_bl)])

### Gain Sampling Times
g_times = get_times(times, g_l)
gains_induce = vmap(jnp.interp, in_axes=(None, None, 1))(
    g_times, times_fine, gains_ants[:, :, 0]
)

### RFI Sampling Times
rfi_times = get_times(times, rfi_l)
rfi_induce = vmap(jnp.interp, in_axes=(None, None, 1))(
    rfi_times, times_fine, rfi_A_app[:, :, 0]
)

print()
print("Number of parameters per antenna/baseline")
print(f"Gains : {len(g_times): 4}")
print(f"RFI   : {len(rfi_times): 4}")
print(f"AST   : {n_vis_times.min(): 4} - {n_vis_times.max()}")
print()
print(f"Number of parameters : {N_ant * len(g_times) - 1 + N_bl * n_vis_times.max() + N_ant * len(rfi_times)}")
print(f"Number of data points: {2* N_bl * N_time}")


### Define RFI Kernel
rfi_amp = jnp.abs(vis_rfi)[:, :, 0]
rfi_phasor = vis_rfi[:, :, 0] / rfi_amp
rfi_kernel = averaging(
    rfi_phasor[:, None, :]
    * resampling_kernel(rfi_times, times_fine, rfi_var, rfi_l, 1e-6)[:, :, None],
    N_int_samples,
)


### Define True Parameters
true_params = {
    **{"g_amp_induce": jnp.abs(gains_induce)},
    **{"g_phase_induce": jnp.angle(gains_induce[:-1])},
    # **{f"vis_r_{i}": vis_ast_induce[i].real for i in range(N_bl)},
    # **{f"vis_i_{i}": vis_ast_induce[i].imag for i in range(N_bl)},
    **{"vis_r": vis_ast_induce.real},
    **{"vis_i": vis_ast_induce.imag},
    **{"rfi_r_induce": rfi_induce.real},
}

# Set Constant Parameters
params = {
    "noise": noise if noise > 0 else 0.2,
    "vis_obs": jnp.concatenate(
        [vis_obs[:, :, 0].real, vis_obs[:, :, 0].imag], axis=0
    ).T,
    "vis_ast_true": vis_ast_true.T,
    "vis_rfi_true": vis_rfi_true.T,
    "gains_true": gains_true.T,
    "N_time": N_time,
    "N_ants": N_ant,
    "N_bl": N_bl,
    "a1": a1,
    "a2": a2,
}

### Define Prior Parameters
prior_params = {
    "mu_G_amp": true_params["g_amp_induce"],
    "mu_G_phase": true_params["g_phase_induce"],
    # "mu_vis": {i: jnp.zeros(len(vis_ast_times[i])) for i in range(N_bl)},
    "mu_vis": jnp.zeros(len(vis_ast_times)),
    "mu_rfi": jnp.zeros(len(rfi_times)),
    # 'inv_cov_G_amp': inv_kernel(g_times[:,None],
    #                             g_amp_var, g_l),
    # 'inv_cov_G_phase': inv_kernel(g_times[:,None],
    #                               g_phase_var, g_l),
    # 'inv_cov_vis': {i: inv_kernel(vis_ast_times[i][:,None], vis_var[i],
    #                               vis_l[i]) for i in range(N_bl)},
    # 'inv_cov_rfi': inv_kernel(rfi_times[:,None], rfi_var, rfi_l),
    "cov_G_amp": kernel(g_times, g_times, g_amp_var, g_l),
    "cov_G_phase": kernel(g_times, g_times, g_phase_var, g_l),
    # "cov_vis": {
    #     i: kernel(vis_ast_times[i], vis_ast_times[i], vis_var[i], vis_l[i]) for i in range(N_bl)
    # },
    # "cov_rfi": kernel(rfi_times, rfi_times, rfi_var, rfi_l),
    "L_G_amp": jnp.linalg.cholesky(kernel(g_times, g_times, g_amp_var, g_l)),
    "L_G_phase": jnp.linalg.cholesky(kernel(g_times, g_times, g_phase_var, g_l)),
    # "L_vis": {
    #     i: jnp.linalg.cholesky(
    #         kernel(vis_ast_times[i], vis_ast_times[i], vis_var[i], vis_l[i])
    #     )
    #     for i in range(N_bl)
    # },
    "L_vis": jnp.linalg.cholesky(kernel(vis_ast_times, vis_ast_times, vis_var, vis_l)),
    "L_RFI_amp": jnp.linalg.cholesky(kernel(rfi_times, rfi_times, rfi_var, rfi_l)),
    # "resample_vis": {
    #     i: resampling_kernel(vis_ast_times[i], times, vis_var[i], vis_l[i], 1e-3)
    #     for i in range(N_bl)
    # },
    "resample_vis": resampling_kernel(vis_ast_times, times, vis_var, vis_l, 1e-3),
    "resample_g_amp": resampling_kernel(g_times, times, g_amp_var, g_l, 1e-8),
    "resample_g_phase": resampling_kernel(g_times, times, g_phase_var, g_l, 1e-8),
    "rfi_kernel": rfi_kernel,
}

params.update(prior_params)

vis_rfi_test = get_rfi_vis(true_params["rfi_r_induce"], rfi_kernel, a1, a2)
rel_rmse = rmse(vis_rfi_test / vis_rfi_true, 1, axis=(0, 1))

print()
print(
    "Assuming an orbit and compressing all resampling and \nphase multiplication and averaging into a single matrix"
)
print("-----------------------------------------------------")
print(f"Relative Root Mean Squared Error : {rel_rmse: .2E}")

print()
end = datetime.now()
print(f"End Time   : {end}")
print(f"Total Time : {end - start}")

guides = {
    "map": "AutoDelta",
    "normal": "AutoNormal", 
    "multinormal": "AutoMultivariateNormal"
    }

### Define Model
model = fixed_orbit_real_rfi

### Check and Plot Model at true parameters
if true_plot:
    pred = Predictive(model, posterior_samples=tree_map(lambda x: x[None, :], true_params))
    init_pred = pred(random.PRNGKey(2), params=params)
    plot_predictions(times, init_pred, params, type="true")

### Check and Plot Model at prior parameters
if prior_plot:
    pred = Predictive(model, num_samples=100)
    prior_pred = pred(random.PRNGKey(2), params=params)
    plot_predictions(times, prior_pred, params, type="prior")

print()
end_plot = datetime.now()
print(f"End Time  : {end_plot}")
print(f"Plot Time : {end_plot - end}")

### Run Inference
if mcmc:
    num_warmup = 500
    num_samples = 1000

    nuts_kernel = NUTS(model, dense_mass=False)#[('g_phase_0', 'g_phase_1')])
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples)
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, params, extra_fields=('potential_energy',), init_params=true_params)

    pred = Predictive(model, posterior_samples=mcmc.get_samples())
    mcmc_pred = pred(random.PRNGKey(2), params=params)
    plot_predictions(times, mcmc_pred, params, type="mcmc")

if opt:
    guide_family = guides[guide]
    num_samples = 100
    vi_results, vi_guide = run_svi(model, params, max_iter=100_000, guide_family=guide_family, init_params=true_params)
    vi_pred = svi_predict(model, vi_guide, vi_results.params, params=params, num_samples=num_samples)
    plot_predictions(times, vi_pred, params, type=guide)


print()
end_mcmc = datetime.now()
print(f"End Time  : {end_mcmc}")
print(f"MCMC Time : {end_mcmc - end_plot}")