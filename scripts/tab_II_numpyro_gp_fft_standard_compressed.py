from datetime import datetime

print()
start = datetime.now()
print(f"Start Time : {start}")

import os

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # add this
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax

# jax.config.update("jax_platform_name", "cpu")

import jax.profiler

import jax.numpy as jnp
from jax import random, vmap, jit
from jax.tree_util import tree_map

from jax.flatten_util import ravel_pytree as flatten

from numpyro.infer import MCMC, NUTS, Predictive

from tabascal.jax.interferometry import ants_to_bl
from tab_opt.data import extract_data
from tab_opt.opt import run_svi, svi_predict, f_model_flat, flatten_obs, post_samples
from tab_opt.gp import (
    get_times,
    kernel,
    resampling_kernel,
)
from tab_opt.plot import plot_predictions
from tab_opt.vis import averaging, get_rfi_phase
from tab_opt.models import (
    fixed_orbit_rfi_fft_standard,
    fixed_orbit_rfi_compressed_fft_standard_model,
)
from tab_opt.transform import affine_transform_full_inv, affine_transform_diag_inv

import matplotlib.pyplot as plt

key, subkey = random.split(random.PRNGKey(1))

init_plot = 0
true_plot = 0
prior_plot = 0
mcmc = 0
opt = 1
fisher = 1
guide = "map"
epsilon = 5e-2
max_iter = 7_000

num_samples = 10
max_cg_iter = 10_000  # None

N_ant = 64
N_time = 450
sampling = 1

# N_ant = 16
# epsilon = 6e-2
# max_iter = 5_000

# N_ant = 32
# epsilon = 5e-2
# max_iter = 7_000

mem_i = 0

### Define Model
vis_model = fixed_orbit_rfi_compressed_fft_standard_model
model_name = "fixed_orbit_rfi_compressed_fft_standard"

print(f"Model : {model_name}")


def model(args, v_obs=None):
    return fixed_orbit_rfi_fft_standard(args, vis_model, v_obs)


# sim_dir = "/Users/chrisfinlay/Documents/PhD/tabascal/tabascal/examples/data"
sim_dir = "/home/users/f/finlay/tabascal/tabascal/examples/data"
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

################
del bl_uvw
# del ants_uvw
# del ants_xyz
# del freqs
# del rfi_orbit
#################

gains_true = vmap(jnp.interp, in_axes=(None, None, 1))(
    times, times_fine, gains_ants[:, :, 0]
).T
vis_ast_true = vis_ast.reshape(N_time, N_int_samples, N_bl).mean(axis=1)
vis_rfi_true = vis_rfi.reshape(N_time, N_int_samples, N_bl).mean(axis=1)
vis_obs = (
    (ants_to_bl(gains_ants, a1, a2) * vis_ast + vis_rfi)
    .reshape(N_time, N_int_samples, N_bl, 1)
    .mean(axis=1)
) + noise_data
vis_cal = (vis_obs / ants_to_bl(gains_true[:, :, None], a1, a2))[:, :, 0]
flag_rate = (
    100 * jnp.where(jnp.abs(vis_ast_true - vis_cal) > 3 * noise, True, False).mean()
)

bl = jnp.arange(N_bl)

print()
print(
    f"Mean RFI Amp. : {jnp.mean(jnp.abs(vis_rfi_true)):.0f} Jy\nFlag Rate :     {flag_rate:.2f} %"
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


@jit
def f(k, P0=1e3, k0=1e-3, gamma=1.0):
    k_ = (k / k0) ** 2
    return P0 * 0.5 * (jnp.exp(-0.5 * k_) + 1.0 / ((1.0 + k_) ** (gamma / 2)))


### GP Parameters

g_amp_var = (1.0e-2) ** 2  # 1%
g_phase_var = jnp.deg2rad(1.0) ** 2  # 1 degree
# g_amp_var = (0.1e-2) ** 2  # 1%
# g_phase_var = jnp.deg2rad(0.1) ** 2  # 1 degree
g_l = 3 * 60.0 * 60.0  # 3 hours

# rfi_var, rfi_l = 1e3, 15.0
rfi_var, rfi_l = jnp.abs(vis_obs).max() / 20.0, 15.0


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
@jit
def rfi_kernel_fn(v_rfi, rfi_I, resample_rfi):
    return averaging((v_rfi / rfi_I)[:, None] * resample_rfi, N_int_samples)


key, subkey = random.split(key)
phase_error_std = 0e-2
traj_phase_error = jnp.exp(
    1.0j * phase_error_std * random.normal(key, (N_ant, 1)) * times_fine[None, :]
)

rfi_A_perturb = rfi_A_app[..., 0].T * traj_phase_error


def rfi_kernel(vis_rfi, rfi_A, a1, a2):
    resample_rfi = resampling_kernel(rfi_times, times_fine, rfi_var, rfi_l, 1e-8)
    return jnp.array(
        [
            rfi_kernel_fn(
                vis_rfi[:, i, 0],
                (rfi_A[a1[i]] * jnp.conjugate(rfi_A[a2[i]])),
                resample_rfi,
            )
            for i in range(len(a1))
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
    "times_fine": times_fine,
    "g_times": g_times,
    "N_time": N_time,
    "N_ants": N_ant,
    "N_bl": N_bl,
    "a1": a1,
    "a2": a2,
    "bl": bl,
    "n_int": int(N_int_samples),
}

# pow_spec_args = {"P0": 2e3, "k0": 8e-4, "gamma": 1.1}
# pow_spec_args = {"P0": 2e3, "k0": 2e-3, "gamma": 1.1}
# pow_spec_args = {"P0": 2e3, "k0": 2e-3, "gamma": 1.5}
# pow_spec_args = {"P0": 1e3, "k0": 8e-4, "gamma": 1.1}
# pow_spec_args = {"P0": 2e3, "k0": 1e-3, "gamma": 5.0}
pow_spec_args = {"P0": 1e3, "k0": 1e-3, "gamma": 1.0}

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
        "L_G_amp": jnp.linalg.cholesky(kernel(g_times, g_times, g_amp_var, g_l, 1e-8)),
        "L_G_phase": jnp.linalg.cholesky(
            kernel(g_times, g_times, g_phase_var, g_l, 1e-8)
        ),
        "sigma_ast_k": jnp.array([f(k_ast, **pow_spec_args) for _ in range(N_bl)]),
        "L_RFI": jnp.linalg.cholesky(kernel(rfi_times, rfi_times, rfi_var, rfi_l)),
        "resample_g_amp": resampling_kernel(g_times, times, g_amp_var, g_l, 1e-8),
        "resample_g_phase": resampling_kernel(g_times, times, g_phase_var, g_l, 1e-8),
        "rfi_kernel": rfi_kernel(vis_rfi, rfi_A_perturb, a1, a2),
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


################
del gains_ants
del rfi_A_app
del ast_k
del ast_k_mean
del k_ast
del gains_induce
del rfi_induce
del gains_true
del vis_ast_true
del vis_rfi_true
del vis_cal
del noise_data
################


@jit
def inv_transform(params, loc, inv_scaling):
    params_trans = {
        "rfi_r_induce_base": vmap(affine_transform_full_inv, in_axes=(0, None, 0))(
            params["rfi_r_induce"], inv_scaling["L_RFI"], loc["mu_rfi_r"]
        ),
        "rfi_i_induce_base": vmap(affine_transform_full_inv, in_axes=(0, None, 0))(
            params["rfi_i_induce"], inv_scaling["L_RFI"], loc["mu_rfi_i"]
        ),
        "g_amp_induce_base": vmap(affine_transform_full_inv, in_axes=(0, None, 0))(
            params["g_amp_induce"], inv_scaling["L_G_amp"], loc["mu_G_amp"]
        ),
        "g_phase_induce_base": vmap(affine_transform_full_inv, in_axes=(0, None, 0))(
            params["g_phase_induce"], inv_scaling["L_G_phase"], loc["mu_G_phase"]
        ),
        "ast_k_r_base": vmap(affine_transform_diag_inv, in_axes=(0, 0, 0))(
            params["ast_k_r"], inv_scaling["sigma_ast_k"], loc["mu_ast_k_r"]
        ),
        "ast_k_i_base": vmap(affine_transform_diag_inv, in_axes=(0, 0, 0))(
            params["ast_k_i"], inv_scaling["sigma_ast_k"], loc["mu_ast_k_i"]
        ),
    }
    return params_trans


inv_scaling = {
    "L_RFI": jnp.linalg.inv(args["L_RFI"]),
    "L_G_amp": jnp.linalg.inv(args["L_G_amp"]),
    "L_G_phase": jnp.linalg.inv(args["L_G_phase"]),
    "sigma_ast_k": 1.0 / args["sigma_ast_k"],
}

true_params_base = inv_transform(true_params, args, inv_scaling)

init_params_base = inv_transform(init_params, args, inv_scaling)


print()
end_start = datetime.now()
print(f"End Time   : {end_start}")
print(f"Total Time : {end_start - start}")

mem_i += 1
jax.profiler.save_device_memory_profile(
    f"memory_profiles/memory_{N_ant:02}A_{mem_i}.prof"
)

guides = {
    "map": "AutoDelta",
}


def reduced_chi2(pred, true, noise):
    rchi2 = ((jnp.abs(pred - true) / noise) ** 2).sum() / (2 * true.size)
    return rchi2


### Check and Plot Model at init params
pred = Predictive(
    model=model,
    posterior_samples=tree_map(lambda x: x[None, :], init_params_base),
    batch_ndims=1,
)
key, subkey = random.split(key)
init_pred = pred(subkey, args=args)
rchi2 = reduced_chi2(init_pred["vis_obs"][0], vis_obs[:, :, 0].T, noise)
print()
print(f"Reduced Chi^2 @ init: {rchi2}")
print()

### Check and Plot Model at true parameters
pred = Predictive(
    model=model,
    posterior_samples=tree_map(lambda x: x[None, :], true_params_base),
    batch_ndims=1,
)
key, subkey = random.split(key)
true_pred = pred(subkey, args=args)
rchi2 = reduced_chi2(true_pred["vis_obs"][0], vis_obs[:, :, 0].T, noise)
print()
print(f"Reduced Chi^2 @ true: {rchi2}")
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
print(f"Init/True Plot Time : {end_true - end_start}")

mem_i += 1
jax.profiler.save_device_memory_profile(
    f"memory_profiles/memory_{N_ant:02}A_{mem_i}.prof"
)

### Check and Plot Model at prior parameters
key, subkey = random.split(key)
if prior_plot:
    pred = Predictive(model, num_samples=100)
    prior_pred = pred(subkey, args=args)
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
print(f"Prior Plot Time : {end_prior - end_true}")

### Run Inference
key, *subkeys = random.split(key, 3)
if mcmc:
    num_warmup = 500
    num_samples = 1000

    nuts_kernel = NUTS(model, dense_mass=False)  # [('g_phase_0', 'g_phase_1')])
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(
        subkeys[0],
        args=args,
        v_obs=v_obs_ri,
        extra_fields=("potential_energy",),
        init_params=true_params_base,
    )

    pred = Predictive(model, posterior_samples=mcmc.get_samples())
    mcmc_pred = pred(subkeys[1], args=args)
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
print(f"MCMC Plot Time : {end_mcmc - end_prior}")

mem_i += 1
jax.profiler.save_device_memory_profile(
    f"memory_profiles/memory_{N_ant:02}A_{mem_i}.prof"
)

key, *subkeys = random.split(key, 3)
if opt:
    guide_family = guides[guide]
    vi_results, vi_guide = run_svi(
        model=model,
        args=args,
        obs=v_obs_ri,
        max_iter=max_iter,
        guide_family=guide_family,
        init_params={
            **{k + "_auto_loc": v for k, v in init_params_base.items()},
        },
        epsilon=epsilon,
        key=subkeys[0],
    )
    vi_params = vi_results.params
    vi_pred = svi_predict(
        model=model,
        guide=vi_guide,
        vi_params=vi_params,
        args=args,
        num_samples=1,
        key=subkeys[1],
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
    print(f"Reduced Chi^2 @ opt: {rchi2}")
    print()

    plt.semilogy(vi_results.losses)
    plt.savefig(f"plots/{model_name}_opt_loss.pdf", format="pdf")

    del vi_pred
    del vi_results


print()
end_opt = datetime.now()
print(f"End Time  : {end_opt}")
print(f"Opt Plot Time : {end_opt - end_prior}")

mem_i += 1
jax.profiler.save_device_memory_profile(
    f"memory_profiles/memory_{N_ant:02}A_{mem_i}.prof"
)

key, *subkeys = random.split(key, 3)
if fisher and rchi2 < 1.1:
    f_model = lambda params, args: vis_model(params, args)[0]
    model_flat = lambda params: f_model_flat(f_model, params, args)

    post_mean = {k[:-9]: v for k, v in vi_params.items()} if opt else true_params_base

    dtheta = post_samples(
        model_flat,
        post_mean,
        flatten_obs(vis_obs),
        noise,
        num_samples,
        subkeys[0],
        max_cg_iter,
    )

    samples = tree_map(jnp.add, post_mean, dtheta)

    pred = Predictive(model, posterior_samples=samples)
    fisher_pred = pred(subkeys[1], args=args)
    plot_predictions(
        times=times,
        pred=fisher_pred,
        args=args,
        type="fisher_opt" if opt else "fisher_true",
        model_name=model_name,
        max_plots=5,
    )

print()
end_fisher = datetime.now()
print(f"End Time  : {end_fisher}")
print(f"Fisher Plot Time : {end_fisher - end_opt}")

print()
end_final = datetime.now()
print(f"End Time  : {end_final}")
print(f"Total Time : {end_final - start}")

mem_i += 1
jax.profiler.save_device_memory_profile(
    f"memory_profiles/memory_{N_ant:02}A_{mem_i}.prof"
)
