from datetime import datetime

print()
start = datetime.now()
print(f"Start Time : {start}")

import os

import jax.numpy as jnp
from jax import random, vmap, jacrev, jit
from jax.tree_util import tree_map

from jax.flatten_util import ravel_pytree as flatten
from jax.lax import scan
import jax

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
from tab_opt.vis import averaging
from tab_opt.models import (
    fixed_orbit_real_rfi_compressed5,
)

import matplotlib.pyplot as plt

true_plot = 0
prior_plot = 0
mcmc = 0
opt = 1
natgrad = 0
max_iter = 1_000
# guide = "multinormal"
# guide = "normal"
# guide = "laplace"
guide = "map"
epsilon = 3e-2
max_iter = 3_000

N_ant = 32
N_time = 450
sampling = 1


### Define Model
model = fixed_orbit_real_rfi_compressed5
model_name = "fixed_orbit_real_rfi_compressed5"

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

### GP Parameters
vis_var, vis_l = get_vis_gp_params(ants_uvw, vis_ast, a1, a2, noise)

g_amp_var = (1e-2) ** 2
g_phase_var = jnp.deg2rad(1.0) ** 2
g_l = 1e3

rfi_var, rfi_l = 1e3, 15.0
rfi_var, rfi_l = jnp.abs(vis_obs).max(), 30.0

### Visibility Sampling Times
# vis_ast_times = get_times(times, vis_l.min())
# vis_ast_induce = jnp.array(
#     [jnp.interp(vis_ast_times, times, vis_ast_true[:, i]) for i in range(N_bl)]
# )
# n_vis_times = len(vis_ast_times)


def round_multiple(x, multiple):
    factor = jnp.ceil(x / multiple)
    return factor * multiple


vis_ast_times = [get_times(times, round_multiple(l, 50)) for l in vis_l]
vis_ast_induce = [
    jnp.interp(vis_ast_times[i], times, vis_ast_true[:, i]) for i in range(N_bl)
]
n_vis_times = jnp.array([len(vis_ast_times[i]) for i in range(N_bl)])

# import sys

# sys.exit(0)

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
print(f"AST   : {n_vis_times.min(): 4} - {n_vis_times.max()}")
print()
print(
    f"Number of parameters : {(2 * N_ant - 1) * n_g_times + n_vis_times.sum() + N_ant * n_rfi_times}"
)
print(f"Number of data points: {2* N_bl * N_time}")


### Define RFI Kernel
# rfi_kernel = vmap(averaging, in_axes=(2, None), out_axes=2)(
#     (vis_rfi[:, :, 0] / jnp.abs(vis_rfi)[:, :, 0])[:, None, :]
#     * resampling_kernel(rfi_times, times_fine, rfi_var, rfi_l, 1e-6)[:, :, None],
#     N_int_samples,
# )

resample_rfi = resampling_kernel(rfi_times, times_fine, rfi_var, rfi_l, 1e-8)

# @jit
# def rfi_kernel_fn(v_rfi):
#     return averaging((v_rfi / jnp.abs(v_rfi))[:, None] * resample_rfi, N_int_samples)
# rfi_kernel = jnp.transpose(
#     jnp.array([rfi_kernel_fn(v_rfi) for v_rfi in vis_rfi[:, :, 0].T]), axes=(1, 2, 0)
# )


@jit
def rfi_kernel_fn(v_rfi, rfi_I):
    return averaging((v_rfi / rfi_I)[:, None] * resample_rfi, N_int_samples)


rfi_I = rfi_A_app[:, a1, 0] * jnp.conjugate(rfi_A_app[:, a2, 0])

rfi_kernel = jnp.transpose(
    jnp.array(
        [
            rfi_kernel_fn(v_rfi, rfi_I)
            for v_rfi, rfi_I in zip(vis_rfi[:, :, 0].T, rfi_I.T)
        ]
    ),
    axes=(1, 2, 0),
)

### Define True Parameters
true_params = {
    **{f"g_amp_induce": jnp.abs(gains_induce)},
    **{f"g_phase_induce": jnp.angle(gains_induce[:-1])},
    **{f"rfi_r_induce": rfi_induce.real},
    **{f"vis_r_induce_{i}": v.real for i, v in enumerate(vis_ast_induce)},
    **{f"vis_i_induce_{i}": v.imag for i, v in enumerate(vis_ast_induce)},
}

v_obs_ri = jnp.concatenate([vis_obs[:, :, 0].real, vis_obs[:, :, 0].imag], axis=0).T

# Set Constant Parameters
args = {
    "noise": noise if noise > 0 else 0.2,
    "vis_ast_true": vis_ast_true.T,
    "vis_rfi_true": vis_rfi_true.T,
    "gains_true": gains_true.T,
    "N_time": N_time,
    "N_ants": N_ant,
    "N_bl": N_bl,
    "a1": a1,
    "a2": a2,
    "bl": bl,
    "n_int": N_int_samples,
}

from tab_opt.vis import pad_vis

### Define Prior Parameters
args.update(
    {
        "mu_G_amp": jnp.abs(gains_induce),
        "mu_G_phase": jnp.angle(gains_induce[:-1]),
        # "mu_rfi_r": true_params["rfi_r_induce"],
        "mu_rfi_r": jnp.zeros_like(rfi_induce.real),
        "mu_vis_r": [
            jnp.zeros_like(true_params[f"vis_r_induce_{i}"]) for i in range(N_bl)
        ],
        "mu_vis_i": [
            jnp.zeros_like(true_params[f"vis_i_induce_{i}"]) for i in range(N_bl)
        ],
        # "mu_vis_r": jnp.zeros((N_bl, n_vis_times)),
        # "mu_vis_i": jnp.zeros((N_bl, n_vis_times)),
        "L_G_amp": jnp.linalg.cholesky(kernel(g_times, g_times, g_amp_var, g_l)),
        "L_G_phase": jnp.linalg.cholesky(kernel(g_times, g_times, g_phase_var, g_l)),
        "L_vis": [
            jnp.linalg.cholesky(
                kernel(vis_ast_times[i], vis_ast_times[i], vis_var[i], vis_l[i])
            )
            for i in range(N_bl)
        ],
        "L_RFI_amp": jnp.linalg.cholesky(kernel(rfi_times, rfi_times, rfi_var, rfi_l)),
        # "resample_vis": [
        #     resampling_kernel(vis_ast_times[i], times, vis_var[i], vis_l[i], 1e-6), n_vis_times.max()
        #     for i in range(N_bl)
        # ],
        "resample_vis": jnp.array(
            [
                vmap(lambda x: pad_vis(x, int(n_vis_times.max())))(
                    resampling_kernel(
                        vis_ast_times[i], times, vis_var[i], vis_l[i], 1e-6
                    )
                )
                for i in range(N_bl)
            ]
        ),
        "resample_g_amp": resampling_kernel(g_times, times, g_amp_var, g_l, 1e-8),
        "resample_g_phase": resampling_kernel(g_times, times, g_phase_var, g_l, 1e-8),
        "rfi_kernel": rfi_kernel,
    }
)


def transform_params(x, loc, scale_tril):
    affine = lambda x, loc, scale_tril: scale_tril @ x + loc
    x_trans = tree_map(vmap(affine), x, loc, scale_tril)
    return x_trans


def inv_transform_params(x, loc, scale_tril_inv):
    affine = lambda x, loc, scale_tril_inv: scale_tril_inv @ (x - loc)
    x_trans = tree_map(vmap(affine), x, loc, scale_tril_inv)
    return x_trans


prior_mu = {
    **{"g_amp_induce": args["mu_G_amp"]},
    **{"g_phase_induce": args["mu_G_phase"]},
    **{"rfi_r_induce": args["mu_rfi_r"]},
    **{f"vis_r_induce_{i}": mu for i, mu in enumerate(args["mu_vis_r"])},
    **{f"vis_i_induce_{i}": mu for i, mu in enumerate(args["mu_vis_i"])},
}

param_scale_tril = {
    **{
        f"g_amp_induce": jnp.broadcast_to(
            args["L_G_amp"], (N_ant, n_g_times, n_g_times)
        )
    },
    **{
        f"g_phase_induce": jnp.broadcast_to(
            args["L_G_phase"], (N_ant - 1, n_g_times, n_g_times)
        )
    },
    **{
        f"rfi_r_induce": jnp.broadcast_to(
            args["L_RFI_amp"], (N_ant, n_rfi_times, n_rfi_times)
        )
    },
    **{f"vis_r_induce_{i}": L for i, L in enumerate(args["L_vis"])},
    **{f"vis_i_induce_{i}": L for i, L in enumerate(args["L_vis"])},
}
param_scale_tril_inv = {
    **{
        f"g_amp_induce": jnp.broadcast_to(
            jnp.linalg.inv(args["L_G_amp"]), (N_ant, n_g_times, n_g_times)
        )
    },
    **{
        f"g_phase_induce": jnp.broadcast_to(
            jnp.linalg.inv(args["L_G_phase"]), (N_ant - 1, n_g_times, n_g_times)
        )
    },
    **{
        f"rfi_r_induce": jnp.broadcast_to(
            jnp.linalg.inv(args["L_RFI_amp"]), (N_ant, n_rfi_times, n_rfi_times)
        )
    },
    **{f"vis_r_induce_{i}": jnp.linalg.inv(L) for i, L in enumerate(args["L_vis"])},
    **{f"vis_i_induce_{i}": jnp.linalg.inv(L) for i, L in enumerate(args["L_vis"])},
}

# true_params_base = inv_transform_params(true_params, prior_mu, param_scale_tril_inv)
# true_params_base = {k + "_base": v for k, v in true_params_base.items()}

# affine_inv = lambda x, loc, scale_tril_inv: scale_tril_inv @ (x - loc)

# true_params_base = {
#     "rfi_r_induce": true_params["rfi_r_induce"],
#     "g_amp_induce": true_params["g_amp_induce"],
#     "g_phase_induce": true_params["g_phase_induce"],

#     "vis_r_induce_base": vmap(affine_inv)(
#         true_params["vis_r_induce"],
#         prior_mu["vis_r_induce"],
#         param_scale_tril_inv["vis_r_induce"],
#     ),
#     "vis_i_induce_base": vmap(affine_inv)(
#         true_params["vis_i_induce"],
#         prior_mu["vis_i_induce"],
#         param_scale_tril_inv["vis_i_induce"],
#     ),
# }

true_params_base = true_params

param_scale = tree_map(vmap(jnp.diag), param_scale_tril)

n_sigma_init = 1e-1

flat_params, unflatten = flatten(true_params)
init_vec = unflatten(
    n_sigma_init * random.normal(random.PRNGKey(1), (len(flat_params),))
)
# init_params = transform_params(init_vec, true_params, param_scale_tril)

init_params = init_vec

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


from numpyro.infer.util import log_density


### Check and Plot Model at true parameters
pred = Predictive(
    model=model,
    posterior_samples=tree_map(lambda x: x[None, :], true_params_base),
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
            **{k + "_auto_scale": v for k, v in param_scale.items()},
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
