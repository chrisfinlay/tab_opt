from datetime import datetime

print()
start = datetime.now()
print(f"Start Time : {start}")

import os

import jax.numpy as jnp
from jax import random, vmap, jacrev, jit
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree
from jax.lax import scan
import jax

from numpyro.infer import MCMC, NUTS, Predictive

from tabascal.jax.interferometry import ants_to_bl
from tab_opt.data import extract_data
from tab_opt.opt import run_svi, svi_predict, fisher_diag_inv, fisher_diag_inv2
from tab_opt.gp import (
    get_times,
    get_vis_gp_params,
    kernel,
    resampling_kernel,
)
from tab_opt.plot import plot_predictions
from tab_opt.vis import averaging, get_rfi_vis1, get_rfi_vis2, rmse
from tab_opt.models import (
    fixed_orbit_real_rfi,
    fixed_orbit_real_rfi_compressed,
    fixed_orbit_real_rfi_compressed2,
)

import matplotlib.pyplot as plt

true_plot = 0
prior_plot = 0
mcmc = 0
opt = 0
natgrad = 1
# guide = "multinormal"
# guide = "normal"
guide = "map"
max_iter = 1_000

N_ant = 4
N_time = 450
sampling = 1


### Define Model
# model = fixed_orbit_real_rfi
# model_name = "fixed_orbit_real_rfi"
# model = fixed_orbit_real_rfi_compressed
# model_name = "fixed_orbit_real_rfi_compressed"
model = fixed_orbit_real_rfi_compressed2
model_name = "fixed_orbit_real_rfi_compressed2"

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
)

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

vis_var = vis_var.max()
vis_l = vis_l.min()

g_amp_var = (1e-2) ** 2
g_phase_var = jnp.deg2rad(1.0) ** 2
g_l = 10e3

rfi_var, rfi_l = 1e3, 15.0

# vis_l = 4
# g_l = 4
# rfi_l = 4

### Visibility Sampling Times
# vis_ast_times = [get_times(times, l) for l in vis_l]
# vis_ast_induce = [
#     jnp.interp(vis_ast_times[i], times, vis_ast_true[:, i]) for i in range(N_bl)
# ]
# n_vis_times = jnp.array([len(vis_ast_times[i]) for i in range(N_bl)])

vis_ast_times = get_times(times, vis_l)
vis_ast_induce = vmap(jnp.interp, in_axes=(None, None, 1))(
    vis_ast_times, times_fine, vis_ast[:, :, 0]
)
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
print(
    f"Number of parameters : {N_ant * len(g_times) - 1 + N_bl * n_vis_times.max() + N_ant * len(rfi_times)}"
)
print(f"Number of data points: {2* N_bl * N_time}")


### Define RFI Kernel
rfi_amp = jnp.abs(vis_rfi)[:, :, 0]
rfi_phasor = vis_rfi[:, :, 0] / rfi_amp
rfi_kernel = vmap(averaging, in_axes=(2, None), out_axes=2)(
    rfi_phasor[:, None, :]
    * resampling_kernel(rfi_times, times_fine, rfi_var, rfi_l, 1e-6)[:, :, None],
    N_int_samples,
)


### Define True Parameters
true_params = {
    **{"g_amp_induce": jnp.abs(gains_induce)},
    **{"g_phase_induce": jnp.angle(gains_induce[:-1])},
    **{"vis_r_induce": vis_ast_induce.real},
    **{"vis_i_induce": vis_ast_induce.imag},
    **{"rfi_r_induce": rfi_induce.real},
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

### Define Prior Parameters
args.update(
    {
        "mu_G_amp": true_params["g_amp_induce"],
        "mu_G_phase": true_params["g_phase_induce"],
        "mu_vis_r": true_params["vis_r_induce"],
        "mu_vis_i": true_params["vis_i_induce"],
        "mu_rfi_r": true_params["rfi_r_induce"],
        # "mu_vis_r": jnp.zeros(len(vis_ast_times)),
        # "mu_vis_i": jnp.zeros(len(vis_ast_times)),
        # "mu_rfi_r": jnp.zeros(len(rfi_times)),
        # 'inv_cov_G_amp': inv_kernel(g_times[:,None],
        #                             g_amp_var, g_l),
        # 'inv_cov_G_phase': inv_kernel(g_times[:,None],
        #                               g_phase_var, g_l),
        # 'inv_cov_vis': {i: inv_kernel(vis_ast_times[i][:,None], vis_var[i],
        #                               vis_l[i]) for i in range(N_bl)},
        # 'inv_cov_rfi': inv_kernel(rfi_times[:,None], rfi_var, rfi_l),
        # "cov_G_amp": kernel(g_times, g_times, g_amp_var, g_l),
        # "cov_G_phase": kernel(g_times, g_times, g_phase_var, g_l),
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
        "L_vis": jnp.linalg.cholesky(
            kernel(vis_ast_times, vis_ast_times, vis_var, vis_l)
        ),
        "L_RFI_amp": jnp.linalg.cholesky(kernel(rfi_times, rfi_times, rfi_var, rfi_l)),
        # "resample_vis": {
        #     i: resampling_kernel(vis_ast_times[i], times, vis_var[i], vis_l[i], 1e-3)
        #     for i in range(N_bl)
        # },
        "resample_vis": resampling_kernel(vis_ast_times, times, vis_var, vis_l, 1e-10),
        "resample_g_amp": resampling_kernel(g_times, times, g_amp_var, g_l, 1e-10),
        "resample_g_phase": resampling_kernel(g_times, times, g_phase_var, g_l, 1e-10),
        "resample_rfi": resampling_kernel(rfi_times, times_fine, rfi_var, rfi_l, 1e-10),
        "rfi_phase": rfi_phasor.T,
        "rfi_kernel": rfi_kernel,
    }
)


# print(true_params["rfi_r_induce"].shape)
# print(args["resample_rfi"].shape)
# print(args["rfi_phase"].shape)

vis_rfi_test1 = get_rfi_vis1(true_params["rfi_r_induce"], rfi_kernel, a1, a2).T
rel_rmse1 = rmse(vis_rfi_test1 / vis_rfi_true, 1, axis=(0, 1))

vis_rfi_test2 = get_rfi_vis2(
    true_params["rfi_r_induce"],
    args["resample_rfi"],
    args["rfi_phase"],
    a1,
    a2,
    bl,
    N_int_samples,
).T
rel_rmse2 = rmse(vis_rfi_test2 / vis_rfi_true, 1, axis=(0, 1))


true_params = {
    **{f"g_amp_induce_{i}": x for i, x in enumerate(jnp.abs(gains_induce))},
    **{f"g_phase_induce_{i}": x for i, x in enumerate(jnp.angle(gains_induce[:-1]))},
    **{f"vis_r_induce_{i}": vis_ast_induce[i].real for i in range(N_bl)},
    **{f"vis_i_induce_{i}": vis_ast_induce[i].imag for i in range(N_bl)},
    **{f"rfi_r_induce_{i}": x for i, x in enumerate(rfi_induce.real)},
}

print()
print(
    "Assuming an orbit and compressing all resampling and \nphase multiplication and averaging into a single matrix"
)
print("-----------------------------------------------------")
print(f"Relative RMSE (compressed) : {rel_rmse1: .2E}")
print(f"Relative RMSE (standard)   : {rel_rmse2: .2E}")

print()
end_start = datetime.now()
print(f"End Time   : {end_start}")
print(f"Total Time : {end_start - start}")

guides = {
    "map": "AutoDelta",
    "normal": "AutoDiagonalNormal",
    "multinormal": "AutoMultivariateNormal",
}

# from numpyro.infer import log_likelihood

# print(
#     2
#     * log_likelihood(
#         model,
#         posterior_samples=tree_map(lambda x: x[None, :], true_params),
#         args=args,
#     )["obs"].sum()
#     / (2 * vis_obs.size)
# )

# plt.plot(
#     log_likelihood(
#         model,
#         posterior_samples=tree_map(lambda x: x[None, :], true_params),
#         args=args,
#     )["obs"][0].T
# )
# plt.show()


def reduced_chi2(pred, true, noise):
    rchi2 = ((jnp.abs(pred - true) / noise) ** 2).sum() / (2 * true.size)
    return rchi2


from jax import hessian
from numpyro.infer.util import log_density


### Check and Plot Model at true parameters
if true_plot:
    pred = Predictive(
        model=model,
        posterior_samples=tree_map(lambda x: x[None, :], true_params),
        batch_ndims=1,
    )
    init_pred = pred(random.PRNGKey(2), args=args)
    rchi2 = reduced_chi2(init_pred["vis_obs"][0], vis_obs[:, :, 0].T, noise)
    print()
    print(f"Reduced Chi^2 : {rchi2}")
    print()

    plot_predictions(
        times=times,
        pred=init_pred,
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
        max_iter=max_iter,
        guide_family=guide_family,
        init_params=true_params,
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

    # plt.semilogy(vi_results.losses)
    # plt.show()

print()
end_opt = datetime.now()
print(f"End Time  : {end_opt}")
print(f"Plot Time : {end_opt - end_prior}")

if natgrad:

    @jit
    def loss(x):
        return (
            -1
            * log_density(
                model,
                model_args=(),
                model_kwargs={**kwargs_model, **kwargs_data},
                params=x,
            )[0]
        )

    @jit
    def grad_and_loss(x):
        value, gradients = jax.value_and_grad(loss)(x)
        return ravel_pytree(gradients)[0], value

    @jit
    def f_inv(x):
        return fisher_diag_inv(model, x, kwargs_model, kwargs_data)

    from tqdm import tqdm

    n_steps = 1000000
    n_inner = 10000
    n_outer = n_steps // n_inner

    losses = jnp.empty(n_steps)

    flatten = lambda x: ravel_pytree(x)[0]
    flat_params, unflatten = ravel_pytree(true_params)

    # params = unflatten(jnp.zeros(len(flat_params)))
    params = true_params
    pred = Predictive(model, num_samples=10)
    prior_pred = pred(random.PRNGKey(2), args=args)
    params = tree_map(lambda x: x.mean(axis=0), prior_pred)
    params = {key: params[key] for key in true_params.keys()}

    pred = Predictive(
        model=model,
        posterior_samples=params,
        batch_ndims=0,
    )
    natgrad_pred = pred(random.PRNGKey(2), args=args)
    rchi2 = reduced_chi2(natgrad_pred["vis_obs"], vis_obs[:, :, 0].T, noise)
    print()
    print(f"Reduced Chi^2 : {rchi2}")
    print()

    step_size = 3e-1

    kwargs_model = {"args": args}
    kwargs_data = {"v_obs": v_obs_ri}

    @jit
    def apply_grads(params_and_losses, i):
        params, losses = params_and_losses
        grads, loss_ = grad_and_loss(params)
        losses = losses.at[i].set(loss_)
        params_new = unflatten(flatten(params) - step_size * grads * f_inv_)
        return (params_new, losses), i + 1

    f_inv_ = f_inv(params)
    new_grads = apply_grads((params, losses), 0)

    print(new_grads)

    pbar = tqdm(range(0, n_outer))
    pbar.set_description(f"NL Post = {loss(params):.0f} ")
    for i in pbar:
        f_inv_ = f_inv(params)
        (params, losses), _ = scan(
            apply_grads, (params, losses), (i * n_inner) + jnp.arange(n_inner)
        )
        current_loss = losses[(i + 1) * n_inner - 1]

        pbar.set_description(f"NL Post = {current_loss:.0f} ")

    pred = Predictive(
        model=model,
        posterior_samples=tree_map(lambda x: x[None, ...], params),
        batch_ndims=1,
    )
    natgrad_pred = pred(random.PRNGKey(2), args=args)
    rchi2 = reduced_chi2(natgrad_pred["vis_obs"][0], vis_obs[:, :, 0].T, noise)
    print()
    print(f"Reduced Chi^2 : {rchi2}")
    print()

    plot_predictions(
        times=times,
        pred=natgrad_pred,
        args=args,
        type="natgrad",
        model_name=model_name,
        max_plots=10,
    )

print()
end_natgrad = datetime.now()
print(f"End Time  : {end_natgrad}")
print(f"Infer Time : {end_natgrad - end_opt}")

plt.semilogy(losses)
plt.axhline(loss(true_params), color="k", linestyle="--")
plt.show()

print()
end_final = datetime.now()
print(f"End Time  : {end_final}")
print(f"Infer Time : {end_final - end_natgrad}")
