import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from tab_opt.dist import MVN
from tab_opt.vis import get_ast_vis, get_ast_vis2, get_obs_vis, get_rfi_vis, get_gains, rmse
from jax import jit
from functools import partial


# @jit
def fixed_orbit_real_rfi(params):
    n_time = params["N_time"]
    n_ant = params["N_ants"]
    n_bl = params["N_bl"]
    a1 = params["a1"]
    a2 = params["a2"]

    with numpyro.plate("N_ants", n_ant):
        rfi_amp = MVN(
            "rfi_r_induce", params["mu_rfi"], params["L_RFI_amp"]
        )
        # G_amp = MVN("g_amp_induce", params["mu_G_amp"], params["L_G_amp"][None,:,:]*jnp.ones((n_ant,1,1)))
        G_amp = numpyro.sample(
            "g_amp_induce",
            dist.MultivariateNormal(
                params["mu_G_amp"],
                params["cov_G_amp"],
            ),
        )

    with numpyro.plate("N_ants-1", n_ant - 1):
        G_phase = numpyro.sample(
            "g_phase_induce",
            dist.MultivariateNormal(params["mu_G_phase"], params["cov_G_phase"]),
        )

    gains = numpyro.deterministic("gains", get_gains(G_amp, G_phase, params["resample_g_amp"], params["resample_g_phase"]))

    rfi_vis = numpyro.deterministic(
        "rfi_vis", get_rfi_vis(rfi_amp, params["rfi_kernel"], a1, a2).T
    )

    # vis_r = [
    #     MVN(f"vis_r_{i}", params["mu_vis"][i], params["L_vis"][i]) for i in range(n_bl)
    # ]
    # vis_i = [
    #     MVN(f"vis_i_{i}", params["mu_vis"][i], params["L_vis"][i]) for i in range(n_bl)
    # ]
    # ast_vis = numpyro.deterministic(
    #     "ast_vis", get_ast_vis(vis_r, vis_i, params["resample_vis"])
    # )

    with numpyro.plate("N_bl", n_bl):
        vis_r = MVN("vis_r", params["mu_vis"], params["L_vis"])
        vis_i = MVN("vis_i", params["mu_vis"], params["L_vis"])

    ast_vis = numpyro.deterministic(
        "ast_vis", get_ast_vis2(vis_r, vis_i, params["resample_vis"])
    )

    vis_obs = numpyro.deterministic(
        "vis_obs", get_obs_vis(ast_vis, rfi_vis, gains, a1, a2)
    )

    numpyro.deterministic(
        "rmse_ast",
        rmse(ast_vis, params["vis_ast_true"]) / jnp.sqrt(2),
    )
    numpyro.deterministic(
        "rmse_rfi",
        rmse(rfi_vis, params["vis_rfi_true"]) / jnp.sqrt(2),
    )
    numpyro.deterministic(
        "rmse_gains",
        rmse(gains, params["gains_true"]) / jnp.sqrt(2),
    )

    return numpyro.sample(
        "obs",
        dist.Normal(
            jnp.concatenate([vis_obs.real, vis_obs.imag], axis=1), params["noise"]
        ),
        obs=params["vis_obs"],
    )

def fixed_orbit_complex_rfi(params):
    n_time = params["N_time"]
    n_ant = params["N_ants"]
    n_bl = params["N_bl"]
    a1 = params["a1"]
    a2 = params["a2"]

    with numpyro.plate("N_ants", n_ant):
        rfi_amp = MVN(
            "rfi_amp_induce", params["mu_rfi_amp"], params["L_RFI_amp"]
        )
        G_amp = numpyro.sample(
            "g_amp_induce",
            dist.MultivariateNormal(
                params["mu_G_amp"],
                params["cov_G_amp"],
            ),
        )

    with numpyro.plate("N_ants-1", n_ant - 1):
        rfi_phase = MVN(
            "rfi_phase", params["mu_rfi_phase"], params["L_RFI_phase"]
        )
        G_phase = numpyro.sample(
            "g_phase_induce",
            dist.MultivariateNormal(params["mu_G_phase"], params["cov_G_phase"]),
        )

    G_amp = numpyro.deterministic("g_amp", G_amp @ params["resample_g_amp"].T)
    G_phase = numpyro.deterministic(
        "g_phase",
        jnp.concatenate(
            [G_phase @ params["resample_g_phase"].T, jnp.zeros((1, n_time))], axis=0
        ),
    )

    gains = numpyro.deterministic("gains", G_amp * jnp.exp(1.0j * G_phase))

    rfi_vis = numpyro.deterministic(
        "rfi_vis", get_rfi_vis(rfi_amp * jnp.exp(1.0j * rfi_phase), params["rfi_kernel"], a1, a2).T
    )

    vis_r = [
        MVN(f"vis_r_{i}", params["mu_vis"][i], params["L_vis"][i]) for i in range(n_bl)
    ]
    vis_i = [
        MVN(f"vis_i_{i}", params["mu_vis"][i], params["L_vis"][i]) for i in range(n_bl)
    ]

    ast_vis = numpyro.deterministic(
        "ast_vis", get_ast_vis(vis_r, vis_i, params["resample_vis"])
    )

    vis_obs = numpyro.deterministic(
        "vis_obs", get_obs_vis(ast_vis, rfi_vis, gains, a1, a2)
    )

    numpyro.deterministic(
        "rmse_ast",
        rmse(ast_vis, params["vis_ast_true"]) / jnp.sqrt(2),
    )
    numpyro.deterministic(
        "rmse_rfi",
        rmse(rfi_vis, params["vis_rfi_true"]) / jnp.sqrt(2),
    )
    numpyro.deterministic(
        "rmse_gains",
        rmse(gains, params["gains_true"]) / jnp.sqrt(2),
    )

    return numpyro.sample(
        "obs",
        dist.Normal(
            jnp.concatenate([vis_obs.real, vis_obs.imag], axis=1), params["noise"]
        ),
        obs=params["vis_obs"],
    )