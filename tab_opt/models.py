import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from tab_opt.dist import MVN, Normal
from tab_opt.vis import (
    get_ast_vis,
    get_ast_vis2,
    get_obs_vis,
    get_rfi_vis1,
    get_rfi_vis2,
    get_gains,
    rmse,
)
from jax import jit, vmap
from functools import partial

import numpy as np


def fixed_orbit_real_rfi(args, v_obs=None):
    n_time = args["N_time"]
    n_ant = args["N_ants"]
    n_bl = args["N_bl"]
    a1 = args["a1"]
    a2 = args["a2"]

    # with numpyro.plate("N_ants", n_ant):
    #     # rfi_amp = MVN("rfi_r_induce", args["mu_rfi_r"], args["L_RFI_amp"])
    #     rfi_amp = numpyro.sample(
    #         "rfi_r_induce",
    #         dist.MultivariateNormal(
    #             loc=args["mu_rfi_r"],
    #             scale_tril=args["L_RFI_amp"] / 1e1,
    #         ),
    #     )
    #     G_amp = numpyro.sample(
    #         "g_amp_induce",
    #         dist.MultivariateNormal(
    #             loc=args["mu_G_amp"],
    #             scale_tril=args["L_G_amp"],
    #         ),
    #     )

    rfi_amp = numpyro.sample(
        "rfi_r_induce",
        dist.MultivariateNormal(
            loc=args["mu_rfi_r"],
            scale_tril=args["L_RFI_amp"] / 1e1,
        ),
    )
    G_amp = numpyro.sample(
        "g_amp_induce",
        dist.MultivariateNormal(
            loc=args["mu_G_amp"],
            scale_tril=args["L_G_amp"],
        ),
    )

    # with numpyro.plate("N_ants-1", n_ant - 1):
    #     G_phase = numpyro.sample(
    #         "g_phase_induce",
    #         dist.MultivariateNormal(
    #             loc=args["mu_G_phase"], scale_tril=args["L_G_phase"]
    #         ),
    #     )

    G_phase = numpyro.sample(
        "g_phase_induce",
        dist.MultivariateNormal(loc=args["mu_G_phase"], scale_tril=args["L_G_phase"]),
    )

    gains = numpyro.deterministic(
        "gains",
        get_gains(G_amp, G_phase, args["resample_g_amp"], args["resample_g_phase"]),
    )

    rfi_vis = numpyro.deterministic(
        "rfi_vis",
        get_rfi_vis2(
            rfi_amp,
            args["resample_rfi"],
            args["rfi_phase"],
            a1,
            a2,
            args["bl"],
            args["n_int"],
        ),
    )

    # vis_r = [
    #     MVN(f"vis_r_{i}", args["mu_vis"][i], args["L_vis"][i]) for i in range(n_bl)
    # ]
    # vis_i = [
    #     MVN(f"vis_i_{i}", args["mu_vis"][i], args["L_vis"][i]) for i in range(n_bl)
    # ]
    # ast_vis = numpyro.deterministic(
    #     "ast_vis", get_ast_vis1(vis_r, vis_i, args["resample_vis"])
    # )

    # with numpyro.plate("N_bl", n_bl):
    #     vis_r = numpyro.sample(
    #         "vis_r_induce",
    #         dist.MultivariateNormal(
    #             loc=args["mu_vis_r"], scale_tril=args["L_vis"] / 10
    #         ),
    #     )
    #     vis_i = numpyro.sample(
    #         "vis_i_induce",
    #         dist.MultivariateNormal(
    #             loc=args["mu_vis_i"], scale_tril=args["L_vis"] / 10
    #         ),
    #     )
    # vis_r = MVN("vis_r_induce", args["mu_vis_r"], args["L_vis"])
    # vis_i = MVN("vis_i_induce", args["mu_vis_i"], args["L_vis"])

    vis_r = numpyro.sample(
        "vis_r_induce",
        dist.MultivariateNormal(loc=args["mu_vis_r"], scale_tril=args["L_vis"] / 10),
    )
    vis_i = numpyro.sample(
        "vis_i_induce",
        dist.MultivariateNormal(loc=args["mu_vis_i"], scale_tril=args["L_vis"] / 10),
    )

    ast_vis = numpyro.deterministic(
        "ast_vis", get_ast_vis2(vis_r, vis_i, args["resample_vis"])
    )

    vis_obs = numpyro.deterministic(
        "vis_obs", get_obs_vis(ast_vis, rfi_vis, gains, a1, a2)
    )

    numpyro.deterministic(
        "rmse_ast",
        rmse(ast_vis, args["vis_ast_true"]) / jnp.sqrt(2),
    )
    numpyro.deterministic(
        "rmse_rfi",
        rmse(rfi_vis, args["vis_rfi_true"]) / jnp.sqrt(2),
    )
    numpyro.deterministic(
        "rmse_gains",
        rmse(gains, args["gains_true"]) / jnp.sqrt(2),
    )

    if v_obs is not None:
        return numpyro.sample(
            "obs",
            dist.Normal(
                jnp.concatenate([vis_obs.real, vis_obs.imag], axis=1), args["noise"]
            ),
            obs=v_obs,
        )


###############################################################################


# @jit
def fixed_orbit_real_rfi_compressed(args, v_obs=None):
    n_time = args["N_time"]
    n_ant = args["N_ants"]
    n_bl = args["N_bl"]
    a1 = args["a1"]
    a2 = args["a2"]

    # with numpyro.plate("N_ants", n_ant):
    #     rfi_amp = MVN("rfi_r_induce", args["mu_rfi"], args["L_RFI_amp"])
    #     # rfi_amp = Normal("rfi_r_induce", args["mu_rfi_r"], 1e3)
    #     # G_amp = MVN("g_amp_induce", args["mu_G_amp"], args["L_G_amp"][None,:,:]*jnp.ones((n_ant,1,1)))
    #     G_amp = numpyro.sample(
    #         "g_amp_induce",
    #         dist.MultivariateNormal(
    #             loc=args["mu_G_amp"],
    #             scale_tril=args["L_G_amp"],
    #         ),
    #     )

    rfi_amp = numpyro.sample(
        "rfi_r_induce",
        dist.MultivariateNormal(
            loc=args["mu_rfi_r"],
            scale_tril=args["L_RFI_amp"],
        ),
    )
    G_amp = numpyro.sample(
        "g_amp_induce",
        dist.MultivariateNormal(
            loc=args["mu_G_amp"],
            scale_tril=args["L_G_amp"],
        ),
    )

    # with numpyro.plate("N_ants-1", n_ant - 1):
    #     G_phase = numpyro.sample(
    #         "g_phase_induce",
    #         dist.MultivariateNormal(
    #             loc=args["mu_G_phase"],
    #             scale_tril=args["L_G_phase"],
    #         ),
    #     )

    G_phase = numpyro.sample(
        "g_phase_induce",
        dist.MultivariateNormal(
            loc=args["mu_G_phase"],
            scale_tril=args["L_G_phase"],
        ),
    )

    gains = numpyro.deterministic(
        "gains",
        get_gains(G_amp, G_phase, args["resample_g_amp"], args["resample_g_phase"]),
    )

    rfi_vis = numpyro.deterministic(
        "rfi_vis", get_rfi_vis1(rfi_amp, args["rfi_kernel"], a1, a2)
    )

    # vis_r = [
    #     MVN(f"vis_r_{i}", args["mu_vis"][i], args["L_vis"][i]) for i in range(n_bl)
    # ]
    # vis_i = [
    #     MVN(f"vis_i_{i}", args["mu_vis"][i], args["L_vis"][i]) for i in range(n_bl)
    # ]
    # ast_vis = numpyro.deterministic(
    #     "ast_vis", get_ast_vis(vis_r, vis_i, args["resample_vis"])
    # )

    # with numpyro.plate("N_bl", n_bl):
    #     vis_r = MVN("vis_r_induce", args["mu_vis_r"], args["L_vis"])
    #     vis_i = MVN("vis_i_induce", args["mu_vis_i"], args["L_vis"])

    vis_r = numpyro.sample(
        "vis_r_induce",
        dist.MultivariateNormal(loc=args["mu_vis_r"], scale_tril=args["L_vis"]),
    )
    vis_i = numpyro.sample(
        "vis_i_induce",
        dist.MultivariateNormal(loc=args["mu_vis_i"], scale_tril=args["L_vis"]),
    )

    ast_vis = numpyro.deterministic(
        "ast_vis", get_ast_vis2(vis_r, vis_i, args["resample_vis"])
    )

    vis_obs = numpyro.deterministic(
        "vis_obs", get_obs_vis(ast_vis, rfi_vis, gains, a1, a2)
    )

    numpyro.deterministic(
        "rmse_ast",
        rmse(ast_vis, args["vis_ast_true"]) / jnp.sqrt(2),
    )
    numpyro.deterministic(
        "rmse_rfi",
        rmse(rfi_vis, args["vis_rfi_true"]) / jnp.sqrt(2),
    )
    numpyro.deterministic(
        "rmse_gains",
        rmse(gains, args["gains_true"]) / jnp.sqrt(2),
    )

    if v_obs is not None:
        return numpyro.sample(
            "obs",
            dist.Normal(
                jnp.concatenate([vis_obs.real, vis_obs.imag], axis=1), args["noise"]
            ),
            obs=v_obs,
        )


###############################################################################


# def fixed_orbit_complex_rfi_compressed(args):
#     n_time = args["N_time"]
#     n_ant = args["N_ants"]
#     n_bl = args["N_bl"]
#     a1 = args["a1"]
#     a2 = args["a2"]

#     with numpyro.plate("N_ants", n_ant):
#         rfi_amp = MVN("rfi_amp_induce", args["mu_rfi_amp"], args["L_RFI_amp"])
#         G_amp = numpyro.sample(
#             "g_amp_induce",
#             dist.MultivariateNormal(
#                 args["mu_G_amp"],
#                 args["cov_G_amp"],
#             ),
#         )

#     with numpyro.plate("N_ants-1", n_ant - 1):
#         rfi_phase = MVN("rfi_phase", args["mu_rfi_phase"], args["L_RFI_phase"])
#         G_phase = numpyro.sample(
#             "g_phase_induce",
#             dist.MultivariateNormal(args["mu_G_phase"], args["cov_G_phase"]),
#         )

#     G_amp = numpyro.deterministic("g_amp", G_amp @ args["resample_g_amp"].T)
#     G_phase = numpyro.deterministic(
#         "g_phase",
#         jnp.concatenate(
#             [G_phase @ args["resample_g_phase"].T, jnp.zeros((1, n_time))], axis=0
#         ),
#     )

#     gains = numpyro.deterministic("gains", G_amp * jnp.exp(1.0j * G_phase))

#     rfi_vis = numpyro.deterministic(
#         "rfi_vis",
#         get_rfi_vis1(
#             rfi_amp * jnp.exp(1.0j * rfi_phase), args["rfi_kernel"], a1, a2
#         ).T,
#     )

#     vis_r = [
#         MVN(f"vis_r_{i}", args["mu_vis"][i], args["L_vis"][i]) for i in range(n_bl)
#     ]
#     vis_i = [
#         MVN(f"vis_i_{i}", args["mu_vis"][i], args["L_vis"][i]) for i in range(n_bl)
#     ]

#     ast_vis = numpyro.deterministic(
#         "ast_vis", get_ast_vis(vis_r, vis_i, args["resample_vis"])
#     )

#     vis_obs = numpyro.deterministic(
#         "vis_obs", get_obs_vis(ast_vis, rfi_vis, gains, a1, a2)
#     )

#     numpyro.deterministic(
#         "rmse_ast",
#         rmse(ast_vis, args["vis_ast_true"]) / jnp.sqrt(2),
#     )
#     numpyro.deterministic(
#         "rmse_rfi",
#         rmse(rfi_vis, args["vis_rfi_true"]) / jnp.sqrt(2),
#     )
#     numpyro.deterministic(
#         "rmse_gains",
#         rmse(gains, args["gains_true"]) / jnp.sqrt(2),
#     )

#     return numpyro.sample(
#         "obs",
#         dist.Normal(
#             jnp.concatenate([vis_obs.real, vis_obs.imag], axis=1), args["noise"]
#         ),
#         obs=args["vis_obs"],
#     )
