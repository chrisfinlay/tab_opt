# from RFIx.utils.dict import *
import jax.numpy as jnp
from jax import jit, random, vmap
from functools import partial
from jax.lax import fori_loop, cond
from jax.config import config

config.update("jax_enable_x64", True)


# @partial(jit, static_argnums=(0,1,7,8))
# def HMC_block_diag(U, delU, qi, params, M_inv, L, delta_T, T, N_samples, key):

#     def acpt_prop(q_options):
#         return q_options[0]
#     def reject_prop(q_options):
#         return q_options[1]

#     # def adapt_dT(deltas):
#     #     dT, dE = deltas
#     #     error = jnp.min(jnp.array([1, jnp.exp(dE)])) - 0.65
#     #     return dT*jnp.exp(0.5*error)
#     # def leave_dT(deltas):
#     #     dT, dE = deltas
#     #     return dT

#     def new_sample(i, loop_var):
#         samples, acpt, dE, key = loop_var
#         q_old = get_sample(samples, i-1)
#         q_prop, delta_E, key = HMC_prop_block_diag(U, delU, q_old, params, M_inv, L, delta_T, T, key)
#         metrop = jnp.log(random.uniform(key))<delta_E
#         key = random.split(key)[0]
#         q_new = cond(metrop, acpt_prop, reject_prop, [q_prop, q_old])
#         acpt = acpt.at[i].set(jnp.int8(metrop))
#         dE = dE.at[i].set(delta_E)
#         # dT = cond(burn_in, adapt_dT, leave_dT, [dT, delta_E])
#         return [assign_sample(samples, q_new, i), acpt, dE, key]

#     samples = assign_sample(unflatten(jnp.empty((len(flatten(qi)), N_samples))), qi, 0)
#     acpt = jnp.empty(N_samples, dtype=jnp.int8).at[0].set(1)
#     dE = jnp.empty(N_samples, dtype=jnp.float64).at[0].set(0.0)

#     samples, acpt, dE, key = fori_loop(1, N_samples, new_sample, [samples, acpt, dE, key])

#     return samples, acpt, dE, key

# @partial(jit, static_argnums=(0,1,7))
# def HMC_prop_block_diag(U, delU, qi, params, M_inv, L, delta_T, T, key):
#     '''
#     Sample from a posterior as defined by the exponential of the
#     negative of the potential energy function, p = exp(-U).
#     '''

#     def update_qp(i, qp):
#         qq = vec_add(qp[0], scalar_prod(block_diag_prod(M_inv, qp[1]), delta_T))
#         pp = vec_add(qp[1], scalar_prod(delU(qq, params), -delta_T))
#         return [qq, pp]

#     pi = block_diag_prod(L, unflatten(random.normal(key, (len(flatten(qi)),))))
#     key = random.split(key)[0]

#     q = qi
#     p = pi

#     p = vec_add(p, scalar_prod(delU(q, params), -delta_T/2))

#     qp = fori_loop(1, T, update_qp, [q,p])

#     q = vec_add(qp[0],  scalar_prod(block_diag_prod(M_inv, qp[1]), delta_T))
#     p = vec_add(qp[1], scalar_prod(delU(q, params), -delta_T/2))

#     Ei = U(qi, params) + vec_sum(vec_prod(pi, block_diag_prod(M_inv, scalar_prod(pi, 0.5))))
#     Ef = U(q, params) + vec_sum(vec_prod(p, block_diag_prod(M_inv, scalar_prod(p, 0.5))))

#     delta_E = Ei - Ef

#     return q, delta_E, key


@jit
def update_dt(var):
    acceptance_rate, delta_T = var
    error = acceptance_rate - 0.65
    factor = jnp.exp(0.5 * error)
    return factor * delta_T


@jit
def inv_sym(x):
    x_inv = jnp.linalg.inv(x + 1e-9 * jnp.eye(x.shape[0]))
    return 0.5 * (x_inv + x_inv.T)


@jit
def kernel(x, var, l, noise=1e-5):
    """
    x: array (n_points, n_dim)
    """
    chi = jnp.linalg.norm(x[None, :, :] - x[:, None, :], axis=-1) / (l)
    cov = jnp.abs(var) * jnp.exp(-0.5 * chi**2) + noise * jnp.eye(x.shape[0])
    return cov


@jit
def log_normal(x, mu, sigma):
    """
    Calculate the natural logarithm of the Normal distribution.

    Parameters:
    -----------
    x: float
        Value of the random variable that is distributed normally.
    mu: float
        Location parameter.
    sig: float
        Scale parameter

    Returns:
    --------
    log_p: float
        The log of the likelihood value.
    """
    scaled_error = (x - mu) / sigma
    log_p = -0.5 * scaled_error**2

    return log_p


@jit
def log_multinorm(x, mu, inv_cov):
    reduced_mean = x - mu
    log_p = -0.5 * reduced_mean @ inv_cov @ reduced_mean

    return log_p


@jit
def inv_kernel(x, var, l):
    return inv_sym(kernel(x, var, l))


@jit
def inv_kernel_vmap(x, var, l):
    return vmap(inv_kernel, in_axes=(None, 0, 0))(x, var, l)


@jit
def log_multinorm_vmap(x, mu, inv_cov):
    return vmap(log_multinorm, in_axes=(0, 0, 0))(x, mu, inv_cov)


@jit
def log_multinorm_sum(x, mu, inv_cov):
    return jnp.sum(log_multinorm_vmap(x, mu, inv_cov))


@jit
def log_normal_vmap(x, mu, sigma):
    return vmap(log_normal, in_axes=(0, 0, 0))(x, mu, sigma)


@jit
def log_normal_sum(x, mu, sigma):
    return jnp.sum(log_normal_vmap(x, mu, sigma))
