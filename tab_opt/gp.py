import jax.numpy as jnp
from jax import jit


# @jit
# def kernel(x, var, l, noise=1e-5):
#     """
#     x: array (n_points, n_dim)
#     """
#     chi = jnp.linalg.norm(x[None, :, :] - x[:, None, :], axis=-1) / (l)
#     cov = jnp.abs(var) * jnp.exp(-0.5 * chi**2) + noise * jnp.eye(x.shape[0])
#     return cov

# @jit
# def inv_kernel(x, var, l):
#     return inv_sym(kernel(x, var, l))


# @jit
# def inv_kernel_vmap(x, var, l):
#     return vmap(inv_kernel, in_axes=(None, 0, 0))(x, var, l)


@jit
def kernel(x, x_, var, l, noise=1e-3):
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


@jit
def gp_resample(y, x, x_, var, l, noise=1e-3):
    K_inv = jnp.linalg.inv(kernel(x, x, var, l, noise))
    K_s = kernel(x, x_, var, l)
    return K_s @ K_inv @ y


@jit
def l_from_uv(uv, l0=7e2, a=6e-4):
    return l0 * jnp.exp(-a * uv)


def get_times(times, gp_l):
    int_time = times[1] - times[0]
    t_i = times[0] - int_time / 2
    t_f = times[-1] + int_time / 2
    n_times = jnp.ceil(2.0 * ((t_f - t_i) / gp_l) + 1).astype(int)
    n_times = jnp.where(n_times < 3, 3, n_times)
    sample_times = jnp.linspace(t_i, t_f, n_times)
    return sample_times


def get_vis_gp_params(ants_uvw, vis_ast, a1, a2, noise):
    mag_uvw = jnp.linalg.norm(ants_uvw[0, a1] - ants_uvw[0, a2], axis=-1)
    vis_var = (jnp.abs(vis_ast).max(axis=(0, 2)) + noise) ** 2
    vis_l = l_from_uv(mag_uvw, l0=5e2)
    return vis_var, vis_l
