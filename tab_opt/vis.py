from functools import partial

import jax.numpy as jnp
from jax import jit, vmap


@partial(jit, static_argnums=(1,))
def averaging(x, n_avg):
    shape = x.shape
    n = shape[0] // n_avg
    x_avg = jnp.mean(jnp.reshape(x, newshape=(n, n_avg, *shape[1:])), axis=1)
    return x_avg


@jit
def get_rfi_vis(rfi_amp, rfi_kernel, a1, a2):
    rfi_I = (rfi_amp[a1] * jnp.conjugate(rfi_amp[a2])).T[None, :, :]
    vis_rfi = (rfi_kernel * rfi_I).sum(axis=1)
    return vis_rfi

@jit
def get_rfi_vis_fft1(rfi_k, a1, a2, rfi_k_kernel):
    rfi_amp = jnp.fft.ifft(rfi_k, axis=0)
    rfi_kI = jnp.fft.fft(rfi_amp[:,a1]*rfi_amp[:,a2], axis=0)
    
    rfi_vis = (rfi_k_kernel*rfi_kI).sum(axis=1).T
    
    return rfi_vis

@partial(jit, static_argnums=(4,5,6))
def get_rfi_vis_fft2(rfi_k, a1, a2, rfi_phasor, N_pad, NN, N_int_samples):
    rfi_amp = vmap(fft_inv_even, in_axes=(1,None,None))(rfi_k, N_pad, NN).T
    rfi_I = rfi_amp[:,a1]*rfi_amp[:,a2]
    rfi_vis = averaging(rfi_I*rfi_phasor, N_int_samples)
    
    return rfi_vis


@jit
def get_ast_vis(vis_r, vis_i, resample_vis):
    n_bl = len(vis_r)
    vis_ast = jnp.array(
        [resample_vis[i] @ (vis_r[i] + 1.0j * vis_i[i]) for i in range(n_bl)]
    )
    return vis_ast

@jit
def get_ast_vis2(vis_r, vis_i, resample_vis):
    vis_ast = (vis_r + 1.0j * vis_i) @ resample_vis.T
    return vis_ast

@jit
def get_gains(g_amp, g_phase, resample_g_amp, resample_g_phase):
    n_time = resample_g_amp.shape[0]
    g_amp = g_amp @ resample_g_amp.T
    g_phase = jnp.concatenate([g_phase @ resample_g_phase.T, jnp.zeros((1, n_time))], axis=0)
    gains = g_amp * jnp.exp(1.0j * g_phase)
    return gains


@jit
def get_obs_vis(ast_vis, rfi_vis, gains, a1, a2):
    vis_obs = gains[a1] * jnp.conjugate(gains[a2]) * (ast_vis + rfi_vis)
    return vis_obs


@partial(jit, static_argnums=(2,))
def rmse(x, x_true, axis=1):
    return jnp.sqrt(jnp.mean(jnp.abs(x - x_true) ** 2, axis=axis))


@partial(jit, static_argnums=(1, 2))
def fft_inv_even(X, n_pad, n_out):
    n_k = len(X)
    n_k_missing = n_out - n_k
    factor = n_out / n_k
    # assert n_k%2 == 0
    X_ = X * factor
    X_ = jnp.concatenate([X_[: n_k // 2], jnp.zeros(n_k_missing), X_[n_k // 2 :]])
    x = jnp.fft.ifft(X_)[n_pad:-n_pad]
    return x

@jit
def construct_real_fourier(x):
    N = len(x)
    x_r = x[:N//2+1] + 1.j*jnp.concatenate([jnp.zeros(1), x[N//2+1:], jnp.zeros(1)])
    x_recon = jnp.concatenate([x_r[:N//2+1], jnp.conjugate(x_r[-2:0:-1])])
    return x_recon


def eye(N, idx):
    x = jnp.zeros(N)
    x = x.at[idx].set(1.0)
    return x
