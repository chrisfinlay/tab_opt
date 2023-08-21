from numpyro.infer import Predictive, SVI, Trace_ELBO, autoguide
import numpyro
from jax import random, jit
import jax
import jax.numpy as jnp

from jax.flatten_util import ravel_pytree
from functools import partial
from numpyro.infer import log_likelihood
from numpyro.infer.util import log_density

import optax

H = jit(optax.hessian_diag, static_argnums=(0,))
# H = optax.hessian_diag


@partial(jit, static_argnums=(0,))
def fisher_vp(f, w, v):
    """Calculate the Fisher vector product. Taken from https://gebob19.github.io/natural-gradient/"""
    # J v
    _, Jv = jax.jvp(f, (w,), (v,))
    # (J v)^T J = v^T (J^T J)
    _, f_vjp = jax.vjp(f, w)
    return f_vjp(Jv)[0]


@partial(jit, static_argnums=(0,))
def fisher_diag(nll, params):
    """Calculate the diagonal of the Fisher information matrix. Taken from Optax second_order.py (hessian_diag)."""
    params_flat, unflatten = ravel_pytree(params)
    N = len(params_flat)
    vs = jnp.eye(N)
    comp = lambda v: jnp.vdot(v, ravel_pytree(fisher_vp(nll, params, unflatten(v)))[0])
    return jax.vmap(comp)(vs)


@partial(jit, static_argnums=(0,))
def fisher_diag_inv(
    model,
    params: dict,
    kwargs_model: dict,
    kwargs_data: dict,
    likelihood_site_name: str = "obs",
):
    """Calculate the inverse of the diagonal of the Fisher information matrix.

    Parameters:
    -----------
    model: NumPyro model
        A numpyro model that is called as model(**kwargs_model, **kwargs_data).
    params: dict
        Parameter values at which to evaluate the posterior fisher information.
    kwargs_model: dict
        Keyword arguments to be passed ot the model.
    kwargs_data: dict
        Keyword arguments where the observed data is passed to the model.
    likelihood_site_name: str
        The model site name where the observed data is passed to in the form of 'obs=kwargs_data.values()'.

    Returns:
    --------
    fisher_diag_inv:
        Flattened array of the inverse diagonal fisher.
    """
    # _, unflatten = ravel_pytree(params)
    # params = jax.tree_map(lambda x: jnp.atleast_1d(x)[None, :], params)
    nlp = (
        lambda x, _, __: -1
        * log_density(model, model_args=(), model_kwargs=kwargs_model, params=x)[0]
    )
    nll = lambda x: log_likelihood(
        model, posterior_samples=x, batch_ndims=0, **kwargs_model, **kwargs_data
    )[likelihood_site_name]
    fisher_prior = jnp.abs(H(nlp, params, None, None))
    fisher_likelihood = fisher_diag(nll, params)

    return 1.0 / (fisher_prior + fisher_likelihood)


@partial(jit, static_argnums=(0,))
def fisher_diag_inv2(
    model,
    params: dict,
    kwargs_model: dict,
    kwargs_data: dict,
    likelihood_site_name: str = "obs",
):
    nlpost = (
        lambda x, _, __: -1
        * log_density(
            model, model_args=(), model_kwargs={**kwargs_model, **kwargs_data}, params=x
        )[0]
    )
    fisher_post = H(nlpost, params, None, None)

    return 1.0 / jnp.abs(fisher_post)


def run_svi(model, args, max_iter=1_000, guide_family="AutoDelta", init_params=None):
    if guide_family == "AutoDelta":
        guide = autoguide.AutoDelta(model)
    elif guide_family == "AutoDiagonalNormal":
        guide = autoguide.AutoDiagonalNormal(model)
    elif guide_family == "AutoLaplaceApproximation":
        guide = autoguide.AutoLaplaceApproximation(model)
    elif guide_family == "AutoMultivariateNormal":
        guide = autoguide.AutoMultivariateNormal(model)
    else:
        raise ValueError(f"Unknown guide_family: {guide_family}")

    optimizer = numpyro.optim.Adam(1e-5)
    svi = SVI(model, guide, optimizer, Trace_ELBO())
    svi_results = svi.run(
        random.PRNGKey(1), max_iter, args=args, init_params=init_params
    )
    # params = svi_results.params
    # losses = svi_results.losses

    return svi_results, guide


def svi_predict(model, guide, vi_params, args, num_samples=100):
    predictive = Predictive(
        model=model, guide=guide, params=vi_params, num_samples=num_samples
    )
    predictions = predictive(random.PRNGKey(2), args=args)

    return predictions
