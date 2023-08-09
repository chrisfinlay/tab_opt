from numpyro.infer import Predictive, SVI, Trace_ELBO, autoguide
import numpyro
from jax import random

def run_svi(model, params, max_iter=1_000, guide_family="AutoDelta", init_params=None):
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

    optimizer = numpyro.optim.Adam(1e-1)
    svi = SVI(model, guide, optimizer, Trace_ELBO())
    svi_results = svi.run(random.PRNGKey(1), max_iter, params=params, init_params=init_params)
    # params = svi_results.params
    # losses = svi_results.losses

    return svi_results, guide

def svi_predict(model, guide, vi_params, params, num_samples=100):
    predictive = Predictive(
        model=model, guide=guide, params=vi_params, num_samples=num_samples
    )
    predictions = predictive(random.PRNGKey(2), params=params)
    
    return predictions