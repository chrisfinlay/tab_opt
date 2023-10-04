import numpyro.distributions as dist
import jax.numpy as jnp
import numpyro


def MVN(name, mu, L):
    rv = numpyro.sample(
        name,
        dist.TransformedDistribution(
            dist.Normal(jnp.zeros(mu.shape), jnp.ones(mu.shape)),
            [
                dist.transforms.LowerCholeskyAffine(mu, L),
            ],
        ),
    )
    return rv


def Normal(name, mu, sigma):
    rv = numpyro.sample(
        name,
        dist.TransformedDistribution(
            dist.Normal(jnp.zeros(mu.shape), jnp.ones(mu.shape)),
            [
                dist.transforms.AffineTransform(mu, sigma),
            ],
        ),
    )
    return rv
