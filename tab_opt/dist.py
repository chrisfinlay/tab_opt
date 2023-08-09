import numpyro.distributions as dist
import jax.numpy as jnp
import numpyro


def MVN(name, mu, L):
    n = len(mu)
    rv = numpyro.sample(
        name,
        dist.TransformedDistribution(
            dist.Normal(0, jnp.ones(n)),
            [
                dist.transforms.LowerCholeskyAffine(mu, L),
            ],
        ),
    )
    return rv


def Normal(name, mu, sigma):
    n = len(mu)
    rv = numpyro.sample(
        name,
        dist.TransformedDistribution(
            dist.Normal(0, jnp.ones(n)),
            [
                dist.transforms.AffineTransform(mu, sigma),
            ],
        ),
    )
    return rv
