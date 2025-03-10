from tab_opt.vis import get_rfi_vis_full
import jax.numpy as jnp
from jax import random
from frozendict import frozendict

r_key = 0

# 64 Antenna - Medium RFI strength
n_ant = 64  # Up to 256
n_time = 450  # 100 to 450 - Maybe larger in future requires further analysis
n_int = 10  # 1 to 1000 - Maybe larger in future requires further analysis
n_sat = 5  # 1 to 100 - Maybe larger in future requires further analysis
n_rfi_time = 20  # 2 to 100 - Maybe larger in future requires further analysis

a1, a2 = jnp.triu_indices(n_ant, 1)
rfi_phase = random.uniform(random.PRNGKey(r_key), (n_sat, n_ant, n_time * n_int + 1))
resample_rfi = random.uniform(
    random.PRNGKey(r_key + 1), (n_time * n_int + 1, n_rfi_time)
)

rfi_amp = random.normal(
    random.PRNGKey(r_key + 2), (n_sat, n_ant, n_rfi_time)
) + 1.0j * random.normal(random.PRNGKey(r_key + 3), (n_sat, n_ant, n_rfi_time))

args = frozendict({"n_int_samples": n_int})

array_args = {
    "a1": a1,
    "a2": a2,
    "rfi_phase": rfi_phase,
    "resample_rfi": resample_rfi,
}

rfi_vis = get_rfi_vis_full(rfi_amp, args, array_args)

print(jnp.sum(rfi_vis))
