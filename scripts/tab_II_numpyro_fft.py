import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from tab_opt.dist import Normal
from tab_opt.vis import get_ast_vis, get_ast_vis2, get_obs_vis, get_rfi_vis, get_gains, rmse, fft_inv_even, get_rfi_vis_fft1, get_rfi_vis_fft2
from jax import jit, random, vmap
from functools import partial





@jit
def pow_spec(k, P0=1e7, k0=3e-3, gamma=2):
    return P0 / ((1. + (jnp.abs(k)/k0)**2)**(gamma / 2))




def model(params):
    
    # n_g = params['N_g_times']
    # n_rfi_k = params['N_rfi_k']
    n_time = params['N_time']
    n_ant = params['N_ants']
    n_bl = params['N_bl']
    a1 = params['a1']
    a2 = params['a2']
    k_rfi = params['k_rfi']
    k_ast = params['k_ast']
    dt = params['dt']
    n_rfi_k = len(k_rfi)
    n_ast_k = len(k_ast)
    

    # with numpyro.plate('g_amps', n_ant) as i:
    #     G_amp = numpyro.sample('g_amp', dist.MultivariateNormal(params['mu_G_amp'][i], params['cov_G_amp']))
    
#     with numpyro.plate('g_phases', n_ant-1) as i:
#         G_phase = numpyro.sample(f'g_phase', dist.MultivariateNormal(params['mu_G_phase'][i], params['cov_G_phase']))
    
#     G_amp = jnp.array([MVN(f'g_amp_{i}', params['mu_G_amp'][i], params['L_G_amp']) for i in range(n_ant)])
#     G_phase = jnp.array([MVN(f'g_phase_{i}', params['mu_G_phase'][i], params['L_G_phase']) for i in range(n_ant-1)])# + \
#                         # [numpyro.sample(f'g_phase_{n_ant-1}', dist.Delta(jnp.zeros(len(params['mu_G_phase'][0]))))])
    
#     G_amp = numpyro.deterministic('g_amp', G_amp@params['resample_g_amp'].T)
#     G_phase = numpyro.deterministic('g_phase', jnp.concatenate([G_phase@params['resample_g_phase'].T, jnp.zeros((1,n_time))], axis=0))
        
    G_amp = Normal('g_amp_', params['mu_G_amp'][:,:1], 0.01)
    G_phase = Normal('g_phase_', params['mu_G_phase'][:,:1], jnp.deg2rad(1))
    
    G_amp = numpyro.deterministic('g_amp', G_amp)
    G_phase = numpyro.deterministic('g_phase', jnp.concatenate([G_phase, jnp.zeros((1,1))], axis=0))
    
    G = numpyro.deterministic('gains', G_amp * jnp.exp(1.j*G_phase) * jnp.ones((n_ant, n_time)))
        
    rfi_Pk_root = jnp.sqrt(pow_spec(k_rfi, **params['rfi_Pk_params']))
    
    # rfi_k_r = Normal('rfi_k_r', jnp.zeros((n_rfi_k,n_ant)), rfi_Pk_root[:,None]*jnp.ones((n_rfi_k,n_ant)))
    # rfi_k_i = Normal('rfi_k_i', jnp.zeros((n_rfi_k,n_ant)), rfi_Pk_root[:,None]*jnp.ones((n_rfi_k,n_ant)))
    
    # rfi_k = numpyro.deterministic('rfi_k', rfi_k_r + 1.j*rfi_k_i)
    
    rfi_k_r = Normal('rfi_k_r', jnp.zeros((n_rfi_k,n_ant)), rfi_Pk_root[:,None]*jnp.ones((n_rfi_k,n_ant)))
    rfi_k = numpyro.deterministic('rfi_k', vmap(construct_real_fourier, in_axes=(1,))(rfi_k_r).T)
    
    rfi_vis = numpyro.deterministic('rfi_vis', get_rfi_vis1(rfi_k, a1, a2, params['rfi_k_kernel']))
    # rfi_vis = numpyro.deterministic('rfi_vis', get_rfi_vis2(rfi_k, a1, a2, params['rfi_phasor']).T)
    
#     rfi_I = numpyro.deterministic('rfi_I', jnp.fft.ifft(rfi_k[:,a1], axis=0)*jnp.fft.ifft(rfi_k[:,a2], axis=0))
#     rfi_kI = numpyro.deterministic('rfi_kI', jnp.fft.fft(rfi_I, axis=0))
        
#     rfi_vis = numpyro.deterministic('rfi_vis', (params['rfi_k_kernel']*rfi_kI).sum(axis=1).T)
    
#     vis_r = [MVN(f'vis_r_{i}', params['mu_vis'][i], params['L_vis'][i]) for i in range(n_bl)]
#     vis_i = [MVN(f'vis_i_{i}', params['mu_vis'][i], params['L_vis'][i]) for i in range(n_bl)]
    
#     vis_r = jnp.array([params['resample_vis'][i]@vis_r[i] for i in range(n_bl)])
#     vis_i = jnp.array([params['resample_vis'][i]@vis_i[i] for i in range(n_bl)])

#     ast_vis = numpyro.deterministic('ast_vis', vis_r + 1.j*vis_i)

    ast_Pk_root = jnp.sqrt(pow_spec(k_ast, **params['ast_Pk_params']))

    ast_k_r = Normal('ast_k_r', jnp.zeros((n_ast_k,n_bl)), ast_Pk_root[:,None]*jnp.ones((n_ast_k,n_bl)))
    ast_k_i = Normal('ast_k_i', jnp.zeros((n_ast_k,n_bl)), ast_Pk_root[:,None]*jnp.ones((n_ast_k,n_bl)))
    
    ast_k = numpyro.deterministic('ast_k', ast_k_r + 1.j*ast_k_i)
    
    ast_vis = numpyro.deterministic('ast_vis', vmap(fft_inv_even, in_axes=(1,None,None))(ast_k, N_pad_ast, NN_ast))
    # ast_vis = numpyro.deterministic('ast_vis', jnp.fft.ifft(ast_k, axis=0).T/dt)
    
    vis_obs = numpyro.deterministic('vis_obs', G[a1]*jnp.conjugate(G[a2]) * ( ast_vis + rfi_vis))
    vis_obs = jnp.concatenate([vis_obs.real, vis_obs.imag], axis=1)
    
    numpyro.deterministic('rmse_ast', jnp.sqrt(jnp.mean( jnp.abs( (ast_vis-params['vis_ast_true'])) **2, axis=1)) )
    
    return numpyro.sample('obs', dist.Normal(vis_obs, params['noise']), obs=params['vis_obs'].T)
    
    
    
rfi_amp = jnp.abs(vis_rfi)[:,:,0]
rfi_phasor = vis_rfi[:,:,0]/rfi_amp

vis_rfi_true = vis_rfi.reshape(N_time, N_int_samples, N_bl).mean(axis=1)

N = N_time*N_int_samples
N_pad = 5*N_int_samples
NN = N + 2*N_pad

dt = jnp.diff(times_fine)[0]
k = jnp.fft.fftfreq(NN, dt)

k_lim = 2e-1
k_idx = jnp.where(jnp.abs(k)<k_lim)[0]
nn = len(k_idx)
k_idx = jnp.concatenate([k_idx[:(nn+1)//2], k_idx[(nn+1)//2+1:]])
k_ = k[k_idx]

nn = len(k_idx)
factor = NN/nn

t_idx = jnp.round(jnp.arange(0, NN, factor), 0).astype(int)
N_pad_ = int(N_pad/factor)

print(f'# of Fourier modes :   {nn}')
print(f'Memory saving factor : {factor:.1f}')

pad = partial(jnp.pad, pad_width=N_pad, mode='reflect', reflect_type='odd')

rfi_amp_pad = vmap(pad)(rfi_A_app[:,:,0].T).T
# rfi_amp_pad = vmap(pad)(rfi_amp[:,:].T).T
rfi_k = jnp.fft.fft(rfi_amp_pad, axis=0)/factor
rfi_k_ = rfi_k[k_idx]

print(f'RFI Parameter Ratio : {nn/N_rfi_time:.2f}')

vis_rfi_test = (rfi_k_kernel*rfi_kI_[None,:,:]).sum(axis=1)

rfi_k1 = vmap(construct_real_fourier, in_axes=(1,))(jnp.concatenate([rfi_k_[:nn//2+1].real, rfi_k_[1:nn//2].imag], axis=0)).T

vis_rfi_test = get_rfi_vis1(rfi_k1, a1, a2, rfi_k_kernel).T
# vis_rfi_test = get_rfi_vis1(rfi_k_, a1, a2, rfi_k_kernel).T
# vis_rfi_test = get_rfi_vis2(rfi_k_, a1, a2, rfi_phasor)

rel_rmse = jnp.sqrt(jnp.mean( jnp.abs(vis_rfi_test/vis_rfi_true - 1)**2 ))

print('Assuming an orbit and compressing all resampling and phase multiplication and averaging into a single matrix')
print('-----------------------------------------------------')
print(f'Relative Root Mean Squared Error : {rel_rmse: .2E}')

noise = 0.5*random.normal(random.PRNGKey(1), (N_time, N_bl), dtype=complex)
vis_ast_true = averaging(vis_ast[:,:,0], N_int_samples) + noise

NN_ast = 2**9
NN_ast = N_time + 20
# NN_ast = N_time
N_pad_ast = (NN_ast - N_time) // 2
dt_ast = jnp.diff(times)[0]
k_ast = jnp.fft.fftfreq(NN_ast, dt_ast)

k_lim_ast = 3e-2
k_idx = jnp.where(jnp.abs(k_ast)<k_lim_ast)[0]
nn_ast = len(k_idx)
k_idx = jnp.concatenate([k_idx[:nn_ast//2], k_idx[-nn_ast//2 + 1:]])
nn_ast = len(k_idx)
factor_ast = NN_ast/nn_ast
k_ast_ = k_ast[k_idx]

pad = partial(jnp.pad, pad_width=N_pad_ast, mode='reflect', reflect_type='odd')

ast_k = jnp.fft.fft(vmap(pad)(vis_ast_true.T).T, axis=0)
ast_k_ = ast_k[k_idx]

params.update(
    {'k_rfi': k_,
     'k_rfi': jnp.concatenate([k_[:nn//2+1], k_[1:nn//2]], axis=0),
     'rfi_k_kernel': rfi_k_kernel,
     'rfi_phasor': rfi_phasor,
     'k_ast': k_ast_,
     'dt': dt,
     'rfi_Pk_params': {'P0': 5e4, 'k0': 5e-3, 'gamma': 3},
     # 'ast_Pk_params': {'P0': 5e6, 'k0': 3e-3, 'gamma': 2.5},
     'ast_Pk_params': {'P0': 1e7, 'k0': 5e-3, 'gamma': 6},
    }         
     )

true_params = {
    # **{f'g_amp_{i}': true_values['g_amp'][i][:1] for i in range(N_ant)},
    # **{f'g_phase_{i}': true_values['g_phase'][i][:1] for i in range(N_ant-1)},
    **{'g_amp_': jnp.abs(G).mean(axis=1)[:,None]},
    **{'g_phase_': jnp.angle(G).mean(axis=1)[:N_ant-1,None]},
    # **{'rfi_amp': jnp.array([true_values['rfi_amp'][i] for i in range(N_ant)])},
    # **{'rfi_k_r': rfi_k_.real},
    # **{'rfi_k_i': rfi_k_.imag},
    **{'rfi_k_r': jnp.concatenate([rfi_k_[:nn//2+1].real, rfi_k_[1:nn//2].imag], axis=0)},
    # **{f'vis_r_{i}': true_values['v_real'][i] for i in range(N_bl)},
    # **{f'vis_i_{i}': true_values['v_imag'][i] for i in range(N_bl)},
    **{'ast_k_r': ast_k_.real},
    **{'ast_k_i': ast_k_.imag},
    }