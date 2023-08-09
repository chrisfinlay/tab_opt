import xarray as xr


def extract_data(sim_path: str, sampling: int = 1, N_time: int = 300):
    ds = xr.open_zarr(sim_path)

    N_int_samples = ds.attrs["n_int_samples"] // sampling
    N_ant = ds.attrs["n_ant"]
    N_bl = ds.attrs["n_bl"]
    a1 = ds.antenna1.data.compute()
    a2 = ds.antenna2.data.compute()
    times = ds.coords["time"].data[:N_time]
    times_fine = ds.coords["time_fine"].data[: N_time * N_int_samples : sampling]
    bl_uvw = ds.bl_uvw.data.compute()[: N_time * N_int_samples : sampling]
    ants_uvw = ds.ants_uvw.data.compute()[: N_time * N_int_samples : sampling]
    ants_xyz = ds.ants_xyz.data.compute()[: N_time * N_int_samples : sampling]
    vis_ast = ds.vis_ast.data.compute()[: N_time * N_int_samples : sampling]
    vis_rfi = ds.vis_rfi.data.compute()[: N_time * N_int_samples : sampling]
    vis_obs = ds.vis_obs.data.compute()[:N_time]
    noise = ds.noise_std.data.compute()[:N_time]
    noise_data = ds.noise_data.data.compute()[:N_time]
    int_time = ds.attrs["int_time"]
    freqs = ds.coords["freq"].data
    gains_ants = ds.gains_ants.data.compute()[: N_time * N_int_samples : sampling]
    rfi_A_app = ds.rfi_sat_A.data.compute()[0, : N_time * N_int_samples : sampling]
    rfi_orbit = ds.rfi_sat_orbit.data.compute()[0]

    return (
        N_int_samples,
        N_ant,
        N_bl,
        a1,
        a2,
        times,
        times_fine,
        bl_uvw,
        ants_uvw,
        ants_xyz,
        vis_ast,
        vis_rfi,
        vis_obs,
        noise,
        noise_data,
        int_time,
        freqs,
        gains_ants,
        rfi_A_app,
        rfi_orbit,
    )
