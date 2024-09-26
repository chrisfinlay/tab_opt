from daskms import xds_to_table, xds_from_ms
import xarray as xr
import dask.array as da
import dask


def extract_data(zarr_path: str, sampling: int = 1, N_time: int = -1, freq_idx: int = 0):
    """Extract

    Parameters:
    -----------
    zarr_path: str
        Path to the zarr file containing the simulation data and parameters.
    sampling: int
        The stride length for which to extract the high time resolution data.
        `sampling=1` provides all the data.
    N_time: int
        The total number of time points sampled at the integration time.
        `N_time=-1` returns all the data.

    Returns:
    --------
    data: list
        List of data
    """
    ds = xr.open_zarr(zarr_path)

    if N_time == -1:
        N_time = ds.attrs["n_time"]

    N_int_samples = ds.attrs["n_int_samples"] // sampling
    N_ant = ds.attrs["n_ant"]
    N_bl = ds.attrs["n_bl"]
    a1 = ds.antenna1.data
    a2 = ds.antenna2.data
    times = ds.coords["time"].data[:N_time]
    times_fine = ds.coords["time_fine"].data[: N_time * N_int_samples : sampling]
    bl_uvw = ds.bl_uvw.data[: N_time * N_int_samples : sampling]
    ants_uvw = ds.ants_uvw.data[: N_time * N_int_samples : sampling]
    ants_xyz = ds.ants_xyz.data[: N_time * N_int_samples : sampling]
    vis_ast = ds.vis_ast.data[: N_time * N_int_samples : sampling,:,freq_idx]
    vis_rfi = ds.vis_rfi.data[: N_time * N_int_samples : sampling,:,freq_idx]
    vis_obs = ds.vis_obs.data[:N_time,:,freq_idx]
    vis_cal = ds.vis_calibrated.data[:N_time,:,freq_idx]
    noise = ds.noise_std.data[:N_time]
    noise_data = ds.noise_data.data[:N_time,:,freq_idx]
    int_time = ds.attrs["int_time"]
    freqs = ds.coords["freq"].data
    gains_ants = ds.gains_ants.data[: N_time * N_int_samples : sampling,:,freq_idx]
    rfi_A_app = ds.rfi_sat_A.data[:, : N_time * N_int_samples : sampling,:,freq_idx]
    rfi_orbit = ds.rfi_sat_orbit.data

    data = [
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
        vis_cal,
        noise,
        noise_data,
        int_time,
        freqs,
        gains_ants,
        rfi_A_app,
        rfi_orbit,
    ]

    data = [
        x.compute() if hasattr(x, "compute") and callable(x.compute) else x
        for x in data
    ]

    return data


def write_corrected_data_zarr(zarr_path, corr_data):
    """Write the corrected data to the zarr Dataset.

    Parameters:
    zarr_path: str
        Path to the zarr file to wite to.
    corr_data: array_like (n_time, n_bl, n_freq)
        Corrected visibilities.
    """
    xds = xr.open_zarr(zarr_path)
    xds = xds.assign(
        vis_corrected=(
            ("time", "bl", "freq"),
            da.asarray(corr_data, chunks=xds.vis_obs.chunks),
        )
    )
    xds.to_zarr(zarr_path, mode="a")

    return xds


def write_corrected_data_zarr(zarr_path, corr_data):
    """Write the corrected data to the zarr Dataset.

    Parameters:
    zarr_path: str
        Path to the zarr file to wite to.
    corr_data: array_like (n_time, n_bl, n_freq)
        Corrected visibilities.
    """
    xds = xr.open_zarr(zarr_path)
    xds = xds.assign(
        vis_corrected=(
            ("time", "bl", "freq"),
            da.asarray(corr_data, chunks=xds.vis_obs.chunks),
        )
    )
    xds.to_zarr(zarr_path, mode="a")

    return xds


def write_corrected_data_ms(ms_path, corr_data, flags):
    """Write the corrected data back to the MS file.

    Parameters:
    -----------
    ms_path: str
        Path to the Measurement Set to write to.
    corr_data: array-like (n_time*n_bl, n_freq)
        Corrected visibilities.
    flags: array-like (n_time*n_bl, n_freq)
        Data flags.
    """
    xds = xds_from_ms(ms_path)[0]

    if isinstance(corr_data, da.Array):
        corr_data = corr_data[:, :, None].rechunk(xds.CORRECTED_DATA.chunks)
    else:
        corr_data = da.asarray(corr_data[:, :, None], chunks=xds.CORRECTED_DATA.chunks)

    if isinstance(flags, da.Array):
        flags = flags[:, :, None].rechunk(xds.CORRECTED_DATA.chunks)
    else:
        flags = da.asarray(flags[:, :, None], chunks=xds.CORRECTED_DATA.chunks)

    xds = xds.assign(CORRECTED_DATA=(("row", "chan", "corr"), corr_data))
    xds = xds.assign(FLAG=(("row", "chan", "corr"), flags))
    writes = xds_to_table(
        [
            xds,
        ],
        ms_path,
        ["CORRECTED_DATA", "FLAG"],
    )
    dask.compute(writes)
