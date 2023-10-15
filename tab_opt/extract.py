from bdsf import process_image
import pandas as pd
import numpy as np

from astropy.coordinates import SkyCoord
from astropy.io import fits
from glob import glob
import argparse
import os
import sys

import xarray as xr

from tab_opt.image import image_ms
from tab_opt.data import write_corrected_data_ms

import dask.array as da


def extract(img_path: str, zarr_path: str, ms_path: str = None):
    img_name = os.path.splitext(img_path)[0]

    if ms_path is not None:
        image_ms(ms_path, img_name)

    image = process_image(img_path, quiet=True)
    # image = process_image(
    #     img_path + ".fits", quiet=True, thresh_isl=2.0, thresh_pix=1.0
    # )
    image.export_image(outfile=img_name + ".gauss_resid.fits", img_type="gaus_resid")
    image.write_catalog(
        outfile=img_name + ".pybdsf.csv", format="csv", catalog_type="srl", clobber=True
    )

    with fits.open(img_name + ".resid.fits") as hdul:
        noise = np.nanstd(hdul[0].data[0, 0])  # *image.pixel_beamarea()

    xds = xr.open_zarr(zarr_path)

    df = pd.read_csv(img_name + ".pybdsf.csv", skiprows=5)

    keys1 = [
        " Isl_id",
        " RA",
        " E_RA",
        " DEC",
        " E_DEC",
        " Total_flux",
        " E_Total_flux",
        " Maj",
        " E_Maj",
        " Min",
        " E_Min",
        " PA",
        " E_PA",
    ]
    keys2 = [" RA", " DEC", " Total_flux", " E_Total_flux"]
    image_df = (
        df[df[" Total_flux"] > 1e-5][keys1]
        .sort_values(" Total_flux")
        .reset_index()[keys1]
    )
    true_df = (
        pd.DataFrame(
            data=np.concatenate(
                [
                    xds.ast_radec.data,
                    xds.ast_I.data[:, 0, :],
                    np.zeros((xds.ast_I.data.shape[0], 1)),
                ],
                axis=1,
            ),
            columns=[" RA", " DEC", " Total_flux", " E_Total_flux"],
        )
        .sort_values(" Total_flux")
        .reset_index()[keys2]
    )

    c = SkyCoord(ra=image_df[" RA"], dec=image_df[" DEC"], unit="deg")
    catalog = SkyCoord(ra=true_df[" RA"], dec=true_df[" DEC"], unit="deg")
    idx, d2d, d3d = c.match_to_catalog_sky(catalog)

    true_df1 = true_df.iloc[idx].reset_index()[keys2]
    image_df1 = image_df[keys2]
    error_df = np.abs((image_df1 - true_df1))
    error_df = pd.DataFrame(
        data=np.abs((image_df1 - true_df1)[[" RA", " DEC"]] * 3600).values,
        columns=["E_RA [as]", "E_DEC [as]"],
    )
    error_df["E_Total_flux [%]"] = (
        100 * (image_df1 - true_df1)[" Total_flux"] / true_df1[" Total_flux"]
    )
    error_df[["RA [deg]", "DEC [deg]"]] = true_df1[[" RA", " DEC"]]
    error_df["Total_flux_true [mJy]"] = 1e3 * true_df1[" Total_flux"]
    error_df["Total_flux_image [mJy]"] = 1e3 * image_df1[" Total_flux"]
    error_df["SNR"] = image_df1[" Total_flux"] / noise
    error_df["Total_flux_std [mJy]"] = 1e3 * image_df1[" E_Total_flux"]
    error_df["Image_noise [mJy/beam]"] = (
        1e3 * image_df1[" Total_flux"] / image_df1[" Total_flux"] * noise
    )
    error_df["E_RADEC [as]"] = d2d.arcsec

    mask = d2d.arcsec < 10
    keys = [
        "RA [deg]",
        "E_RA [as]",
        "DEC [deg]",
        "E_DEC [as]",
        "Total_flux_true [mJy]",
        "E_Total_flux [%]",
        "E_RADEC [as]",
        "Total_flux_image [mJy]",
        "Total_flux_std [mJy]",
        "SNR",
        "Image_noise [mJy/beam]",
    ]
    error_df = error_df[keys].iloc[mask]
    error_df.to_csv(img_name + ".csv", index=False)


if __name__ == "__main__":
    program_desc = "Extract and measure sources from FITS file using PyBDSF."
    parser = argparse.ArgumentParser(description=program_desc)
    # Output File Arguments
    parser.add_argument("--zarr_path")
    # parser.add_argument("--img_path", help="Path to directory of images.")
    parser.add_argument("--ms_path", default=None, help="Path to directory of images.")
    parser.add_argument(
        "--type",
        default="ideal",
        help="Data type to image. {'ideal', 'flag', 'tabascal'}",
    )
    args = parser.parse_args()

    zarr_path = args.zarr_path
    # img_path = args.img_path
    ms_path = args.ms_path
    data_type = args.type.lower()
    img_path = os.path.join(zarr_path, f"{data_type}.fits")

    if data_type not in ["ideal", "flag", "tabascal"]:
        print(
            f"'{data_type}' not an option for 'type'."
            + " Choose from  {'ideal', 'flag', 'tabascal'}."
        )
        sys.exit(0)

    print(img_path)
    print(zarr_path)
    print(ms_path)

    # img_paths = glob(os.path.join(img_dir, "*.fits"))
    # img_paths = [os.path.splitext(x)[0] for x in img_paths if x.find("resid") == -1]
    xds = xr.open_zarr(zarr_path)
    if data_type == "ideal":
        corr_data = (xds.vis_model.data + xds.noise_data.data).reshape(-1, xds.n_freq)
        flags = da.zeros_like(corr_data, dtype=bool)
    elif data_type == "flag":
        corr_data = xds.vis_calibrated.data.reshape(-1, xds.n_freq)
        flags = xds.flags.data.reshape(-1, xds.n_freq)
    elif data_type == "tabascal":
        corr_data = xds.vis_corrected.data.reshape(-1, xds.n_freq)
        flags = da.zeros_like(corr_data, dtype=bool)

    write_corrected_data_ms(ms_path, corr_data, flags)
    extract(img_path, zarr_path, ms_path)

    # for img_path in img_paths:
    #     extract(img_path, zarr_path, ms_path)
