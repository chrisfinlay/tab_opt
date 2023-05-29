from setuptools import setup, find_packages

description = """Trajectory based Radio Frequency Interference (RFI) subtraction
                 and calibration using Bayesian methods for radio
                 interferometeric data."""

setup(
    name="tab_opt",
    version="0.0.1",
    description=description,
    url="http://github.com/chrisfinlay/tab_opt",
    author="Chris Finlay",
    author_email="christopher.finlay@unige.ch",
    license="MIT",
    packages=find_packages(),
    # package_data={"tabascal": ["tabascal/data/*"]},
    # entry_points="""
    #     [console_scripts]
    #     sim-vis=tabascal.examples.sim:cli
    # """,
    # install_requires=["jax", "jaxlib", "dask", "xarray", "dask-ms"],
    zip_safe=False,
)
