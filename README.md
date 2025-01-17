[![pipeline status](https://git.ligo.org/colm.talbot/bilby-cython/badges/main/pipeline.svg)](https://git.ligo.org/colm.talbot/bilby-cython/-/commits/main)
[![coverage report](https://git.ligo.org/colm.talbot/bilby-cython/badges/main/coverage.svg)](https://git.ligo.org/colm.talbot/bilby-cython/-/commits/main)
[![Latest Release](https://git.ligo.org/colm.talbot/bilby-cython/-/badges/release.svg)](https://git.ligo.org/colm.talbot/bilby-cython/-/releases)
[![Conda Forge](https://anaconda.org/conda-forge/bilby.cython/badges/version.svg)](https://anaconda.org/conda-forge/bilby.cython)
[![License](https://anaconda.org/conda-forge/bilby.cython/badges/license.svg)](https://anaconda.org/conda-forge/bilby.cython)

# bilby-cython

Optimized `cython` implementations of specific `Bilby` operations.

For very fast waveform models, computing the antenna response and time delays
uses a significant amount of run time, see, e.g., [this issue](https://git.ligo.org/lscsoft/bilby/-/issues/576).

This repo provides optimized `cython` implementations of the existing `Python`
functions.
Most of the speed up comes from leveraging the fact that these are operations
of 3-element vectors.

## Installation

`bilby.cython` is available through `conda-forge` and `pypi` and can be installed with either of the following

```console
$ conda install -c conda-forge bilby.cython
$ python -m pip install bilby.cython
```

There are prebuilt versions for many architectures/OSs, however, non-standard systems will require a version of `Cython` to build from the source with

## Usage

In order to use the functions implemented here you can import them from the
`bilby.cython package`
```python
from bilby_cython import geometry
geometry.get_polarization_tensor(ra=0.0, dec=0.0, time=0.0, psi=0.0, mode="plus")
```

## Ported functions

- `time_delay_geocentric`: calculating time delays between two interferometers
- `get_polarization_tensor`: calculation of polarization tensors
- `three_by_three_matrix_contraction`: rojecting polarization tensors against
   detector response tensors
- `calculate_arm`: calculate an interferometer arm vector
- `zenith_azimuth_to_theta_phi`: rotating the reference from from
   detector-based coordinates to celestial

## New functions

- `time_delay_from_geocenter`: calculate the time delay between an
   interferometer and the geocenter. This removes an array allocation that was
   slow for some reason.
- `get_polarization_tensor_multiple_modes`: compute multiple polarization
   tensors in a single function call. This reduces the overheads per call.

## Testing

`test/test_geometry.py` verifies that the new functions agree with the old
versions at sufficient precision.
The old code is deliberately copied from the current `Bilby` to enable testing
after `Bilby` switches to use this code.

## Timing

This is the output of `test/timing.py` when run on a 2020 M1 MacBook Pro.
The new functions are `5-40x` faster than the old versions.
These times are comparable with the lal` equivalents with greater flexibility.

```
Timing time delay calculation over 10000 trials.
Cython time: 2.366e-06
Numpy time: 1.026e-05
Timing polarization tensor calculation over 1000 trials.
Cython time: 2.470e-06
Numpy time: 4.247e-05
Timing two mode polarization tensor calculation over 1000 trials.
Cython time: 1.790e-06
Numpy time: 2.907e-05
Timing six mode polarization tensor calculation over 1000 trials.
Cython time: 1.250e-06
Numpy time: 4.111e-05
Timing antenna response calculation over 1000 trials.
Cython time: 4.269e-06
Numpy time: 4.709e-05
Timing frame conversion calculation over 1000 trials.
Cython time: 2.516e-06
Numpy time: 2.243e-05
```

Profiling the standard `Bilby` `fast_tutorial.py` example for a very fast
waveform model reduces the amount of time spent computing the antenna response
from 16% to 3% and computing time delays from 3% to 1%.

