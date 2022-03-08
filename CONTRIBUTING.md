# Contributing Guidelines

## Should I add a function to `bilby_cython`?

This package is intended to provide optimized implementations of functions
specific to gravitaitonal-wave data analysis in `Bilby`.
As one of the core driving principles of `Bilby` is to write software that
is flexible and simple for new users to understand and develop into the
decision to move functions to this optimized package must have a high
barrier.
As a general metric, one should show that the function being ported is:
- taking `O(%)` of the total run time for a not-too-contrived run configuration. 
  This can be verified using the
  `Python` profiler `cProfile` (see, e.g.,
  [this SO](https://stackoverflow.com/questions/582336/how-can-you-profile-a-python-script)
  for more details.)
- porting to `Cython` reduces the run time associated with that function by
  more than a factor of 4.

As an example, the primary motivation for beginning this package was that when
using a very fast source model evaluating the response of the gravitational-wave
detectors contributed a significant amount of run time due to having small
array operations being not very well optimized in `numpy`.
By hard-coding 3x3 matrix operations a significant acceleration is possible.

## How do I add a function?

For general details about how `Cython` works, we recommend looking for
information online, the official documentaiton and stack overflow (SO)
are a great source of information.

If you add a new function you should always add new tests that compare the
result of the new function and the existing pure-Python `Bilby` implementation.
This will require copying the existing code from `Bilby` as the current `Bilby`
implementation will be replaced by the `Cython` version.

You should also make sure to add the function to the `README` in this repository.
