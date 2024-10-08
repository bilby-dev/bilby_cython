import concurrent.futures
import numpy as np
from bilby_cython import geometry


def test_polarization_tensor_threadsafe():
    """
    A basic test of thread safety for the polarization tensor calculation.
    Previously, this was not thread safe due to the use of global variables
    to store intermediate results.
    """

    def dummy_func(val):
        return geometry.get_polarization_tensor(*val, "plus")

    values = np.random.uniform(0, 1, (10000, 4))

    truths = np.array([geometry.get_polarization_tensor(*val, "plus") for val in values])

    results = truths.copy()
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        jobs = {executor.submit(dummy_func, val): ii for ii, val in enumerate(values)}
        for job in concurrent.futures.as_completed(jobs):
            results[jobs[job]] = job.result()

    assert np.allclose(truths, results)