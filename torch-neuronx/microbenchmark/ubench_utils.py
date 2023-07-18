from numbers import Number
import time


class Timer:
    """
    A simple Timer with high-enough granularity for performance
    measurments. The timer is catered towards using as a context manager.

    Example usage:
          with ubench_utils.Timer() as benchmark_timer:
              time.sleep(1)
              time.sleep(4)

          act_time = benchmark_timer()
          print("Sleeping for 5 seconds actualy took {:2g} seconds".format(act_time)
    """

    def __enter__(self):
        self.start = time.perf_counter()
        self.end = 0.0
        return lambda: self.end - self.start

    def __exit__(self, *args):
        self.end = time.perf_counter()

