import time

import numpy as np
import pytest

from lccd_python.blob_detector import BlobDetector


TEST_TIMES = 2

class TestBlobDetector:

    blob_detector_gpu = BlobDetector(**{'filtersize1': 100,
        'filtersize2': 4,
        'sigma': 1.25,
        'fsize': 7,
        'min_area': 20,
        'max_area': 50, 'gpu': True, 'matlab_conv': False, 'debug': False})
    blob_detector_cpu = BlobDetector(**{'filtersize1': 100,
        'filtersize2': 4,
        'sigma': 1.25,
        'fsize': 7,
        'min_area': 20,
        'max_area': 50, 'gpu': False, 'matlab_conv': False, 'debug': False})

    test_variables = []
    for _ in range(TEST_TIMES):
        X, Y = np.random.randint(300, 600, 2)
        T = 100
        tv = np.random.rand(X, Y, T).astype(np.float32)
        tv = tv.reshape(-1, T)
        test_variables.append((tv, X, Y))

    def test_preprocess(self):
        gpu_time, cpu_time = 0, 0
        for tv, X, Y in self.test_variables:
            before = time.time()
            max_lumi_map_gpu = self.blob_detector_gpu.preprocess(tv, X, Y)
            gpu_time += time.time()-before
            before = time.time()
            max_lumi_map_cpu = self.blob_detector_cpu.preprocess(tv, X, Y)
            cpu_time += time.time()-before
            np.testing.assert_array_almost_equal(max_lumi_map_gpu, max_lumi_map_cpu)
        print(f"GPU is {round(cpu_time / gpu_time, 3)} faster than CPU. CPU took {round(cpu_time)} s.")

    def test_apply(self):
        gpu_time, cpu_time = 0, 0
        for tv, X, Y in self.test_variables:
            tv = tv.reshape(X, Y, -1)
            before = time.time()
            ROI_gpu = self.blob_detector_gpu.apply(tv)
            gpu_time += time.time()-before
            before = time.time()
            ROI_cpu = self.blob_detector_cpu.apply(tv)
            cpu_time += time.time()-before
            np.testing.assert_array_almost_equal(ROI_gpu, ROI_cpu)
        print(f"GPU is {round(cpu_time / gpu_time, 3)} faster. CPU took {round(cpu_time)} s.")
