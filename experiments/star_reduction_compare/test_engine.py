import unittest
import numpy as np
from engine import midtones, reduce_local


class ReductionTests(unittest.TestCase):
    def test_identity(self):
        original=np.random.default_rng(5).random((20,30,3),dtype=np.float32)
        np.testing.assert_array_equal(reduce_local(original,original*.7,0),original)
        np.testing.assert_allclose(midtones(original,.5),original,atol=1e-7)

    def test_monotonic_background_and_color(self):
        base=np.full((30,40,3),.13,np.float32)
        stars=np.random.default_rng(4).uniform(.01,.6,(30,40,3)).astype(np.float32)
        original=base+(1-base)*stars
        mild=reduce_local(original,base,.3)
        strong=reduce_local(original,base,.8)
        self.assertTrue(np.all(strong>=base) and np.all(strong<=mild) and np.all(mild<=original))
        residual=(strong-base)/(1-base)
        gains=residual/stars
        np.testing.assert_allclose(gains[...,0],gains[...,1],atol=2e-6)
        np.testing.assert_array_equal(reduce_local(base,base,1),base)

    def test_white_and_black_finite(self):
        for value in (0.,1.):
            original=np.full((3,4,3),value,np.float32)
            result=reduce_local(original,original*.8,1)
            self.assertTrue(np.isfinite(result).all())
            self.assertTrue(np.all(result>=0) and np.all(result<=1))

if __name__=='__main__':unittest.main()
