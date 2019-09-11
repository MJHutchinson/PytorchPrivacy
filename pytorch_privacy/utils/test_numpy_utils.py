from .numpy_utils import *

def test_clip():
    assert np.array_equal( clip(np.array([1.,2.,3.,4.]), 2.5), np.array([1., 2., 2.5, 2.5]))
