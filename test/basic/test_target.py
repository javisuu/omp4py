import pytest
import math
import ctypes
import os
from omp4py import *

# ---------------------------------------------------------
# Test 1: Direct GPU Compilation (Bypassing the CPU Harness)
# ---------------------------------------------------------
def test_target_reduction_pi():
    n = 100_000 
    
    # We are going to trigger the omp4py parser, but NOT its internal compiler.
    # By omitting `compile=True`, it just processes the AST and triggers your gpu.py hook!
    @omp(debug=True)
    def target_reduction_pi(n: int):
        w = 1.0 / n
        pi_value = 0.0
        with omp("target"):
            for i in range(n):
                local_x = (i + 0.5) * w
                pi_value += 4.0 / (1.0 + local_x * local_x)
        return pi_value * w

    # Run the function!
    result = target_reduction_pi(n)
    
    assert math.isclose(result, math.pi, rel_tol=1e-5)


# ---------------------------------------------------------
# Test 2: Cython GIL Syntax Catch
# ---------------------------------------------------------
def test_target_syntax_error():
    # We expect Cython to throw an error because it's invalid C code.
    with pytest.raises(Exception):
        @omp(debug=True)
        def target_syntax_error():
            with omp("target"):
                return undefined_variable 
        
        target_syntax_error()
