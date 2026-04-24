import math
import time
import numpy as np
from omp4py import omp

@omp(debug=True)
def target_reduction_pi(n: int):
    w = 1.0 / n
    pi_value = 0.0
    
    with omp("target map(pi_value)"):
        for i in range(n):
            local_x = (i + 0.5) * w
            pi_value += 4.0 / (1.0 + local_x * local_x)
            
    return pi_value * w

if __name__ == "__main__":
    print("🚀 Starting compilation and execution on A100...")
    start = time.time()
    
    # Run the function!
    n = 100_000
    result = target_reduction_pi(n)
    
    end = time.time()
    print(f"✅ Compilation & Execution finished in {end - start:.2f} seconds.")
    print(f"🔢 Calculated Pi: {result}")
    print(f"🎯 Math module Pi: {math.pi}")
    
    if math.isclose(result, math.pi, rel_tol=1e-5):
        print("🎉 SUCCESS! The GPU calculated Pi correctly!")
    else:
        print("❌ FAILURE! The math doesn't match.")
