import sys
# Assuming your standard omp4py import
from omp4py import omp

def main():
    print("🧪 Starting PCIe Memory Map Test (to / from)...")
    
    # 1. Initialize variables on the Host (CPU)
    input_val = 42.0
    output_val = 0.0
    
    print(f"📦 [CPU] Before offload: input_val = {input_val}, output_val = {output_val}")
    
    # 2. Offload to Device (A100 GPU)
    # - map(to: input_val)   -> Send 42.0 to the GPU's memory.
    # - map(from: output_val)-> Retrieve the result back to the CPU's memory.
    with omp("target map(to: input_val) map(from: output_val)"):
        # This execution happens entirely on the silicon of the A100
        output_val = input_val * 2.0
        
    # 3. Verify the result back on the Host (CPU)
    print(f"📥 [CPU] After offload: output_val = {output_val}")
    
    # The ultimate jury proof:
    if output_val == 84.0:
        print("🎉 SUCCESS! The JIT Compiler successfully routed memory across the PCIe bus!")
    else:
        print("❌ FAILURE! The host variable was not updated.")

if __name__ == "__main__":
    n = 1000000 # Dummy variable if your backend still requires it in scope
    main()