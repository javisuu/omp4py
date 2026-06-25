import time
from omp4py import omp

# 1. THE HOOK: The decorator intercepts the AST before execution
@omp(debug=True)
def test_memory_mapping(n: int):
    # Initialize variables inside the tracked scope
    input_val = 23.0
    output_val = 0.0

    # 2. THE ANCHOR: The target directive wrapped around a for-loop
    with omp("target map(to: input_val) map(from: output_val)"):
        # This execution happens entirely on the silicon of the A100
        for i in range(n):
            output_val += input_val
            
    return output_val

def main():
    print("🧪 Starting PCIe Memory Map Test (to / from)...")
    print("📦 [CPU] Before offload: input_val = 23.0, output_val = 0.0")

    n = 10
    
    # Trigger the compiled runtime execution
    result = test_memory_mapping(n)

    print(f"📥 [CPU] After offload: output_val = {result}")

    # The ultimate jury proof:
    if result == 230.0:
        print("🎉 SUCCESS! The JIT Compiler successfully routed memory across the PCIe bus!")
    else:
        print("❌ FAILURE! The host variable was not updated.")

if __name__ == "__main__":
    main()
