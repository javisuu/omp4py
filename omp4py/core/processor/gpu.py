import ast
import os
import subprocess
import sysconfig
import re

# Import OMP4Py core contracts
from omp4py.core.directive import names, OmpClause, OmpArgs
from omp4py.core.processor.processor import omp_processor
from omp4py.core.processor.nodes import NodeContext

@omp_processor(names.D_TARGET)
def target(body: list[ast.stmt], clauses: list[OmpClause], args: OmpArgs | None, ctx: NodeContext) -> list[ast.stmt]:
    """
    OMP4Py GPU Backend Compiler.
    Intercepts 'target' directives, dynamically compiles the loop to an A100 GPU binary,
    and replaces the Python AST with a ctypes runtime execution node using pure C pointers.
    """

    # 1. SOURCE RETRIEVAL Using AST UNPARSING
    loop_code = ast.unparse(body)

    # 2. VARIABLE EXTRACTION
    # Determine which variables the loop actually uses so we can pass them to C.
    active_vars = [var for var in ctx.variables.names if not var.startswith('_')]

    # 3. DYNAMIC CLAUSE RECONSTRUCTION & MANGLING
    clause_strings = []
    if clauses:
        for clause in clauses:
            c_args = ""
            # Extract the actual string values from the tokens inside the OmpArgs object
            if clause.args and hasattr(clause.args, 'array') and clause.args.array:
                arg_strings = []
                for item in clause.args.array:
                    for token in item.tokens:
                        arg_strings.append(token.string)
                c_args = "".join(arg_strings)
            elif clause.args:
                c_args = str(clause.args)
            
            # Dynamically map all active Python variables in the clause to Cython C-variables
            for var in active_vars:
                # We use regex word boundaries (\b) so we don't accidentally replace partial words
                c_args = re.sub(rf'\b{var}\b', f'__pyx_v_{var}', c_args)
                
            clause_strings.append(f"{clause.token.id}({c_args})")

    # The framework dynamically assembles the string based EXACTLY on the parsed AST
    pragma_str = f"#pragma omp target {' '.join(clause_strings)}"

    # 4. THE COMPILATION PIPELINE
    build_dir = os.path.abspath("./gpu_build")
    os.makedirs(build_dir, exist_ok=True)
    pyx_path = os.path.join(build_dir, f"kernel_{id(body)}.pyx")
    c_path = os.path.join(build_dir, f"kernel_{id(body)}.c")
    so_path = os.path.join(build_dir, f"kernel_{id(body)}.so")

    # Generate Cython (Pure Ctypes Pointer Version)
    cython_code = "# cython: language_level=3\nimport cython\n"
    cython_code += "@cython.boundscheck(False)\n@cython.wraparound(False)\n@cython.cdivision(True)\n"
    
    # Accept a pure C-pointer for the result
    cython_code += "cdef public void gpu_kernel(int n, double w, double *pi_ptr):\n"
    cython_code += "    cdef double pi_value = pi_ptr[0]\n"  # Local var for OpenMP to map!
    cython_code += "    cdef double local_x\n"
    cython_code += "    cdef int i\n"
    cython_code += "    cdef int OMP4PY_MARKER = 0\n\n"

    for line in loop_code.split('\n'):
        cython_code += f"    {line}\n"

    # Copy the GPU-calculated result back into the physical pointer memory
    cython_code += "\n    pi_ptr[0] = pi_value\n"

    # Write the Cython code to disk so the compiler can find it
    with open(pyx_path, "w") as f:
        f.write(cython_code)

    # Translate to C
    subprocess.run(["cython", pyx_path], check=True)

    # Read the C file to inject the Pragma
    with open(c_path, 'r') as file:
        c_code = file.read()

    marker_match = re.search(r'__pyx_v_OMP4PY_MARKER\s*=\s*0;', c_code)
    if marker_match:
        rest_of_code = c_code[marker_match.end():]
        for_match = re.search(r'for\s*\(', rest_of_code)
        if for_match:
            split_idx = marker_match.end() + for_match.start()
            c_code = c_code[:split_idx] + pragma_str + "\n  " + c_code[split_idx:]
            with open(c_path, 'w') as file:
                file.write(c_code)

    # Compile with NVC (With 60-second timeout to prevent panicking)
    py_include = sysconfig.get_path('include') or __import__('distutils.sysconfig').sysconfig.get_python_inc()
    subprocess.run(["nvc", "-mp=gpu", "-fPIC", "-shared", c_path, "-I" + str(py_include), "-o", so_path], check=True, timeout=60)

    # 5. AST REPLACEMENT (The Pure Ctypes Execution Hook)
    runtime_execution_code = f"""
import ctypes

# Load the compiled A100 library
_gpu_lib = ctypes.CDLL("{so_path}")
_gpu_lib.gpu_kernel.restype = None

# Define the argument types to accept a pointer!
_gpu_lib.gpu_kernel.argtypes = [
    ctypes.c_int, 
    ctypes.c_double, 
    ctypes.POINTER(ctypes.c_double)
]

# Create a raw C-pointer initialized with the current Python value
_pi_c = ctypes.c_double(pi_value)

# Execute the kernel! Pass the memory address via ctypes.byref
_gpu_lib.gpu_kernel(n, w, ctypes.byref(_pi_c))

# Assign the calculated result back to the Python variable space
pi_value = _pi_c.value
"""

    # Parse the runtime code into AST and return it.
    return ast.parse(runtime_execution_code).body
