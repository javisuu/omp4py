import ast
import os
import subprocess
import sysconfig
import re

# Import OMP4Py core contracts
from omp4py.core.directive import names, OmpClause, OmpArgs
from omp4py.core.processor.processor import omp_processor
from omp4py.core.processor.nodes import NodeContext

 
# Dinamic construction of the OMP pragma string
# @args:
#   clauses: List of OmpClause objects representing the clauses in the directive.
#  mangle: A dictionary mapping original variable names to their mangled counterparts.
def _build_pragma_string(clauses: list[OmpClause], mangle:dict[str,str])-> str:
    parts = [] # Other directives includind map, parallel, etc.

    for clause in clauses:
        # 1. If the clause doesnt have args we just add it to parts[]
        if not clause.args or not clause.args.array:
            parts.append(clause.token.id)
        
        # 2.  Extract and format the modifiers (e.g. to -> "to:")
        prefix=""
        if clause.args.modifiers: 
            for m in clause.args.modifiers: #Explore before ":"
                if m.name in (names.M_TO, names.M_FROM, names.M_TOFROM):
                    prefix = m.map + ":"
                    break  # Only one for now
        
        # 3. Extract the arguments (e.g. "x, y, z") and apply mangling
        #  if needed (e.g. "x" -> "x_mangled")
        mangled_args = []
        for arg in clause.args.array:
            arg_name= ast.unparse(arg).strip()
            # Get the mangled name if it exists, if not use the default name
            mangled_name = mangle.get(arg_name, arg_name)
            mangled_args.append(mangled_name)

        vars_str = ", ".join(mangled_args)

        # 4. Combine the clause + prefix + arguments and create the pragma
        parts.append(f"{clause.token.id}({prefix}{vars_str})")

        return "#pragma omp target " + " ".join(parts)

# Helper function in compilation_pipeline to inject the pragma into the generated C code
# @args:
#   c_path: Path to the generated C file from Cython
#   body_id: Unique identifier for the loop body to find the correct marker
#   pragma_str: The constructed OpenMP pragma string to inject
def __inject_pragma_into_c_code(c_path:str, body_id:int, pragma_str:str):
    with open(c_path, 'r') as file:
        c_code = file.read()

    #Search the unique marker 
    marker_match = re.search(r'__pyx_v_OMP4PY_MARKER_' + str(body_id) + r'\s*=\s*0;', c_code)
    if marker_match:
        rest_of_code = c_code[marker_match.end():]
        for_match = re.search(r'for\s*\(', rest_of_code)
        if for_match: #This is the loop afected by the pragma
            split_idx = marker_match.end() + for_match.start()
            c_code = c_code[:split_idx] + pragma_str + "\n  " + c_code[split_idx:]
            with open(c_path, 'w') as file:
                file.write(c_code)  

# Main function to create A100 binary from Cython
# @args:
#   body_id: Unique identifier for the loop body to find the correct marker
#   loop_code: The original loop code extracted from the AST, to be included in the Cython function
#   pragma_str: The constructed OpenMP pragma string to inject into the C code
def compilation_pipeline(body_id:int, loop_code:str,pragma_str:str)->str:
    """ Cython generation + C pragma injection + NVC compilation"""
    # Check if the build directory exists, if not create it
    build_dir = os.path.abspath("./gpu_build")
    os.makedirs(build_dir, exist_ok=True)

    # Paths for the generated files
    pyx_path = os.path.join(build_dir, f"kernel_{body_id}.pyx")
    c_path = os.path.join(build_dir, f"kernel_{body_id}.c")
    so_path = os.path.join(build_dir, f"kernel_{body_id}.so")

    # 1. Generate Cython
    #Using the array injection strategy

    cython_lines = [
        "# cython: language_level=3",
        "import cython",
        "@cython.boundscheck(False)",
        "@cython.wraparound(False)",
        "@cython.cdivision(True)",
        "cdef public void gpu_kernel(int n, double w, double *pi_ptr):",
        "    cdef double pi_value = pi_ptr[0]",
        "    cdef double local_x",
        "    cdef int i",
        f"    cdef int OMP4PY_MARKER_{body_id} = 0",
        ""
    ]

    # Append the original loop code, properly indented
    for line in loop_code.split('\n'):
        cython_lines.append(f"    {line}")

    # Append the pointer copy-back
    cython_lines.extend([
        "",
        "    pi_ptr[0] = pi_value",
        ""
    ])

    #Write the Cython code to disk
    with open(pyx_path, "w") as f:
        f.write("\n".join(cython_lines))

    # 2. Translate to C
    subprocess.run(["cython", pyx_path], check=True)

    #3. Inyect the pragma into the generated C code
    __inject_pragma_into_c_code(c_path, body_id, pragma_str)

    # 4. Compile to A100 .so Library
    py_include = sysconfig.get_path('include') or __import__('distutils.sysconfig').sysconfig.get_python_inc()
    subprocess.run(["nvc", "-mp=gpu", "-fPIC", "-shared", c_path, "-I" + str(py_include), "-o", so_path], check=True, timeout=60)
    
    return so_path

# MAIN FUNCTION: OMP4Py GPU Backend Compiler


@omp_processor(names.D_TARGET)
def target(body: list[ast.stmt], clauses: list[OmpClause], args: OmpArgs | None, ctx: NodeContext) -> list[ast.stmt]:
    """
    OMP4Py GPU Backend Compiler.
    Intercepts 'target' directives, dynamically compiles the loop to an A100 GPU binary,
    and replaces the Python AST with a ctypes runtime execution node using pure C pointers.
    """
    # 1. Setup & Code Extraction
    body_id = id(body)
    loop_code = ast.unparse(body)

    # 2. Variable Mangling Setup
    # Extract all active variables and map them to their Cython C-equivalents
    active_vars = [var for var in ctx.variables.names if not var.startswith('_')]
    mangle_dict = {var: f"__pyx_v_{var}" for var in active_vars}

    # 3. Pragma Construction
    pragma_str = _build_pragma_string(clauses, mangle_dict)

    # 4. Compilation Pipeline
    so_path = compilation_pipeline(body_id, loop_code, pragma_str)

    # 5. AST Replacement with ctypes execution node 
    # TODO -> Make dinamic to support different signatures (e.g. more args, different types, etc.)
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

