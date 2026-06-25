import ast
import os
import sys
import subprocess
import sysconfig
import re
import time #TEMP
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

        #Support teams clause + distribute - TODO
        if clause.token.id==names.D_TEAMS:
            parts.append("teams distribute")
            continue

        # 1. If the clause doesnt have args we just add it to parts[]
        if not clause.args or not clause.args.array:
            parts.append(clause.token.id)
            continue
        
        # 2.  Extract and format the modifiers (e.g. to -> "to:")
        prefix=""
        if clause.args.modifiers:
            if clause.token.id=="reduction":
                prefix="+:"
            else:
            #for m in clause.args.modifiers: #Explore before ":"
                #if m.name in (names.M_TO, names.M_FROM, names.M_TOFROM):
                m_name=clause.args.modifiers[0].name 
                prefix = m_name + ":"
                
        
        # 3. Extract the arguments (e.g. "x, y, z") and apply mangling
        #  if needed (e.g. "x" -> "x_mangled")
        mangled_args = []
        for arg in clause.args.array:
            arg_name= "".join([t.string for t in arg.tokens]).strip()
            # Get the mangled name if it exists, if not use the default name
            mangled_name = mangle.get(arg_name, arg_name)
            mangled_args.append(mangled_name)

        vars_str = ", ".join(mangled_args)

        # 4. Combine the clause + prefix + arguments and create the pragma
        parts.append(f"{clause.token.id}({prefix}{vars_str})")

    return "#pragma omp target " + " ".join(parts)


# Helper function to extrtact the ctypes of a python variable (double as default)
#@args:
#   var_name: The name of the Python variable
#   ctx: The node context containing type information
def _get_c_type_info(var_name:str, ctx:NodeContext)->dict:
    """ If no type hint is given, we assuem dobuble for simplicity"""

    if var_name in ctx.variables.current_types:
        py_type=ctx.variables.current_types[var_name]
        if isinstance(py_type, ast.Name): #If the type is a simple name (e.g. int, float, etc.)
            if py_type.id == "int":
                return {"c_type": "int", "ctype_obj": "ctypes.c_int"}
            elif py_type.id == "float":
                return {"c_type": "double", "ctype_obj": "ctypes.c_double"}

    # Default to double if no type hint is found
    return {"c_type": "double", "ctype_obj": "ctypes.c_double"}

# Helper function to detect pointer variables
#@args:   
# clauses: List of OmpClause objects representing the clauses in the directive.
# @return A set of variable names that are mapped as 'from' or 'tofrom' and require C-pointers for output

def _get_pointer_variables(clauses:list[OmpClause])->set[str]:
    """
    Scan the OpenMP clause to find variables mapped as 'from' or 'tofrom' 
    Theese variables require C-pointer to return their modified values back
    """
    pointer_vars= set()

    for clause in clauses:
        # We look only for map clauses with arguments
        if clause.token.id=="map" and clause.args and hasattr(clause.args, "array") and clause.args.array:

            is_output=False
            if clause.args.modifiers:
                for m in clause.args.modifiers:
                    if m.name in (names.M_FROM, names.M_TOFROM):
                        is_output=True
                        break
                # If we found an output we add its variables to the set
            else:
                # If no modifiers are given, we assume it's a default map(tofrom)
                is_output=True
            if is_output:
                for arg in clause.args.array:
                    arg_name= "".join([t.string for t in arg.tokens]).strip() #.tokens gives the raw string in pieces
                    pointer_vars.add(arg_name)
        elif clause.token.id=="reduction" and clause.args and hasattr(clause.args, "array") and clause.args.array:
            for arg in clause.args.array:
                arg_name= "".join([t.string for t in arg.tokens]).strip()
                pointer_vars.add(arg_name)

    return pointer_vars


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

#Auxiliary class to dinamically extrac the variables used in the loop body and their types from the context, to be passed as arguments to the Cython function and then to the C kernel. This is needed to avoid passing functions/modules or unused variables that can cause compilation errors or runtime errors in the GPU execution.
class LocalVariableExtractor(ast.NodeVisitor):
    def __init__(self):
        self.assigned_vars = set()
        self.loop_vars = set()
    
    def visit_For(self, node):
        # Extract loop variable (e.g. 'i' in 'for i in range(...)')
        if isinstance(node.target, ast.Name):
            self.assigned_vars.add(node.target.id)
            self.loop_vars.add(node.target.id)
        self.generic_visit(node)

    def visit_Name(self,node):
        # We consider as local variables those that are assigned within the loop
        if isinstance(node.ctx, ast.Store):
            self.assigned_vars.add(node.id)
        self.generic_visit(node)

#Helper function to generate the cython inits
def _generate_cython_inits(loop_code:str,active_vars:list,ctx:NodeContext, body_id:int)->list[str]:
    """
    Generate the Cython variable initializations for the active variables, including the unique marker for pragma injection.
    """
    extractor = LocalVariableExtractor()
    extractor.visit(ast.parse(loop_code))

    # Loop variables 
    loop_vars= extractor.assigned_vars - set(active_vars) #We consider loop variables as local variables, so we exclude them from the active variables list

    inits=[]

    for var in loop_vars:
        if var in extractor.loop_vars:
            # Loop index
            c_type = "int"
        else:
            # Guess the type for standard assignments
            type_info = _get_c_type_info(var, ctx)
            c_type = type_info["c_type"]
        inits.append(f"    cdef {c_type} {var}")

    # Add the framework tracking marker
    inits.append(f"    cdef int OMP4PY_MARKER_{body_id} = 0")
    inits.append("") # Empty line for spacing

    return inits




# Main function to create A100 binary from Cython
# @args:
#   body_id: Unique identifier for the loop body to find the correct marker
#   loop_code: The original loop code extracted from the AST, to be included in the Cython function
#   pragma_str: The constructed OpenMP pragma string to inject into the C code
#   active_vars: List of active variables in the loop
#   ctx: Node context containing type information
def compilation_pipeline(body_id:int, loop_code:str,pragma_str:str, active_vars:list,ctx:NodeContext,pointer_vars:set[str])->str:
    """ Cython generation + C pragma injection + NVC compilation"""
    # Check if the build directory exists, if not create it
    build_dir = os.path.abspath("./gpu_build")
    os.makedirs(build_dir, exist_ok=True)

    # Paths for the generated files
    pyx_path = os.path.join(build_dir, f"kernel_{body_id}.pyx")
    c_path = os.path.join(build_dir, f"kernel_{body_id}.c")
    so_path = os.path.join(build_dir, f"kernel_{body_id}.so")

    t0_cython=time.perf_counter() #TEMP
    # 0. Dynamic construction of the Cython function with pointer arguments for active variables
    
    cython_args = []
    cython_body_inits = _generate_cython_inits(loop_code, active_vars, ctx, body_id) # Generate the Cython inits for local variables and the marker

    cython_body_teardowns = [] # In this simple example we only have one output variable (pi_value), but this can be extended to multiple variables if needed

    for var in active_vars:
        type_info=_get_c_type_info(var, ctx)
        c_type=type_info["c_type"]

        # We still asume that pi_value is the output pointer
        if var in pointer_vars:
            cython_args.append(f"{c_type} *{var}_ptr")
            cython_body_inits.insert(0, f"    cdef {c_type} {var} = {var}_ptr[0]")
            cython_body_teardowns.append(f"    {var}_ptr[0] = {var}")
        else:
            cython_args.append(f"{c_type} {var}")
    
    signature = ", ".join(cython_args)


    # 1. Generate Cython
    #Using the array injection strategy

    cython_lines = [
        "# cython: language_level=3",
        "import cython",
        "@cython.boundscheck(False)",
        "@cython.wraparound(False)",
        "@cython.cdivision(True)",
        f"cdef public void gpu_kernel({signature}):" # Dinamic signature
    ]
    # We append also the cython body inits
    cython_lines.extend(cython_body_inits) 

    # Append the original loop code, properly indented
    for line in loop_code.split('\n'):
        cython_lines.append(f"    {line}")

    cython_lines.append("") #Empty line between body and teardowns
    cython_lines.extend(cython_body_teardowns) # Append the teardowns at the end of the function

    #Write the Cython code to disk
    with open(pyx_path, "w") as f:
        f.write("\n".join(cython_lines))

    # 2. Translate to C
    subprocess.run(["cython", pyx_path], check=True)

    #3. Inyect the pragma into the generated C code
    __inject_pragma_into_c_code(c_path, body_id, pragma_str)

    t1_cython = time.perf_counter() #TEMP
    print(f"[TELEMETRÍA] 2. Generación Cython/C: {(t1_cython - t0_cython) * 1000:.3f} ms", file=sys.stderr)
    t0_nvc = time.perf_counter()
    # 4. Compile to A100 .so Library
    py_include = sysconfig.get_path('include') or __import__('distutils.sysconfig').sysconfig.get_python_inc()
    try:
        subprocess.run(["nvc", "-mp=gpu", "-fPIC", "-shared", c_path,
                        "-I" + str(py_include),
                        # Python 3.11+ reubicó longintrepr.h al subdir cpython/
                        "-I" + os.path.join(str(py_include), "cpython"),
                        # Interruptor maestro: fuerza a Cython a usar SOLO API pública
                        # portable (sin tocar structs internos de CPython 3.12 que
                        # nvc 21.2 no soporta: ob_digit, curexc_type, _frame, etc.)
                        "-DCYTHON_COMPILING_IN_CPYTHON=0",
                        "-o", so_path], check=True, timeout=60, capture_output=True, text=True)

    except (subprocess.CalledProcessError,FileNotFoundError) as e:
        # [DEBUG-REVERT] Destapar el error real del compilador en vez de tragarlo
        print(f"[JIT-FAIL] cython/nvc falló:\n{getattr(e, 'stderr', '')}\n{getattr(e, 'stdout', '')}", file=sys.stderr)
        return None
        
    t1_nvc = time.perf_counter() #TEMP
    print(f"[TELEMETRÍA] 3. Compilación NVC:     {(t1_nvc - t0_nvc) * 1000:.3f} ms\n", file=sys.stderr)

    return so_path

# MAIN FUNCTION: OMP4Py GPU Backend Compiler


@omp_processor(names.D_TARGET)
def target(body: list[ast.stmt], clauses: list[OmpClause], args: OmpArgs | None, ctx: NodeContext) -> list[ast.stmt]:
    """
    OMP4Py GPU Backend Compiler.
    Intercepts 'target' directives, dynamically compiles the loop to an A100 GPU binary,
    and replaces the Python AST with a ctypes runtime execution node using pure C pointers.
    """
    #print("\n" + "="*45)
    #print("INICIANDO COMPILACIÓN JIT (omp4py)")
    #print("="*45)

    t0_ast = time.perf_counter()

    # 1. Setup & Code Extraction
    body_id = id(body)
    loop_code = ast.unparse(body)

    # 2. Variable Scope Filtering
    # Identify variables actually used inside the loop AST to avoid passing functions/modules
    loop_names = set()
    for stmt in body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name):
                loop_names.add(node.id)

    # Extract all active variables: must be in the context AND actually used in the loop
    active_vars = [var for var in ctx.variables.names if not var.startswith('_') and var in loop_names]
    mangle_dict = {var: f"__pyx_v_{var}" for var in active_vars}

    # 3. Pragma Construction
    pragma_str = _build_pragma_string(clauses, mangle_dict)

    # NEW: 3.5 Detect which variables need physical memory pointers
    pointer_vars = _get_pointer_variables(clauses)

    t1_ast = time.perf_counter()
    print(f"[TELEMETRÍA] 1. Frontend AST/Tipos:  {(t1_ast - t0_ast) * 1000:.3f} ms", file=sys.stderr)

    # 4. Compilation Pipeline
    so_path = compilation_pipeline(body_id, loop_code, pragma_str, active_vars, ctx, pointer_vars)

    # If compilation fails so_path is none and the execution is in the CPU
    if so_path is None:
        raise RuntimeError("🚨 [CRITICAL] Fallback detectado en el pipeline del JIT")
        return body
    # 5. AST Replacement with ctypes execution node 
    ctypes_argtypes = [] # List of ctypes types for the function arguments, to be used in the runtime execution node
    ctypes_call_args = [] # List of arguments to pass to the ctypes function call in the runtime execution node
    ptr_setup= [] # List of lines to setup the C pointers for the active variables in the runtime execution node
    ptr_teardown= [] # List of lines to copy back the results from the C pointers to the original Python variables in the runtime execution node

    for var in active_vars:
        type_info=_get_c_type_info(var, ctx)
        
        if var in pointer_vars:
            ctypes_argtypes.append(f"ctypes.POINTER({type_info['ctype_obj']})")
            ptr_setup.append(f"_{var}_c={type_info['ctype_obj']}({var})") #Allocate the variable in a physical C chunck of memory
            ctypes_call_args.append(f"ctypes.byref(_{var}_c)") #byref id the memory address
            ptr_teardown.append(f"{var} = _{var}_c.value")

        else:
            ctypes_argtypes.append(type_info["ctype_obj"])
            ctypes_call_args.append(var)

    # Join the dynamic lists into formatted strings
    argtypes_str = ",\n    ".join(ctypes_argtypes)
    call_args_str = ", ".join(ctypes_call_args)
    setup_str = "\n".join(ptr_setup)
    teardown_str = "\n".join(ptr_teardown)

    runtime_execution_code = f"""
import ctypes

# Load the compiled A100 library
_gpu_lib = ctypes.CDLL("{so_path}")
_gpu_lib.gpu_kernel.restype = None

# Dynamic argtypes mapping
_gpu_lib.gpu_kernel.argtypes = [
    {argtypes_str}
]

# Pointer initialization
{setup_str}

# Execute kernel
_gpu_lib.gpu_kernel({call_args_str})

# Extract pointer values back to Python
{teardown_str}
"""
    # Calculamos el tiempo total de todo el JIT
    t_total_fin = time.perf_counter()
    print("-" * 45, file=sys.stderr)
    print(f"[TELEMETRÍA] OVERHEAD JIT TOTAL:     {(t_total_fin - t0_ast) * 1000:.3f} ms", file=sys.stderr)
    print("="*45 + "\n", file=sys.stderr)

    # Parse the runtime code into AST and return it.
    return ast.parse(runtime_execution_code).body

