"""
Deprecated compatibility wrapper.

This file used to contain the LS-DYNA setup implementation. The implementation
has been moved to `primemesh.py`. Importing `lsdyna_setup` will delegate to
`primemesh` so existing imports continue to work. Running this file as a
script will call `primemesh.main()`.
"""

import os
import sys
import importlib.util

print("⚠ WARNING: 'lsdyna_setup' has been renamed to 'primemesh'. Delegating to primemesh.")

# Load primemesh.py from the same directory as this file
this_dir = os.path.dirname(__file__)
primemesh_path = os.path.join(this_dir, 'primemesh.py')

if not os.path.exists(primemesh_path):
    raise FileNotFoundError(f"primemesh.py not found next to {__file__}; cannot delegate")

spec = importlib.util.spec_from_file_location('primemesh', primemesh_path)
primemesh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(primemesh)

# Re-export commonly used symbols for backward compatibility
setup_lsdyna = primemesh.setup_lsdyna
mesh_with_pyprimemesh = primemesh.mesh_with_pyprimemesh
update_metadata = primemesh.update_metadata
load_metadata = primemesh.load_metadata


if __name__ == '__main__':
    primemesh.main()
