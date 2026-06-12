import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.auto_encoder import encode_mesh_multi_island
from core.reconstruct import reconstruct
from core.aero_3d import solve_3d_panel_method
from core.visualise import plot_3d_pressure
import matplotlib.pyplot as plt

def main():
    print("Testing 3D Source Panel Method on NetFold geometry...")
    
    # 1. Encode the nosecone into a NetFold object
    obj_path = os.path.join(os.path.dirname(__file__), 'test', 'nosecone.obj')
    print(f"Loading {obj_path}...")
    nf = encode_mesh_multi_island(obj_path, name="nosecone", normalize=True)
    
    # 2. Reconstruct the 3D mesh
    print("Reconstructing 3D mesh...")
    placed = reconstruct(nf, auto_refine=True, refine_iterations=200)
    
    # 3. Define freestream velocity (flow along X-axis from front to back)
    # The nosecone tip is at X=1.4, so flow should be towards -X.
    # We will use v_inf = [-1.0, 0.0, 0.0]
    v_inf = np.array([-1.0, 0.0, 0.0])
    
    # 4. Solve for 3D Pressure Distribution
    print("Solving 3D Aerodynamics (Constant-Strength Source Panel Method)...")
    cp_array, v_field = solve_3d_panel_method(placed, v_inf)
    
    print(f"Cp Range: min={cp_array.min():.3f}, max={cp_array.max():.3f}")
    
    # 5. Visualize
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'nosecone_3d_pressure.png')
    
    print("Plotting 3D pressure map...")
    plot_3d_pressure(placed, nf, cp_array, v_inf, save_path=save_path)
    print("Done!")

if __name__ == "__main__":
    main()
