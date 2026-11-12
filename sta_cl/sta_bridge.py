import torch
import numpy as np
from .sta_tensor import STATensor

class STABridge:
    """
    Binary serialization module v1.2.
    Exports 16D tensors from PyTorch device memory to 1D arrays for WebGL and Three.js.
    """
    
    @staticmethod
    def export_positions_to_buffer(tensor_obj):
        """
        Extracts grade 1 spatial coordinates (x, y, z) from a batch.
        Returns a 1D NumPy array [x1, y1, z1, x2, y2, z2...] formatted for BufferGeometry in JavaScript.
        """
        # Extract indices: g1(2), g2(4), g3(8)
        pos_3d = tensor_obj.data[:, [2, 4, 8]].detach().cpu().numpy()
        
        # Flatten array to 1D (C-contiguous) in float32 format
        return pos_3d.astype(np.float32).ravel()

    @staticmethod
    def export_bivectors_to_normals(tensor_obj):
        """
        Extracts spatial bivectors (g1^g2, g2^g3, g3^g1) as normal vectors.
        Index mapping: g2^g3(12)->x_norm, g3^g1(10)->y_norm, g1^g2(6)->z_norm.
        """
        # Extract spatial bivector components
        normals_3d = tensor_obj.data[:, [12, 10, 6]].detach().cpu().numpy()
        
        # Normalize vectors along axis 1
        norms = np.linalg.norm(normals_3d, axis=1, keepdims=True)
        norms[norms == 0] = 1.0 
        normals_3d = normals_3d / norms
        
        return normals_3d.astype(np.float32).ravel()

    @staticmethod
    def serialize_to_file(tensor_obj, filename="simulation_frame.bin"):
        """
        Writes 3D spatial coordinates to a binary file.
        """
        buffer_data = STABridge.export_positions_to_buffer(tensor_obj)
        with open(filename, 'wb') as f:
            f.write(buffer_data.tobytes())
        print(f"Frame exported: {filename} ({len(buffer_data)//3} particles).")
