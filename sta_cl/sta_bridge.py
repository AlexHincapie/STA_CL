import torch
import numpy as np
from .sta_tensor import STATensor

class STABridge:
    """
    Puente de Serialización Binaria v1.1.
    Exporta geometrías masivas 16D desde la VRAM de PyTorch a Buffers planos para WebGL/Three.js.
    """
    
    @staticmethod
    def export_positions_to_buffer(tensor_obj):
        """
        Extrae exclusivamente el grado 1 (Cuadrivector espacial x, y, z) de un lote masivo.
        Retorna un arreglo Numpy plano [x1, y1, z1, x2, y2, z2...] listo para `BufferGeometry` en JS.
        """
        # Extraer grados: g1(2), g2(4), g3(8)
        pos_3d = tensor_obj.data[:, [2, 4, 8]].detach().cpu().numpy()
        
        # Aplanar la matriz a 1D (C-contiguous) para transmisión binaria
        return pos_3d.astype(np.float32).ravel()

    @staticmethod
    def export_bivectors_to_normals(tensor_obj):
        """
        Extrae los bivectores de Pauli (Áreas orientadas) para usarlos como normales de renderizado.
        g1^g2(6)->z_norm, g2^g3(12)->x_norm, g3^g1(10)->y_norm
        """
        # Extraer campos magnéticos / áreas espaciales puras
        normals_3d = tensor_obj.data[:, [12, 10, 6]].detach().cpu().numpy()
        
        # Normalizar los vectores para el cálculo de iluminación en shaders
        norms = np.linalg.norm(normals_3d, axis=1, keepdims=True)
        # Evitar división por cero
        norms[norms == 0] = 1.0 
        normals_3d = normals_3d / norms
        
        return normals_3d.astype(np.float32).ravel()

    @staticmethod
    def serialize_to_file(tensor_obj, filename="simulation_frame.bin"):
        """
        Vuelca las posiciones espaciales 3D puras en un archivo binario ultraligero.
        Ideal para cargar cuadros de simulación estáticos en el front-end web.
        """
        buffer_data = STABridge.export_positions_to_buffer(tensor_obj)
        with open(filename, 'wb') as f:
            f.write(buffer_data.tobytes())
        print(f"📦 Frame exportado: {filename} ({len(buffer_data)//3} partículas).")
