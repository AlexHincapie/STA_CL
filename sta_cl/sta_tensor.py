import torch

class STATensor:
    def __init__(self):
        # Verifica si logró acceder a la GPU (CUDA)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✅ Motor Espacio-Temporal STA_CL enlazado. Dispositivo activo: {self.device}")
