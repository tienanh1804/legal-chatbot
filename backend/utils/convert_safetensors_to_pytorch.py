import os
import torch
from safetensors import safe_open

def convert_safetensors_to_pytorch(model_dir):
    """
    Convert a model.safetensors file to pytorch_model.bin
    
    Args:
        model_dir: Directory containing the model.safetensors file
    """
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    pytorch_path = os.path.join(model_dir, "pytorch_model.bin")
    
    if not os.path.exists(safetensors_path):
        print(f"Error: {safetensors_path} does not exist")
        return False
    
    print(f"Converting {safetensors_path} to {pytorch_path}")
    
    # Load tensors from safetensors file
    tensors = {}
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    
    # Save tensors to pytorch_model.bin
    torch.save(tensors, pytorch_path)
    print(f"Successfully converted to {pytorch_path}")
    return True

if __name__ == "__main__":
    model_dir = "fine_tuned_embedding_model"
    convert_safetensors_to_pytorch(model_dir)
