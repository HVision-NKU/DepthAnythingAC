# convert_to_onnx.py
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depth_anything.dpt import DepthAnything_AC

def load_model(model_path, encoder='vits'):
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024], 'version': 'v2'},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768], 'version': 'v2'},
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384], 'version': 'v2'}
    }
    model = DepthAnything_AC(model_configs[encoder])
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model

def export_to_onnx(model, onnx_path, input_size=(518, 518), encoder='vits'):
    # Dummy input with dynamic batch size
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
    
    # Since model returns a dict, we need to wrap it to extract 'out'
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, x):
            output = self.model(x)
            return output['out']  # Extract the 'out' tensor from the dictionary
    
    wrapped_model = ModelWrapper(model)
    wrapped_model.eval() # Good practice to set wrapper to eval mode too
    
    # Export to ONNX with dynamic axes
    torch.onnx.export(
        wrapped_model, # <--- CORRECTED: Use the wrapper
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,  # Opset 12 or higher is generally better if your TRT supports it
        do_constant_folding=True,
        input_names=['input'],
        output_names=['depth'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'depth': {0: 'batch_size', 2: 'height', 3: 'width'}
        },
        verbose=False # Set to True for debugging if needed
    )
    print(f"Model exported to ONNX: {onnx_path}")

if __name__ == '__main__':
    model_path = 'checkpoints/depth_anything_AC_vits.pth'  # Replace with your model path
    onnx_path = 'depth_anything_AC_vits.onnx'  # Output ONNX file
    encoder = 'vits'  # Adjust if needed
    model = load_model(model_path, encoder)
    export_to_onnx(model, onnx_path, encoder=encoder)