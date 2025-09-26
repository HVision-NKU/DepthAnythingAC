import onnxruntime as ort
import cv2
import numpy as np
import torch # <--- IMPORT TORCH
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm

def normalize_depth(depth):
    # This function normalizes a 2D depth map for visualization
    eps = 1e-6
    depth_min = np.min(depth)
    depth_max = np.max(depth)
    normalized_depth = (depth - depth_min) / (depth_max - depth_min + eps)
    return normalized_depth

def preprocess_image(image_path, target_size=518):
    raw_image = cv2.imread(image_path)
    if raw_image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    # The model expects a square input, so we resize directly
    image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    
    image_float = image_resized.astype(np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_normalized = (image_float - mean) / std
    
    # HWC to NCHW
    image_transposed = image_normalized.transpose(2, 0, 1)
    image_tensor = np.expand_dims(image_transposed, axis=0).astype(np.float32)
    
    return image_tensor, (raw_image.shape[0], raw_image.shape[1])

def postprocess_depth(depth_tensor_numpy, original_size):
    """
    Postprocesses the raw 4D model output to a 2D depth map.
    - Converts numpy to torch tensor.
    - Resizes to original image dimensions.
    - Converts back to numpy.
    """
    # 1. Convert the numpy array to a torch tensor
    # The input is already in the correct 4D shape (1, 1, H, W)
    depth_tensor_torch = torch.from_numpy(depth_tensor_numpy)
    
    h, w = original_size
    
    # 2. Interpolate using torch.nn.functional
    # align_corners=False is the modern default and generally recommended
    depth_resized = F.interpolate(depth_tensor_torch, size=(h, w), mode='bilinear', align_corners=False)
    
    # 3. Squeeze, convert back to a CPU numpy array
    depth_output = depth_resized.squeeze().cpu().numpy()
    
    return depth_output

def save_depth_map(depth, output_path, colormap='inferno'):
    # Assumes depth is already normalized to 0-1 range
    depth_raw_path = output_path.replace('.png', '_raw.npy')
    np.save(depth_raw_path, depth)

    if colormap == 'inferno':
        depth_colored = (plt.get_cmap(colormap)(depth)[:, :, :3] * 255).astype(np.uint8)
    elif colormap == 'spectral':
        spectral_cmap = cm.get_cmap('Spectral_r')
        depth_colored = (spectral_cmap(depth) * 255).astype(np.uint8)[:, :, :3]
    else: # Grayscale
        depth_colored = (depth * 255).astype(np.uint8)
        
    # Convert to BGR for OpenCV
    depth_colored_bgr = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, depth_colored_bgr)
    print(f"Depth map saved: {output_path}")
    print(f"Raw depth data saved: {depth_raw_path}")

def infer_onnx(onnx_path, image_path, output_path, colormap='inferno'):
    print(f"Processing: {image_path}")
    # Use CPUExecutionProvider as requested by CUDA_VISIBLE_DEVICES=-1
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    
    image_tensor, original_size = preprocess_image(image_path)
    print(f"Input tensor shape: {image_tensor.shape}")
    print(f"Original image size: {original_size}")
    
    # The model outputs disparity, which is inversely proportional to depth
    disparity = sess.run([output_name], {input_name: image_tensor})[0]
    print(f"Model output shape (disparity): {disparity.shape}")
    
    # Postprocess (resize to original size)
    # disparity is a 4D numpy array (1, 1, H, W)
    depth_resized = postprocess_depth(disparity, original_size)
    print(f"Resized depth map shape: {depth_resized.shape}")

    # Normalize the final depth map for visualization
    depth_normalized = normalize_depth(depth_resized)
    
    save_depth_map(depth_normalized, output_path, colormap)

if __name__ == '__main__':
    onnx_path = '/home/e300/code/DepthAnythingAC/depth_anything_AC_vits.onnx'
    image_path = '/home/e300/Downloads/WhatsApp Image 2025-09-26 at 1.11.48 PM.jpeg'
    output_path = 'syed_depth.png' # Changed output name for clarity
    
    infer_onnx(onnx_path, image_path, output_path)