import argparse
import os
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depth_anything.dpt import DepthAnything_AC


def normalize_depth(disparity_tensor):
    """
    Convert disparity to depth using normalization method.
    
    Args:
        disparity_tensor (torch.Tensor): Predicted disparity values
        
    Returns:
        torch.Tensor: depth values
    """
    eps = 1e-6
    
    disparity_min = disparity_tensor.min()
    disparity_max = disparity_tensor.max()
    normalized_disparity = (disparity_tensor - disparity_min) / (disparity_max - disparity_min + eps)
    
    
    return normalized_disparity


def preprocess_image(image_path, target_size=518):
    """Preprocess input image"""
    
    raw_image = cv2.imread(image_path)
    if raw_image is None:
        raise ValueError(f"Cannot read image: {image_path}") 
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = image.shape[:2]
    scale = target_size / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    new_h = ((new_h + 13) // 14) * 14
    new_w = ((new_w + 13) // 14) * 14
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std

    image = torch.from_numpy(image.transpose(2, 0, 1)).float()
    image = image.unsqueeze(0)  
    
    return image, (h, w)


def postprocess_depth(depth_tensor, original_size):
    """Postprocess depth map"""
    depth_tensor = depth_tensor.unsqueeze(1) 
    h, w = original_size
    
    try:
        depth = F.interpolate(depth_tensor, size=(h, w), mode='bilinear', align_corners=True)
        depth = depth.squeeze().cpu().numpy()
        # print(f"Final depth map shape: {depth.shape}")
        return depth
    except Exception as e:
        print(f"Interpolation failed: {str(e)}")
        return None


def save_depth_map(depth, output_path, colormap='inferno'):
    """Save depth map"""
    depth_raw_path = output_path.replace('.png', '_raw.npy')
    np.save(depth_raw_path, depth)

    if colormap == 'inferno':
        depth_colored = cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    elif colormap == 'spectral':
        spectral_cmap = cm.get_cmap('Spectral_r')  
        depth_colored = (spectral_cmap(depth) * 255).astype(np.uint8)
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_RGBA2BGR)
    else:
        # Grayscale
        depth_colored = (depth * 255).astype(np.uint8)
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(output_path, depth_colored)
    print(f"Depth map saved: {output_path}")
    print(f"Raw depth data saved: {depth_raw_path}")


def infer_single_image(model, image_path, output_path, colormap='inferno', depth_cap=80):
    """Perform depth estimation inference on a single image"""
    print(f"Processing: {image_path}")

    image_tensor, original_size = preprocess_image(image_path)
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
    with torch.no_grad():
        prediction = model(image_tensor)
        disparity_tensor = prediction['out']
        
        print(f"Model output shape (disparity): {disparity_tensor.shape}")
        print(f"Original image size: {original_size}")
        
        depth_tensor = normalize_depth(disparity_tensor)
        print(f"Converted to depth, shape: {depth_tensor.shape}")
    
    depth = postprocess_depth(depth_tensor, original_size)
            
    if depth is None:
        print("Reprocessing depth tensor...")
        if depth_tensor.dim() == 1:
            h, w = original_size
            expected_size = h * w
            if depth_tensor.shape[0] == expected_size:
                depth_tensor = depth_tensor.view(1, 1, h, w)
            else:
                import math
                side_length = int(math.sqrt(depth_tensor.shape[0]))
                if side_length * side_length == depth_tensor.shape[0]:
                    depth_tensor = depth_tensor.view(1, 1, side_length, side_length)
                else:
                    raise ValueError(f"Cannot reshape 1D tensor with size {depth_tensor.shape[0]}")
        depth = postprocess_depth(depth_tensor, original_size)
        if depth is None:
            raise ValueError("Cannot handle depth tensor shape problem")
    save_depth_map(depth, output_path, colormap)
    
    return depth


def infer_batch_images(model, input_dir, output_dir, colormap='inferno', depth_cap=80):
    """Batch process images"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    if not image_files:
        print(f"No image files found in directory {input_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)
    print(f"Found {len(image_files)} images, starting batch processing...")
    for i, image_path in enumerate(image_files):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_depth.png")
        try:
            infer_single_image(model, image_path, output_path, colormap, depth_cap)
            print(f"Progress: {i+1}/{len(image_files)}")
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
    
    print("Batch processing completed!")


def infer_video(model, video_path, output_path, colormap='inferno', depth_cap=80, fps=None):
    """
    Perform depth estimation inference on a video file
    
    Args:
        model: Trained depth estimation model
        video_path (str): Path to input video file
        output_path (str): Path to output video file
        colormap (str): Colormap for depth visualization ('inferno', 'spectral', 'gray')
        depth_cap (float): Maximum depth value for capping
        fps (float): Output video FPS (if None, uses input video FPS)
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Processing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file: {video_path}")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if fps is None:
        fps = input_fps
    
    print(f"Video properties: {total_frames} frames, {input_fps} FPS, {width}x{height}")
    print(f"Output FPS: {fps}")
    
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Cannot create output video: {output_path}")
        cap.release()
        return False
    
    frame_count = 0

    pbar = tqdm(total=total_frames, desc="Processing video", unit="frame")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            temp_frame_path = "temp_frame.png"
            cv2.imwrite(temp_frame_path, frame)
            
            try:
                image_tensor, original_size = preprocess_image(temp_frame_path)
                if torch.cuda.is_available():
                    image_tensor = image_tensor.cuda()
                
                with torch.no_grad():
                    prediction = model(image_tensor)
                    disparity_tensor = prediction['out']
                    depth_tensor = normalize_depth(disparity_tensor)
                
                depth = postprocess_depth(depth_tensor, original_size)
                
                if depth is None:
                    if depth_tensor.dim() == 1:
                        h, w = original_size
                        expected_size = h * w
                        if depth_tensor.shape[0] == expected_size:
                            depth_tensor = depth_tensor.view(1, 1, h, w)
                        else:
                            import math
                            side_length = int(math.sqrt(depth_tensor.shape[0]))
                            if side_length * side_length == depth_tensor.shape[0]:
                                depth_tensor = depth_tensor.view(1, 1, side_length, side_length)
                    depth = postprocess_depth(depth_tensor, original_size)
                
                if depth is None:
                    print(f"Warning: Failed to process frame {frame_count}, using black frame")
                    depth_frame = np.zeros((height, width, 3), dtype=np.uint8)
                else:
                    if colormap == 'inferno':
                        depth_frame = cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
                    elif colormap == 'spectral':
                        spectral_cmap = cm.get_cmap('Spectral_r')
                        depth_frame = (spectral_cmap(depth) * 255).astype(np.uint8)
                        depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_RGBA2BGR)
                    else:
                        depth_frame = (depth * 255).astype(np.uint8)
                        depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_GRAY2BGR)
                
                out.write(depth_frame)
                
            except Exception as e:
                print(f"Error processing frame {frame_count}: {str(e)}")
                black_frame = np.zeros((height, width, 3), dtype=np.uint8)
                out.write(black_frame)
            
            pbar.update(1)
            
            if os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)
                
    except KeyboardInterrupt:
        print("\nVideo processing interrupted by user")
    except Exception as e:
        print(f"Unexpected error during video processing: {str(e)}")
    finally:
        pbar.close()
        cap.release()
        out.release()
        if os.path.exists("temp_frame.png"):
            os.remove("temp_frame.png")
    
    print(f"Video processing completed! Output saved to: {output_path}")
    return True


def is_video_file(filepath):
    """
    Check if the given file is a video file based on its extension
    
    Args:
        filepath (str): Path to the file
        
    Returns:
        bool: True if file is a video, False otherwise
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
    _, ext = os.path.splitext(filepath.lower())
    return ext in video_extensions


def find_files_recursive(directory, extensions, recursive=True):
    """
    Recursively find files with specified extensions
    
    Args:
        directory (str): Root directory to search
        extensions (list): List of file extensions to search for (e.g., ['.mp4', '.avi'])
        recursive (bool): Whether to search recursively in subdirectories
        
    Returns:
        list: List of found file paths
    """
    files = []
    
    if recursive:
        for root, dirs, filenames in os.walk(directory):
            for filename in filenames:
                _, ext = os.path.splitext(filename.lower())
                if ext in extensions:
                    files.append(os.path.join(root, filename))
    else:
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                _, ext = os.path.splitext(filename.lower())
                if ext in extensions:
                    files.append(filepath)
    
    return sorted(files)


def infer_batch_videos(model, input_dir, output_dir, colormap='inferno', depth_cap=80, fps=None, recursive=True):
    """
    Batch process videos in a directory
    
    Args:
        model: Trained depth estimation model
        input_dir (str): Input directory containing videos
        output_dir (str): Output directory for processed videos
        colormap (str): Colormap for depth visualization
        depth_cap (float): Maximum depth value for capping
        fps (float): Output video FPS (if None, uses input video FPS)
        recursive (bool): Whether to search recursively in subdirectories
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
    video_files = find_files_recursive(input_dir, video_extensions, recursive)
    
    if not video_files:
        print(f"No video files found in directory {input_dir}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Found {len(video_files)} videos, starting batch processing...")
    
    for i, video_path in enumerate(video_files):
        rel_path = os.path.relpath(video_path, input_dir)
        base_name = os.path.splitext(rel_path)[0]
        output_path = os.path.join(output_dir, f"{base_name}_depth.mp4")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"\n[{i+1}/{len(video_files)}] Processing: {video_path}")
        try:
            success = infer_video(model, video_path, output_path, colormap, depth_cap, fps)
            if success:
                print(f"✓ Completed: {output_path}")
            else:
                print(f"✗ Failed: {video_path}")
        except Exception as e:
            print(f"✗ Error processing {video_path}: {str(e)}")
    
    print("\nBatch video processing completed!")


def infer_batch_mixed(model, input_dir, output_dir, colormap='inferno', depth_cap=80, fps=None, recursive=True):
    """
    Batch process both images and videos in a directory
    
    Args:
        model: Trained depth estimation model
        input_dir (str): Input directory containing images and videos
        output_dir (str): Output directory for processed files
        colormap (str): Colormap for depth visualization
        depth_cap (float): Maximum depth value for capping
        fps (float): Output video FPS (if None, uses input video FPS)
        recursive (bool): Whether to search recursively in subdirectories
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = find_files_recursive(input_dir, image_extensions, recursive)
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
    video_files = find_files_recursive(input_dir, video_extensions, recursive)
    
    if not image_files and not video_files:
        print(f"No image or video files found in directory {input_dir}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    total_files = len(image_files) + len(video_files)
    print(f"Found {len(image_files)} images and {len(video_files)} videos ({total_files} total), starting batch processing...")
    
    file_count = 0
    
    # Process images
    if image_files:
        print(f"\nProcessing {len(image_files)} images...")
        for image_path in tqdm(image_files, desc="Processing images"):
            file_count += 1
            
            # Create relative path structure in output directory
            rel_path = os.path.relpath(image_path, input_dir)
            base_name = os.path.splitext(rel_path)[0]
            output_path = os.path.join(output_dir, f"{base_name}_depth.png")
            
            # Create output subdirectory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            try:
                infer_single_image(model, image_path, output_path, colormap, depth_cap)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
    
    # Process videos
    if video_files:
        print(f"\nProcessing {len(video_files)} videos...")
        for i, video_path in enumerate(video_files):
            file_count += 1
            
            # Create relative path structure in output directory
            rel_path = os.path.relpath(video_path, input_dir)
            base_name = os.path.splitext(rel_path)[0]
            output_path = os.path.join(output_dir, f"{base_name}_depth.mp4")
            
            # Create output subdirectory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            print(f"\n[{i+1}/{len(video_files)}] Processing: {video_path}")
            try:
                success = infer_video(model, video_path, output_path, colormap, depth_cap, fps)
                if success:
                    print(f"✓ Completed: {output_path}")
                else:
                    print(f"✗ Failed: {video_path}")
            except Exception as e:
                print(f"✗ Error processing {video_path}: {str(e)}")
    
    print(f"\nBatch processing completed! Processed {total_files} files total.")


def main():
    parser = argparse.ArgumentParser(description='Depth Anything AC Depth Estimation Inference')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input image/video path or directory')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output path (file or directory)')
    parser.add_argument('--model', '-m', type=str, 
                        default='checkpoints/depth_anything_AC_vits.pth',
                        help='Model weights path')
    parser.add_argument('--encoder', type=str, default='vits',
                        choices=['vits', 'vitb', 'vitl'],
                        help='Encoder type')
    parser.add_argument('--colormap', type=str, default='spectral',
                        choices=['inferno', 'spectral', 'gray'],
                        help='Depth map colormap')
    parser.add_argument('--depth_cap', type=float, default=80,
                        help='Maximum depth value for capping')
    parser.add_argument('--fps', type=float, default=None,
                        help='Output video FPS (for video processing, defaults to input FPS)')
    parser.add_argument('--recursive', '-r', action='store_true',
                        help='Search recursively in subdirectories when processing a directory')
    parser.add_argument('--mode', type=str, default='mixed',
                        choices=['images', 'videos', 'mixed'],
                        help='Processing mode for directories: images only, videos only, or both')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file does not exist: {args.model}")
        return
    
    try:
        model = DepthAnything_AC.from_pretrained(args.model, args.encoder)
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        return
    
    if os.path.isfile(args.input):
        if is_video_file(args.input):
            if not is_video_file(args.output):
                args.output = os.path.splitext(args.output)[0] + '.mp4'
            
            os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
            
            try:
                success = infer_video(model, args.input, args.output, args.colormap, args.depth_cap, args.fps)
                if success:
                    print("Video inference completed!")
                else:
                    print("Video inference failed!")
            except Exception as e:
                print(f"Video inference failed: {str(e)}")
        else:
            if not os.path.splitext(args.output)[1]:
                args.output += '.png'
            os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
            try:
                infer_single_image(model, args.input, args.output, args.colormap, args.depth_cap)
                print("Image inference completed!")
            except Exception as e:
                print(f"Image inference failed: {str(e)}")
    
    elif os.path.isdir(args.input):
        if args.mode == 'images':
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            image_files = find_files_recursive(args.input, image_extensions, args.recursive)
            
            if not image_files:
                print(f"No image files found in directory {args.input}")
                return
                
            os.makedirs(args.output, exist_ok=True)
            print(f"Found {len(image_files)} images, starting batch processing...")
            
            for image_path in tqdm(image_files, desc="Processing images"):
                rel_path = os.path.relpath(image_path, args.input)
                base_name = os.path.splitext(rel_path)[0]
                output_path = os.path.join(args.output, f"{base_name}_depth.png")

                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                try:
                    infer_single_image(model, image_path, output_path, args.colormap, args.depth_cap)
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
            
            print("Image batch processing completed!")
            
        elif args.mode == 'videos':
            infer_batch_videos(model, args.input, args.output, args.colormap, args.depth_cap, args.fps, args.recursive)
        else:
            infer_batch_mixed(model, args.input, args.output, args.colormap, args.depth_cap, args.fps, args.recursive)
    else:
        print(f"Error: Input path does not exist: {args.input}")



if __name__ == '__main__':
    main() 
