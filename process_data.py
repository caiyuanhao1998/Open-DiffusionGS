# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).


import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
import json
import multiprocessing as mp
import logging
import time
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def process_single_file(args):
    """
    Helper function to process a single file (needed for multiprocessing)
    
    Args:
        args (tuple): Tuple containing (file_path, output_dir)
    """
    file_path, output_dir = args
    return process_torch_file(file_path, output_dir)

def process_torch_file(file_path, output_dir):
    """
    Process a .torch file and save images and poses
    
    Args:
        file_path (str): Path to the .torch file
        output_dir (str): Base directory to save outputs
    """
    try:
        # Create output directories
        images_dir = os.path.join(output_dir, 'images')
        meta_dir = os.path.join(output_dir, 'metadata')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(meta_dir, exist_ok=True)
        
        # Load the torch file
        data = torch.load(file_path)
        
        # Process each scene in parallel using ThreadPool
        for cur_scene in data:
            # Get the key from the first element to use as subdirectory name
            scene_name = cur_scene['key']
            if isinstance(scene_name, torch.Tensor):
                scene_name = scene_name.item()
            
            # Create subdirectories for this specific sequence
            seq_images_dir = os.path.join(images_dir, str(scene_name))
            os.makedirs(seq_images_dir, exist_ok=True)
            
            cur_info_dict = {
                'scene_name': scene_name,
                'frames': []
            }
            
            cur_pose_info = cur_scene['cameras']
            
            # Pre-allocate lists for better memory efficiency
            frames = []
            num_images = len(cur_scene['images'])
            
            # Process each element in the list
            for img_idx, img_data in enumerate(cur_scene['images']):
                try:
                    # Convert tensor to numpy if needed
                    if isinstance(img_data, torch.Tensor):
                        img_data = img_data.numpy()
                    
                    # Convert PIL image to numpy array more efficiently
                    img_array = np.frombuffer(img_data.tobytes(), dtype=np.uint8)
                    img_array = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if img_array is None:
                        raise ValueError("Failed to decode image data")
                    
                    h, w = img_array.shape[:2]
                    
                    # Save as PNG using cv2 (faster than plt.imsave)
                    img_path = os.path.join(seq_images_dir, f'{img_idx:05d}.png')
                    if not cv2.imwrite(img_path, img_array):
                        raise ValueError(f"Failed to write image to {img_path}")
                    
                    # Convert pose info tensors to regular Python types if needed
                    pose_data = cur_pose_info[img_idx]
                    if isinstance(pose_data, torch.Tensor):
                        pose_data = pose_data.tolist()
                    
                    # Calculate camera parameters
                    fx, fy, cx, cy = map(float, [
                        pose_data[0] * w,
                        pose_data[1] * h,
                        pose_data[2] * w,
                        pose_data[3] * h
                    ])
                    
                    # Calculate world to camera transform
                    w2c = np.array(pose_data[6:], dtype=np.float32).reshape(3, 4)
                    w2c = np.vstack([w2c, [0, 0, 0, 1]])
                    
                    frame_info = {
                        'image_path': os.path.join(seq_images_dir, f'{img_idx:05d}.png'),
                        'fxfycxcy': [fx, fy, cx, cy],
                        'w2c': w2c.tolist()
                    }
                    frames.append(frame_info)
                    
                except Exception as e:
                    logging.error(f"Error processing image {img_idx} in {file_path}: {str(e)}")
                    continue
            
            cur_info_dict['frames'] = frames
            
            # Save metadata
            meta_path = os.path.join(meta_dir, f'{scene_name}.json')
            with open(meta_path, 'w') as f:
                json.dump(cur_info_dict, f, indent=4)
                
        return True, file_path
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        return False, file_path

def process_directory(input_dir, output_dir, num_processes=None, chunk_size=1):
    """
    Process all .torch files in a directory using multiprocessing
    
    Args:
        input_dir (str): Directory containing .torch files
        output_dir (str): Base directory to save outputs
        num_processes (int, optional): Number of processes to use. Defaults to CPU count - 1
        chunk_size (int, optional): Number of files to process per worker at once. Defaults to 1
    """
    # Get all .torch files in the directory
    torch_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.torch')]
    
    torch_files.sort()

    total_files = len(torch_files)
    logging.info(f"Found {total_files} files to process in {input_dir}")
    
    # Set up multiprocessing
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 1)
    
    # Prepare arguments for multiprocessing
    args = [(f, output_dir) for f in torch_files]
    
    # Process files in parallel with progress bar
    start_time = time.time()
    with mp.Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_file, args, chunksize=chunk_size),
            total=total_files,
            desc=f"Processing files with {num_processes} processes"
        ))
    
    # Log results
    successful = sum(1 for success, _ in results if success)
    failed = [(success, path) for success, path in results if not success]
    
    elapsed_time = time.time() - start_time
    logging.info(f"Processing completed in {elapsed_time:.2f} seconds")
    logging.info(f"Successfully processed {successful}/{total_files} files")
    
    if failed:
        logging.warning(f"Failed to process {len(failed)} files:")
        for _, path in failed:
            logging.warning(f"  - {path}")

def generate_full_list(base_path, output_dir):
    # find all .json files in the base_path and generate a full list saving their absolute paths and store it in a txt file in the output_dir
    json_files = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('.json')]
    json_files = [os.path.abspath(f) for f in json_files]
    json_files.sort()
    with open(os.path.join(output_dir, 'full_list.txt'), 'w') as f:
        for file in json_files:
            f.write(file + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--chunk_size", type=int, default=10)
    parser.add_argument("--num_processes", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default='/share/phoenix/nfs06/S9/hj453/DATA/re10k/')
    parser.add_argument("--base_path", type=str, default='/share/phoenix/nfs06/S9/hj453/DATA/re10k_raw/')
    
    args = parser.parse_args()
    # Example usage
    cur_mode = args.mode
    input_dir = os.path.join(args.base_path, cur_mode)
    # output_dir = os.path.join('./', 'preprocessed_data', cur_mode)
    output_dir = os.path.join(args.output_dir, cur_mode)
    # Process test data only
    logging.info("Starting test data processing...")
    process_directory(input_dir, output_dir, chunk_size=args.chunk_size, num_processes=args.num_processes)  
    logging.info("Processing completed!") 
    search_list_dir = os.path.join(args.output_dir, cur_mode, 'metadata')
    save_dir = os.path.join(args.output_dir, cur_mode)
    generate_full_list(search_list_dir, save_dir)
    logging.info("Full list generated!")