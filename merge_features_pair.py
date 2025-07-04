import os
import re
import numpy as np
import glob
import h5py
from tqdm import tqdm
import concurrent.futures
from multiprocessing import cpu_count

def get_identifier(file_path, feature_type):
    """
    파일 경로와 타입에 따라 정규표현식을 사용하여 공통 식별자를 생성합니다.
    """
    basename = os.path.splitext(os.path.basename(file_path))[0]
    match = re.search(r'^(\d+).*?(seg\d+)$', basename)
    if match:
        video_name = match.group(1)
        segment_id = match.group(2)
        return f"{video_name}_{segment_id}"
    return None

def find_feature_triplets(video_dir, text_dir, image_dir):
    """
    세 디렉토리를 비교하고, 누락된 파일 식별자를 로그 파일로 저장합니다.
    """
    print("Finding matching feature triplets (video, text, image)...")
    
    video_files = glob.glob(os.path.join(video_dir, "*.npz"))
    text_files = glob.glob(os.path.join(text_dir, "*.npy"))
    image_files = glob.glob(os.path.join(image_dir, "*.npz"))

    print(f"Found {len(video_files)} video, {len(text_files)} text, and {len(image_files)} image features.")

    video_map = {identifier: p for p in video_files if (identifier := get_identifier(p, 'video'))}
    text_map = {identifier: p for p in text_files if (identifier := get_identifier(p, 'text'))}
    image_map = {identifier: p for p in image_files if (identifier := get_identifier(p, 'image'))}
    
    print(f"Successfully extracted identifiers for {len(video_map)} video, {len(text_map)} text, and {len(image_map)} image files.")

    video_ids = set(video_map.keys())
    text_ids = set(text_map.keys())
    image_ids = set(image_map.keys())
    
    # --- 🧐 누락된 식별자 로깅 기능 ---
    missing_text_ids = (video_ids.intersection(image_ids)) - text_ids
    if missing_text_ids:
        log_filename = 'missing_text_features.log'
        print(f"🚨 Found {len(missing_text_ids)} items with missing TEXT features. Saving identifiers to '{log_filename}'")
        with open(log_filename, 'w') as f:
            for item_id in sorted(list(missing_text_ids)):
                f.write(f"{item_id}\n")
    # --- 로깅 기능 끝 ---

    common_ids = sorted(list(video_ids.intersection(text_ids).intersection(image_ids)))
    
    triplet_files = []
    for identifier in common_ids:
        triplet_files.append({
            'video': video_map[identifier],
            'text': text_map[identifier],
            'image': image_map[identifier]
        })
            
    if not triplet_files:
        print("⚠️ Warning: No matching feature triplets were found.")
    else:
        print(f"✅ Successfully paired {len(triplet_files)} triplets.")
    
    return triplet_files

# (이하 load_feature_triplet, merge_features_to_hdf5, main 함수는 이전과 동일)

def load_feature_triplet(triplet_dict):
    with np.load(triplet_dict['video']) as data:
        video_feature = data['feature']
        label = data['label']
    text_feature = np.load(triplet_dict['text'])
    with np.load(triplet_dict['image']) as data:
        image_feature = data['feature']
    return video_feature, label, text_feature, image_feature

def merge_features_to_hdf5(video_dir, text_dir, image_dir, output_path, batch_size=32, n_workers=None):
    triplet_files = find_feature_triplets(video_dir, text_dir, image_dir)
    if not triplet_files:
        return
    num_triplets = len(triplet_files)
    if n_workers is None:
        n_workers = min(cpu_count(), 8)
    print(f"Using {n_workers} workers for parallel processing.")
    first_video_feat, first_label, first_text_feat, first_image_feat = load_feature_triplet(triplet_files[0])
    video_shape, video_dtype = first_video_feat.shape, first_video_feat.dtype
    label_dtype = first_label.dtype
    text_shape, text_dtype = first_text_feat.shape, first_text_feat.dtype
    image_shape, image_dtype = first_image_feat.shape, first_image_feat.dtype
    print(f"Detected Video Feature Shape: {video_shape}, Dtype: {video_dtype}")
    print(f"Detected Text Feature Shape: {text_shape}, Dtype: {text_dtype}")
    print(f"Detected Image Feature Shape: {image_shape}, Dtype: {image_dtype}")
    print(f"Detected Label Dtype: {label_dtype}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Creating HDF5 file at: {output_path}")
    with h5py.File(output_path, 'w') as hf:
        video_dset = hf.create_dataset('video_features', shape=(num_triplets,) + video_shape, dtype=video_dtype, compression='lzf')
        text_dset = hf.create_dataset('text_features', shape=(num_triplets,) + text_shape, dtype=text_dtype, compression='lzf')
        image_dset = hf.create_dataset('visual_features', shape=(num_triplets,) + image_shape, dtype=image_dtype, compression='lzf')
        label_dset = hf.create_dataset('labels', shape=(num_triplets,) + first_label.shape, dtype=label_dtype, compression='lzf')
        with tqdm(total=num_triplets, desc="Processing feature triplets") as pbar:
            for i in range(0, num_triplets, batch_size):
                batch_triplets = triplet_files[i:i+batch_size]
                with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                    results = list(executor.map(load_feature_triplet, batch_triplets))
                for j, (video_feat, label, text_feat, image_feat) in enumerate(results):
                    idx = i + j
                    video_dset[idx] = video_feat
                    text_dset[idx] = text_feat
                    image_dset[idx] = image_feat
                    label_dset[idx] = label
                pbar.update(len(batch_triplets))
        print("\nHDF5 preprocessing complete.")
        print(f"Final Video dataset shape: {video_dset.shape}")
        print(f"Final Text dataset shape: {text_dset.shape}")
        print(f"Final Visual dataset shape: {image_dset.shape}")
        print(f"Final Labels dataset shape: {label_dset.shape}")

if __name__ == '__main__':
    BATCH_SIZE = 32
    N_WORKERS = 8
    VIDEO_INPUT_DIR = '../../dataset/dvd/DVD_Competition/Training/features_pooled_16'
    TEXT_INPUT_DIR = '../../dataset/dvd/save/train/text_features_resequenced'
    IMAGE_INPUT_DIR = '../../dataset/dvd/DVD_Competition/Training/features_clip_16x768'
    OUTPUT_PATH = '../../dataset/dvd/DVD_Competition/Training/preprocessed_hdf5/training_data_triplet.hdf5'
    merge_features_to_hdf5(
        video_dir=VIDEO_INPUT_DIR,
        text_dir=TEXT_INPUT_DIR,
        image_dir=IMAGE_INPUT_DIR,
        output_path=OUTPUT_PATH,
        batch_size=BATCH_SIZE,
        n_workers=N_WORKERS
    )