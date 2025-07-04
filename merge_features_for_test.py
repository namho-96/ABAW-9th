import os
import re
import numpy as np
import glob
import h5py
from tqdm import tqdm
import concurrent.futures
from multiprocessing import cpu_count

def get_identifier(file_path):
    """
    파일 경로에서 비디오 이름과 세그먼트 ID를 추출하여 식별자를 생성합니다.
    예: '102-30-640x360_seg000.npz' -> ('102-30-640x360', 'seg000', '102-30-640x360_seg000')
    """
    basename = os.path.splitext(os.path.basename(file_path))[0]
    # 정규표현식을 사용하여 video_name과 segment_id를 분리합니다.
    match = re.search(r'^(.*?)_seg(\d+)$', basename)
    if match:
        video_name = match.group(1)
        segment_id = f"seg{match.group(2)}"
        identifier = f"{video_name}_{segment_id}"
        return video_name, segment_id, identifier
    return None, None, None

def find_feature_triplets(video_dir, text_dir, image_dir):
    """
    세 디렉토리에서 공통된 파일(triplets)을 찾고,
    각 triplet에 대한 피처 경로, 비디오 이름, 세그먼트 ID를 반환합니다.
    """
    print("Finding matching feature triplets (video, text, image)...")

    video_files = glob.glob(os.path.join(video_dir, "*.np[yz]"))
    text_files = glob.glob(os.path.join(text_dir, "*.np[yz]"))
    image_files = glob.glob(os.path.join(image_dir, "*.np[yz]"))

    print(f"Found {len(video_files)} video, {len(text_files)} text, and {len(image_files)} image features.")

    # 각 파일의 식별자, 비디오 이름, 세그먼트 ID를 매핑합니다. (for 루프로 수정)
    video_map = {}
    for p in video_files:
        vn, sid, identifier = get_identifier(p)
        if identifier:
            video_map[identifier] = (p, vn, sid)

    text_map = {}
    for p in text_files:
        _, _, identifier = get_identifier(p)
        if identifier:
            text_map[identifier] = p

    image_map = {}
    for p in image_files:
        _, _, identifier = get_identifier(p)
        if identifier:
            image_map[identifier] = p

    print(f"Successfully extracted identifiers for {len(video_map)} video, {len(text_map)} text, and {len(image_map)} image files.")

    video_ids = set(video_map.keys())
    text_ids = set(text_map.keys())
    image_ids = set(image_map.keys())

    # 공통 식별자를 찾습니다.
    common_ids = sorted(list(video_ids.intersection(text_ids, image_ids)))

    triplet_data = []
    for identifier in common_ids:
        video_path, video_name, segment_id = video_map[identifier]
        triplet_data.append({
            'video_path': video_path,
            'text_path': text_map[identifier],
            'image_path': image_map[identifier],
            'video_name': video_name,
            'segment_id': segment_id
        })

    if not triplet_data:
        print("⚠️ Warning: No matching feature triplets were found.")
    else:
        print(f"✅ Successfully paired {len(triplet_data)} triplets.")

    return triplet_data

def load_feature_triplet(triplet_info):
    """
    주어진 triplet 정보에서 각 피처를 로드하고,
    비디오 이름과 세그먼트 ID를 함께 반환합니다. (레이블 로드 제거)
    """
    # .npz와 .npy 파일 모두 처리
    def load_feature(path):
        if path.endswith('.npz'):
            with np.load(path) as data:
                # 'feature' 또는 'features' 키를 가정
                return data['feature'] if 'feature' in data else data['features']
        else:
            return np.load(path)

    video_feature = load_feature(triplet_info['video_path'])
    text_feature = load_feature(triplet_info['text_path'])
    image_feature = load_feature(triplet_info['image_path'])
    
    video_name = triplet_info['video_name']
    segment_id = triplet_info['segment_id']
    
    return video_feature, text_feature, image_feature, video_name, segment_id

def merge_features_to_hdf5(video_dir, text_dir, image_dir, output_path, batch_size=32, n_workers=None):
    """
    찾아낸 feature triplet들을 병합하여 하나의 HDF5 파일로 저장합니다.
    레이블 대신 video_name과 segment_id를 저장합니다.
    """
    triplet_data = find_feature_triplets(video_dir, text_dir, image_dir)
    if not triplet_data:
        return
        
    num_triplets = len(triplet_data)
    if n_workers is None:
        n_workers = min(cpu_count(), 8)
        
    print(f"Using {n_workers} workers for parallel processing.")
    
    # 첫 번째 데이터를 로드하여 shape과 dtype을 확인합니다.
    first_video_feat, first_text_feat, first_image_feat, _, _ = load_feature_triplet(triplet_data[0])
    
    video_shape, video_dtype = first_video_feat.shape, first_video_feat.dtype
    text_shape, text_dtype = first_text_feat.shape, first_text_feat.dtype
    image_shape, image_dtype = first_image_feat.shape, first_image_feat.dtype
    # 문자열 저장을 위한 h5py 특별 dtype
    string_dtype = h5py.special_dtype(vlen=str)

    print(f"Detected Video Feature Shape: {video_shape}, Dtype: {video_dtype}")
    print(f"Detected Text Feature Shape: {text_shape}, Dtype: {text_dtype}")
    print(f"Detected Image Feature Shape: {image_shape}, Dtype: {image_dtype}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Creating HDF5 file at: {output_path}")
    
    with h5py.File(output_path, 'w') as hf:
        # 데이터셋 생성
        video_dset = hf.create_dataset('video_features', shape=(num_triplets,) + video_shape, dtype=video_dtype, compression='lzf')
        text_dset = hf.create_dataset('text_features', shape=(num_triplets,) + text_shape, dtype=text_dtype, compression='lzf')
        image_dset = hf.create_dataset('visual_features', shape=(num_triplets,) + image_shape, dtype=image_dtype, compression='lzf')
        video_name_dset = hf.create_dataset('video_names', shape=(num_triplets,), dtype=string_dtype, compression='lzf')
        segment_id_dset = hf.create_dataset('segment_ids', shape=(num_triplets,), dtype=string_dtype, compression='lzf')
        
        with tqdm(total=num_triplets, desc="Processing feature triplets") as pbar:
            # ProcessPoolExecutor를 사용하여 병렬로 데이터 로드
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                # 데이터를 순서대로 처리하기 위해 map 사용
                results = executor.map(load_feature_triplet, triplet_data)
                
                for idx, (video_feat, text_feat, image_feat, video_name, segment_id) in enumerate(results):
                    video_dset[idx] = video_feat
                    text_dset[idx] = text_feat
                    image_dset[idx] = image_feat
                    video_name_dset[idx] = video_name
                    segment_id_dset[idx] = segment_id
                    pbar.update(1)

    print("\nHDF5 preprocessing complete.")
    print(f"Final Video dataset shape: {video_dset.shape}")
    print(f"Final Text dataset shape: {text_dset.shape}")
    print(f"Final Visual dataset shape: {image_dset.shape}")
    print(f"Final Video Names dataset shape: {video_name_dset.shape}")
    print(f"Final Segment IDs dataset shape: {segment_id_dset.shape}")

if __name__ == '__main__':
    BATCH_SIZE = 32
    N_WORKERS = 8 # 시스템 환경에 맞게 조절하세요.
    
    # --- ‼️ 테스트 데이터 경로로 수정해주세요 ‼️ ---
    VIDEO_INPUT_DIR = '../../dataset/affectnet/features_new/test/video_pooled'
    TEXT_INPUT_DIR = '../../dataset/affectnet/features_new/test/text_L'
    IMAGE_INPUT_DIR = '../../dataset/affectnet/features_new/test/image'
    OUTPUT_PATH = '../../dataset/affectnet/features_new/test/test_data_triplet.hdf5'
    
    merge_features_to_hdf5(
        video_dir=VIDEO_INPUT_DIR,
        text_dir=TEXT_INPUT_DIR,
        image_dir=IMAGE_INPUT_DIR,
        output_path=OUTPUT_PATH,
        batch_size=BATCH_SIZE,
        n_workers=N_WORKERS
    )