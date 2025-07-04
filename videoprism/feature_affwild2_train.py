import os
import sys
import numpy as np
import pandas as pd
import gc
import cv2
from tqdm import tqdm
import multiprocessing as mp

import jax
import jax.numpy as jnp
import tensorflow as tf

# 'videoprism' 라이브러리가 설치되어 있다고 가정합니다.
from videoprism import models as vp

# -------------------------------------------------------------------
# Configuration Section
# -------------------------------------------------------------------
mp.set_start_method('spawn', force=True)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
tf.config.set_visible_devices([], 'GPU')
jax.config.update('jax_enable_x64', False)


def init_model(model_name='videoprism_public_v1_base'):
    """JAX 모델과 사전 학습된 가중치를 로드하는 함수"""
    global flax_model, loaded_state
    flax_model = vp.MODELS[model_name]()
    loaded_state = vp.load_pretrained_weights(model_name)


@jax.jit
def forward_fn(inputs, train=False):
    """JIT 컴파일된 JAX 모델의 forward pass 함수"""
    return flax_model.apply(loaded_state, inputs, train=train)

# -------------------------------------------------------------------
# Frame Processing Functions
# -------------------------------------------------------------------

def read_image_files(frame_paths: list, target_size: tuple=(288, 288)) -> np.ndarray | None:
    """주어진 경로 리스트에서 이미지 파일들을 읽어 numpy 배열로 반환합니다."""
    frames = []
    for path in frame_paths:
        try:
            frame = cv2.imread(path)
            if frame is None:
                print(f'⚠️ 프레임 파일 읽기 실패 (파일이 없거나 손상됨): {path}')
                continue
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, target_size)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        except Exception as e:
            print(f"🚨 프레임 처리 중 오류 발생: {path}, 오류: {e}")
            continue

    if not frames:
        return None
        
    return np.array(frames)


def extract_features_from_frames(
    frame_dir: str, 
    out_dir: str, 
    anno_path: str | None,
    num_samples: int = 16, 
    window_size: int = 64, 
    frame_size: int = 288
):
    """
    폴더에 있는 실제 프레임 목록을 기준으로 피처를 추출합니다.
    anno_path가 주어지면(Train/Validation) 라벨을 함께 처리하고, None이면(Test) 피처만 추출합니다.
    """
    is_test_set = (anno_path is None)
    label_df = None

    if not is_test_set:
        try:
            label_df = pd.read_csv(anno_path)
        except FileNotFoundError:
            print(f"🚨 어노테이션 파일을 찾을 수 없습니다: {anno_path}")
            return
        except Exception as e:
            print(f"🚨 어노테이션 파일 읽기 오류: {anno_path}, {e}")
            return

    os.makedirs(out_dir, exist_ok=True)
    video_name = os.path.basename(frame_dir)

    try:
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.lower().endswith('.jpg')])
        n_frames = len(frame_files)
    except FileNotFoundError:
        print(f"🚨 프레임 디렉토리를 찾을 수 없습니다: {frame_dir}")
        return

    if n_frames < window_size:
        print(f"⚠️ 전체 프레임 수가 {window_size}개 미만이므로 처리를 건너뜁니다: {video_name}")
        return

    for seg_idx, start_list_idx in enumerate(range(0, n_frames, window_size)):
        end_list_idx = start_list_idx + window_size
        
        if end_list_idx > n_frames:
            continue

        segment_frame_files = frame_files[start_list_idx:end_list_idx]
        frame_paths_to_read = []
        final_label = None

        if is_test_set:
            all_segment_paths = [os.path.join(frame_dir, f) for f in segment_frame_files]
            picker_indices = np.linspace(0, len(all_segment_paths) - 1, num=num_samples, dtype=int)
            frame_paths_to_read = [all_segment_paths[i] for i in picker_indices]
        else:
            valid_frame_paths = []
            invalid_frame_paths = []
            for filename in segment_frame_files:
                try:
                    frame_number = int(os.path.splitext(filename)[0])
                    label = label_df.iloc[frame_number - 1]
                    frame_path = os.path.join(frame_dir, filename)
                    
                    if label['valence'] != -5:
                        valid_frame_paths.append(frame_path)
                    else:
                        invalid_frame_paths.append(frame_path)
                except (ValueError, IndexError):
                    print(f"⚠️ 라벨 조회 실패 (파일명: {filename}), 이 프레임은 무시됩니다.")
                    continue
            
            if len(valid_frame_paths) >= num_samples:
                num_valid = len(valid_frame_paths)
                picker_indices = np.linspace(0, num_valid - 1, num=num_samples, dtype=int)
                frame_paths_to_read = [valid_frame_paths[i] for i in picker_indices]

                total_valence, total_arousal = 0, 0
                for path in frame_paths_to_read:
                    frame_number = int(os.path.splitext(os.path.basename(path))[0])
                    label = label_df.iloc[frame_number - 1]
                    total_valence += label['valence']
                    total_arousal += label['arousal']
                final_label = np.array([total_valence / num_samples, total_arousal / num_samples])
            else:
                all_segment_paths = valid_frame_paths + invalid_frame_paths
                if not all_segment_paths: continue
                num_total = len(all_segment_paths)
                picker_indices = np.linspace(0, num_total - 1, num=min(num_samples, num_total), dtype=int)
                frame_paths_to_read = [all_segment_paths[i] for i in picker_indices]
                final_label = np.array([-5.0, -5.0])

        if not frame_paths_to_read:
            continue

        frames = read_image_files(frame_paths_to_read, target_size=(frame_size, frame_size))
        
        if frames is None:
            print(f'세그먼트 {seg_idx} 처리 실패 (이미지 로딩 실패): {video_name}')
            continue

        inputs = jnp.asarray(frames[None, ...])
        embeddings, _ = forward_fn(inputs)
        feat_np = np.asarray(embeddings)
        
        filepath = os.path.join(out_dir, f"{video_name}_seg{seg_idx:03d}.npz")
        
        if is_test_set:
            np.savez_compressed(filepath, feature=feat_np)
        else:
            np.savez_compressed(filepath, feature=feat_np, label=final_label)

# -------------------------------------------------------------------
# Multiprocessing Section
# -------------------------------------------------------------------

def dedicated_worker(gpu_id: int, task_queue: mp.Queue, done_queue: mp.Queue, worker_args: dict):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    init_model()
    print(f"[Worker on GPU:{gpu_id}] 초기화 완료. 작업 시작.")
    sys.stdout.flush()

    for args in iter(task_queue.get, 'STOP'):
        frame_folder_path, anno_root_dir, output_dir = args
        video_name = os.path.basename(frame_folder_path)
        
        anno_path = None
        if anno_root_dir:
            anno_path = os.path.join(anno_root_dir, f"{video_name}.txt")
        
        extract_features_from_frames(
            frame_dir=frame_folder_path,
            out_dir=output_dir,
            anno_path=anno_path,
            num_samples=worker_args['num_samples'],
            window_size=worker_args['window_size'],
            frame_size=worker_args['frame_size']
        )
        done_queue.put(video_name)

# -------------------------------------------------------------------
# Parallel Processing Setup
# -------------------------------------------------------------------

def process_dataset_parallel(
    frame_root_dir: str, 
    output_dir: str, 
    anno_root_dir: str | None,
    num_samples: int = 16, 
    window_size: int = 64, 
    frame_size: int = 288, 
    num_workers: int = 4,
    is_test_set: bool = False,
    list_file_path: str | None = None # ★★★ Test set 목록 파일 경로 추가
):
    """데이터셋을 병렬로 처리합니다."""
    os.makedirs(output_dir, exist_ok=True)
    
    all_video_basenames = []
    dataset_name = "Unknown"

    # ★★★ Test set의 경우, 목록 파일에서 비디오 리스트를 읽어오도록 수정 ★★★
    if is_test_set:
        if not list_file_path:
            print("🚨 Test Set 처리를 위해 'list_file_path' 인자가 필요합니다.")
            return
        dataset_name = f"Test_Set_from_file"
        print(f"📁 {dataset_name} 처리 시작 (목록 파일: {os.path.basename(list_file_path)})...")
        try:
            with open(list_file_path, 'r') as f:
                # 파일의 각 줄을 읽어 공백을 제거하고, 비어있지 않은 줄만 리스트에 추가
                all_video_basenames = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"🚨🚨 Test set 목록 파일을 찾을 수 없습니다: {list_file_path}")
            return
    else: # Train/Validation Set
        if not anno_root_dir:
            print("🚨 Train/Validation Set 처리를 위해 'anno_root_dir' 인자가 필요합니다.")
            return
        dataset_name = os.path.basename(anno_root_dir)
        print(f"📁 Annotation-based 처리 시작 ({dataset_name})...")
        try:
            all_anno_files = sorted([f for f in os.listdir(anno_root_dir) if f.lower().endswith('.txt')])
            all_video_basenames = [os.path.splitext(f)[0] for f in all_anno_files]
        except FileNotFoundError:
            print(f"🚨🚨 어노테이션 루트 디렉토리를 찾을 수 없습니다: {anno_root_dir}")
            return

    processed_video_basenames = set()
    if os.path.exists(output_dir):
        for f_name in os.listdir(output_dir):
            if f_name.endswith('.npz'):
                processed_video_basenames.add(f_name.split('_seg')[0])

    basenames_to_process = [basename for basename in all_video_basenames if basename not in processed_video_basenames]

    if not basenames_to_process:
        print(f"✅ [{dataset_name}] 처리할 새로운 비디오가 없습니다.")
        return
        
    print(f"  - 총 비디오 수: {len(all_video_basenames)}개")
    print(f"  - 새로 처리할 비디오: {len(basenames_to_process)}개")

    try:
        num_available_gpus = jax.device_count()
        if num_workers > num_available_gpus:
            print(f"경고: 워커 수({num_workers})가 사용 가능한 GPU 수({num_available_gpus})보다 많습니다. {num_available_gpus}로 조정합니다.")
            num_workers = num_available_gpus
    except Exception as e:
        print(f"JAX GPU 감지 실패: {e}. 설정된 num_workers({num_workers})를 사용합니다.")

    task_queue = mp.Queue()
    done_queue = mp.Queue()
    
    tasks_added = 0
    for basename in basenames_to_process:
        frame_folder_path = os.path.join(frame_root_dir, basename)
        
        if not os.path.isdir(frame_folder_path):
            print(f"⚠️ 프레임 폴더를 찾을 수 없어 건너뜁니다: {frame_folder_path}")
            continue

        args = (frame_folder_path, anno_root_dir if not is_test_set else None, output_dir)
        task_queue.put(args)
        tasks_added += 1

    for _ in range(num_workers):
        task_queue.put('STOP')

    if tasks_added == 0:
        print(f"✅ 처리할 비디오 목록에 해당하는 프레임 폴더가 하나도 없습니다.")
        return

    worker_args = {'num_samples': num_samples, 'window_size': window_size, 'frame_size': frame_size}
    processes = []
    for i in range(num_workers):
        p = mp.Process(target=dedicated_worker, args=(i, task_queue, done_queue, worker_args))
        processes.append(p)
        p.start()

    for _ in tqdm(range(tasks_added), desc=f'Processing {dataset_name}', unit='video'):
        done_queue.get()
        
    for p in processes:
        p.join()

# -------------------------------------------------------------------
# Main Execution Section
# -------------------------------------------------------------------

def main():
    print(f'JAX Platform: {jax.lib.xla_bridge.get_backend().platform}, OpenCV Version: {cv2.__version__}')

    datasets = [
        {
            'type': 'train',
            'frame_dir': '../../../dataset/affectnet/new_frames',
            'anno_dir': '../../../dataset/affectnet/9th ABAW Annotations/VA_Estimation_Challenge/Train_Set',
            'out_dir': '../../../dataset/affectnet/features_new/train/video',
            'list_file': None # Train set은 사용 안함
        },
        {
            'type': 'validation',
            'frame_dir': '../../../dataset/affectnet/new_frames',
            'anno_dir': '../../../dataset/affectnet/9th ABAW Annotations/VA_Estimation_Challenge/Validation_Set',
            'out_dir': '../../../dataset/affectnet/features_new/valid/video',
            'list_file': None # Validation set은 사용 안함
        },
        # ★★★ Test set에 목록 파일 경로 지정 ★★★
        {
            'type': 'test',
            'frame_dir': '../../../dataset/affectnet/new_frames', # Test set 프레임들이 있는 상위 폴더
            'anno_dir': None, 
            'out_dir': '../../../dataset/affectnet/features_new/test/video',
            'list_file': '../../../dataset/affectnet/names_of_videos_in_each_test_set/Valence_Arousal_Estimation_Challenge_test_set_release.txt' # Test set 비디오 이름 목록 파일
        }
    ]
    
    for d in datasets:
        print("-" * 50)
        is_test = (d['type'] == 'test')
        
        process_dataset_parallel(
            frame_root_dir=d['frame_dir'], 
            anno_root_dir=d['anno_dir'], 
            output_dir=d['out_dir'],
            num_samples=16,
            window_size=64,
            num_workers=4,
            is_test_set=is_test,
            list_file_path=d['list_file'] # Test set 목록 파일 경로 전달
        )
        gc.collect()
        
    print('🎉 모든 처리가 완료되었습니다.')


if __name__ == '__main__':
    main()