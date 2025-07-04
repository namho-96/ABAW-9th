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

from videoprism import models as vp

# -------------------------------------------------------------------
# Configuration Section (기존과 동일)
# -------------------------------------------------------------------
mp.set_start_method('spawn', force=True)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
tf.config.set_visible_devices([], 'GPU')
jax.config.update('jax_enable_x64', False)

def configure_gpu_memory():
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        pass

def init_model(model_name='videoprism_public_v1_base'):
    global flax_model, loaded_state
    flax_model = vp.MODELS[model_name]()
    loaded_state = vp.load_pretrained_weights(model_name)

@jax.jit
def forward_fn(inputs, train=False):
    return flax_model.apply(loaded_state, inputs, train=train)

# -------------------------------------------------------------------
# Video/Frame Processing Functions (테스트 데이터용으로 수정됨)
# -------------------------------------------------------------------

def get_video_info(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"🚨 [파일 열기 실패] 다음 비디오를 열 수 없습니다: {video_path}")
        return None, None, None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return total_frames, fps, (width, height)

def read_frames_at_indices(video_path: str, frame_indices: list, target_size: tuple=(288,288)) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'비디오 열기 실패: {video_path}')
        return None
    frames = []
    for idx in sorted(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f'프레임 {idx} 읽기 실패')
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, target_size)
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
    cap.release()
    if len(frames) != len(frame_indices):
        print(f'요청 {len(frame_indices)}개 프레임 중 {len(frames)}개만 읽음')
        return None
    return np.array(frames)

def process_video_segments(video_path: str, num_samples: int=16, window_size: int=64, frame_size: int=288):
    total_frames, _, _ = get_video_info(video_path)
    if total_frames is None or total_frames < window_size:
        return []

    results = []
    for start in range(0, total_frames, window_size):
        if start + window_size > total_frames:
            continue

        end = start + window_size - 1
        indices = np.linspace(start, end, num_samples, dtype=int)
        frames = read_frames_at_indices(video_path, indices.tolist(), target_size=(frame_size, frame_size))

        if frames is None:
            continue

        inputs = jnp.asarray(frames[None, ...])
        embeddings, _ = forward_fn(inputs)
        feat_np = np.asarray(embeddings)
        results.append({'feature': feat_np})

    return results

def extract_features_single_video(video_path: str, out_dir: str, num_samples: int=16, window_size: int=64, frame_size: int=288):
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(video_path))[0]
    # <<< 변경점: anno_path 없이 함수 호출 >>>
    segments = process_video_segments(video_path, num_samples, window_size, frame_size)
    for idx, seg in enumerate(segments):
        filepath = os.path.join(out_dir, f"{name}_seg{idx:03d}.npz")
        # <<< 변경점: feature만 저장 >>>
        np.savez_compressed(filepath, feature=seg['feature'])

# -------------------------------------------------------------------
# Multiprocessing Section (테스트 데이터용으로 수정됨)
# -------------------------------------------------------------------

def dedicated_worker(gpu_id: int, task_queue: mp.Queue, done_queue: mp.Queue, worker_args: dict):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    init_model()
    print(f"[Worker on GPU:{gpu_id}] 초기화 완료. 작업 시작.")
    sys.stdout.flush()

    # <<< 변경점: task_queue에서 anno_dir를 받지 않음 >>>
    for args in iter(task_queue.get, 'STOP'):
        video_path, output_dir = args
        
        # <<< 변경점: a_path (어노테이션 경로) 생성 로직 제거 >>>
        base = os.path.splitext(os.path.basename(video_path))[0]

        extract_features_single_video(
            video_path, output_dir, # <<< 변경점: a_path 인자 제거
            num_samples=worker_args['num_samples'],
            window_size=worker_args['window_size'],
            frame_size=worker_args['frame_size']
        )
        done_queue.put(base)

# -------------------------------------------------------------------
# Parallel Processing Setup (테스트 데이터용으로 수정됨)
# -------------------------------------------------------------------

def process_dataset_parallel(input_dir: str, output_dir: str, num_samples: int=16, window_size: int=64, frame_size: int=288, num_workers: int=4):
    os.makedirs(output_dir, exist_ok=True)
    
    all_vids_in_dir = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))])
    
    processed_video_basenames = set()
    if os.path.exists(output_dir):
        for f_name in os.listdir(output_dir):
            if f_name.endswith('.npz'):
                base_name = f_name.split('_seg')[0]
                processed_video_basenames.add(base_name)

    vids_to_process = []
    for video_filename in all_vids_in_dir:
        video_basename = os.path.splitext(video_filename)[0]
        if video_basename not in processed_video_basenames:
            vids_to_process.append(video_filename)

    if not vids_to_process:
        print(f"✅ [{os.path.basename(input_dir)}] 처리할 새로운 비디오가 없습니다. (총 {len(all_vids_in_dir)}개 비디오 모두 처리 완료)")
        return
        
    print(f"📁 디렉토리: {os.path.basename(input_dir)}")
    print(f"  - 총 비디오: {len(all_vids_in_dir)}개")
    print(f"  - 이미 처리된 비디오: {len(processed_video_basenames)}개")
    print(f"  - 새로 처리할 비디오: {len(vids_to_process)}개")

    try:
        num_available_gpus = jax.device_count()
        if num_workers > num_available_gpus:
            print(f"경고: 워커 수({num_workers})가 사용 가능한 GPU 수({num_available_gpus})보다 많습니다. 워커 수를 {num_available_gpus}로 조정합니다.")
            num_workers = num_available_gpus
    except Exception as e:
        print(f"JAX에서 GPU 장치를 감지하지 못했습니다: {e}. 설정된 num_workers({num_workers})를 그대로 사용합니다.")

    task_queue = mp.Queue()
    done_queue = mp.Queue()

    # <<< 변경점: task 큐에 video 경로와 output 경로만 넣음 >>>
    for v in vids_to_process:
        args = (os.path.join(input_dir, v), output_dir)
        task_queue.put(args)

    for _ in range(num_workers):
        task_queue.put('STOP')

    worker_args = {
        'num_samples': num_samples,
        'window_size': window_size,
        'frame_size': frame_size
    }

    processes = []
    for i in range(num_workers):
        p = mp.Process(target=dedicated_worker, args=(i, task_queue, done_queue, worker_args))
        processes.append(p)
        p.start()

    for _ in tqdm(range(len(vids_to_process)), desc=f'Processing {os.path.basename(input_dir)}', unit='video'):
        done_queue.get()
        
    for p in processes:
        p.join()

# -------------------------------------------------------------------
# Main Execution Section (테스트 데이터용으로 수정됨)
# -------------------------------------------------------------------

def main():
    print(f'JAX Platform: {jax.lib.xla_bridge.get_backend().platform}, OpenCV Version: {cv2.__version__}')

    # <<< 변경점: 테스트 데이터셋 경로 리스트 (입력 비디오 폴더, 출력 피처 폴더) >>>
    test_datasets = [
        ('../../../dataset/dvd/DVD_Competition/Testing/converted_videos',
         '../../../dataset/dvd/DVD_Competition/Testing/features'),
        # 다른 테스트 데이터셋이 있다면 여기에 추가
        # ('/path/to/your/test_videos_2', '/path/to/your/features_2'),
    ]
    
    # <<< 변경점: anno_dir 없이 함수 호출 >>>
    for inp, out in test_datasets:
        process_dataset_parallel(inp, out, num_workers=4)
        gc.collect()
        
    print('🎉 모든 처리가 완료되었습니다.')


if __name__ == '__main__':
    main()
    