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
# Configuration Section (기존과 동일)
# -------------------------------------------------------------------

# 멀티프로세싱 시작 방식을 'spawn'으로 설정하여 CUDA 초기화 오류를 방지합니다.
# force=True는 이미 설정된 경우에도 강제로 재설정합니다.
mp.set_start_method('spawn', force=True)

# JAX 메모리 최적화 설정
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8' # 시스템에 따라 조정 가능
# TensorFlow가 GPU를 사용하지 않도록 설정
tf.config.set_visible_devices([], 'GPU')
# JAX에서 64비트 정밀도 비활성화 (일반적으로 필요 없음)
jax.config.update('jax_enable_x64', False)


def configure_gpu_memory():
    """GPU 메모리 동적 할당 설정 (주로 TensorFlow 용)"""
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # TensorFlow가 GPU를 못찾는 것은 의도된 것이므로 이 오류는 무시해도 됩니다.
        pass

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
# Video/Frame Processing Functions (기존과 동일)
# -------------------------------------------------------------------

def get_video_info(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # ★★★ 파일 열기 실패 시, 어떤 파일인지 경로를 명확히 출력하도록 수정 ★★★
        print(f"🚨 [파일 열기 실패] 다음 비디오를 열 수 없습니다. 파일이 손상되었을 가능성이 높습니다: {video_path}")
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

def process_video_segments(video_path: str, anno_path: str, num_samples: int=16, window_size: int=64, frame_size: int=288):
    total_frames, _, _ = get_video_info(video_path)
    if total_frames is None or total_frames < window_size:
        # print(f'비디오 처리 불가 (프레임 부족): {video_path}')
        return []
    try:
        anno_df = pd.read_csv(anno_path)
        total_frames = min(total_frames, len(anno_df))
    except Exception as e:
        print(f'어노테이션 로드 실패: {anno_path}, 오류: {e}')
        return []
    results = []
    for start in range(0, total_frames, window_size):
        end = min(start + window_size, total_frames) - 1
        if end - start + 1 < num_samples:
            continue
        indices = np.linspace(start, end, num_samples, dtype=int)
        frames = read_frames_at_indices(video_path, indices.tolist(), target_size=(frame_size, frame_size))
        if frames is None:
            continue
        inputs = jnp.asarray(frames[None, ...])
        embeddings, _ = forward_fn(inputs)
        feat_np = np.asarray(embeddings)
        labels = anno_df.iloc[indices]['Label'].values
        vr = (labels == 1).mean()
        lbl = 1 if vr >= 0.25 else 0
        results.append({'feature': feat_np, 'label': lbl})
    return results

def extract_features_single_video(video_path: str, anno_path: str, out_dir: str, num_samples: int=16, window_size: int=64, frame_size: int=288):
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(video_path))[0]
    segments = process_video_segments(video_path, anno_path, num_samples, window_size, frame_size)
    for idx, seg in enumerate(segments):
        filepath = os.path.join(out_dir, f"{name}_seg{idx:03d}.npz")
        np.savez_compressed(filepath, feature=seg['feature'], label=seg['label'])

# -------------------------------------------------------------------
# Multiprocessing Section (기존과 동일)
# -------------------------------------------------------------------

def dedicated_worker(gpu_id: int, task_queue: mp.Queue, done_queue: mp.Queue, worker_args: dict):
    """
    특정 GPU에 할당되어 작업을 처리하는 워커 프로세스.
    이 함수는 각 자식 프로세스의 진입점(entry point)입니다.
    """
    # 1. (가장 중요) 이 프로세스가 사용할 GPU를 환경 변수로 지정합니다.
    #    이 작업은 JAX나 다른 라이브러리가 초기화되기 전에 수행되어야 합니다.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # 2. 모델을 초기화합니다. 이제 JAX는 지정된 단일 GPU만 인식하고 그곳에 모델을 로드합니다.
    init_model()
    print(f"[Worker on GPU:{gpu_id}] 초기화 완료. 작업 시작.")
    sys.stdout.flush() # 자식 프로세스의 출력이 즉시 보이도록 버퍼를 비웁니다.

    # 3. 큐에서 'STOP' 신호를 받을 때까지 계속 작업을 가져와 처리합니다.
    for args in iter(task_queue.get, 'STOP'):
        video_path, anno_dir, output_dir = args
        
        # 4. 영상 파일을 처리합니다.
        base = os.path.splitext(os.path.basename(video_path))[0]
        if base.endswith('_converted'):
            base = base[:-10]
        a_path = os.path.join(anno_dir, f'{base}.csv')
        
        extract_features_single_video(
            video_path, a_path, output_dir,
            num_samples=worker_args['num_samples'],
            window_size=worker_args['window_size'],
            frame_size=worker_args['frame_size']
        )
        # 5. 처리 완료된 파일 이름을 done_queue에 넣어 메인 프로세스에 알립니다.
        done_queue.put(base)

# -------------------------------------------------------------------
# Parallel Processing Setup (★★★★★ 여기가 핵심 변경 부분)
# -------------------------------------------------------------------

def process_dataset_parallel(input_dir: str, anno_dir: str, output_dir: str, num_samples: int=16, window_size: int=64, frame_size: int=288, num_workers: int=4):
    """
    multiprocessing.Process 와 Queue를 사용하여 각 워커를 특정 GPU에 할당하고 작업을 분배합니다.
    ★★ 이미 처리된 비디오는 건너뜁니다. ★★
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 입력 디렉토리에서 모든 비디오 파일 목록을 가져옵니다.
    all_vids_in_dir = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))])

    # 2. 출력 디렉토리에서 이미 처리된 비디오의 기본 이름(확장자 제외)을 집합(set)으로 만듭니다.
    #    예: "my_video_seg001.npz" 파일에서 "my_video"를 추출하여 집합에 추가합니다.
    #    이렇게 하면 중복 없이 빠르게 조회할 수 있습니다.
    processed_video_basenames = set()
    if os.path.exists(output_dir):
        for f_name in os.listdir(output_dir):
            if f_name.endswith('.npz'):
                # "_seg" 이전의 부분을 비디오의 기본 이름으로 간주합니다.
                base_name = f_name.split('_seg')[0]
                processed_video_basenames.add(base_name)

    # 3. 전체 비디오 목록에서 아직 처리되지 않은 비디오만 필터링합니다.
    vids_to_process = []
    for video_filename in all_vids_in_dir:
        video_basename = os.path.splitext(video_filename)[0]
        if video_basename not in processed_video_basenames:
            vids_to_process.append(video_filename)

    # 4. 처리할 비디오가 없는 경우, 메시지를 출력하고 함수를 종료합니다.
    if not vids_to_process:
        print(f"✅ [{os.path.basename(input_dir)}] 처리할 새로운 비디오가 없습니다. (총 {len(all_vids_in_dir)}개 비디오 모두 처리 완료)")
        return
        
    print(f"📁 디렉토리: {os.path.basename(input_dir)}")
    print(f"  - 총 비디오: {len(all_vids_in_dir)}개")
    print(f"  - 이미 처리된 비디오: {len(processed_video_basenames)}개")
    print(f"  - 새로 처리할 비디오: {len(vids_to_process)}개")

    # 사용 가능한 GPU 수보다 워커 수가 많지 않도록 조정합니다.
    try:
        num_available_gpus = jax.device_count()
        if num_workers > num_available_gpus:
            print(f"경고: 워커 수({num_workers})가 사용 가능한 GPU 수({num_available_gpus})보다 많습니다. 워커 수를 {num_available_gpus}로 조정합니다.")
            num_workers = num_available_gpus
    except Exception as e:
        print(f"JAX에서 GPU 장치를 감지하지 못했습니다: {e}. 설정된 num_workers({num_workers})를 그대로 사용합니다.")

    task_queue = mp.Queue()
    done_queue = mp.Queue()

    # 필터링된, 즉 처리해야 할 비디오 목록만 큐에 추가합니다.
    for v in vids_to_process:
        args = (os.path.join(input_dir, v), anno_dir, output_dir)
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

    # 진행 표시줄(tqdm)의 전체 개수를 실제 처리할 비디오 수로 설정합니다.
    for _ in tqdm(range(len(vids_to_process)), desc=f'Processing {os.path.basename(input_dir)}', unit='video'):
        done_queue.get()
        
    for p in processes:
        p.join()

# -------------------------------------------------------------------
# Main Execution Section (기존과 동일)
# -------------------------------------------------------------------

def main():
    print(f'JAX Platform: {jax.lib.xla_bridge.get_backend().platform}, OpenCV Version: {cv2.__version__}')

    # 처리할 데이터셋 경로 리스트
    datasets = [
        ('../../../dataset/dvd/DVD_Competition/Training/converted_videos',
         '../../../dataset/dvd/DVD_Competition/Training/annotations',
         '../../../dataset/dvd/DVD_Competition/Training/features'),
        ('../../../dataset/dvd/DVD_Competition/Validation/converted_videos',
         '../../../dataset/dvd/DVD_Competition/Validation/annotations',
         '../../../dataset/dvd/DVD_Competition/Validation/features'),
    ]
    
    for inp, ano, out in datasets:
        process_dataset_parallel(inp, ano, out, num_workers=4)
        gc.collect()
        
    print('🎉 모든 처리가 완료되었습니다.')


if __name__ == '__main__':
    # `spawn` 시작 방식을 사용하는 경우, main 실행 코드는 이 블록 안에 위치해야 합니다.
    main()