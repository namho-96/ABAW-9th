import os
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from tqdm import tqdm
import gc
import multiprocessing as mp
import argparse
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path

# -------------------------------------------------------------------
# Configuration and Optimizations
# -------------------------------------------------------------------

# CUDA 최적화 설정
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# 메모리 최적화를 위한 전역 설정
BATCH_SIZE = 8  # GPU 메모리에 따라 조정
MAX_WORKERS_PER_GPU = 2  # GPU당 워커 수 증가
PREFETCH_BUFFER_SIZE = 16  # 이미지 프리페치 버퍼 크기

# -------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------

def uniform_sample_range(start, end, num=4):
    """주어진 범위 내에서 균등하게 샘플링할 인덱스를 반환합니다."""
    return np.linspace(start, end, num=num, dtype=int)

def clean_caption(text):
    """생성된 캡션 텍스트를 정리합니다."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    joined = " ".join(lines)
    if joined and joined[-1] not in {'.', '!', '?'}:
        joined += '.'
    return joined

def preload_images_batch(frame_paths):
    """이미지들을 배치로 미리 로드합니다."""
    images = []
    valid_paths = []
    
    def load_single_image(path):
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            print(f"⚠️ 이미지 로드 실패 {path}: {e}")
            return None
    
    # ThreadPoolExecutor로 이미지 로딩 병렬화
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(load_single_image, frame_paths))
    
    for img, path in zip(results, frame_paths):
        if img is not None:
            images.append(img)
            valid_paths.append(path)
    
    return images, valid_paths

# -------------------------------------------------------------------
# Optimized Captioning Core Function
# -------------------------------------------------------------------

def extract_captions_for_frames_optimized(chat, frame_dir, out_dir):
    """배치 처리와 최적화된 메모리 관리로 캡션을 추출합니다."""
    os.makedirs(out_dir, exist_ok=True)
    video_name = os.path.basename(frame_dir)
    
    try:
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    except FileNotFoundError:
        print(f"🚨 [PID:{os.getpid()}] 프레임 폴더를 찾을 수 없습니다: {frame_dir}")
        return

    n_frames = len(frame_files)
    if n_frames < 4:
        return

    # 세그먼트별 처리를 배치로 최적화
    for seg_idx, seg_start in enumerate(range(0, n_frames, 64)):
        seg_end = min(seg_start + 64, n_frames)
        if seg_end - seg_start < 4:
            continue
        
        frame_indices_to_process = uniform_sample_range(seg_start, seg_end - 1, 4)
        frame_paths = [os.path.join(frame_dir, frame_files[f_idx]) for f_idx in frame_indices_to_process]
        
        # 배치로 이미지 미리 로드
        images, valid_paths = preload_images_batch(frame_paths)
        
        if not images:
            continue
            
        seg_captions = []
        
        # 배치 처리로 최적화
        for i in range(0, len(images), BATCH_SIZE):
            batch_images = images[i:i + BATCH_SIZE]
            batch_paths = valid_paths[i:i + BATCH_SIZE]
            
            batch_captions = process_image_batch(chat, batch_images, batch_paths)
            seg_captions.extend(batch_captions)
            
            # 배치 처리 후 메모리 정리
            del batch_images
            torch.cuda.empty_cache()
        
        # 남은 이미지들 정리
        for img in images:
            del img
        del images
        
        # 결과 저장
        out_file = os.path.join(out_dir, f"{video_name}_seg{seg_idx:03d}.txt")
        with open(out_file, 'w', encoding='utf-8') as f_out:
            f_out.write("\n".join(seg_captions))

def process_image_batch(chat, images, paths):
    """이미지 배치를 효율적으로 처리합니다."""
    from minigpt4.conversation.conversation import CONV_VISION_Vicuna0
    
    captions = []
    prompt = (
        "Describe the facial expression in this image using valence and arousal dimensions. "
        "Indicate whether the valence (positive or negative emotion) and arousal (calm or excited) are high, medium, or low. "
        "Explain briefly why you think so."
    )
    
    for img, path in zip(images, paths):
        try:
            # 대화 상태 재사용 최적화
            chat_state = CONV_VISION_Vicuna0.copy()
            img_list = []
            
            # 이미지 업로드 및 인코딩
            chat.upload_img(img, chat_state, img_list)
            chat.encode_img(img_list)
            
            chat.ask(prompt, chat_state)
            
            # 추론 파라미터 최적화 - do_sample 파라미터 제거
            llm_message = chat.answer(
                conv=chat_state, 
                img_list=img_list, 
                num_beams=1,  # beam search 줄여서 속도 향상
                temperature=0.7,
                max_new_tokens=30, 
                max_length=500
                # do_sample=False 파라미터 제거 - 이 파라미터가 오류의 원인
            )[0]
            
            captions.append(clean_caption(llm_message))
            
            # 즉시 메모리 정리
            del chat_state, img_list
            
        except Exception as e:
            print(f"⚠️ [PID:{os.getpid()}] 프레임 처리 실패 {os.path.basename(path)}: {e}")
            captions.append("Error generating caption for this frame.")
    
    return captions

# -------------------------------------------------------------------
# Enhanced Multiprocessing Worker
# -------------------------------------------------------------------

def dedicated_worker_optimized(gpu_id, task_queue, done_queue, cfg_path):
    """
    최적화된 워커 프로세스 - 모델 로딩 최적화 및 메모리 관리 개선
    """
    from transformers import StoppingCriteriaList
    from minigpt4.common.config import Config
    from minigpt4.common.registry import registry
    from minigpt4.conversation.conversation import Chat, StoppingCriteriaSub
    from minigpt4 import models
    from minigpt4 import processors

    chat = None
    try:
        print(f"[Worker on GPU:{gpu_id}] 초기화를 시작합니다...")
        sys.stdout.flush()
        
        # GPU 설정 최적화
        torch.cuda.set_device(gpu_id)
        
        args = argparse.Namespace(cfg_path=cfg_path, gpu_id=gpu_id, options=[])
        cfg = Config(args)
        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        
        # 모델 로딩 시 메모리 최적화
        with torch.cuda.device(gpu_id):
            model_cls = registry.get_model_class(model_config.arch)
            model = model_cls.from_config(model_config).to(f'cuda:{gpu_id}')
            
            # 모델을 evaluation 모드로 설정하여 dropout 등 비활성화
            model.eval()
            
            # 메모리 효율성을 위해 모델을 half precision으로 변환 (선택사항)
            # model = model.half()  # 정확도에 영향을 줄 수 있으므로 주의
            
        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        
        stop_words_ids = [torch.tensor(ids).to(f'cuda:{gpu_id}') for ids in [[835], [2277, 29937]]]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        chat = Chat(model, vis_processor, device=f'cuda:{gpu_id}', stopping_criteria=stopping_criteria)
        
        print(f"[Worker on GPU:{gpu_id}] 초기화 완료. 작업을 시작합니다.")
        sys.stdout.flush()

    except Exception as e:
        print(f"🚨 [Worker on GPU:{gpu_id}] 모델 초기화 실패: {e}", file=sys.stderr)
        while True:
            task = task_queue.get()
            if task == 'STOP': break
            done_queue.put('init_failed_task')
        return

    # 성능 통계를 위한 변수
    processed_count = 0
    start_time = time.time()
    
    while True:
        task = task_queue.get()
        if task == 'STOP':
            break
        
        frame_dir, out_dir = task
        task_start_time = time.time()
        
        try:
            extract_captions_for_frames_optimized(chat, frame_dir, out_dir)
            processed_count += 1
            task_duration = time.time() - task_start_time
            
            if processed_count % 10 == 0:  # 10개마다 진행 상황 출력
                avg_time = (time.time() - start_time) / processed_count
                print(f"[GPU:{gpu_id}] 처리 완료: {processed_count}개, 평균 시간: {avg_time:.2f}초/비디오")
                
        except Exception as e:
            print(f"🚨 [Worker on GPU:{gpu_id}] 작업 처리 중 오류 발생 (폴더: {os.path.basename(frame_dir)}): {e}", file=sys.stderr)
        finally:
            done_queue.put(frame_dir)
            
            # 주기적 메모리 정리
            if processed_count % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()

# -------------------------------------------------------------------
# Optimized Parallel Processing Orchestrator
# -------------------------------------------------------------------

def process_dataset_parallel_optimized(dataset_config, cfg_path, num_workers):
    """
    최적화된 병렬 처리 - 작업 분배 및 로드 밸런싱 개선
    """
    d_type = dataset_config['type']
    frame_root_dir = dataset_config['frame_dir']
    anno_root_dir = dataset_config['anno_dir']
    list_file_path = dataset_config['list_file']
    out_dir = dataset_config['out_dir']
    
    os.makedirs(out_dir, exist_ok=True)
    all_video_basenames = []
    dataset_name = "Unknown"

    print("-" * 50)
    if d_type == 'test':
        dataset_name = f"Test Set ({os.path.basename(list_file_path)})"
        print(f"📁 {dataset_name} 처리 시작...")
        try:
            with open(list_file_path, 'r') as f:
                all_video_basenames = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"🚨🚨 Test set 목록 파일을 찾을 수 없습니다: {list_file_path}")
            return
    else:
        dataset_name = os.path.basename(anno_root_dir)
        print(f"📁 Annotation 기반 처리 시작 ({dataset_name})...")
        try:
            all_anno_files = sorted([f for f in os.listdir(anno_root_dir) if f.lower().endswith('.txt')])
            all_video_basenames = [os.path.splitext(f)[0] for f in all_anno_files]
        except FileNotFoundError:
            print(f"🚨🚨 어노테이션 루트 디렉토리를 찾을 수 없습니다: {anno_root_dir}")
            return
            
    # 이미 처리된 비디오 확인 최적화
    processed_videos = set()
    if os.path.exists(out_dir):
        existing_files = os.listdir(out_dir)
        processed_videos = {f.split('_seg')[0] for f in existing_files if f.endswith('.txt')}

    basenames_to_process = [b for b in all_video_basenames if b not in processed_videos]

    if not basenames_to_process:
        print(f"✅ [{dataset_name}] 처리할 새로운 비디오가 없습니다.")
        return
        
    print(f"  - 총 비디오 수: {len(all_video_basenames)}개")
    print(f"  - 새로 처리할 비디오: {len(basenames_to_process)}개")

    # 큐 크기 최적화
    task_queue = mp.Queue(maxsize=num_workers * 4)  # 큐 크기 제한으로 메모리 사용량 조절
    done_queue = mp.Queue()

    # 유효한 작업들을 미리 필터링하여 성능 향상
    valid_tasks = []
    for video_name in basenames_to_process:
        frame_dir = os.path.join(frame_root_dir, video_name)
        if os.path.isdir(frame_dir):
            # 프레임 수 미리 확인하여 불필요한 작업 제거
            try:
                frame_count = len([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
                if frame_count >= 4:
                    valid_tasks.append((frame_dir, out_dir))
            except:
                continue
        else:
            print(f"⚠️ 프레임 폴더 누락, 건너뜁니다: {frame_dir}")

    if not valid_tasks:
        print(f"✅ 처리할 유효한 비디오 폴더가 없습니다.")
        return

    print(f"  - 실제 처리할 유효한 비디오: {len(valid_tasks)}개")

    # 작업을 큐에 추가 (별도 스레드에서 진행하여 블로킹 방지)
    def add_tasks_to_queue():
        for task in valid_tasks:
            task_queue.put(task)
        for _ in range(num_workers):
            task_queue.put('STOP')

    task_thread = ThreadPoolExecutor(max_workers=1)
    task_future = task_thread.submit(add_tasks_to_queue)

    # 워커 프로세스 시작
    processes = []
    for i in range(num_workers):
        gpu_id = i % torch.cuda.device_count()  # GPU 순환 할당
        p = mp.Process(target=dedicated_worker_optimized, args=(gpu_id, task_queue, done_queue, cfg_path))
        p.start()
        processes.append(p)

    # 진행률 모니터링 (더 상세한 정보 제공)
    start_time = time.time()
    completed_tasks = 0
    
    for _ in tqdm(range(len(valid_tasks)), desc=f"Processing {dataset_name}"):
        result = done_queue.get()
        completed_tasks += 1
        
        # 주기적으로 성능 통계 출력
        if completed_tasks % 50 == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_task = elapsed_time / completed_tasks
            remaining_tasks = len(valid_tasks) - completed_tasks
            estimated_remaining_time = remaining_tasks * avg_time_per_task
            
            print(f"\n📊 진행 상황: {completed_tasks}/{len(valid_tasks)} "
                  f"(평균 {avg_time_per_task:.2f}초/비디오, "
                  f"예상 남은 시간: {estimated_remaining_time/60:.1f}분)")

    # 정리 작업
    task_future.result()  # 작업 추가 스레드 완료 대기
    task_thread.shutdown()
    
    for p in processes:
        p.join()
    
    total_time = time.time() - start_time
    print(f"✅ {dataset_name} 처리 완료: {len(valid_tasks)}개 비디오, 총 시간: {total_time/60:.1f}분")

# -------------------------------------------------------------------
# Main Execution Section
# -------------------------------------------------------------------

def main():
    """
    메인 실행 함수. 설정을 정의하고 데이터셋 처리를 시작합니다.
    """
    CFG_PATH = "eval_configs/minigpt4_eval.yaml"
    
    try:
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print("🚨 GPU가 없습니다. 이 스크립트는 GPU가 필요합니다.")
            return
        print(f"🚀 사용 가능한 GPU 수: {num_gpus}개")
        
        # GPU당 워커 수 증가로 처리량 향상
        num_workers = num_gpus * MAX_WORKERS_PER_GPU
        print(f"🔧 총 워커 수: {num_workers}개 (GPU당 {MAX_WORKERS_PER_GPU}개)")
        
    except Exception as e:
        print(f"🚨 GPU 확인 중 오류 발생: {e}. 워커 수를 1로 설정합니다.")
        num_workers = 1

    # --- 처리할 데이터셋 목록 정의 ---
    datasets = [
        {
            'type': 'test',
            'frame_dir': '../../../dataset/affectnet/new_frames',
            'anno_dir': None,
            'out_dir': '../../../dataset/affectnet/text/test',
            'list_file': '../../../dataset/affectnet/names_of_videos_in_each_test_set/Valence_Arousal_Estimation_Challenge_test_set_release.txt'
        }
    ]

    overall_start_time = time.time()
    
    for d_config in datasets:
        dataset_start_time = time.time()
        process_dataset_parallel_optimized(d_config, CFG_PATH, num_workers)
        
        dataset_duration = time.time() - dataset_start_time
        print(f"⏱️ 데이터셋 처리 시간: {dataset_duration/60:.1f}분")
        
        # 데이터셋 간 메모리 정리
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # 모든 GPU 메모리 정리
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()

    total_duration = time.time() - overall_start_time
    print(f'🎉 모든 처리가 완료되었습니다. 총 소요 시간: {total_duration/60:.1f}분')

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
        print("✅ 멀티프로세싱 시작 방식을 'spawn'으로 설정했습니다.")
    except RuntimeError:
        print("ⓘ 멀티프로세싱 시작 방식이 이미 설정되었습니다.")
    
    # 프로세스 우선순위 설정 (Linux에서)
    try:
        import psutil
        p = psutil.Process()
        p.nice(-10)  # 높은 우선순위 설정
        print("✅ 프로세스 우선순위를 높게 설정했습니다.")
    except:
        pass
    
    main()