import os
import cv2
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from typing import List, Dict

# ==================================================================
# ✨ 경로 및 VA Task 설정 하드코딩 ✨
# 아래 변수들의 값을 실제 환경에 맞게 수정해주세요.
# ==================================================================

# 1. 입력 H5 파일 경로 (VA 특징이 추출된 파일)
H5_FILE_PATH = "/mnt/data/abaw/dataset/affectnet/features_new/test/test_data_triplet.hdf5"

# 2. VA Task로 학습된 모델(.pth) 파일 경로
MODEL_PATH = "/mnt/data/abaw/2025/code/output/20250704_002633_va_image_text_video_5fold/fold_1/best_model_fold1_image_text_video.pth"

# 3. 원본 동영상 파일(.mp4, .avi 등)이 있는 폴더 경로
VIDEO_DIR = "/mnt/data/abaw/dataset/affectnet/video"

# 4. 최종 예측 결과를 저장할 폴더 경로
OUTPUT_DIR = "/mnt/data/abaw/dataset/affectnet/predictions/folds/fold1"

# 5. 모델이 학습된 모달리티 조합 (학습 시와 동일하게)
MODALITIES = "video_text_visual" # 예: "video_text_visual"

# 6. 예측에 사용할 장치 ('cuda' 또는 'cpu')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ==================================================================
#  모델 정의 (train_total_v2.py에서 VA Task 부분 포함하여 수정)
#  (이전 코드와 동일하므로 생략)
# ==================================================================

class SequenceEncoder(nn.Module):
    """GRU 기반 범용 시퀀스 인코더"""
    def __init__(self, input_dim=768, hidden_dim=512, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, 
                          dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.drop = nn.Dropout(dropout)
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), 
            nn.Tanh(), 
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        if x.dim() == 4: 
            x = x.squeeze(1)
        out, _ = self.gru(x)
        out = self.norm(out)
        out = self.drop(out)
        w = self.attn(out)
        w = F.softmax(w, dim=1)
        return torch.sum(out * w, dim=1)

class UnifiedClassifier(nn.Module):
    """VA Task를 지원하도록 수정된 멀티모달 분류기"""
    def __init__(self, 
                 modalities: List[str], 
                 feat_dims: Dict[str, int], 
                 task: str, 
                 hidden_dim: int = 768, 
                 n_cls: int = 2, # violence task용, va에서는 사용 안함
                 dropout: float = 0.3, 
                 num_attn_heads: int = 8,
                 image_centric: bool = True):
        super().__init__()
        self.modalities = modalities
        self.task = task
        self.model_dim = hidden_dim
        self.image_centric = image_centric and 'image' in modalities
        enc_hidden_dim = self.model_dim // 2
        self.encoders = nn.ModuleDict()
        modalities_internal = ['image' if m == 'visual' else m for m in self.modalities]
        if 'image' in modalities_internal: self.encoders['image'] = SequenceEncoder(feat_dims.get('visual', feat_dims.get('image')), enc_hidden_dim, dropout=dropout)
        if 'text' in modalities_internal: self.encoders['text'] = SequenceEncoder(feat_dims['text'], enc_hidden_dim, dropout=dropout)
        if 'video' in modalities_internal: self.encoders['video'] = SequenceEncoder(feat_dims['video'], enc_hidden_dim, dropout=dropout)
        self.feature_refiners = nn.ModuleDict()
        for modality in modalities_internal: self.feature_refiners[modality] = nn.Sequential(nn.Linear(self.model_dim, self.model_dim), nn.GELU(), nn.Dropout(dropout))
        if len(self.modalities) > 1:
            if self.image_centric:
                self.auxiliary_attention = nn.ModuleDict()
                aux_modalities = [m for m in modalities_internal if m != 'image']
                for modality in aux_modalities: self.auxiliary_attention[modality] = nn.MultiheadAttention(self.model_dim, num_attn_heads, batch_first=True, dropout=dropout)
                if aux_modalities:
                    self.fusion_gate = nn.Sequential(nn.Linear(self.model_dim * (len(aux_modalities) + 1), self.model_dim), nn.Sigmoid())
                    self.layer_norm = nn.LayerNorm(self.model_dim)
            else:
                self.cross_attention = nn.ModuleDict({m: nn.MultiheadAttention(self.model_dim, num_attn_heads, batch_first=True, dropout=dropout) for m in modalities_internal})
                self.layer_norms = nn.ModuleDict({m: nn.LayerNorm(self.model_dim) for m in modalities_internal})
                self.fusion_mlp = nn.Sequential(nn.Linear(self.model_dim * len(self.modalities), self.model_dim), nn.GELU(), nn.Dropout(dropout))
        
        if self.task == 'violence':
            self.classifier = nn.Sequential(
                nn.Linear(self.model_dim, self.model_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.model_dim // 2, n_cls)
            )
        elif self.task == 'va':
            self.valence_head = nn.Sequential(
                nn.Linear(self.model_dim, self.model_dim // 2), nn.GELU(), nn.Linear(self.model_dim // 2, 1)
            )
            self.arousal_head = nn.Sequential(
                nn.Linear(self.model_dim, self.model_dim // 2), nn.GELU(), nn.Linear(self.model_dim // 2, 1)
            )
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def forward(self, **kwargs: torch.Tensor) -> torch.Tensor:
        refined_embeddings = {}
        kwargs_internal = {'image' if k == 'visual' else k: v for k, v in kwargs.items()}
        modalities_internal = ['image' if m == 'visual' else m for m in self.modalities]
        for modality in modalities_internal:
            if modality in kwargs_internal:
                raw_embedding = self.encoders[modality](kwargs_internal[modality])
                refined_embeddings[modality] = self.feature_refiners[modality](raw_embedding)
        if len(self.modalities) == 1: final_features = list(refined_embeddings.values())[0]
        elif self.image_centric and 'image' in refined_embeddings:
            image_features = refined_embeddings['image']
            attended_features = [image_features]
            aux_modalities = [m for m in modalities_internal if m != 'image' and m in refined_embeddings]
            for modality in aux_modalities:
                image_query, modality_kv = image_features.unsqueeze(1), refined_embeddings[modality].unsqueeze(1)
                auxiliary_info, _ = self.auxiliary_attention[modality](image_query, modality_kv, modality_kv)
                attended_features.append(auxiliary_info.squeeze(1))
            if len(attended_features) > 1:
                all_features = torch.cat(attended_features, dim=1)
                fusion_weight = self.fusion_gate(all_features)
                auxiliary_sum = sum(attended_features[1:]) if len(attended_features) > 1 else 0
                final_features = self.layer_norm(image_features + fusion_weight * auxiliary_sum)
            else: final_features = image_features
        else:
            ordered_embeddings = [refined_embeddings[m] for m in modalities_internal if m in refined_embeddings]
            if len(ordered_embeddings) > 1:
                all_embs, refined_embs = torch.stack(ordered_embeddings, dim=1), []
                for i, modality in enumerate([m for m in modalities_internal if m in refined_embeddings]):
                    query = ordered_embeddings[i].unsqueeze(1)
                    context, _ = self.cross_attention[modality](query, all_embs, all_embs)
                    refined_emb = self.layer_norms[modality](ordered_embeddings[i] + context.squeeze(1))
                    refined_embs.append(refined_emb)
                final_features = self.fusion_mlp(torch.cat(refined_embs, dim=1))
            else: final_features = ordered_embeddings[0]

        if self.task == 'violence':
            return self.classifier(final_features)
        elif self.task == 'va':
            valence = self.valence_head(final_features)
            arousal = self.arousal_head(final_features)
            return torch.cat([valence, arousal], dim=1)

# ==================================================================
# VA Task 예측 실행을 위한 메인 로직
# ==================================================================

def main():
    """VA Task에 대한 예측을 수행하고 txt 파일로 저장합니다."""
    TASK = 'va' # Task 고정
    
    print(f"🚀 Starting VA prediction with hardcoded settings.")
    print(f"   - HDF5 File: {H5_FILE_PATH}")
    print(f"   - Model Path: {MODEL_PATH}")
    print(f"   - Video Dir: {VIDEO_DIR}")
    print(f"   - Modalities: {MODALITIES}")
    print(f"   - Device: {DEVICE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    modalities_list = sorted(MODALITIES.split('_'))
    
    try:
        video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        if not video_files:
            print(f"Error: No video files found in '{VIDEO_DIR}'. Please check the path.")
            return
    except FileNotFoundError:
        print(f"Error: Video directory not found at '{VIDEO_DIR}'. Please check the path.")
        return

    # 1. H5 파일에서 데이터 로드
    print("\n[Step 1/4] Loading data from HDF5 file...")
    try:
        with h5py.File(H5_FILE_PATH, 'r') as h5f:
            video_feats = h5f['video_features'][:]
            text_feats = h5f['text_features'][:]
            visual_feats = h5f['visual_features'][:]
            video_names_bytes = h5f['video_names'][:]
            segment_ids = h5f['segment_ids'][:]
            video_names_h5 = [name.decode('utf-8') for name in video_names_bytes]
            feat_dims = {'video': video_feats.shape[-1], 'text': text_feats.shape[-1], 'visual': visual_feats.shape[-1]}
    except Exception as e:
        print(f"Error loading HDF5 file: {e}")
        return
    print(f"Feature dimensions loaded: {feat_dims}")

    # 2. 모델 로드
    print("\n[Step 2/4] Loading the trained model...")
    model = UnifiedClassifier(modalities=modalities_list, feat_dims=feat_dims, task=TASK)
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)['model_state_dict']
        if all(k.startswith('module.') for k in state_dict.keys()): state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        return

    # 3. 예측 수행
    print("\n[Step 3/4] Running model predictions...")
    video_predictions = defaultdict(list)
    with torch.no_grad():
        num_segments = len(video_names_h5)
        for i in tqdm(range(num_segments), desc="Predicting segments"):
            features = {}
            if 'video' in modalities_list: features['video'] = torch.from_numpy(video_feats[i]).unsqueeze(0).float().to(DEVICE)
            if 'text' in modalities_list: features['text'] = torch.from_numpy(text_feats[i]).unsqueeze(0).float().to(DEVICE)
            if 'visual' in modalities_list: features['visual'] = torch.from_numpy(visual_feats[i]).unsqueeze(0).float().to(DEVICE)

            outputs = model(**features)
            va_values = outputs.squeeze().cpu().numpy()
            video_predictions[video_names_h5[i]].append((segment_ids[i], va_values))
    
    # 4. 동영상별로 .txt 파일 생성
    print("\n[Step 4/4] Generating prediction TXT files...")
    # tqdm의 `bar_format`을 수정하여 기본 진행률 표시줄을 제거하고 설명만 남김
    pbar = tqdm(sorted(list(video_predictions.keys())),
                desc="Writing TXT files", 
                bar_format='{desc}: {n_fmt}/{total_fmt} |{postfix}')
    
    for h5_video_name in pbar:
        # tqdm 설명 업데이트
        pbar.set_description(f"Processing '{h5_video_name}'")
        
        matched_video_file = None
        for vf in video_files:
            if os.path.splitext(vf)[0].startswith(h5_video_name):
                matched_video_file = vf
                break
        
        if not matched_video_file:
            print(f"  - ❌ Warning: No matching video file found for '{h5_video_name}'. Skipping.")
            continue
            
        video_path = os.path.join(VIDEO_DIR, matched_video_file)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  - ❌ Warning: Could not open video file '{video_path}'. Skipping.")
            continue
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # ✨ [출력 추가] 원본 비디오의 프레임 수 출력
        pbar.set_postfix_str(f"Source video has {total_frames} frames.")
        
        segment_preds = sorted(video_predictions[h5_video_name], key=lambda x: x[0])
        if not segment_preds: continue
        pred_map = {seg_id: va_pair for seg_id, va_pair in segment_preds}
        
        last_known_va = segment_preds[0][1] 
        
        output_path = os.path.join(OUTPUT_DIR, f"{h5_video_name}.txt")
        written_frames_count = 0
        with open(output_path, 'w') as f:
            f.write("image_location,valence,arousal\n")
            
            for frame_idx in range(total_frames):
                target_segment_id = frame_idx // 64
                current_va = pred_map.get(target_segment_id, last_known_va)
                last_known_va = current_va
                
                valence, arousal = current_va[0], current_va[1]
                
                image_loc = f"{h5_video_name}/{frame_idx + 1:05d}.jpg"
                f.write(f"{image_loc},{valence:.4f},{arousal:.4f}\n")
                written_frames_count += 1
        
        # ✨ [출력 추가] 최종적으로 작성된 프레임 수와 일치 여부 출력
        match_status = "✅ OK" if written_frames_count == total_frames else f"⚠️ MISMATCH ({written_frames_count} vs {total_frames})"
        pbar.set_postfix_str(f"Source: {total_frames} frames -> Written: {written_frames_count} frames. [{match_status}]")

    print(f"\n\n✅ VA prediction complete! Results are saved in '{OUTPUT_DIR}'.")

if __name__ == '__main__':
    main()