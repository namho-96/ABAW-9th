# train_total_v2.py
# ────────────────────────────────────────────────────────────────────────
# ✨ 최종 기능 요약:
# 1. Cross-modal Attention 기반 Fusion 메커니즘
# 2. 'violence', 'va' Task 동적 지원
# 3. K-Fold 각 Fold 종료 시 즉시 Testset 평가 및 로깅
# 4. K-Fold 미사용 시 일반 학습/검증 모드 지원
# 5. 데이터 증강(Noise, Mixup) 기법을 argparse로 개별 제어
# 6. 평가 전용 모드(--eval_only) 지원
# ────────────────────────────────────────────────────────────────────────
import os
import random
import logging
import h5py
from datetime import datetime
from collections import Counter
from typing import Optional, Dict, List, Tuple
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset, ConcatDataset
from torch_ema import ExponentialMovingAverage
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, KFold

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 재현성 설정
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class FlexiblePairedFeatureDataset(Dataset):
    """다양한 모달리티 조합과 학습 Task, 개별 제어 가능한 증강을 지원하는 데이터셋 클래스"""
    def __init__(
        self, hdf5_path: str,
        modalities: List[str],
        task: str = 'violence',
        normalize: bool = True,
        precomputed_stats: Optional[Dict] = None,
        use_noise: bool = False,
        use_mixup: bool = False,
        mixup_alpha: float = 0.2
    ):
        self.hdf5_path = hdf5_path
        self.modalities = modalities
        self.task = task
        self.normalize = normalize
        self.use_noise = use_noise
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.h5_file = None
        self.all_modalities = ['video', 'text', 'image']
        
        if self.task == 'va':
            self.feat_keys = {'video': 'video_features', 'text': 'text_features', 'image': 'image_features'}
        else: # 'violence' task (기본값)
            self.feat_keys = {'video': 'video_features', 'text': 'text_features', 'image': 'visual_features'}

        with h5py.File(hdf5_path, "r") as hf:
            self.dataset_len = len(hf["labels"])
            self.labels = hf["labels"][:]
            self.feat_shapes = {m: hf[self.feat_keys[m]].shape[1:] for m in self.all_modalities if self.feat_keys[m] in hf}
            if normalize and precomputed_stats is None:
                print(f"Calculating normalization stats from {hdf5_path} for {self.modalities}...")
                sample_size = min(5000, self.dataset_len)
                indices = np.random.choice(self.dataset_len, sample_size, replace=False)
                indices.sort()
                stats = {}
                for m in self.modalities:
                    if m in self.feat_keys and self.feat_keys[m] in hf:
                        data = hf[self.feat_keys[m]][indices]
                        # ✨✨✨ 수정된 부분 ✨✨✨
                        med = np.median(data)
                        mad = np.median(np.abs(data - med))
                        stats[m] = {'median': med, 'mad': mad}
                self.stats = stats
                print("Normalization stats ready.")
            elif normalize:
                self.stats = precomputed_stats

    def __len__(self):
        return self.dataset_len

    def _open_hdf5(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, 'r')

    def __getitem__(self, idx: int) -> Dict:
        self._open_hdf5()
        
        if self.task == 'violence': label = torch.tensor(self.labels[idx]).long()
        elif self.task == 'va': label = torch.tensor(self.labels[idx]).float()
        else: raise ValueError(f"Unknown task: {self.task}")

        features = {}
        for m in self.modalities:
            feat = torch.from_numpy(self.h5_file[self.feat_keys[m]][idx]).float()
            if self.normalize and m in self.stats:
                med, mad = self.stats[m]['median'], self.stats[m]['mad']
                feat = (feat - med) / (mad + 1e-8)
            features[m] = feat
        
        if self.use_noise and random.random() < 0.4:
            if 'video' in features: features['video'] += torch.randn_like(features['video']) * 0.02
            if 'text' in features: features['text'] += torch.randn_like(features['text']) * 0.01
            if 'image' in features: features['image'] += torch.randn_like(features['image']) * 0.02
        
        if self.use_mixup and random.random() < 0.5:
            idx2 = random.randint(0, len(self) - 1)
            label2 = torch.tensor(self.labels[idx2]).long() if self.task == 'violence' else torch.tensor(self.labels[idx2]).float()
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            for m in self.modalities:
                feat2 = torch.from_numpy(self.h5_file[self.feat_keys[m]][idx2]).float()
                if self.normalize and m in self.stats:
                    med, mad = self.stats[m]['median'], self.stats[m]['mad']
                    feat2 = (feat2 - med) / (mad + 1e-8)
                features[m] = lam * features[m] + (1 - lam) * feat2
            return {'features': features, 'label': label, 'label2': label2, 'lam': lam}

        return {'features': features, 'label': label, 'label2': label, 'lam': 1.0}

    def __del__(self):
        if self.h5_file:
            self.h5_file.close()


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


class GatedFusion(nn.Module):
    """
    두 피처 x, y를 받아서
      gate = sigmoid(W x + b)
      output = gate * y + (1-gate) * x
    형태로 융합
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gate_fc = nn.Linear(dim, dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x, y: [batch, dim]
        gate = self.sigmoid(self.gate_fc(x))      # [batch, dim]
        return gate * y + (1 - gate) * x         # [batch, dim]

class UnifiedClassifier(nn.Module):
    """유연한 멀티모달 분류기 - 공정한 비교를 위한 동일한 기본 구조"""
    
    def __init__(self, 
                 modalities: List[str], 
                 feat_dims: Dict[str, int], 
                 task: str = 'violence', 
                 hidden_dim: int = 768, 
                 n_cls: int = 2, 
                 dropout: float = 0.3, 
                 num_attn_heads: int = 8,
                 auxiliary_weight: float = 0.3,
                 image_centric: bool = True):
        super().__init__()
        
        self.modalities = modalities
        self.task = task
        self.model_dim = hidden_dim
        self.auxiliary_weight = auxiliary_weight
        self.image_centric = image_centric and 'image' in modalities
        
        # 각 모달리티별 인코더 (동일한 구조)
        enc_hidden_dim = self.model_dim // 2
        self.encoders = nn.ModuleDict()
        
        if 'image' in self.modalities:
            self.encoders['image'] = SequenceEncoder(feat_dims['image'], enc_hidden_dim, dropout=dropout)
        if 'text' in self.modalities:
            self.encoders['text'] = SequenceEncoder(feat_dims['text'], enc_hidden_dim, dropout=dropout)
        if 'video' in self.modalities:
            self.encoders['video'] = SequenceEncoder(feat_dims['video'], enc_hidden_dim, dropout=dropout)
        
        # 모든 모달리티에 동일한 정제 레이어 적용 (더 간단하게)
        self.feature_refiners = nn.ModuleDict()
        for modality in self.modalities:
            self.feature_refiners[modality] = nn.Sequential(
                nn.Linear(self.model_dim, self.model_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        
        # 멀티모달 융합 레이어들 (간단한 구조)
        if len(self.modalities) > 1:
            if self.image_centric:
                # 이미지 중심 융합 - 간단한 어텐션
                self.auxiliary_attention = nn.ModuleDict()
                
                aux_modalities = [m for m in self.modalities if m != 'image']
                for modality in aux_modalities:
                    # 단순한 어텐션만 사용
                    self.auxiliary_attention[modality] = nn.MultiheadAttention(
                        self.model_dim, num_attn_heads, batch_first=True, dropout=dropout
                    )
                
                # 간단한 가중 융합
                if aux_modalities:
                    self.fusion_gate = nn.Sequential(
                        nn.Linear(self.model_dim * (len(aux_modalities) + 1), self.model_dim),
                        nn.Sigmoid()
                    )
                    self.layer_norm = nn.LayerNorm(self.model_dim)
            else:
                # 일반적인 cross-attention 융합 - 더 간단하게
                self.cross_attention = nn.ModuleDict({
                    m: nn.MultiheadAttention(self.model_dim, num_attn_heads, batch_first=True, dropout=dropout) 
                    for m in self.modalities
                })
                self.layer_norms = nn.ModuleDict({
                    m: nn.LayerNorm(self.model_dim) for m in self.modalities
                })
                # 더 간단한 융합
                self.fusion_mlp = nn.Sequential(
                    nn.Linear(self.model_dim * len(self.modalities), self.model_dim),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
        
        # 태스크별 분류 헤드 (동일한 구조)
        if self.task == 'violence':
            self.classifier = nn.Sequential(
                nn.Linear(self.model_dim, self.model_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.model_dim // 2, n_cls)
            )
        elif self.task == 'va':
            self.valence_head = nn.Sequential(
                nn.Linear(self.model_dim, self.model_dim // 2),
                nn.GELU(),
                nn.Linear(self.model_dim // 2, 1)
            )
            self.arousal_head = nn.Sequential(
                nn.Linear(self.model_dim, self.model_dim // 2),
                nn.GELU(),
                nn.Linear(self.model_dim // 2, 1)
            )
        else:
            raise ValueError(f"Unknown task: {self.task}")
    
    def forward(self, **kwargs: torch.Tensor) -> torch.Tensor:
        # 1. 각 모달리티별 특성 추출 및 정제 (모든 모달리티 동일한 처리)
        refined_embeddings = {}
        for modality in self.modalities:
            if modality in kwargs:
                # 기본 인코딩
                raw_embedding = self.encoders[modality](kwargs[modality])
                # 정제 레이어 적용 (모든 모달리티 동일)
                refined_embeddings[modality] = self.feature_refiners[modality](raw_embedding)
        
        # 2. 단일 모달리티인 경우
        if len(self.modalities) == 1:
            final_features = list(refined_embeddings.values())[0]
        
        # 3. 이미지 중심 멀티모달 처리 (간단한 구조)
        elif self.image_centric and 'image' in refined_embeddings:
            image_features = refined_embeddings['image']
            
            # 보조 모달리티들의 특성 추출
            attended_features = [image_features]  # 이미지를 기본으로
            aux_modalities = [m for m in self.modalities if m != 'image' and m in refined_embeddings]
            
            for modality in aux_modalities:
                # 이미지를 Query로, 다른 모달리티를 Key/Value로 사용
                image_query = image_features.unsqueeze(1)
                modality_kv = refined_embeddings[modality].unsqueeze(1)
                
                auxiliary_info, _ = self.auxiliary_attention[modality](
                    image_query, modality_kv, modality_kv
                )
                attended_features.append(auxiliary_info.squeeze(1))
            
            # 간단한 가중 융합
            if len(attended_features) > 1:
                # 모든 특성을 concat하여 가중치 계산
                all_features = torch.cat(attended_features, dim=1)
                fusion_weight = self.fusion_gate(all_features)
                
                # 기본 이미지 특성과 융합
                auxiliary_sum = sum(attended_features[1:]) if len(attended_features) > 1 else 0
                final_features = self.layer_norm(
                    image_features + fusion_weight * auxiliary_sum
                )
            else:
                final_features = image_features
        
        # 4. 일반적인 멀티모달 처리
        else:
            ordered_embeddings = [refined_embeddings[m] for m in self.modalities if m in refined_embeddings]
            
            if len(ordered_embeddings) > 1:
                # 기존 cross-attention 방식
                all_embs = torch.stack(ordered_embeddings, dim=1)
                refined_embs = []
                
                for i, modality in enumerate([m for m in self.modalities if m in refined_embeddings]):
                    query = ordered_embeddings[i].unsqueeze(1)
                    context, _ = self.cross_attention[modality](query, all_embs, all_embs)
                    refined_emb = self.layer_norms[modality](
                        ordered_embeddings[i] + context.squeeze(1)
                    )
                    refined_embs.append(refined_emb)
                
                final_features = self.fusion_mlp(torch.cat(refined_embs, dim=1))
            else:
                final_features = ordered_embeddings[0]
        
        # 5. 태스크별 예측
        if self.task == 'violence':
            return self.classifier(final_features)
        elif self.task == 'va':
            valence = self.valence_head(final_features)
            arousal = self.arousal_head(final_features)
            return torch.cat([valence, arousal], dim=1)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__(); self.alpha, self.gamma, self.reduction = alpha, gamma, reduction
    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none'); pt = torch.exp(-ce); fl = self.alpha * (1 - pt) ** self.gamma * ce
        if self.reduction == 'mean': return fl.mean()
        if self.reduction == 'sum': return fl.sum()
        return fl

def mixup_loss(criterion, pred, y_a, y_b, lam):
    loss_a = criterion(pred, y_a); loss_b = criterion(pred, y_b)
    lam_device = lam.to(loss_a.device)
    while len(lam_device.shape) < len(loss_a.shape): lam_device = lam_device.unsqueeze(-1)
    return lam_device * loss_a + (1. - lam_device) * loss_b

def compute_ccc(y_true, y_pred):
    y_true_mean, y_pred_mean = torch.mean(y_true), torch.mean(y_pred)
    vx, vy = y_true - y_true_mean, y_pred - y_pred_mean
    rho = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)) + 1e-8)
    std_true, std_pred = torch.std(y_true), torch.std(y_pred)
    ccc = (2 * rho * std_true * std_pred) / (std_true**2 + std_pred**2 + (y_true_mean - y_pred_mean)**2 + 1e-8)
    return ccc

def train_model_unified(model, train_loader, valid_loader, device, out_dir, logger, task='violence', epochs=50, max_lr=3e-4, fold_num=None):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=1e-2)
    scheduler = OneCycleLR(optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=len(train_loader), pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler(); ema = ExponentialMovingAverage(model.parameters(), decay=0.9995)

    if task == 'violence':
        criterion, val_criterion = FocalLoss(reduction='none').to(device), FocalLoss(reduction='mean').to(device)
        best_metric, metric_name = 0.0, 'F1'
    elif task == 'va':
        criterion, val_criterion = nn.MSELoss(reduction='none').to(device), nn.MSELoss(reduction='mean').to(device)
        best_metric, metric_name = 0.0, 'CCC'
    else: raise ValueError(f"Unknown task: {task}")
    
    wait, patience = 0, 10
    for e in range(1, epochs + 1):
        model.train()
        tloss, all_preds, all_labels = 0.0, [], []
        pbar_desc = f"[Train Fold {fold_num} E{e:02d}]" if fold_num else f"[Train E{e:02d}]"
        for batch in tqdm(train_loader, desc=pbar_desc):
            features = {k: v.to(device) for k, v in batch['features'].items()}
            la, lb, lam = batch['label'].to(device), batch['label2'].to(device), batch['lam'].to(device).float()
            with torch.cuda.amp.autocast():
                outputs = model(**features)
                loss = mixup_loss(criterion, outputs, la, lb, lam).mean()
            optimizer.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step(); ema.update()
            tloss += loss.item() * la.size(0)
            if task == 'violence': all_preds.extend(outputs.argmax(1).cpu().numpy()); all_labels.extend(la.cpu().numpy())
        log_msg = f"Epoch {e:02d} Train     loss={tloss/len(train_loader.dataset):.4f}"
        if task == 'violence': log_msg += f" f1={f1_score(all_labels, all_preds, average='macro', zero_division=0):.4f}"
        logger.info(log_msg + f" LR={scheduler.get_last_lr()[0]:.2e}")

        ema.store(); ema.copy_to(); model.eval()
        vloss, vp, vl = 0.0, [], []
        pbar_desc_val = f"[Valid Fold {fold_num} E{e:02d}]" if fold_num else f"[Valid E{e:02d}]"
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=pbar_desc_val):
                features = {k: v.to(device) for k, v in batch['features'].items()}
                true_labels = batch['label'].to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(**features)
                    loss = val_criterion(outputs, true_labels)
                vloss += loss.item() * true_labels.size(0)
                vp.append(outputs.cpu()); vl.append(true_labels.cpu())
        ema.restore()
        val_preds_all, val_labels_all = torch.cat(vp, dim=0), torch.cat(vl, dim=0)
        log_msg_val, current_metric = f"Epoch {e:02d} Valid loss={vloss/len(valid_loader.dataset):.4f}", 0
        if task == 'violence':
            preds_class, labels_class = val_preds_all.argmax(1).numpy(), val_labels_all.numpy()
            val_f1 = f1_score(labels_class, preds_class, average='macro', zero_division=0)
            log_msg_val += f" f1={val_f1:.4f}"; current_metric = val_f1
        elif task == 'va':
            ccc_v, ccc_a = compute_ccc(val_labels_all[:, 0], val_preds_all[:, 0]).item(), compute_ccc(val_labels_all[:, 1], val_preds_all[:, 1]).item()
            avg_ccc = (ccc_v + ccc_a) / 2
            log_msg_val += f" ccc_v={ccc_v:.4f} ccc_a={ccc_a:.4f} avg_ccc={avg_ccc:.4f}"; current_metric = avg_ccc
        logger.info(log_msg_val)

        if current_metric > best_metric:
            best_metric, wait = current_metric, 0
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            mode_str = "_".join(model.modalities if not isinstance(model, nn.DataParallel) else model.module.modalities)
            filename = f"best_model_fold{fold_num}_{mode_str}.pth" if fold_num else f"best_model_{mode_str}.pth"
            path = os.path.join(out_dir, filename)
            torch.save({'model_state_dict': model_state, 'epoch': e, f'best_{metric_name}': best_metric}, path)
            logger.info(f"🔥 Best model saved: {path} ({metric_name}={best_metric:.4f})")
            if task == 'violence': logger.info(f"Classification Report:\n{classification_report(labels_class, preds_class, digits=4, zero_division=0)}")
        else:
            wait += 1
            if wait >= patience: logger.info(f"⏹️ Early stopping (patience={patience})"); break
    return best_metric

def setup_logger(log_dir: str):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(); fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    ch.setFormatter(fmt); fh.setFormatter(fmt)
    logger.addHandler(ch); logger.addHandler(fh)
    return logger

def collate_fn(batch: List[Dict]) -> Dict:
    collated_batch = {}
    feature_keys = batch[0]['features'].keys()
    collated_features = {key: torch.stack([d['features'][key] for d in batch]) for key in feature_keys}
    collated_batch['features'] = collated_features
    collated_batch['label'] = torch.stack([d['label'] for d in batch])
    collated_batch['label2'] = torch.stack([d['label2'] for d in batch])
    collated_batch['lam'] = torch.tensor([d['lam'] for d in batch])
    return collated_batch

def evaluate_on_test_set(model: nn.Module, model_path: str, test_loader: DataLoader, device: str, logger: logging.Logger, task: str, fold_num: Optional[int]) -> Tuple[float, torch.Tensor, torch.Tensor]:
    fold_str = f"Fold {fold_num}" if fold_num is not None else "Model"
    logger.info(f"--- Loading best model for {fold_str} from {model_path} for Test Evaluation ---")
    try:
        state_dict = torch.load(model_path, map_location=device)['model_state_dict']
        if all(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}. Error: {e}")
        return 0.0, None, None

    fold_outputs, fold_labels = [], []
    with torch.no_grad():
        pbar_desc = f"[Test Fold {fold_num}]" if fold_num is not None else "[Test]"
        for batch in tqdm(test_loader, desc=pbar_desc):
            features = {k: v.to(device) for k, v in batch['features'].items()}
            labels = batch['label']
            with torch.cuda.amp.autocast():
                outputs = model(**features)
            fold_outputs.append(outputs.cpu())
            fold_labels.append(labels)

    all_outputs = torch.cat(fold_outputs, dim=0)
    all_labels = torch.cat(fold_labels, dim=0)
    
    metric = 0.0
    if task == 'violence':
        preds_class = all_outputs.argmax(1).numpy()
        labels_class = all_labels.numpy()
        metric = f1_score(labels_class, preds_class, average='macro', zero_division=0)
        logger.info(f"🧪 {fold_str} | Test F1 Score: {metric:.4f}")
        logger.info(f"{fold_str} Test Classification Report:\n{classification_report(labels_class, preds_class, digits=4, zero_division=0)}")
    elif task == 'va':
        ccc_v = compute_ccc(all_labels[:, 0], all_outputs[:, 0]).item()
        ccc_a = compute_ccc(all_labels[:, 1], all_outputs[:, 1]).item()
        metric = (ccc_v + ccc_a) / 2
        logger.info(f"🧪 {fold_str} | Test CCC: Valence={ccc_v:.4f}, Arousal={ccc_a:.4f}, Average={metric:.4f}")

    return metric, all_outputs, all_labels


def main():
    parser = argparse.ArgumentParser(description='Flexible Multi-Modal Classifier/Regressor Training')
    # --- 기본 인자 ---
    parser.add_argument('--task', type=str, default='violence', choices=['violence', 'va'], help="Task: 'violence' or 'va'.")
    parser.add_argument('--mode', type=str, default='image_video_text', help="Modalities separated by '_' (e.g., 'image_text').")
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--run_dir', type=str, default=None, help="Directory of a previous run to use for --eval_only.")
    
    # --- 학습 하이퍼파라미터 ---
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--num_workers', type=int, default=8)

    # --- 학습 방식 제어 ---
    parser.add_argument('--kfold', action='store_true', help="Enable K-Fold cross-validation.")
    parser.add_argument('--n_splits', type=int, default=5, help="Number of splits for K-Fold.")
    # ✨✨✨ 수정된 부분: 특정 폴드만 학습하기 위한 인자 추가 ✨✨✨
    parser.add_argument('--target_fold', type=int, default=None, help="Specify a single fold to train (e.g., 1, 2, ...).")
    parser.add_argument('--eval_only', action='store_true', help="Run evaluation only on existing models from --run_dir.")

    # --- 데이터 증강(Augmentation) 제어 인자 ---
    parser.add_argument('--use_noise', action='store_true', help="Enable noise augmentation.")
    parser.add_argument('--use_mixup', action='store_true', help="Enable MixUp augmentation.")
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help="Alpha parameter for MixUp's beta distribution.")
    
    args = parser.parse_args()

    # --- 출력 디렉토리 및 로거 설정 ---
    if args.eval_only:
        if not args.run_dir: raise ValueError("For --eval_only mode, you must specify the run directory with --run_dir.")
        if not os.path.isdir(args.run_dir): raise FileNotFoundError(f"The specified run directory does not exist: {args.run_dir}")
        out_root = args.run_dir
        logger = setup_logger(out_root)
        logger.info("="*50 + f"\n🚀 Running in EVALUATION-ONLY mode on directory: {out_root}\n" + "="*50)
    else:
        modalities_str = sorted(args.mode.split('_'))
        if 'only' in modalities_str: modalities_str.remove('only')
        run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.task}_{'_'.join(modalities_str)}"
        if args.kfold: run_name += f"_{args.n_splits}fold"
        out_root = os.path.join(args.output_dir, run_name)
        os.makedirs(out_root, exist_ok=True)
        logger = setup_logger(out_root)
        logger.info(f"Run arguments: {args}")

    modalities = sorted(args.mode.split('_'))
    if 'only' in modalities: modalities.remove('only')
    logger.info(f"Selected Task: {args.task.upper()}")
    logger.info(f"Selected modalities: {modalities}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # --- 데이터셋 경로 설정 ---
    if args.task == 'violence':
        train_h5_path = '../../dataset/dvd/DVD_Competition/Training/preprocessed_hdf5/training_data_triplet.hdf5'
        valid_h5_path = '../../dataset/dvd/DVD_Competition/Validation/preprocessed_hdf5/training_data_triplet.hdf5'
    elif args.task == 'va':
        train_h5_path = '/mnt/data/abaw/dataset/affectnet/features/train/preprocessed_hdf5/va_train_data.hdf5'
        valid_h5_path = '/mnt/data/abaw/dataset/affectnet/features/valid/preprocessed_hdf5/va_validation_data.hdf5'
    else:
        raise ValueError(f"Task '{args.task}' has no defined HDF5 paths.")
    
    logger.info(f"Train HDF5: {train_h5_path}")
    logger.info(f"Validation HDF5: {valid_h5_path}")

    # --- 데이터셋 및 피처 정보 로드 ---
    base_train_ds = FlexiblePairedFeatureDataset(
        train_h5_path, modalities=modalities, task=args.task, normalize=True
    )
    normalization_stats = base_train_ds.stats
    feat_dims = {m: base_train_ds.feat_shapes[m][-1] for m in modalities}
    logger.info(f"Feature dimensions for model input: {feat_dims}")

    # K-Fold를 위해 Train/Validation 데이터셋을 통합
    if args.kfold:
        logger.info("Combining Train and Validation datasets for K-Fold Cross-Validation.")
        
        train_ds_no_aug = base_train_ds 
        valid_ds_no_aug = FlexiblePairedFeatureDataset(
            valid_h5_path, modalities=modalities, task=args.task, normalize=True, 
            precomputed_stats=normalization_stats
        )
        combined_ds_for_valid = ConcatDataset([train_ds_no_aug, valid_ds_no_aug])

        train_ds_with_aug = FlexiblePairedFeatureDataset(
            train_h5_path, modalities, task=args.task, normalize=True, precomputed_stats=normalization_stats,
            use_noise=args.use_noise, use_mixup=args.use_mixup, mixup_alpha=args.mixup_alpha
        )
        valid_ds_with_aug = FlexiblePairedFeatureDataset(
            valid_h5_path, modalities, task=args.task, normalize=True, precomputed_stats=normalization_stats,
            use_noise=args.use_noise, use_mixup=args.use_mixup, mixup_alpha=args.mixup_alpha
        )
        combined_ds_for_train = ConcatDataset([train_ds_with_aug, valid_ds_with_aug])

        combined_labels = np.concatenate([train_ds_no_aug.labels, valid_ds_no_aug.labels])
        n_classes = len(np.unique(combined_labels)) if args.task == 'violence' else None
    else:
        # K-Fold를 사용하지 않을 경우
        with h5py.File(train_h5_path, 'r') as hf: labels_all = hf['labels'][:]
        n_classes = len(Counter(labels_all)) if args.task == 'violence' else None

        test_loader = DataLoader(
            FlexiblePairedFeatureDataset(valid_h5_path, modalities=modalities, task=args.task, normalize=True, precomputed_stats=normalization_stats),
            batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
        )

    if args.kfold:
        if not args.eval_only:
            logger.info(f"🔄 Starting {args.n_splits}-Fold Cross Validation Training on combined dataset ({len(combined_labels)} samples)...")
            if args.target_fold:
                logger.info(f"🎯 Target fold specified: FOLD {args.target_fold}")
            logger.info(f"Augmentation settings: Noise={args.use_noise}, MixUp={args.use_mixup}, MixUp Alpha={args.mixup_alpha}")
            
            split_iterator = None
            if args.task == 'va':
                kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=SEED)
                split_iterator = kf.split(np.arange(len(combined_labels)))
            else: # violence
                kf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=SEED)
                split_iterator = kf.split(np.arange(len(combined_labels)), combined_labels)

            fold_val_scores = []
            
            for fold, (tr_idx, val_idx) in enumerate(split_iterator, 1):
                # ✨✨✨ 수정된 부분: 지정된 폴드가 아니면 건너뛰기 ✨✨✨
                if args.target_fold is not None and fold != args.target_fold:
                    continue

                logger.info("\n" + "-"*60 + f"\n📁 Fold {fold}/{args.n_splits} Training Start\n" + "-"*60)
                
                train_sub = Subset(combined_ds_for_train, tr_idx)
                valid_sub = Subset(combined_ds_for_valid, val_idx)
                
                logger.info(f"Fold {fold}: Train samples={len(train_sub)}, Validation samples={len(valid_sub)}")

                sampler = None
                if args.task == 'violence':
                    train_fold_labels = combined_labels[tr_idx]
                    counts = Counter(train_fold_labels)
                    weights_map = {c: len(train_fold_labels) / (n_classes * cnt) for c, cnt in counts.items()}
                    ws = [weights_map[label] for label in train_fold_labels]
                    sampler = WeightedRandomSampler(ws, num_samples=len(ws), replacement=True)

                train_loader = DataLoader(train_sub, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None), num_workers=args.num_workers, pin_memory=True, drop_last=True, collate_fn=collate_fn)
                valid_loader = DataLoader(valid_sub, batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
                
                model = UnifiedClassifier(modalities=modalities, feat_dims=feat_dims, task=args.task, n_cls=n_classes)
                if torch.cuda.device_count() > 1:
                    logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
                    model = nn.DataParallel(model)

                fold_dir = os.path.join(out_root, f"fold_{fold}")
                os.makedirs(fold_dir, exist_ok=True)
                
                val_metric = train_model_unified(model, train_loader, valid_loader, device, fold_dir, logger, task=args.task, epochs=args.epochs, max_lr=args.lr, fold_num=fold)
                fold_val_scores.append(val_metric)

            metric_name = 'F1' if args.task == 'violence' else 'CCC'
            # ✨✨✨ 수정된 부분: 전체 폴드 실행 시에만 최종 요약 출력 ✨✨✨
            if args.target_fold is None:
                logger.info("\n" + "="*60 + f"\n K-FOLD FINAL SUMMARY \n" + "="*60)
                logger.info(f"Validation scores per fold: {[f'{s:.4f}' for s in fold_val_scores]}")
                logger.info(f"🏆 Avg Validation {metric_name} across {args.n_splits} folds: {np.mean(fold_val_scores):.4f} ± {np.std(fold_val_scores):.4f}")
            else:
                logger.info("\n" + "="*60 + f"\n ✅ Finished training for target fold: {args.target_fold} \n" + "="*60)
                logger.info(f"Validation {metric_name} for Fold {args.target_fold}: {fold_val_scores[0]:.4f}")

        else: # K-Fold & Eval-Only
            logger.info("🧪 K-Fold Evaluation-Only Mode...")
            if args.target_fold:
                logger.info(f"🎯 Evaluating target fold: FOLD {args.target_fold}")

            test_loader = DataLoader(
                FlexiblePairedFeatureDataset(valid_h5_path, modalities, task=args.task, normalize=True, precomputed_stats=normalization_stats),
                batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn
            )

            all_fold_outputs, test_labels_list, fold_test_scores = [], None, []
            model_for_eval = UnifiedClassifier(modalities=modalities, feat_dims=feat_dims, task=args.task, n_cls=n_classes)
            
            for fold in range(1, args.n_splits + 1):
                # ✨✨✨ 수정된 부분: 지정된 폴드가 아니면 평가도 건너뛰기 ✨✨✨
                if args.target_fold is not None and fold != args.target_fold:
                    continue

                fold_dir = os.path.join(out_root, f"fold_{fold}")
                model_path = os.path.join(fold_dir, f"best_model_fold{fold}_{'_'.join(modalities)}.pth")
                if not os.path.exists(model_path):
                    logger.warning(f"Model for fold {fold} not found at {model_path}, skipping.")
                    continue
                test_metric, fold_preds, fold_labels = evaluate_on_test_set(
                    model=model_for_eval, model_path=model_path, test_loader=test_loader,
                    device=device, logger=logger, task=args.task, fold_num=fold
                )
                if fold_preds is not None:
                    fold_test_scores.append(test_metric)
                    all_fold_outputs.append(F.softmax(fold_preds, dim=1) if args.task == 'violence' else fold_preds)
                    if test_labels_list is None: test_labels_list = fold_labels
            
            # ✨✨✨ 수정된 부분: 전체 폴드 평가 시에만 앙상블 수행 ✨✨✨
            if all_fold_outputs and args.target_fold is None:
                logger.info("\n✨ Performing Ensemble on Test Set...")
                ensembled_outputs = torch.stack(all_fold_outputs, dim=0).mean(dim=0)
                if args.task == 'violence':
                    ensemble_preds = ensembled_outputs.argmax(1).numpy()
                    ensemble_metric = f1_score(test_labels_list.numpy(), ensemble_preds, average='macro', zero_division=0)
                    logger.info(f"🏆 FINAL ENSEMBLE F1 SCORE: {ensemble_metric:.4f}")
                    logger.info(f"Ensemble Classification Report:\n{classification_report(test_labels_list.numpy(), ensemble_preds, digits=4, zero_division=0)}")
                elif args.task == 'va':
                    ccc_v = compute_ccc(test_labels_list[:, 0], ensembled_outputs[:, 0]).item()
                    ccc_a = compute_ccc(test_labels_list[:, 1], ensembled_outputs[:, 1]).item()
                    avg_ccc = (ccc_v + ccc_a) / 2
                    logger.info(f"🏆 FINAL ENSEMBLE CCC: Valence={ccc_v:.4f}, Arousal={ccc_a:.4f}, Average={avg_ccc:.4f}")
    else:
        # --- No K-Fold ---
        if args.eval_only:
            logger.info(f"🧪 Evaluating a single model from {args.run_dir}")
            model = UnifiedClassifier(modalities=modalities, feat_dims=feat_dims, task=args.task, n_cls=n_classes)
            model_path = os.path.join(args.run_dir, f"best_model_{'_'.join(modalities)}.pth")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Could not find model file at {model_path}")
            evaluate_on_test_set(model, model_path, test_loader, device, logger, args.task, fold_num=None)
        else:
            logger.info("🚀 Starting training (No K-Fold)...")
            logger.info(f"Augmentation settings: Noise={args.use_noise}, MixUp={args.use_mixup}, MixUp Alpha={args.mixup_alpha}")
            
            train_ds_aug = FlexiblePairedFeatureDataset(
                train_h5_path, modalities=modalities, task=args.task, normalize=True, precomputed_stats=normalization_stats,
                use_noise=args.use_noise, use_mixup=args.use_mixup, mixup_alpha=args.mixup_alpha
            )
            
            sampler = None
            if args.task == 'violence':
                counts = Counter(labels_all)
                weights_map = {c: len(labels_all) / (n_classes * cnt) for c, cnt in counts.items()}
                ws = [weights_map[label] for label in labels_all]
                sampler = WeightedRandomSampler(ws, num_samples=len(ws), replacement=True)

            train_loader = DataLoader(
                train_ds_aug, batch_size=args.batch_size, sampler=sampler, 
                shuffle=(sampler is None), num_workers=args.num_workers, 
                pin_memory=True, drop_last=True, collate_fn=collate_fn
            )
            
            valid_loader = test_loader
            logger.info(f"Train set: {len(train_loader.dataset)} samples. Validation set: {len(valid_loader.dataset)} samples.")

            model = UnifiedClassifier(modalities=modalities, feat_dims=feat_dims, task=args.task, n_cls=n_classes)
            if torch.cuda.device_count() > 1:
                logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
                model = nn.DataParallel(model)

            logger.info("--- Training model on full training data, validating on test data ---")
            best_metric = train_model_unified(
                model=model, train_loader=train_loader, valid_loader=valid_loader,
                device=device, out_dir=out_root, logger=logger, task=args.task,
                epochs=args.epochs, max_lr=args.lr, fold_num=None
            )
            metric_name = 'F1' if args.task == 'violence' else 'CCC'
            logger.info(f"🏆 Training finished. Best score on test set ({metric_name}): {best_metric:.4f}")


if __name__ == "__main__":
    main()
