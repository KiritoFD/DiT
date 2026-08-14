import os
import torch
import torch.nn as nn
from eval_models import MultiTaskCalligraphyEvalNet

class CalligraphyEvaluator:
    """
    Evaluator module for generated calligraphy images.
    Calculates Character Recognition Accuracy (OCR), Calligrapher Style Accuracy,
    Script Accuracy, and extracts visual feature vectors for FID/PR.
    """
    def __init__(
        self,
        checkpoint_path="pretrained_models/eval_classifier.pt",
        backbone="dinov2_vits14",
        device=None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model = MultiTaskCalligraphyEvalNet(
            num_characters=7765,
            num_calligraphers=2243,
            num_scripts=12,
            backbone=backbone,
            pretrained=False
        ).to(self.device)
        
        if os.path.exists(checkpoint_path):
            print(f"[CalligraphyEvaluator] Loading evaluation model weights from {checkpoint_path}...")
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            self.model.load_state_dict(state_dict, strict=False)
        else:
            print(f"[CalligraphyEvaluator] Warning: Checkpoint {checkpoint_path} not found. Using initialized model.")
            
        self.model.eval()

    @torch.no_grad()
    def evaluate_batch(self, generated_imgs, target_chars, target_callig, target_scripts):
        """
        generated_imgs: (B, 3, 256, 256) normalized in [-1, 1] or [0, 1]
        target_chars: (B,) tensor of target character IDs
        target_callig: (B,) tensor of target calligrapher IDs
        target_scripts: (B,) tensor of target script IDs
        
        returns dict of accuracy metrics:
        - ocr_acc: float
        - callig_acc: float
        - script_acc: float
        """
        imgs = generated_imgs.to(self.device)
        
        with torch.amp.autocast('cuda'):
            logits_char, logits_callig, logits_script = self.model(imgs)
            
        pred_char = logits_char.argmax(dim=-1)
        pred_callig = logits_callig.argmax(dim=-1)
        pred_script = logits_script.argmax(dim=-1)
        
        ocr_correct = (pred_char == target_chars.to(self.device)).float().mean().item() * 100.0
        callig_correct = (pred_callig == target_callig.to(self.device)).float().mean().item() * 100.0
        script_correct = (pred_script == target_scripts.to(self.device)).float().mean().item() * 100.0
        
        return {
            "ocr_acc": ocr_correct,
            "callig_acc": callig_correct,
            "script_acc": script_correct
        }

    @torch.no_grad()
    def get_features(self, imgs):
        """
        Extracts 512-dim visual features for FID or Precision/Recall metrics.
        """
        imgs = imgs.to(self.device)
        with torch.amp.autocast('cuda'):
            feats = self.model.extract_features(imgs)
        return feats
