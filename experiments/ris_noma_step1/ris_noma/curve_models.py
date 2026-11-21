"""7g checkpoints bind architecture, nested train prefix and validation-only selection."""
import torch
from .multik_learning_models import build_model, error_metrics, save_torch

FORMAT = "nested_learning_checkpoint_7g_v1"


# Checkpoint 7g ràng buộc trọng số với kiến trúc, prefix bố trí train,
# normalizer và số K đã huấn luyện. Không lưu kết quả test làm tín hiệu chọn mạng.
def checkpoint(model, state, kind, seed, epoch, plan, bounds, validation):
    return {"format": FORMAT, "state_dict": state, "model_kind": kind, "seed": seed,
            "epoch": epoch, "training_config": plan["config"], "bounds": bounds,
            "normalization": plan["normalization"],
            "trained_public_counts": plan["training_public_counts"],
            "training_layout_ids": plan["training_layout_ids"],
            "architecture_name": plan["architecture_name"],
            "validation_errors": validation, "plan_sha256": plan["plan_sha256"],
            "source_7g_stage_sha256": plan["stage_completion_sha256"],
            "source_corpus_plan_sha256": plan["corpus_plan_sha256"],
            "test_used_for_training_or_selection": False, "physical_gradient": False}


# Dựng lại đúng kiến trúc từ checkpoint và nạp state_dict; strict=True
# sẽ từ chối thiếu/thừa trọng số thay vì âm thầm nạp một phần.
def load_model(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (saved.get("format") != FORMAT or saved.get("physical_gradient") is not False
            or saved.get("test_used_for_training_or_selection") is not False):
        raise ValueError("Not a frozen 7g checkpoint")
    model = build_model(saved["model_kind"], saved["bounds"], saved["training_config"])
    model.load_state_dict(saved["state_dict"], strict=True)
    # Chuyển sang chế độ đánh giá; lệnh này không tự thay đổi hoặc huấn luyện trọng số.
    model.eval()
    return model, saved
