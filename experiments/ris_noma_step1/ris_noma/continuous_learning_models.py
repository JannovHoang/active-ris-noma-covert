"""7f keeps the 7c architectures and weighted loss; checkpoints bind to 7e provenance."""
import torch
from .multik_learning_models import build_model,error_metrics,save_torch

CHECKPOINT_FORMAT='continuous_learning_checkpoint_7f_v1'


def checkpoint(model,state,kind,seed,epoch,plan,bounds,validation):
    return {'format':CHECKPOINT_FORMAT,'state_dict':state,'model_kind':kind,'seed':seed,'epoch':epoch,
        'training_config':plan['config'],'bounds':bounds,'normalization':plan['normalization'],
        'trained_public_counts':plan['training_public_counts'],'validation_errors':validation,
        'test_used_for_training_or_selection':False,'physical_gradient':False,
        'plan_sha256':plan['plan_sha256'],'source_7e_completion_sha256':plan['dataset_sha256']['completion.json'],
        'physics_source_sha256':plan['dataset_sha256']['source_physics_plan.json'],
        'input_domain':{'power_dbm_range':plan['power_dbm_range'],'dep_range':plan['dep_range'],
                        'profile':plan['profile'],'geometry_scope':plan['geometry_scope']}}


def load_model(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if (saved.get('format')!=CHECKPOINT_FORMAT or saved.get('test_used_for_training_or_selection') is not False
            or saved.get('physical_gradient') is not False):raise ValueError('Not a supported frozen 7f checkpoint')
    model=build_model(saved['model_kind'],saved['bounds'],saved['training_config'])
    model.load_state_dict(saved['state_dict'],strict=True);model.eval()
    return model,saved
