from unittest.mock import patch
from torch import nn
from src.eval.eval_models import MultiTaskCalligraphyEvalNet


def test_full_classifier_load_uses_cached_code_without_downloading_pretrained_weights(tmp_path):
    repo=tmp_path/'facebookresearch_dinov2_main';repo.mkdir()
    (repo/'hubconf.py').touch()
    fake=nn.Identity();fake.embed_dim=384
    with patch('torch.hub.get_dir',return_value=str(tmp_path)),patch('torch.hub.load',return_value=fake) as load:
        model=MultiTaskCalligraphyEvalNet(3,4,2,pretrained=False)
    load.assert_called_once_with(str(repo),'dinov2_vits14',source='local',pretrained=False)
    assert model.char_head[-1].out_features==3
