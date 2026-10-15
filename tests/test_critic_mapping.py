import torch
from src.loss.critic_loss import CriticLoss


def test_remap_not_csv_order_and_script_ids(tmp_path):
    path = tmp_path / 'train.csv'
    path.write_text('calligrapher_id,calligrapher,script_id,script,aug\n'
                    '9001,赵孟頫,3,行,\n956,颜真卿,3,行,\n', encoding='utf-8')
    critic = CriticLoss.__new__(CriticLoss)
    torch.nn.Module.__init__(critic)
    critic.slots = ['赵孟頫·行', '颜真卿·行', 'std·行']
    lut = critic.build_id2slot(str(path), {9001: 9, 956: 8}, {3: 1})
    assert lut == {(9, 1): '赵孟頫·行', (8, 1): '颜真卿·行'}
    # Defaults match sorted raw IDs and retain raw script ID, not appearance order.
    assert critic.build_id2slot(str(path)) == {(1, 3): '赵孟頫·行', (0, 3): '颜真卿·行'}
