import sys
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.deform_skel import DeformSkel

def test_stroke_mod():
    print("[1] 测试 DeformSkel 构造与 step0 恒等性...")
    B, C, H, W = 4, 4, 32, 32
    cond_dim = 128
    g = torch.randn(B, C, H, W)
    style = torch.randn(B, cond_dim)
    dt = torch.rand(B, 1, H, W) # 0=笔画, 1=背景

    # 1. zero-init 恒等性测试
    m_no_mod = DeformSkel(cond_dim=cond_dim, ch=C, grid=H, width=32, stroke_mod=0)
    m_mod = DeformSkel(cond_dim=cond_dim, ch=C, grid=H, width=32, stroke_mod=1, gate_radius=0.25)

    # 同步基础权重
    m_mod.load_state_dict(m_no_mod.state_dict(), strict=False)

    out_no_mod = m_no_mod(g, style, dt=dt)
    out_mod = m_mod(g, style, dt=dt)

    diff = (out_mod - out_no_mod).abs().max().item()
    print(f"    step0 输出最大差异 (应为 0.0): {diff:.8f}")
    assert diff < 1e-6, f"zero-init 不恒等: {diff}"

    # 2. 空白背景保护测试
    print("[2] 测试空白背景绝对不造墨门控保护...")
    # 手动给 stroke_conv 赋非零随机权重
    nn.init.normal_(m_mod.stroke_conv.weight, std=1.0)
    nn.init.normal_(m_mod.stroke_conv.bias, std=1.0)

    # 构造一块完全空白的区域 (dt=1.0)
    dt_far = torch.ones(B, 1, H, W) # 全是背景
    out_far = m_mod(g, style, dt=dt_far)
    out_base = m_no_mod(g, style, dt=dt_far)
    diff_far = (out_far - out_base).abs().max().item()
    print(f"    远距离背景区域附加墨量 (应为 0.0): {diff_far:.8f}")
    assert diff_far == 0.0, f"空白背景泄漏墨量! {diff_far}"

    # 3. 笔画区域确实生效
    print("[3] 测试笔画区域调制生效...")
    dt_near = torch.zeros(B, 1, H, W) # 笔画中心
    out_near = m_mod(g, style, dt=dt_near)
    diff_near = (out_near - out_base).abs().max().item()
    print(f"    笔画中心区域附加调制幅值: {diff_near:.4f}")
    assert diff_near > 0.01, f"笔画区域未产生调制! {diff_near}"

    # 4. 反向传播梯度回传测试
    print("[4] 测试梯度回传...")
    loss = out_near.sum()
    loss.backward()
    assert m_mod.stroke_conv.weight.grad is not None, "stroke_conv 缺少梯度!"
    assert m_mod.stroke_style.weight.grad is not None, "stroke_style 缺少梯度!"
    print("    stroke_conv 梯度 norm:", m_mod.stroke_conv.weight.grad.norm().item())
    print("    stroke_style 梯度 norm:", m_mod.stroke_style.weight.grad.norm().item())

    # 5. regularizers 测试
    print("[5] 测试 regularizers...")
    regs = m_mod.regularizers()
    assert 'tv_stroke' in regs, "缺少 tv_stroke 正则项!"
    print("    tv_stroke:", regs['tv_stroke'].item())

    print("[PASS] 所有测试通过!")

if __name__ == "__main__":
    test_stroke_mod()
