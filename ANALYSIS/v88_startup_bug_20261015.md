# Bug 分析报告

## 一、错误现象

v88 初次启动误进入 StrongStyle，因同时启用书家表训练而报错退出，未开始正式训练。
复现：生成 `strong_style_mode=""` 的配置后启动训练。错误堆栈和配置齐全，输入校验通过。该错误来自本次新增配置，未提交。

## 二、根因分析

自顶向下定位到 `src/train/train.py`：启用判断为 `strong_style_mode != 'off'`，不是字符串真值。空字符串因此被视为启用。仅本次 v88 配置受影响。

## 三、最小化Diff格式修复补丁

```diff
-    strong_style_mode='', w_critic_writer=0., w_critic_away=0.,
+    strong_style_mode='off', w_critic_writer=0., w_critic_away=0.,
```

修复位于 `tools/analysis/prepare_v88.py`，不改已有训练模式语义。

## 四、防回归测试用例

`tests/test_writer_glyph.py::test_training_and_eval_share_new_architecture_config` 同时检查关闭后训练、无 warm-start、训练与评测构造参数一致：

```python
assert args.strong_style_mode == 'off'
assert args.w_critic_writer == args.w_critic_away == 0
assert not args.warm_start_from
```

GPU 启动检查也断言模式必须为 `off`。

## 五、预防建议

禁用模式使用训练入口约定的枚举值，不使用空字符串替代。GPU 检查复用正式训练的 compile 路径，否则显存估计失真。正式训练日志记录新分支门控与 P/Q 梯度，区分“模块存在”和“模块参与学习”。
