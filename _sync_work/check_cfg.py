import sys
sys.path.insert(0, "/root/Workspace/xy/DiT")
import src.train.train as T

CFG = sys.argv[1]
CAPTURED = {}


def fake_main(args):
    CAPTURED.update(vars(args))
    print(f"\n===== RESOLVED ARGS = {CFG} =====")
    keys = ["experiment_name", "results_dir", "model", "condition_fusion",
            "glyph_vec_cond", "glyph_vec_dim", "glyph_vec_pool",
            "no_char_cond", "use_char_cond",
            "cond_drop_all_prob", "cond_drop_one_prob", "cond_drop_which_glyph_prob",
            "glyph_drop_prob", "glyph_inject_mode", "glyph_inject_layers",
            "data_csv", "global_batch_size", "lr", "max_steps", "w_repa",
            "resume_full"]
    for k in keys:
        print(f"  {k:32s} = {CAPTURED.get(k, '<ABSENT>')}")


T.main = fake_main
# 注意: main_from_cli 里 `parser.parse_known_args()` 不带参数 -> 读 sys.argv,
# 而不是传给 main_from_cli 的 argv。必须通过 sys.argv 指定 --config。
sys.argv = ["train.py", "--config", CFG]
T.main_from_cli()

assert CAPTURED.get("condition_fusion") in ("factorized_cat", "factorized_add"), \
    f"condition_fusion NOT applied! got {CAPTURED.get('condition_fusion')}"
if "cat" in CFG:
    assert CAPTURED.get("glyph_vec_cond") is True, "glyph_vec_cond NOT applied!"
print(f">>> OK: {CFG} resolved, condition_fusion={CAPTURED['condition_fusion']}, "
      f"glyph_vec_cond={CAPTURED.get('glyph_vec_cond')}")
