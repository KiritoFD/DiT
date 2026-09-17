"""决定性测试: 把三个新键写进 config, 确认它们真的落进 args (不是被静默丢弃)。"""
import importlib.util, json, os, sys, tempfile

sys.path.insert(0, "/root/Workspace/xy/DiT")

src = "src/train/configs/v12_pretrain_S_cat_fame_kxl_tj_px60.json"
d = json.load(open(src, encoding="utf-8"))
d["w_latent_skel"] = 0.05
d["latent_skel_probe"] = "assets/structure_probes/latent_skel_probe_v1/best.pt"
d["latent_skel_max_t"] = 0.25

tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                  dir="src/train/configs", encoding="utf-8")
json.dump(d, tmp, ensure_ascii=False, indent=2)
tmp.close()

spec = importlib.util.spec_from_file_location("T", "src/train/train.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

captured = {}
T.main = lambda a=None: captured.update(vars(a))
sys.argv = ["train.py", "--config", tmp.name]
try:
    T.main_from_cli()
except Exception as e:
    print("cli err:", type(e).__name__, e)

os.unlink(tmp.name)

want = {"w_latent_skel": 0.05,
        "latent_skel_probe": "assets/structure_probes/latent_skel_probe_v1/best.pt",
        "latent_skel_max_t": 0.25}
allok = True
for k, v in want.items():
    got = captured.get(k, "<ABSENT>")
    ok = (got == v)
    allok &= ok
    print(("OK  " if ok else "FAIL"), k, "=", repr(got), "期望", repr(v))
print("\nRESULT:", "全部落位" if allok else "存在静默丢弃")
