import importlib
for m in ["albumentations", "scipy", "cv2", "skimage"]:
    try:
        mod = importlib.import_module(m)
        print(m, "OK", getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, "MISSING", e)