"""Prints versions and GPU status. Run after installing: python -m src.check_env"""
import platform
import sys

print("python      :", sys.version.split()[0], "|", platform.platform())
for name in ["numpy", "pandas", "pyarrow", "sklearn", "statsmodels", "xgboost", "transformers"]:
    try:
        mod = __import__(name)
        print(f"{name:12}:", getattr(mod, "__version__", "ok"))
    except ImportError:
        print(f"{name:12}: NOT INSTALLED")
try:
    import torch
    print("torch       :", torch.__version__)
    print("cuda ready  :", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("gpu         :", p.name, f"({p.total_memory / 1e9:.1f} GB)")
except ImportError:
    print("torch       : NOT INSTALLED (install it first, see README)")
