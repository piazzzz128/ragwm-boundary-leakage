import json
from pathlib import Path

# GPT 是当前实验真正需要的模型，必须导入
from .GPT import GPT

# 以下模型是原论文备用/对比模型。当前 doc 阶段不用，缺依赖时不能让主程序崩。
try:
    from .PaLM2 import PaLM2
except Exception:
    PaLM2 = None

try:
    from .Vicuna import Vicuna
except Exception:
    Vicuna = None

try:
    from .Llama import Llama
except Exception:
    Llama = None


def create_model(config_path):
    print("load json:", config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    provider = config.get("model_info", {}).get("provider", "").lower()
    name = config.get("model_info", {}).get("name", "").lower()

    if provider == "gpt" or "gpt" in name:
        return GPT(config)

    if provider in ["palm", "palm2", "gemini"]:
        if PaLM2 is None:
            raise ImportError("PaLM2 is requested, but google.generativeai is not available.")
        return PaLM2(config)

    if provider == "vicuna" or "vicuna" in name:
        if Vicuna is None:
            raise ImportError("Vicuna is requested, but fastchat is not available.")
        return Vicuna(config)

    if provider == "llama" or "llama" in name:
        if Llama is None:
            raise ImportError("Llama is requested, but its dependencies are not available.")
        return Llama(config)

    raise ValueError(f"Unknown model provider/name: provider={provider}, name={name}")
