from __future__ import annotations
import json
from pathlib import Path
import multiprocessing
from typing import Optional
from llama_cpp import Llama

_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


class _ModelSingleton:
    _llm: Optional[Llama] = None

    @classmethod
    def get(cls) -> Llama:
        if cls._llm is None:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            repo_id = cfg.get("repo_id")
            filename = cfg.get("filename")
            n_ctx = int(cfg.get("n_ctx", 4096))
            n_gpu_layers = int(cfg.get("n_gpu_layers", 0))
            n_threads = multiprocessing.cpu_count() if int(
                cfg.get("n_threads", 0)) == 0 else int(cfg.get("n_threads", 0))
            cls._llm = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                verbose=False
            )
        return cls._llm


def get_model() -> Llama:
    return _ModelSingleton.get()
