"""Fast-dVLM via SGLang (NVlabs vendored fork) — V3 prompt path.

Uses sgl.Engine with HierarchyBlock (MDM block diffusion) or SpeculativeBlock
(self-speculative) decoding. SGLang's API does NOT expose template-style mask
fill at the client level, so this loader takes the V3 prompt verbatim and
asks the model to GENERATE the JSON output as a continuation (no pre-positioned
masks). The model's strong instruction-following + Qwen2.5-VL backbone should
produce valid JSON.

This is the "fast path" — block-diffusion speculative decoding gives Tok/NFE
~2.63 plus SGLang server gives extra parallelism (KV cache reuse, etc.).
"""
from __future__ import annotations

import os
import sys
import time

# Drop our scripts dir from sys.path so `import sglang` doesn't shadow.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _HERE]

DEFAULT_PATH = "/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/huggingface/Fast_dVLM_3B"
PROCESSOR_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"

ALGO_MAP = {
    "mdm": "HierarchyBlock",
    "spec": "SpeculativeBlock",
}


def _strip_template_from_prompt(prompt: str) -> str:
    """Drop the TEMPLATE (..) tail from the V3 prompt — SGLang generates
    free-form, so we don't need the JSON scaffold-with-masks sent to the model.
    """
    i = prompt.find("TEMPLATE (")
    return prompt[:i].rstrip() if i > 0 else prompt


def load(model_path=None, algorithm="spec", mem_fraction_static=0.75,
          quantization=None, max_running_requests=1, chunked_prefill_size=16384):
    """Load Fast-dVLM as sgl.Engine. Takes ~30-60s for model load + CUDA graph capture."""
    os.environ.setdefault("SGLANG_DISABLE_CUDNN_CHECK", "1")
    import sglang as sgl
    from transformers import AutoProcessor, AutoTokenizer

    path = model_path or DEFAULT_PATH
    processor = AutoProcessor.from_pretrained(PROCESSOR_PATH, use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    processor.tokenizer = tokenizer

    dllm_algo = ALGO_MAP[algorithm]
    engine_kwargs = dict(
        model_path=path,
        trust_remote_code=True,
        dtype="bfloat16",
        mem_fraction_static=mem_fraction_static,
        max_running_requests=max_running_requests,
        chunked_prefill_size=chunked_prefill_size,
        dllm_algorithm=dllm_algo,
        disable_cuda_graph=False,
        log_level="warning",
        enable_metrics=True,
        mm_attention_backend="triton_attn",
    )
    if quantization:
        engine_kwargs["quantization"] = quantization

    engine = sgl.Engine(**engine_kwargs)
    return {
        "engine": engine,
        "processor": processor,
        "tokenizer": tokenizer,
        "algorithm": algorithm,
        "model_path": path,
    }


def _build_input_ids(processor, image_path, prompt_text):
    """Build input_ids for sgl.Engine. Matches run_chatbot_sglang.build_inputs."""
    from qwen_vl_utils import process_vision_info
    content = []
    if image_path:
        content.append({"type": "image", "image": image_path})
    content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    )
    return inputs.input_ids[0].tolist()


def generate(bundle, image_paths, question, gen_length=512, temperature=0.0,
              repetition_penalty=1.05):
    """V3 prompt-only generation via SGLang. Returns (text, latency_s).

    `repetition_penalty=1.05` prevents the model from looping on category
    keys when generating the long V3 JSON (seen on long GO_STRAIGHT samples).
    """
    import torch
    engine = bundle["engine"]
    processor = bundle["processor"]
    image_path = image_paths[0] if image_paths else None

    # V3 prompt has a TEMPLATE block at the end with mask placeholders. SGLang
    # generates free-form, so we strip the template scaffold and keep just the
    # instruction. The model is asked to output the JSON itself.
    user_text = _strip_template_from_prompt(question)

    input_ids = _build_input_ids(processor, image_path, user_text)

    sampling = {
        "max_new_tokens": gen_length,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
    }

    torch.cuda.synchronize()
    t0 = time.time()
    out = engine.generate(
        input_ids=input_ids,
        image_data=[image_path] if image_path else None,
        sampling_params=sampling,
    )
    torch.cuda.synchronize()
    latency = time.time() - t0

    if isinstance(out, list):
        out = out[0]
    text = out.get("text", "") if isinstance(out, dict) else str(out)
    return text, latency


def shutdown(bundle):
    engine = bundle.get("engine")
    if engine is not None:
        engine.shutdown()
