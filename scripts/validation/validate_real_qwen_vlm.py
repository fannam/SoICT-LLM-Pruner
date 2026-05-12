from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUN_ENV_VAR = "CARVE_LM_RUN_REAL_VLM_VALIDATION"


@dataclass(frozen=True)
class ModelSpec:
    family: str
    default_model_id: str
    adapter_name: str


MODEL_SPECS = {
    "qwen2_5_vl": ModelSpec(
        family="qwen2_5_vl",
        default_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        adapter_name="qwen2_5_vl",
    ),
    "qwen3_vl": ModelSpec(
        family="qwen3_vl",
        default_model_id="Qwen/Qwen3-VL-2B-Instruct",
        adapter_name="qwen3_vl",
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Manually gated real-model validation for Qwen2.5-VL and Qwen3-VL pruning. "
            "This script downloads Hugging Face models and is intentionally excluded from CI."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_SPECS),
        default=tuple(MODEL_SPECS),
        help="Model families to validate.",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        choices=("bridge", "vision", "merger"),
        default=("bridge", "vision", "merger"),
        help="Pruning components to validate for each model.",
    )
    parser.add_argument(
        "--qwen2-5-vl-model-id",
        default=MODEL_SPECS["qwen2_5_vl"].default_model_id,
        help="Hugging Face model id for Qwen2.5-VL validation.",
    )
    parser.add_argument(
        "--qwen3-vl-model-id",
        default=MODEL_SPECS["qwen3_vl"].default_model_id,
        help="Hugging Face model id for Qwen3-VL validation.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device for model execution, e.g. cpu or cuda.")
    parser.add_argument(
        "--device-map",
        default=None,
        help=(
            "Optional transformers device_map value such as auto. "
            "When set, the script does not call model.to(device)."
        ),
    )
    parser.add_argument(
        "--dtype",
        default=None,
        choices=(None, "auto", "float32", "float16", "bfloat16"),
        help="Optional model dtype passed to transformers from_pretrained().",
    )
    parser.add_argument("--pruning-ratio", type=float, default=0.125, help="Structured pruning ratio.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(".manual-validation/real-qwen-vlm"),
        help="Directory for temporary save/load artifacts.",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep saved pruned model artifacts after validation.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to Hugging Face loaders.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm that network downloads and large local model execution are intended.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.yes and os.environ.get(RUN_ENV_VAR) != "1":
        print(
            "Refusing to run real-model validation without explicit confirmation.\n"
            "Re-run with --yes or set {}=1. This script downloads large models and may require a GPU.".format(
                RUN_ENV_VAR
            ),
            file=sys.stderr,
        )
        return 2

    torch = _import_torch()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    for family in args.models:
        spec = MODEL_SPECS[family]
        model_id = _model_id_for_args(args, family)
        for component in args.components:
            component_dir = args.work_dir / family / component
            if component_dir.exists():
                shutil.rmtree(component_dir)
            component_dir.mkdir(parents=True, exist_ok=True)
            print("[{}:{}] loading {}".format(family, component, model_id), flush=True)
            _run_component_validation(
                torch=torch,
                spec=spec,
                model_id=model_id,
                component=component,
                output_dir=component_dir,
                args=args,
            )
            _clear_runtime_cache(torch)

    if not args.keep_artifacts:
        shutil.rmtree(args.work_dir)
    return 0


def _model_id_for_args(args: argparse.Namespace, family: str) -> str:
    if family == "qwen2_5_vl":
        return args.qwen2_5_vl_model_id
    if family == "qwen3_vl":
        return args.qwen3_vl_model_id
    raise KeyError(family)


def _run_component_validation(
    *,
    torch,
    spec: ModelSpec,
    model_id: str,
    component: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    model, processor = _load_model_and_processor(
        model_id=model_id,
        torch=torch,
        dtype_name=args.dtype,
        device=args.device,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    try:
        inputs = _build_forward_inputs(torch, processor, _model_input_device(model, args.device))
        _assert_forward_runs(torch, model, inputs, label="dense")

        if component == "bridge":
            result, pruner, pruner_cls, manifest_name, canonical_pruner = _run_bridge_pruning(
                model,
                spec,
                args.device,
                args.pruning_ratio,
            )
        elif component == "vision":
            result, pruner, pruner_cls, manifest_name, canonical_pruner = _run_vision_pruning(
                model,
                spec,
                args.device,
                args.pruning_ratio,
            )
        elif component == "merger":
            result, pruner, pruner_cls, manifest_name, canonical_pruner = _run_merger_pruning(
                model,
                spec,
                args.device,
                args.pruning_ratio,
            )
        else:
            raise KeyError(component)

        _assert_forward_runs(torch, result.model, inputs, label="pruned")
        _save_load_and_verify(
            torch=torch,
            pruner=pruner,
            pruner_cls=pruner_cls,
            result=result,
            output_dir=output_dir,
            inputs=inputs,
            device=args.device,
            manifest_name=manifest_name,
            expected_adapter=spec.adapter_name,
            expected_pruner=canonical_pruner,
        )
        print("[{}:{}] ok".format(spec.family, component), flush=True)
    finally:
        del model
        del processor


def _run_bridge_pruning(model, spec: ModelSpec, device: str, pruning_ratio: float):
    from carve_lm.vlm.components.merger.pruners import BridgeChannelConfig, BridgeChannelPruner

    pruner = BridgeChannelPruner(
        model,
        BridgeChannelConfig(pruning_ratio=pruning_ratio, clone_model=False),
        device=device,
        model_adapter=spec.adapter_name,
    )
    context = pruner.discover()
    result = pruner.apply(pruner.select(_scores_for_context(context)))
    return result, pruner, BridgeChannelPruner, "vlm_merger_pruner_manifest.json", "width.bridge"


def _run_vision_pruning(model, spec: ModelSpec, device: str, pruning_ratio: float):
    from carve_lm.vlm.components.vision.pruners import WidthChannelConfig, WidthChannelPruner

    pruner = WidthChannelPruner(
        model,
        WidthChannelConfig(pruning_ratio=pruning_ratio, clone_model=False),
        device=device,
        model_adapter=spec.adapter_name,
    )
    context = pruner.discover()
    result = pruner.apply(pruner.select(_scores_for_context(context)))
    return result, pruner, WidthChannelPruner, "vlm_vision_pruner_manifest.json", "width.channel"


def _run_merger_pruning(model, spec: ModelSpec, device: str, pruning_ratio: float):
    from carve_lm.vlm.components.merger.pruners import WidthConfig, WidthPruner

    pruner = WidthPruner(
        model,
        WidthConfig(pruning_ratio=pruning_ratio, clone_model=False),
        device=device,
        model_adapter=spec.adapter_name,
    )
    context = pruner.discover()
    result = pruner.apply(pruner.select(_scores_for_context(context)))
    return result, pruner, WidthPruner, "vlm_merger_pruner_manifest.json", "width"


def _save_load_and_verify(
    *,
    torch,
    pruner,
    pruner_cls,
    result,
    output_dir: Path,
    inputs: Mapping[str, Any],
    device: str,
    manifest_name: str,
    expected_adapter: str,
    expected_pruner: str,
) -> None:
    pruner.save_pruned(output_dir, result)

    manifest = json.loads((output_dir / manifest_name).read_text(encoding="utf-8"))
    _assert_manifest(manifest, expected_adapter=expected_adapter, expected_pruner=expected_pruner)

    loaded = pruner_cls.load_pruned(output_dir, device=device)
    loaded_inputs = _move_inputs(inputs, _model_input_device(loaded.model, device))
    _assert_forward_runs(torch, loaded.model, loaded_inputs, label="loaded")


def _assert_manifest(manifest: Mapping[str, Any], *, expected_adapter: str, expected_pruner: str) -> None:
    if manifest.get("version") != 2:
        raise AssertionError("Expected manifest version 2, received {!r}.".format(manifest.get("version")))
    if manifest.get("adapter_name") != expected_adapter:
        raise AssertionError(
            "Expected adapter_name={!r}, received {!r}.".format(expected_adapter, manifest.get("adapter_name"))
        )
    if manifest.get("canonical_pruner") != expected_pruner:
        raise AssertionError(
            "Expected canonical_pruner={!r}, received {!r}.".format(
                expected_pruner,
                manifest.get("canonical_pruner"),
            )
        )
    plan = manifest.get("plan")
    if not isinstance(plan, Mapping) or not plan.get("selected_group_ids"):
        raise AssertionError("Manifest does not contain selected pruning groups.")


def _scores_for_context(context) -> dict[str, float]:
    return {group.group_id: float(group.local_idx + 1) for group in context.groups}


def _load_model_and_processor(
    *,
    model_id: str,
    torch,
    dtype_name: str | None,
    device: str,
    device_map: str | None,
    trust_remote_code: bool,
):
    transformers = _import_transformers()
    model_cls = _resolve_auto_model_cls(transformers)
    processor = transformers.AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    model = _from_pretrained(
        model_cls,
        model_id,
        dtype=_resolve_dtype(torch, dtype_name),
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    if device_map is None:
        model = model.to(device)
    model.eval()
    return model, processor


def _resolve_auto_model_cls(transformers):
    for name in (
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
    ):
        model_cls = getattr(transformers, name, None)
        if model_cls is not None:
            return model_cls
    raise RuntimeError("Transformers does not expose a supported multimodal AutoModel class.")


def _from_pretrained(model_cls, model_id: str, *, dtype, device_map: str | None, trust_remote_code: bool):
    kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if device_map is not None:
        kwargs["device_map"] = device_map
    if dtype is not None:
        kwargs["dtype"] = dtype
    try:
        return model_cls.from_pretrained(model_id, **kwargs)
    except TypeError:
        if "dtype" in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
            return model_cls.from_pretrained(model_id, **kwargs)
        raise


def _build_forward_inputs(torch, processor, device):
    image = torch.full((224, 224, 3), 255, dtype=torch.uint8)
    prompt = "Describe this image in one short sentence."
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if hasattr(processor, "apply_chat_template"):
        try:
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            return _move_inputs(inputs, device)
        except Exception:
            try:
                text = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except Exception:
                text = prompt
    else:
        text = prompt

    try:
        inputs = processor(text=[text], images=[image], return_tensors="pt")
    except TypeError:
        inputs = processor(text=text, images=image, return_tensors="pt")
    return _move_inputs(inputs, device)


def _move_inputs(inputs, device):
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in dict(inputs).items()
        if value is not None
    }


def _assert_forward_runs(torch, model, inputs: Mapping[str, Any], *, label: str) -> None:
    with torch.inference_mode():
        outputs = model(**inputs)
    logits = getattr(outputs, "logits", None)
    if logits is None and isinstance(outputs, Mapping):
        logits = outputs.get("logits")
    if logits is None:
        raise AssertionError("{} forward did not return logits.".format(label))
    if logits.ndim < 2 or logits.numel() == 0:
        raise AssertionError("{} forward returned invalid logits shape {}.".format(label, tuple(logits.shape)))


def _model_input_device(model, fallback: str):
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return fallback


def _resolve_dtype(torch, dtype_name: str | None):
    if dtype_name is None:
        return None
    if dtype_name == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def _clear_runtime_cache(torch) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _import_torch():
    import torch

    return torch


def _import_transformers():
    import transformers

    return transformers


if __name__ == "__main__":
    raise SystemExit(main())
