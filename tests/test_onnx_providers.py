from types import SimpleNamespace

from src import common


def test_resolve_onnx_providers_prefers_coreml_on_apple_silicon(monkeypatch):
    monkeypatch.setattr(common.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(common.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        common,
        "_maybe_import_onnxruntime",
        lambda: SimpleNamespace(get_available_providers=lambda: [
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]),
    )

    assert common.resolve_onnx_providers() == [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_recommended_onnx_threads_is_bounded_on_apple_silicon(monkeypatch):
    monkeypatch.setattr(common.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(common.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(common.os, "cpu_count", lambda: 32)

    assert common.recommended_onnx_threads() == 6


def test_resolve_onnx_providers_can_force_metal(monkeypatch):
    monkeypatch.setattr(common.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(common.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        common,
        "_maybe_import_onnxruntime",
        lambda: SimpleNamespace(get_available_providers=lambda: [
            "CoreMLExecutionProvider",
            "MetalExecutionProvider",
            "CPUExecutionProvider",
        ]),
    )

    assert common.resolve_onnx_providers("metal") == [
        "MetalExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]
