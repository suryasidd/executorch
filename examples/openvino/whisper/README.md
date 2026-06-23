# Whisper with the OpenVINO Backend

This example exports and runs [OpenAI Whisper](https://huggingface.co/openai/whisper-small) speech-to-text models on Intel hardware using ExecuTorch with the OpenVINO backend.

## Overview

The model is split into three separate programs so that the cross-attention K/V projections — which depend only on the encoder output — run once per utterance instead of once per generated token:

- **encoder.pte** — mel features → encoder hidden states
- **cross_kv.pte** — encoder hidden states → per-layer cross-attention K/V
- **decoder.pte** — token-by-token generation with a self-attention KV cache, consuming the pre-computed cross K/V as inputs

The decoder uses plain aten ops (`index_copy_` for the static KV cache, `F.scaled_dot_product_attention`, `F.embedding`) with no backend-specific custom ops, so the graph stays OpenVINO-partitionable. An additive attention mask is supplied by the host each decode step, keeping the decoder graph shape-static.

## Environment Setup

Follow the **Prerequisites** and **Setup** instructions in [backends/openvino/README.md](../../../backends/openvino/README.md) to set up the OpenVINO backend.

### Install dependencies

```bash
pip install transformers soundfile datasets
```

## Export the Model

```bash
python export_whisper.py \
    --model_id openai/whisper-small \
    --output_dir ./whisper_ov \
    --device GPU \
    --max_cache_length 448
```

This writes four files to `./whisper_ov/`:

- `encoder.pte`
- `cross_kv.pte`
- `decoder.pte`
- `metadata.json`

### Export Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model_id` | `openai/whisper-small` | HuggingFace model ID |
| `--output_dir` | `whisper_ov` | Output directory for the `.pte` files |
| `--device` | `CPU` | Target OpenVINO device (`CPU`, `GPU`, `NPU`) |
| `--max_cache_length` | `448` | Maximum decoder sequence length (self-attention cache size) |

## Run Inference

```bash
python run_whisper.py \
    --model_dir ./whisper_ov \
    --use_sample_audio
```

Or with a custom 16kHz audio file:

```bash
python run_whisper.py \
    --model_dir ./whisper_ov \
    --audio /path/to/audio.wav
```

### Run Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model_dir` | required | Directory containing the exported `.pte` files and `metadata.json` |
| `--model_id` | from metadata | HuggingFace model ID used to load the processor/tokenizer |
| `--audio` | None | Path to a 16kHz audio file |
| `--use_sample_audio` | off | Use sample audio from HuggingFace datasets |
| `--max_new_tokens` | `128` | Maximum number of tokens to generate |
