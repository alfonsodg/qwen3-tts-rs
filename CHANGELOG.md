# Changelog (fork modifications)

All changes relative to [TrevorS/qwen3-tts-rs](https://github.com/TrevorS/qwen3-tts-rs) upstream.

## Batched Inference (`src/lib.rs`)

### `synthesize_batch_with_voices()`
- New method accepting per-request `Option<&VoiceClonePrompt>` for mixed batches
- Batched autoregressive loop: single transformer forward for N sequences per frame
- Pre-allocated zero tensor for done sequences
- Batched initial token sampling (argmax across batch in one op)
- Batched EOS detection with single GPU→CPU transfer
- Bulk frame code transfer (stacked tensors, single sync)

### `synthesize_batch()` 
- Now delegates to `synthesize_batch_with_voices()` (backward compatible)

### Prefill Phase (Phase 1)
- Shared tensors computed once outside loop (`role_prefix`, `tts_pad_bos`)
- Serena codec IDs built as template, only language token patched per request
- Voice clone suffix embed computed once, reused

### Vocoder Phase (Phase 4)
- Batched decode: stacks all code tensors `[N, 16, T_max]`, single vocoder forward
- Pads shorter sequences, trims output waveforms to actual length
- Falls back to sequential for batch=1

### Adaptive max_length
- `max_length` set to `~6 frames/word + 50`, capped at 512 (was 2048)
- Reduces KV cache pre-allocation by ~75% for call center text

## Batched Streaming (`src/lib.rs`)

### `synthesize_batch_streaming()`
- Changed signature to accept `(String, Language, Option<SynthesisOptions>)` tuples (was `(String, Language)`)
- Uses `generate_acoustic_codes_batched()` instead of sequential per-request
- Pre-allocated zero tensor for done sequences

## Code Predictor (`src/models/code_predictor.rs`)

### `generate_acoustic_codes_batched()`
- New method: batched acoustic code generation for all active sequences in one pass

## KV Cache (`src/models/kv_cache.rs`)

### `QuantizedKVCache` (new)
- U8 quantization with per-head-per-position scale factors
- Symmetric quantization: `val_u8 = round(val/scale) + 128`
- Integrated into `AnyKVCache` enum
- Disabled by default (dequantize overhead in candle causes regression)

### `AnyKVCache` enum
- Added `Quantized(QuantizedKVCache)` variant
- Updated all match arms (`update`, `reset`, `peek`, `replace_kv`)

## Talker (`src/models/talker.rs`)

### `new_kv_caches_quantized()`
- New factory method for quantized KV caches

### `get_codec_embedding_batch()`
- **Bug fix**: `unsqueeze(0)` instead of `unsqueeze(1)` — was producing `[N, 1, hidden]` instead of `[1, N, hidden]`, causing shape mismatch in ICL voice cloning

## ICL Voice Clone Fixes (`src/lib.rs`)

### Shape mismatch fix
- `get_codec_embedding_batch()` returns `[1, T, hidden]` matching `embed_codes_for_group()` output

### KV cache overflow fix
- Dynamic KV cache allocation for ICL: adds `ref_frames + ref_text_len + input_ids.len() + 16` extra positions

### Warm-up frame skip
- ICL generates warm-up frames for `ref_text` before target text
- Skip `ref_text_len - 3` frames from generated codes before vocoder decode
- Prepend original `ref_codes` for vocoder context, then proportional cut
- The `-3` margin (~240ms) preserves onset of first phoneme

## Thread Safety (`src/lib.rs`)

### `unsafe impl Send + Sync for Qwen3TTS`
- Enables `Arc<Qwen3TTS>` for sharing model weights across batch and streaming worker threads
