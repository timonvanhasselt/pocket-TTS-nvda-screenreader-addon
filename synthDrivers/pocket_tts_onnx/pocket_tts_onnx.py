"""
PocketTTS ONNX - Pure ONNX inference for Pocket TTS

A standalone, production-ready class for text-to-speech with voice cloning.
Supports both offline (batch) and streaming modes with adaptive chunking.

Dependencies:
    - onnxruntime (or onnxruntime-gpu for CUDA)
    - numpy
    - soundfile
    - sentencepiece

Usage:
    from pocket_tts_onnx import PocketTTSOnnx

    # Initialize with INT8 (CPU optimized - default, fastest)
    tts = PocketTTSOnnx()

    # Voice cloning from pre-computed numpy embedding (fastest)
    audio = tts.generate("Hello world!", voice="voices/my_voice.npy")

    # Fallback to audio file
    audio = tts.generate("Hello world!", voice="samples/reference.wav")

    # Streaming with adaptive chunking
    for chunk in tts.stream("Hello world!", voice="samples/reference.wav"):
        play_audio(chunk)  # Process each chunk as it's ready
"""

import os
import queue
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Generator, Optional, Union
import numpy as np
import onnxruntime as ort
import sentencepiece as spm

# Optional imports
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import scipy.signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class PocketTTSOnnx:
    """
    Pure ONNX inference engine for Pocket TTS.

    Supports:
        - Offline (batch) generation
        - Streaming generation with adaptive chunking
        - INT8 and FP32 models
        - Voice cloning from audio files OR .npy embeddings
        - Auto GPU/CPU detection
        - Temperature control for generation diversity

    Args:
        models_dir: Directory containing ONNX models
        tokenizer_path: Path to sentencepiece tokenizer.model
        precision: Model precision - "int8" (CPU optimized, fastest) or "fp32" (full precision)
        device: "auto", "cpu", or "cuda"
        temperature: Sampling temperature (0.0 = deterministic, 0.7 = default, 1.0 = more diverse)
        lsd_steps: Number of flow matching steps (default 10, lower = faster but lower quality)
    """

    SAMPLE_RATE = 24000
    SAMPLES_PER_FRAME = 1920
    FRAME_DURATION = SAMPLES_PER_FRAME / SAMPLE_RATE  # 0.08s per frame

    VALID_PRECISIONS = ("int8", "fp32")

    def __init__(
        self,
        models_dir: str = "onnx",
        tokenizer_path: str = "tokenizer.model",
        precision: str = "int8",
        device: str = "auto",
        temperature: float = 0.7,
        lsd_steps: int = 10,
        eos_threshold: float = -2.0,
    ):
        self.models_dir = Path(models_dir)

        if precision not in self.VALID_PRECISIONS:
            raise ValueError(f"precision must be one of {self.VALID_PRECISIONS}, got '{precision}'")

        self.precision = precision
        self.temperature = temperature
        self.lsd_steps = lsd_steps
        # EOS threshold: logit above this value triggers end-of-speech detection.
        # -4.0 (original) fires too early on long sentences — the model briefly
        # considers stopping at any noun that could end a clause. -2.0 requires
        # a stronger signal and matches the behaviour of the official PyTorch API
        # default. Raise toward 0.0 for shorter pauses; lower toward -4.0 if
        # the model clips the last word of short utterances.
        self.eos_threshold = eos_threshold

        # Setup execution providers
        self.providers = self._get_providers(device)

        # Load tokenizer
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(str(tokenizer_path))

        # Load models
        self._load_models()

        # Pre-compute s/t buffers for flow matching
        self._precompute_flow_buffers()

        # Cache for voice embeddings. Bounded LRU: keeps the most recently used
        # entries and evicts the oldest when the limit is reached, preventing
        # unbounded growth when many voices are used in a session.
        self._voice_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._voice_cache_max = 8

        # Cache of transformer state *after* the voice-conditioning pass,
        # keyed by voice path. The voice prompt is typically 100+ frames of
        # embeddings; prefilling it through the flow LM on every utterance
        # dominated time-to-first-audio (the "lag" after each keypress).
        # With this cache the prefill runs once per voice and each utterance
        # starts from a cheap memcpy of the cached state instead.
        self._voice_state_cache: "OrderedDict[str, dict]" = OrderedDict()
        self._voice_state_cache_max = 3

    def _get_providers(self, device: str) -> list:
        """Get ONNX execution providers based on device setting."""
        if device == "cpu":
            return ["CPUExecutionProvider"]
        elif device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:  # auto
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            return ["CPUExecutionProvider"]

    def _make_session_options(self, num_threads: int, allow_spinning: bool = True) -> ort.SessionOptions:
        """Create optimized session options for ONNX inference.

        Caps intra-op threads to avoid over-subscription overhead on the
        small sequential matmuls in the autoregressive loop.  The flow LM
        (critical path) keeps thread spinning enabled for lowest per-op
        latency; the mimi decoder now runs concurrently in a background
        thread, so it gets fewer threads with spinning disabled to avoid
        starving the flow LM (and the audio thread) on low-core machines.
        """
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if not allow_spinning:
            try:
                opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
            except Exception:
                pass  # older onnxruntime without this config entry
        return opts

    def _load_models(self):
        """Load ONNX models (dual model architecture)."""
        # Select model files based on precision
        suffix = "_int8" if self.precision == "int8" else ""
        flow_main_file = f"flow_lm_main{suffix}.onnx"
        flow_flow_file = f"flow_lm_flow{suffix}.onnx"
        mimi_file = f"mimi_decoder{suffix}.onnx"

        cpu = os.cpu_count() or 4
        # Critical path (runs every frame, sequentially):
        flow_opts = self._make_session_options(min(cpu, 4), allow_spinning=True)
        # Runs concurrently with the flow LM in a background thread:
        mimi_opts = self._make_session_options(2 if cpu > 4 else 1, allow_spinning=False)
        # One-shot / per-utterance helpers:
        aux_opts = self._make_session_options(min(cpu, 2), allow_spinning=False)

        self.mimi_encoder = ort.InferenceSession(
            str(self.models_dir / "mimi_encoder.onnx"),
            sess_options=aux_opts, providers=self.providers
        )
        self.text_conditioner = ort.InferenceSession(
            str(self.models_dir / "text_conditioner.onnx"),
            sess_options=aux_opts, providers=self.providers
        )
        # Dual model split: main (transformer) + flow (flow network)
        self.flow_lm_main = ort.InferenceSession(
            str(self.models_dir / flow_main_file),
            sess_options=flow_opts, providers=self.providers
        )
        self.flow_lm_flow = ort.InferenceSession(
            str(self.models_dir / flow_flow_file),
            sess_options=flow_opts, providers=self.providers
        )
        self.mimi_decoder = ort.InferenceSession(
            str(self.models_dir / mimi_file),
            sess_options=mimi_opts, providers=self.providers
        )

    def _precompute_flow_buffers(self):
        """Pre-compute s/t time step buffers for flow matching."""
        dt = 1.0 / self.lsd_steps
        self._st_buffers = []
        for j in range(self.lsd_steps):
            s = j / self.lsd_steps
            t = s + dt
            self._st_buffers.append((
                np.array([[s]], dtype=np.float32),
                np.array([[t]], dtype=np.float32)
            ))

    def _init_state(self, session: ort.InferenceSession) -> dict:
        """Initialize state tensors for a stateful model."""
        state = {}
        type_map = {
            "tensor(float)": np.float32,
            "tensor(int64)": np.int64,
            "tensor(bool)": np.bool_,
        }
        for inp in session.get_inputs():
            if inp.name.startswith("state_"):
                shape = [s if isinstance(s, int) else 0 for s in inp.shape]
                dtype = type_map.get(inp.type, np.float32)
                state[inp.name] = np.zeros(shape, dtype=dtype)
        return state

    def _increment_step(self, state: dict, n: int):
        """Increment step counters in state dict."""
        for k in state:
            if "step" in k:
                state[k] = (state[k] + n).astype(np.int64)

    # Maximum audio duration (seconds) passed to the voice encoder.
    # The mimi encoder runs on the full audio array at once; very long
    # samples (> ~60 s) can exhaust memory on low-RAM machines.
    # 30 s is more than enough for a high-quality voice embedding.
    MAX_VOICE_SECONDS = 30

    def _load_audio(self, path: Union[str, Path]) -> np.ndarray:
        """Load and preprocess audio file for voice cloning.

        Audio is truncated to MAX_VOICE_SECONDS before encoding to prevent
        OOM errors when large MP3/WAV samples are supplied.
        """
        if not HAS_SOUNDFILE:
            raise ImportError("soundfile required for voice cloning. Install with: pip install soundfile")

        audio, sr = sf.read(str(path))

        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Truncate to MAX_VOICE_SECONDS *before* resampling to keep the
        # source array small.
        max_source_samples = int(self.MAX_VOICE_SECONDS * sr)
        if len(audio) > max_source_samples:
            audio = audio[:max_source_samples]

        # Resample to 24kHz if needed
        if sr != self.SAMPLE_RATE:
            if HAS_SCIPY:
                num_samples = int(len(audio) * self.SAMPLE_RATE / sr)
                audio = scipy.signal.resample(audio, num_samples)
            else:
                # Fallback: numpy linear interpolation (no scipy required)
                num_samples = int(len(audio) * self.SAMPLE_RATE / sr)
                old_indices = np.linspace(0, len(audio) - 1, len(audio))
                new_indices = np.linspace(0, len(audio) - 1, num_samples)
                audio = np.interp(new_indices, old_indices, audio)

        audio = audio.astype(np.float32)
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()

        return audio.reshape(1, 1, -1)

    def encode_voice(self, audio_path: Union[str, Path]) -> np.ndarray:
        """
        Encode an audio file into voice embeddings for cloning.

        Args:
            audio_path: Path to audio file (wav, mp3, etc.)

        Returns:
            Voice embeddings array [1, N, 1024]
        """
        audio = self._load_audio(audio_path)
        embeddings = self.mimi_encoder.run(None, {"audio": audio})[0]
        
        # Normalize dimensions to [1, N, 1024]
        while embeddings.ndim > 3:
            embeddings = embeddings.squeeze(0)
        if embeddings.ndim < 3:
            embeddings = embeddings[None]
            
        return embeddings

    def _get_voice_embeddings(self, voice: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Get voice embeddings from various input types, supporting .npy for speed."""
        # Already embeddings
        if isinstance(voice, np.ndarray):
            return voice

        voice_str = str(voice)

        # Check cache
        if voice_str in self._voice_cache:
            self._voice_cache.move_to_end(voice_str)
            return self._voice_cache[voice_str]

        # Check if it's a pre-computed numpy file (.npy)
        if voice_str.lower().endswith(".npy") and os.path.exists(voice_str):
            embeddings = np.load(voice_str)
        # Audio file fallback
        elif os.path.exists(voice_str):
            embeddings = self.encode_voice(voice_str)
        else:
            raise ValueError(f"Voice file '{voice_str}' not found.")

        # Cache with LRU eviction: move to end on hit, pop oldest when full.
        self._voice_cache[voice_str] = embeddings
        self._voice_cache.move_to_end(voice_str)
        if len(self._voice_cache) > self._voice_cache_max:
            self._voice_cache.popitem(last=False)
        return embeddings

    def _voice_key(self, voice) -> "Optional[str]":
        """Stable cache key for a voice argument (path string), or None for raw arrays."""
        if isinstance(voice, np.ndarray):
            return None
        return str(voice)

    # Phonetic expansions for single letters, matching how a screen reader
    # would expect them to sound when read in isolation.
    _LETTER_NAMES = {
        'a': 'ay', 'b': 'bee', 'c': 'see', 'd': 'dee', 'e': 'ee',
        'f': 'ef', 'g': 'gee', 'h': 'aitch', 'i': 'eye', 'j': 'jay',
        'k': 'kay', 'l': 'el', 'm': 'em', 'n': 'en', 'o': 'oh',
        'p': 'pee', 'q': 'cue', 'r': 'ar', 's': 'ess', 't': 'tee',
        'u': 'you', 'v': 'vee', 'w': 'double-you', 'x': 'ex',
        'y': 'why', 'z': 'zee',
    }

    def _tokenize(self, text: str) -> np.ndarray:
        """Tokenize text for the model, with special handling for single characters."""
        text = text.strip()
        if not text:
            raise ValueError("Text cannot be empty")

        # Single character: expand to phonetic name so the TTS pronounces
        # the letter correctly instead of guessing from a bare "S." etc.
        if len(text) == 1:
            letter = text.lower()
            if letter in self._LETTER_NAMES:
                text = self._LETTER_NAMES[letter].capitalize() + "."
            elif letter.isdigit():
                # Digits are fine as-is; the model handles them.
                text = text + "."
            else:
                text = text + "."
        else:
            # Multi-character: apply light normalisation only.
            # Add terminal punctuation when the last printable character is
            # alphanumeric (prevents abrupt cut-off at end of utterance).
            if text[-1].isalnum():
                text = text + "."
            # Do NOT force capitalisation. NVDA sometimes passes mid-sentence
            # fragments (e.g. "her notifications") as separate speak() calls.
            # Capitalising them makes the model treat them as new sentences,
            # resetting prosody and causing unnatural stress patterns.
            # The model handles lower-case sentence starts fine.

        token_ids = self.tokenizer.Encode(text)
        return np.array(token_ids, dtype=np.int64).reshape(1, -1)

    def _update_state_from_outputs(self, state: dict, result: list, session: ort.InferenceSession):
        """Update state dict from model outputs."""
        for i in range(2, len(session.get_outputs())):
            name = session.get_outputs()[i].name
            if name.startswith("out_state_"):
                idx = int(name.replace("out_state_", ""))
                state[f"state_{idx}"] = result[i]

    def _get_voice_conditioned_state(
        self,
        voice_embeddings: np.ndarray,
        voice_key: "Optional[str]" = None,
    ) -> dict:
        """Return flow-LM state with the voice prompt already prefixed.

        The voice-conditioning prefill is by far the most expensive fixed
        cost per utterance (the voice prompt is typically 100+ frames).
        Its result depends only on the voice, so it is computed once per
        voice and cached; each utterance then starts from a copy of the
        cached state (a fast memcpy) instead of re-running the prefill.
        """
        cached = self._voice_state_cache.get(voice_key) if voice_key else None
        if cached is not None:
            self._voice_state_cache.move_to_end(voice_key)
            return {k: np.copy(v) for k, v in cached.items()}

        state = self._init_state(self.flow_lm_main)
        empty_seq = np.zeros((1, 0, 32), dtype=np.float32)
        res_voice = self.flow_lm_main.run(None, {
            "sequence": empty_seq,
            "text_embeddings": voice_embeddings,
            **state
        })
        self._update_state_from_outputs(state, res_voice, self.flow_lm_main)
        # Note: Step counters are already updated in the model's output states

        if voice_key:
            self._voice_state_cache[voice_key] = state
            self._voice_state_cache.move_to_end(voice_key)
            while len(self._voice_state_cache) > self._voice_state_cache_max:
                self._voice_state_cache.popitem(last=False)
            # The cached copy must never be mutated by generation.
            return {k: np.copy(v) for k, v in state.items()}
        return state

    def _run_flow_lm(
        self,
        voice_embeddings: np.ndarray,
        text_ids: np.ndarray,
        max_frames: int = 500,
        frames_after_eos: int = 1,
        cancel_event: "Optional[threading.Event]" = None,
        voice_key: "Optional[str]" = None,
    ) -> Generator[np.ndarray, None, None]:
        """
        Run flow LM autoregressive generation, yielding latents.

        Uses dual model architecture:
        - flow_lm_main: transformer/conditioner (produces conditioning vector)
        - flow_lm_flow: flow network (Euler integration for latent sampling)

        Yields individual latent frames as they're generated.
        If cancel_event is set, generation stops at the next frame boundary
        so a cancelled utterance stops burning CPU almost immediately.
        """
        # Text conditioning
        text_emb = self.text_conditioner.run(None, {"token_ids": text_ids})[0]
        if text_emb.ndim == 2:
            text_emb = text_emb[None]

        # Voice conditioning (cached per voice)
        state = self._get_voice_conditioned_state(voice_embeddings, voice_key)

        empty_seq = np.zeros((1, 0, 32), dtype=np.float32)
        empty_text = np.zeros((1, 0, 1024), dtype=np.float32)

        # Text conditioning pass
        res_text = self.flow_lm_main.run(None, {
            "sequence": empty_seq,
            "text_embeddings": text_emb,
            **state
        })
        self._update_state_from_outputs(state, res_text, self.flow_lm_main)
        # Note: Step counters are already updated in the model's output states

        # Autoregressive generation
        curr = np.full((1, 1, 32), np.nan, dtype=np.float32)
        dt = 1.0 / self.lsd_steps
        
        eos_step = None

        for step in range(max_frames):
            if cancel_event is not None and cancel_event.is_set():
                return
            # Run main model to get conditioning and EOS
            res_step = self.flow_lm_main.run(None, {
                "sequence": curr,
                "text_embeddings": empty_text,
                **state
            })

            conditioning = res_step[0]  # [1, 1, dim]
            eos_logit = res_step[1]     # [1, 1]

            # Update state (step counters are already updated in model outputs)
            self._update_state_from_outputs(state, res_step, self.flow_lm_main)

            # Check EOS - record when EOS is first detected
            if eos_logit[0][0] > self.eos_threshold and eos_step is None:
                eos_step = step
            
            # Stop only after frames_after_eos additional frames
            if eos_step is not None and step >= eos_step + frames_after_eos:
                break

            # Flow matching with external loop (enables temperature control)
            # Initialize with noise scaled by temperature
            std = np.sqrt(self.temperature) if self.temperature > 0 else 0.0
            x = np.random.normal(0, std, (1, 32)).astype(np.float32) if std > 0 else np.zeros((1, 32), dtype=np.float32)

            # Euler integration over flow network
            for j in range(self.lsd_steps):
                s_arr, t_arr = self._st_buffers[j]
                flow_out = self.flow_lm_flow.run(None, {
                    "c": conditioning,
                    "s": s_arr,
                    "t": t_arr,
                    "x": x
                })
                x = x + flow_out[0] * dt

            latent = x.reshape(1, 1, 32)
            yield latent
            curr = latent

    def _decode_worker(self, latent_queue: queue.Queue, audio_chunks: list,
                       decode_chunk_size: int = 12):
        """Decode latents from a queue in a background thread."""
        mimi_state = self._init_state(self.mimi_decoder)
        buf = []
        decoded = 0

        while True:
            item = latent_queue.get()
            if item is None:
                break
            buf.append(item)

            if len(buf) - decoded >= decode_chunk_size:
                chunk = np.concatenate(buf[decoded:decoded + decode_chunk_size], axis=1)
                result = self.mimi_decoder.run(None, {"latent": chunk, **mimi_state})
                audio_chunks.append(result[0].squeeze())
                for k in range(1, len(self.mimi_decoder.get_outputs())):
                    out_name = self.mimi_decoder.get_outputs()[k].name
                    if out_name.startswith("out_state_"):
                        idx = int(out_name.replace("out_state_", ""))
                        mimi_state[f"state_{idx}"] = result[k]
                decoded += decode_chunk_size

        # Decode remaining
        if decoded < len(buf):
            remaining = np.concatenate(buf[decoded:], axis=1)
            result = self.mimi_decoder.run(None, {"latent": remaining, **mimi_state})
            audio_chunks.append(result[0].squeeze())

    def generate(
        self,
        text: str,
        voice: Union[str, Path, np.ndarray],
        max_frames: int = 1500,
    ) -> np.ndarray:
        """
        Generate audio from text (offline/batch mode).

        Runs flow LM generation and mimi decoding in parallel threads
        for maximum throughput.

        Args:
            text: Text to synthesize
            voice: Audio file path for voice cloning, or pre-computed embeddings
            max_frames: Maximum latent frames to generate

        Returns:
            Audio samples as numpy array (float32, 24kHz)
        """
        voice_emb = self._get_voice_embeddings(voice)
        voice_key = self._voice_key(voice)
        text_ids = self._tokenize(text)

        # Start decode worker thread
        latent_queue = queue.Queue()
        audio_chunks = []
        decoder = threading.Thread(
            target=self._decode_worker,
            args=(latent_queue, audio_chunks),
            daemon=True,
        )
        decoder.start()

        # Generate latents and feed to decoder
        for latent in self._run_flow_lm(voice_emb, text_ids, max_frames, voice_key=voice_key):
            latent_queue.put(latent)
        latent_queue.put(None)  # sentinel

        decoder.join()
        return np.concatenate(audio_chunks)

    def _stream_decode_worker(
        self,
        latent_queue: queue.Queue,
        audio_queue: queue.Queue,
        first_chunk_frames: int,
        max_chunk_frames: int,
        cancel_event: "Optional[threading.Event]" = None,
    ):
        """Decode latents to audio in a background thread (streaming mode).

        Running the mimi decoder concurrently with flow-LM generation is the
        main throughput fix: previously decode time was stolen directly from
        generation time on the same thread, so whenever playback caught up the
        pipeline fell further behind and audio stuttered.

        Chunking is naturally adaptive: whatever latents have accumulated
        while the previous chunk was being decoded are batched into the next
        one (capped at max_chunk_frames), so small chunks are used when the
        decoder is keeping up and larger, more efficient chunks when it lags.
        """
        try:
            mimi_state = self._init_state(self.mimi_decoder)
            out_meta = self.mimi_decoder.get_outputs()
            buf = []
            first = True
            done = False
            while not done:
                item = latent_queue.get()
                if item is None:
                    done = True
                else:
                    buf.append(item)
                # Drain whatever else is already available (natural batching).
                while not done:
                    try:
                        nxt = latent_queue.get_nowait()
                    except queue.Empty:
                        break
                    if nxt is None:
                        done = True
                    else:
                        buf.append(nxt)

                if cancel_event is not None and cancel_event.is_set():
                    return
                # First chunk is decoded as soon as first_chunk_frames are
                # available (controls time-to-first-audio); afterwards decode
                # whatever is pending.
                need = first_chunk_frames if first else 1
                while len(buf) >= need or (done and buf):
                    if cancel_event is not None and cancel_event.is_set():
                        return
                    n = min(len(buf), max_chunk_frames)
                    chunk = np.concatenate(buf[:n], axis=1)
                    del buf[:n]
                    res = self.mimi_decoder.run(None, {"latent": chunk, **mimi_state})
                    for k in range(1, len(out_meta)):
                        name = out_meta[k].name
                        if name.startswith("out_state_"):
                            idx = int(name.replace("out_state_", ""))
                            mimi_state[f"state_{idx}"] = res[k]
                    audio_queue.put(res[0].squeeze())
                    first = False
        except Exception as e:
            audio_queue.put(e)
        finally:
            audio_queue.put(None)

    def stream(
        self,
        text: str,
        voice: Union[str, Path, np.ndarray],
        max_frames: int = 1500,
        first_chunk_frames: int = 1,
        target_buffer_sec: float = None,  # deprecated, kept for compatibility
        max_chunk_frames: int = 12,
        cancel_event: "Optional[threading.Event]" = None,
    ) -> Generator[np.ndarray, None, None]:
        """
        Stream audio generation with pipelined decoding.

        Flow-LM generation runs on the calling thread while the mimi decoder
        runs in a background thread, so the two overlap instead of competing
        for the same thread. This raises sustained throughput (fewer buffer
        underruns = less glitching) and lowers time-to-first-audio.

        Args:
            text: Text to synthesize
            voice: Audio file path for voice cloning, or pre-computed embeddings
            max_frames: Maximum latent frames to generate
            first_chunk_frames: Frames in first chunk (controls TTFB)
            target_buffer_sec: Deprecated; ignored (kept for API compatibility)
            max_chunk_frames: Maximum frames per decoded chunk
            cancel_event: Optional threading.Event; when set, generation and
                decoding stop at the next frame/chunk boundary.

        Yields:
            Audio chunks as numpy arrays (float32, 24kHz)
        """
        voice_emb = self._get_voice_embeddings(voice)
        voice_key = self._voice_key(voice)
        text_ids = self._tokenize(text)

        latent_queue: queue.Queue = queue.Queue()
        audio_queue: queue.Queue = queue.Queue()
        decoder = threading.Thread(
            target=self._stream_decode_worker,
            args=(latent_queue, audio_queue, first_chunk_frames,
                  max_chunk_frames, cancel_event),
            daemon=True,
        )
        decoder.start()

        decoder_finished = False
        try:
            for latent in self._run_flow_lm(
                voice_emb, text_ids, max_frames,
                cancel_event=cancel_event, voice_key=voice_key,
            ):
                latent_queue.put(latent)
                # Yield any audio that is already decoded, without blocking
                # generation.
                while True:
                    try:
                        item = audio_queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is None:
                        decoder_finished = True
                        return
                    if isinstance(item, Exception):
                        decoder_finished = True
                        raise item
                    yield item
        finally:
            # Always unblock the decoder thread, even if the consumer closed
            # this generator early (e.g. on cancel).
            latent_queue.put(None)

        # Generation finished: drain the remaining audio as it is decoded.
        while not decoder_finished:
            item = audio_queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    def save_audio(self, audio: np.ndarray, path: Union[str, Path]):
        """Save audio to file."""
        if not HAS_SOUNDFILE:
            raise ImportError("soundfile required. Install with: pip install soundfile")
        sf.write(str(path), audio, self.SAMPLE_RATE)

    @property
    def device(self) -> str:
        """Return the device being used."""
        if "CUDAExecutionProvider" in self.providers:
            return "cuda"
        return "cpu"

    def __repr__(self) -> str:
        return (
            f"PocketTTSOnnx("
            f"device={self.device!r}, "
            f"precision={self.precision!r}, "
            f"temperature={self.temperature}, "
            f"lsd_steps={self.lsd_steps}, "
            f"eos_threshold={self.eos_threshold}, "
            f"sample_rate={self.SAMPLE_RATE})"
        )
