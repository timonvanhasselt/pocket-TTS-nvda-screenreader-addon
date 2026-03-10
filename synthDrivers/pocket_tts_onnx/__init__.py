import os
import sys
import threading
import queue
import ctypes
import glob
from collections import OrderedDict
from typing import OrderedDict as TOrderedDict

# NVDA Core Imports
import globalVars
from nvwave import WavePlayer, AudioPurpose
from logHandler import log
import synthDriverHandler
from synthDriverHandler import (
    SynthDriver as BaseSynthDriver,
    VoiceInfo,
    synthIndexReached,
    synthDoneSpeaking,
)
from speech.commands import IndexCommand, VolumeCommand, BreakCommand

# --- PATH CONFIGURATION ---
DRIVER_DIR = os.path.dirname(os.path.abspath(__file__))
ADDON_DIR = os.path.dirname(os.path.dirname(DRIVER_DIR))
LIBS_DIR = os.path.join(ADDON_DIR, "libs")

if LIBS_DIR not in sys.path:
    sys.path.insert(0, LIBS_DIR)

try:
    import numpy as np
    from .pocket_tts_onnx import PocketTTSOnnx
    log.info("Pocket TTS ONNX: Numpy and engine loaded.")
except ImportError as e:
    log.error(f"Pocket TTS ONNX: Error loading dependencies: {e}")
    np = None
    PocketTTSOnnx = None

# =========================================================================
# SYNTHESIS QUEUE THREAD
# =========================================================================

class _SynthQueueThread(threading.Thread):
    def __init__(self, driver: 'SynthDriver'):
        super().__init__()
        self.driver = driver
        self.daemon = True
        self.stop_event = threading.Event()
        self.cancel_event = threading.Event()

    def run(self):
        ctypes.windll.ole32.CoInitialize(None)
        while not self.stop_event.is_set():
            try:
                request = self.driver._request_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self.cancel_event.clear()
            text, voice_data, index = request
            
            try:
                if self.cancel_event.is_set() or (not text and index is None) or not self.driver.tts_engine:
                    self._finish_request(index)
                    continue

                # If there is text to speak
                if text and text.strip():
                    # Use the pre-calculated embedding directly for maximum speed
                    audio_stream = self.driver.tts_engine.stream(
                        text=text, 
                        voice=voice_data,
                        target_buffer_sec=0.2
                    )

                    # Optimization: Pre-calculate conversion factor to reduce CPU load in the loop
                    # Factor lowered to 1.2 to prevent digital clipping
                    volume_multiplier = (self.driver._volume / 100.0) * 1.2
                    conversion_factor = volume_multiplier * 32767

                    for chunk in audio_stream:
                        if self.cancel_event.is_set(): 
                            break
                        if chunk is not None and self.driver._player:
                            # Direct processing: Clipping at the integer level is faster and prevents crackling
                            processed_audio = np.clip(chunk * conversion_factor, -32767, 32767).astype(np.int16)
                            self.driver._player.feed(processed_audio.tobytes())
                
                self._finish_request(index)
            except Exception as e:
                log.error(f"Pocket TTS ONNX: Synthesis error: {e}")
                self._finish_request(index)
        ctypes.windll.ole32.CoUninitialize()

    def _finish_request(self, index):
        if index is not None and not self.cancel_event.is_set():
            synthIndexReached.notify(synth=self.driver, index=index)
        self.driver._request_queue.task_done()
        synthDoneSpeaking.notify(synth=self.driver)

# =========================================================================
# MAIN SYNTHDRIVER CLASS
# =========================================================================

class SynthDriver(BaseSynthDriver):
    name = "pocket_tts_onnx"
    description = "Pocket TTS ONNX Synthesizer"

    @classmethod
    def check(cls):
        return np is not None and PocketTTSOnnx is not None

    def __init__(self):
        super(SynthDriver, self).__init__()
        
        # Determine the correct base path for models
        # If in secure mode, we must use the systemConfig path explicitly
        if globalVars.appArgs.secure:
            self.models_root = os.path.join(r"C:\Program Files\NVDA\systemConfig", "pocket_tts")
        else:
            self.models_root = os.path.join(globalVars.appArgs.configPath, "pocket_tts")
        
        # Configure paths based on the root
        self.models_dir = os.path.join(self.models_root, "onnx")
        self.tokenizer_path = os.path.join(self.models_root, "tokenizer.model")
        self.voices_dir = os.path.join(self.models_root, "voices")
        
        # Create directories if they do not exist (only in non-secure mode to avoid permission errors)
        if not globalVars.appArgs.secure:
            if not os.path.exists(self.voices_dir): 
                os.makedirs(self.voices_dir, exist_ok=True)
            if not os.path.exists(self.models_dir):
                os.makedirs(self.models_dir, exist_ok=True)

        self._current_voice = ""
        self._current_embedding = None
        self._volume = 80
        self.tts_engine = None
        self._player = None
        self._available_voices = OrderedDict()
        self._request_queue = queue.Queue()
        self._engine_loaded_event = threading.Event()

        self._scan_voices()
        self._worker_thread = _SynthQueueThread(driver=self)
        self._worker_thread.start()
        threading.Thread(target=self._initialize_async, daemon=True).start()

    supportedCommands = frozenset([IndexCommand, VolumeCommand, BreakCommand])
    supportedNotifications = frozenset([synthIndexReached, synthDoneSpeaking])
    supportedSettings = (BaseSynthDriver.VoiceSetting(), BaseSynthDriver.VolumeSetting())

    def _scan_voices(self):
        """Searches for .wav and .npy files in the new user folder location."""
        self._available_voices.clear()
        # Now only scans in the new external folders
        files = glob.glob(os.path.join(self.voices_dir, "*.*")) + glob.glob(os.path.join(self.models_root, "*.*"))
        seen = set()
        for path in files:
            name, ext = os.path.splitext(os.path.basename(path))
            if ext.lower() in [".wav", ".npy"] and name not in seen:
                self._available_voices[name] = VoiceInfo(name, name.replace("_", " ").title())
                seen.add(name)
        if self._available_voices and not self._current_voice:
            self._current_voice = list(self._available_voices.keys())[0]

    def _load_voice_embedding(self, voice_id):
        """Loads embedding from disk (.npy) or generates it once (.wav) from the user folder."""
        if not self.tts_engine: return
        for ext in [".npy", ".wav"]:
            for folder in [self.voices_dir, self.models_root]:
                path = os.path.join(folder, f"{voice_id}{ext}")
                if os.path.exists(path):
                    try:
                        if ext == ".npy":
                            self._current_embedding = np.load(path)
                            log.info(f"Pocket TTS: Embedding loaded: {path}")
                        else:
                            log.info(f"Pocket TTS: Generating embedding for: {path}")
                            self._current_embedding = self.tts_engine.encode_voice(path)
                        return
                    except Exception as e: log.error(f"Error loading voice {voice_id}: {e}")

    def _initialize_async(self):
        ctypes.windll.ole32.CoInitialize(None)
        try:
            self._player = WavePlayer(channels=1, samplesPerSec=24000, bitsPerSample=16, purpose=AudioPurpose.SPEECH)
            self.tts_engine = PocketTTSOnnx(models_dir=self.models_dir, tokenizer_path=self.tokenizer_path, precision="int8")
            self._load_voice_embedding(self._current_voice)
            self._engine_loaded_event.set()
            log.info("Pocket TTS ONNX: Ready for use.")
        except Exception as e: log.error(f"Initialization failed: {e}")

    def _get_availableVoices(self) -> TOrderedDict[str, VoiceInfo]: return self._available_voices
    def _get_voice(self): return self._current_voice
    def _set_voice(self, value):
        if value in self._available_voices:
            self._current_voice = value
            threading.Thread(target=self._load_voice_embedding, args=(value,), daemon=True).start()

    def _get_volume(self): return self._volume
    def _set_volume(self, value): self._volume = value

    def speak(self, speechSequence):
        """
        Processes the speech sequence, maintaining support for IndexCommands
        which are vital for 'Read All' functionality.
        """
        if not self._engine_loaded_event.is_set() or self._current_embedding is None:
            return

        text_parts = []
        for item in speechSequence:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, IndexCommand):
                # Send accumulated text and the index to the queue
                combined_text = "".join(text_parts)
                self._request_queue.put((combined_text, self._current_embedding, item.index))
                text_parts = []
            elif isinstance(item, VolumeCommand):
                # Optional: Update volume mid-sequence if supported
                self._volume = item.value

        # Send any remaining text after the last index
        remaining_text = "".join(text_parts)
        if remaining_text.strip():
            self._request_queue.put((remaining_text, self._current_embedding, None))

    def cancel(self):
        self._worker_thread.cancel_event.set()
        if self._player:
            try: self._player.stop()
            except: pass
        try:
            while not self._request_queue.empty(): self._request_queue.get_nowait()
        except queue.Empty: pass

    def terminate(self):
        self._worker_thread.stop_event.set()
        if self._player: self._player.close()
        self.tts_engine = None