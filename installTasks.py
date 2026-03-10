# installTasks.py
import os
import requests
import shutil
import globalVars
from logHandler import log

# The repository containing the ONNX models
REPO_URL = "https://huggingface.co/KevinAHM/pocket-tts-onnx/resolve/main"

ONNX_REMOTE_FILES = [
    "onnx/flow_lm_flow_int8.onnx",
    "onnx/flow_lm_main_int8.onnx",
    "onnx/mimi_decoder_int8.onnx",
    "onnx/mimi_encoder.onnx",
    "onnx/text_conditioner.onnx",
    "onnx/LICENSE",
]

ROOT_REMOTE_FILES = [
    "tokenizer.model",
]

def download_file(url, target_path, session):
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        log.info(f"Pocket TTS: {os.path.basename(target_path)} already exists, skipping.")
        return True
    log.info(f"Pocket TTS: Downloading {os.path.basename(target_path)}...")
    try:
        with session.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(target_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk: f.write(chunk)
        return True
    except Exception as e:
        log.error(f"Pocket TTS: Error downloading {url}: {e}")
        return False

def onInstall():
    # Detect addon root directory (where installTasks.py is located)
    addon_root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # User paths
    user_base_dir = os.path.join(globalVars.appArgs.configPath, "pocket_tts")
    user_onnx_dir = os.path.join(user_base_dir, "onnx")
    user_voices_dir = os.path.join(user_base_dir, "voices")

    # Create directories in user profile
    for folder in [user_base_dir, user_onnx_dir, user_voices_dir]:
        os.makedirs(folder, exist_ok=True)

    # 1. DOWNLOADS to user profile
    headers = {"User-Agent": "NVDA-Addon-PocketTTS-ONNX-1.1"}
    with requests.Session() as session:
        session.headers.update(headers)
        for filename in ROOT_REMOTE_FILES:
            download_file(f"{REPO_URL}/{filename}", os.path.join(user_base_dir, filename), session)
        for remote_path in ONNX_REMOTE_FILES:
            download_file(f"{REPO_URL}/{remote_path}", os.path.join(user_onnx_dir, os.path.basename(remote_path)), session)

    # 2. COPY LOCAL FILES (convert.py and voices) to user profile
    local_convert_py = os.path.join(addon_root_dir, "convert.py")
    if os.path.exists(local_convert_py):
        shutil.copy2(local_convert_py, os.path.join(user_base_dir, "convert.py"))
        log.info("Pocket TTS: convert.py copied to user config.")

    local_voices_path = os.path.join(addon_root_dir, "voices")
    if os.path.exists(local_voices_path):
        for item in os.listdir(local_voices_path):
            shutil.copy2(os.path.join(local_voices_path, item), os.path.join(user_voices_dir, item))
        log.info("Pocket TTS: Voices copied to user config.")

    # 3. CLEAN UP ADDON DIRECTORY (Remove after successful copy)
    # Remove convert.py from the addon installation directory
    if os.path.exists(local_convert_py):
        try: os.remove(local_convert_py)
        except: pass

    # Remove voices directory from the addon installation directory
    if os.path.exists(local_voices_path):
        try: shutil.rmtree(local_voices_path)
        except: pass
        
    # Remove tokenizer.model after copying
    local_tokenizer = os.path.join(addon_root_dir, "tokenizer.model")
    if os.path.exists(local_tokenizer):
        try: os.remove(local_tokenizer)
        except: pass

    log.info("Pocket TTS: Installation and cleanup completed.")