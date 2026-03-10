import os
import sys
import wx
import gui
import config
import globalVars
from logHandler import log
from gui.settingsDialogs import SettingsPanel

# Define paths to locate the engine in synthDrivers
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Going up from globalPlugins/pocket_tts_onnx/ to the addon root
ADDON_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
SYNTH_DIR = os.path.join(ADDON_DIR, "synthDrivers")
LIBS_DIR = os.path.join(ADDON_DIR, "libs")

# Add synthDrivers and libs to sys.path
if SYNTH_DIR not in sys.path:
    sys.path.insert(0, SYNTH_DIR)
if LIBS_DIR not in sys.path:
    sys.path.insert(0, LIBS_DIR)

try:
    import numpy as np
    from pocket_tts_onnx import PocketTTSOnnx
    log.info("Pocket TTS Settings: Numpy and engine loaded.")
except ImportError as e:
    log.error(f"Pocket TTS Settings: Error loading dependencies from {SYNTH_DIR} or {LIBS_DIR}: {e}")
    np = None
    PocketTTSOnnx = None

_ = lambda s: s

class PocketTTSSettingsPanel(SettingsPanel):
    title = _("Pocket TTS Voice Manager")

    def makeSettings(self, settingsSizer):
        # Correct way to get the NVDA user config directory
        conf_dir = globalVars.appArgs.configPath
        
        # Configure paths based on the expected folder structure
        self.models_root = os.path.join(conf_dir, "pocket_tts")
        self.voices_dir = os.path.join(self.models_root, "voices")
        self.onnx_dir = os.path.join(self.models_root, "onnx")
        self.tokenizer_path = os.path.join(self.models_root, "tokenizer.model")
        
        # Ensure the voices directory exists
        if not os.path.exists(self.voices_dir):
            try:
                os.makedirs(self.voices_dir, exist_ok=True)
            except Exception as e:
                log.error(f"Pocket TTS: Could not create folder: {e}")

        # --- Section 1: Add/Clone Voice ---
        add_box = wx.StaticBox(self, label=_("Add new voice (Voice Cloning)"))
        add_sizer = wx.StaticBoxSizer(add_box, wx.VERTICAL)
        
        help_text = wx.StaticText(self, label=_("Select an audio file (MP3/WAV) to generate a new .npy voice embedding:"))
        add_sizer.Add(help_text, 0, wx.ALL, 5)
        
        self.btnAdd = wx.Button(self, label=_("&Convert audio file..."))
        self.btnAdd.Bind(wx.EVT_BUTTON, self.onAddVoice)
        add_sizer.Add(self.btnAdd, 0, wx.ALL, 5)
        settingsSizer.Add(add_sizer, 0, wx.EXPAND | wx.ALL, 10)

        # --- Section 2: Manage Voices ---
        manage_box = wx.StaticBox(self, label=_("Manage installed voices"))
        manage_sizer = wx.StaticBoxSizer(manage_box, wx.VERTICAL)
        
        list_label = wx.StaticText(self, label=_("&Select a voice from the list:"))
        manage_sizer.Add(list_label, 0, wx.LEFT | wx.TOP, 5)
        
        self.voiceList = wx.Choice(self, choices=self.get_installed_voices())
        self.voiceList.Bind(wx.EVT_CHOICE, self.onVoiceSelect)
        manage_sizer.Add(self.voiceList, 0, wx.EXPAND | wx.ALL, 5)

        # Rename Section
        name_label = wx.StaticText(self, label=_("&Edit display name:"))
        manage_sizer.Add(name_label, 0, wx.LEFT | wx.TOP, 5)
        
        self.nameEdit = wx.TextCtrl(self)
        manage_sizer.Add(self.nameEdit, 0, wx.EXPAND | wx.ALL, 5)
        
        self.btnRename = wx.Button(self, label=_("&Rename voice file"))
        self.btnRename.Bind(wx.EVT_BUTTON, self.onRenameVoice)
        manage_sizer.Add(self.btnRename, 0, wx.ALL, 5)

        manage_sizer.AddSpacer(10)
        self.btnRemove = wx.Button(self, label=_("&Remove selected voice"))
        self.btnRemove.Bind(wx.EVT_BUTTON, self.onRemoveVoice)
        manage_sizer.Add(self.btnRemove, 0, wx.ALL, 5)
        
        settingsSizer.Add(manage_sizer, 0, wx.EXPAND | wx.ALL, 10)

    def get_installed_voices(self):
        """Scans the voices directory for .npy embedding files."""
        if not os.path.exists(self.voices_dir): return []
        try:
            return [f for f in os.listdir(self.voices_dir) if f.lower().endswith(".npy")]
        except Exception:
            return []

    def onAddVoice(self, evt):
        """Processes the audio file using PocketTTSOnnx and saves the embedding."""
        if PocketTTSOnnx is None or np is None:
            gui.messageBox(_("Dependencies (Numpy or Engine) not loaded. Check NVDA log for details."), _("Error"))
            return

        wildcard = "Audio files (*.mp3;*.wav)|*.mp3;*.wav"
        with wx.FileDialog(self, message=_("Select a voice sample"), wildcard=wildcard, style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fd:
            if fd.ShowModal() == wx.ID_OK:
                audio_src = fd.GetPath()
                base_name = os.path.splitext(os.path.basename(audio_src))[0]
                dest_path = os.path.join(self.voices_dir, f"{base_name}.npy")

                gui.messageBox(_("Generating voice embedding. NVDA might become unresponsive for a moment..."), _("Processing"))

                try:
                    # Initialize the engine with paths defined in the driver
                    tts = PocketTTSOnnx(
                        models_dir=self.onnx_dir, 
                        tokenizer_path=self.tokenizer_path
                    )
                    
                    # Generate embedding from audio
                    embedding = tts.encode_voice(audio_src)
                    
                    # Save as NumPy file
                    np.save(dest_path, embedding)
                    
                    self.refresh_ui()
                    gui.messageBox(_("Voice '{name}' successfully created!").format(name=base_name), _("Success"))
                except Exception as e:
                    log.error(f"Pocket TTS Conversion Error: {e}")
                    gui.messageBox(f"Error during conversion: {e}", _("Error"))

    def onRenameVoice(self, evt):
        """Renames the actual .npy file on the disk."""
        old_filename = self.voiceList.GetStringSelection()
        new_name = self.nameEdit.GetValue().strip()
        if not old_filename or not new_name:
            return

        old_path = os.path.join(self.voices_dir, old_filename)
        new_path = os.path.join(self.voices_dir, f"{new_name}.npy")

        try:
            os.rename(old_path, new_path)
            self.refresh_ui()
            gui.messageBox(_("Voice renamed to {name}").format(name=new_name), _("Success"))
        except Exception as e:
            gui.messageBox(f"Failed to rename: {e}", _("Error"))

    def onRemoveVoice(self, evt):
        """Deletes the .npy voice file after confirmation."""
        sel = self.voiceList.GetStringSelection()
        if not sel:
            return
        
        if gui.messageBox(
            _("Are you sure you want to remove the voice '{name}'?").format(name=sel),
            _("Confirm"),
            wx.YES_NO | wx.ICON_QUESTION
        ) == wx.YES:
            try:
                os.remove(os.path.join(self.voices_dir, sel))
                self.refresh_ui()
            except Exception as e:
                log.error(f"Pocket TTS: Error while removing: {e}")

    def onVoiceSelect(self, evt):
        """Updates the edit field when a voice is selected from the list."""
        filename = self.voiceList.GetStringSelection()
        if filename:
            self.nameEdit.SetValue(os.path.splitext(filename)[0])

    def refresh_ui(self):
        """Reloads the voice list and updates the selection."""
        voices = self.get_installed_voices()
        self.voiceList.Clear()
        self.voiceList.AppendItems(voices)
        if voices:
            self.voiceList.SetSelection(0)
            self.onVoiceSelect(None)
        else:
            self.nameEdit.Clear()

    def onSave(self):
        pass