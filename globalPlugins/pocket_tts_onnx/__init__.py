import globalPluginHandler
import gui
from .pocketTTSSettingsPanel import PocketTTSSettingsPanel

class GlobalPlugin(globalPluginHandler.GlobalPlugin):
    def __init__(self):
        super().__init__()
        # Register the settings panel in NVDA settings
        if PocketTTSSettingsPanel not in gui.settingsDialogs.NVDASettingsDialog.categoryClasses:
            gui.settingsDialogs.NVDASettingsDialog.categoryClasses.append(PocketTTSSettingsPanel)

    def terminate(self):
        # Clean up by removing the settings panel when the plugin terminates
        super().terminate()
        if PocketTTSSettingsPanel in gui.settingsDialogs.NVDASettingsDialog.categoryClasses:
            gui.settingsDialogs.NVDASettingsDialog.categoryClasses.remove(PocketTTSSettingsPanel)