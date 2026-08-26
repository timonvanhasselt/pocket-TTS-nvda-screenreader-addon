import globalPluginHandler
import gui
import scriptHandler
import ui
import wx
from .pocketTTSSettingsPanel import PocketTTSSettingsPanel

try:
    import addonHandler
    addonHandler.initTranslation()
except Exception:
    pass

try:
    from gettext import gettext as _
except Exception:
    _ = lambda text: text

try:
    from .githubReleaseUpdater import GitHubReleaseUpdater
except Exception:
    GitHubReleaseUpdater = None


class GlobalPlugin(globalPluginHandler.GlobalPlugin):
    scriptCategory = _("Pocket-TTS")

    def __init__(self):
        super().__init__()
        self._updater = None
        # Register the settings panel in NVDA settings
        if PocketTTSSettingsPanel not in gui.settingsDialogs.NVDASettingsDialog.categoryClasses:
            gui.settingsDialogs.NVDASettingsDialog.categoryClasses.append(PocketTTSSettingsPanel)
        if GitHubReleaseUpdater:
            self._updater = GitHubReleaseUpdater(
                "Pocket-TTS",
                "Pocket-TTS",
                "timonvanhasselt",
                "pocket-TTS-nvda-screenreader-addon",
            )
            self._updater.start()

    def terminate(self):
        # Clean up by removing the settings panel when the plugin terminates
        if self._updater:
            self._updater.stop()
        if PocketTTSSettingsPanel in gui.settingsDialogs.NVDASettingsDialog.categoryClasses:
            gui.settingsDialogs.NVDASettingsDialog.categoryClasses.remove(PocketTTSSettingsPanel)
        super().terminate()

    @scriptHandler.script(description=_("Check for Pocket-TTS updates"))
    def script_checkForPocketTTSUpdate(self, gesture):
        if self._updater:
            wx.CallAfter(self._updater.checkNow, True)
        else:
            ui.message(_("Updater is not available"))
