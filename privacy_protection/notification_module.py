"""
Cross-platform system notifications + optional beep.

Uses `plyer` so the notification works on Windows, macOS, and most
Linux desktops. Falls back gracefully if the OS notifier is missing.
"""

import sys
import time
from typing import Optional

from . import config


class Notifier:
    """State-aware notifier that respects a cooldown window."""

    def __init__(
        self,
        cooldown: float = config.alert_cooldown,
        play_beep: bool = config.play_beep,
    ) -> None:
        self.cooldown = cooldown
        self.play_beep = play_beep
        self._last_alert_at: float = 0.0

        # Lazy import so the rest of the system still works if plyer
        # isn't installed for some reason.
        try:
            from plyer import notification  # type: ignore
            self._notify = notification
        except Exception as exc:
            print(f"[notify] plyer unavailable: {exc}")
            self._notify = None

    # ------------------------------------------------------------------
    def alert(
        self,
        title: str = "Privacy Alert",
        message: str = "Unknown person detected in camera view",
    ) -> bool:
        """Fire a system notification if the cooldown has elapsed.

        Returns True when the notification was actually dispatched.
        """
        now = time.time()
        if (now - self._last_alert_at) < self.cooldown:
            return False

        self._last_alert_at = now
        self._send(title, message)
        if self.play_beep:
            self._beep()
        return True

    # ------------------------------------------------------------------
    def _send(self, title: str, message: str) -> None:
        if self._notify is None:
            print(f"[ALERT] {title}: {message}")
            return
        try:
            self._notify.notify(
                title=title,
                message=message,
                app_name="Privacy Protection",
                timeout=5,
            )
        except Exception as exc:
            # Some Linux distros need a notification daemon; don't crash.
            print(f"[notify] system notification failed: {exc}")
            print(f"[ALERT] {title}: {message}")

    # ------------------------------------------------------------------
    def _beep(self) -> None:
        """Best-effort short beep. Silent on failure."""
        try:
            if sys.platform.startswith("win"):
                import winsound  # type: ignore
                winsound.Beep(1000, 200)
            else:
                # ASCII BEL — works in most terminals; harmless otherwise.
                sys.stdout.write("\a")
                sys.stdout.flush()
        except Exception:
            pass
