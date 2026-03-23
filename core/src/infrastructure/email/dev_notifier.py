from __future__ import annotations

import logging


logger = logging.getLogger(__name__)


class DevLogNotifier:
    def send_password_reset(self, email: str, token: str) -> None:
        token_preview = f"{token[:6]}..." if token else "<empty>"
        logger.info("password_reset_requested email=%s token_preview=%s", email, token_preview)
