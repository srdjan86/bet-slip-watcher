from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass
class NotifierConfig:
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    ntfy_topic: str = ""


class Notifier:
    def __init__(self, cfg: NotifierConfig) -> None:
        self._cfg = cfg

    async def send(self, title: str, body: str) -> None:
        logger.info("%s\n%s", title, body)
        if self._cfg.telegram_bot_token and self._cfg.telegram_chat_id:
            await self._telegram(f"{title}\n\n{body}")
        if self._cfg.ntfy_topic:
            await self._ntfy(title, body)

    async def _telegram(self, text: str) -> None:
        url = f"https://api.telegram.org/bot{self._cfg.telegram_bot_token}/sendMessage"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                url,
                json={
                    "chat_id": self._cfg.telegram_chat_id,
                    "text": text,
                    "disable_web_page_preview": True,
                },
            )
            resp.raise_for_status()

    async def _ntfy(self, title: str, body: str) -> None:
        topic = self._cfg.ntfy_topic.strip("/")
        url = f"https://ntfy.sh/{topic}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                url,
                content=body.encode("utf-8"),
                headers={"Title": title, "Priority": "high"},
            )
            resp.raise_for_status()
