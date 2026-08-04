"""Desktop-specific XTalk service composition."""

from __future__ import annotations

import logging
from typing import Any

from xtalk import Xtalk
from xtalk.serving.events import TTSPlaybackFinished
from xtalk.serving.modules.tts_playback_manager import TTSPlaybackManager
from xtalk.serving.service import DefaultService


logger = logging.getLogger(__name__)


class DesktopTTSPlaybackManager(TTSPlaybackManager):
    """Finalize completed desktop playback with the full generated response.

    The generic playback manager intentionally commits only text confirmed by
    incremental audio alignment. Desktop output sessions already emit
    ``TTSPlaybackFinished`` after every queued audio chunk has played, so using
    a conservative intermediate alignment prefix at that point truncates chat
    history and leaves the next model turn conditioned on an unfinished reply.

    Interrupted playback continues to use the inherited stop path, which
    commits only the text prefix that the user actually heard.
    """

    async def _publish_response_finish(
        self,
        event: TTSPlaybackFinished,
    ) -> None:
        """Commit the complete generated turn after normal playback finishes.

        Parameters
        ----------
        event : TTSPlaybackFinished
            Confirmation that the desktop output queue finished playing.
        """

        del event
        if not self._received_audio:
            logger.warning(
                "Desktop TTS playback finished without generated audio; "
                "discarding unplayed response - session: %s",
                self.session_id,
            )
            self._reset_all_state()
            return

        played_text = self._build_reported_text()
        if len(self._reported_text) > len(played_text):
            played_text = self._reported_text
        final_text = self._pending_text or played_text
        if not final_text:
            logger.warning(
                "Desktop TTS playback finished without response text - session: %s",
                self.session_id,
            )
            self._reset_all_state()
            return

        if played_text and final_text != played_text:
            logger.debug(
                "Desktop TTS normal completion promoted generated text - "
                "session: %s, aligned_length: %s, generated_length: %s",
                self.session_id,
                len(played_text),
                len(final_text),
            )
        try:
            await self._commit_playback_text(final_text)
        finally:
            self._reset_all_state()


class DesktopService(DefaultService):
    """Use desktop playback finalization with the standard XTalk managers."""

    MANAGER_CLASSES = [
        DesktopTTSPlaybackManager
        if manager_class is TTSPlaybackManager
        else manager_class
        for manager_class in DefaultService.MANAGER_CLASSES
    ]


class DesktopXtalk(Xtalk):
    """Build XTalk sessions around :class:`DesktopService`."""

    @classmethod
    def _build_from_config_dict(cls, config: dict[str, Any]) -> DesktopXtalk:
        """Build a desktop runtime from an effective XTalk configuration.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration after all public builder transforms have run.

        Returns
        -------
        DesktopXtalk
            Runtime whose sessions use desktop playback semantics.
        """

        models = cls.create_models_from_config(config_path_or_dict=config)
        service_config = config.get("service_config", {})
        if not isinstance(service_config, dict):
            raise ValueError("service_config must be an object")

        max_sessions: int | None = None
        if "max_connections" in config:
            max_sessions = int(config["max_connections"])
        service_prototype = DesktopService(
            models=models,
            service_config=service_config,
        )
        return cls(
            service_prototype=service_prototype,
            max_sessions=max_sessions,
        )
