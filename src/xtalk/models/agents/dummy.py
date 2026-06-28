"""Lightweight dummy agent implementation for tests and sample wiring."""

from typing import Any, AsyncIterator, Iterable

from .interfaces import Agent, AgentContext, AgentOutput
from ..registry import model


@model
class DummyAgent(Agent):
    """Dummy agent that always returns the same response text.

    Parameters
    ----------
    default_response : str, optional
        Response text yielded for every input turn.
    """

    def __init__(
        self,
        default_response: str = 'The term "psychology" can refer to the entirety of humans\' internal mental activities. It can also denote an organism\'s subjective reflection of the objective world, as well as the processes and phenomena related to mental activity, such as emotion, thinking, and behavior. In addition, "psychology" is often used to refer to the academic discipline that studies human psychological phenomena, mental functions, and behavior.',
    ) -> None:
        """Initialize the dummy agent."""
        self.default_response = default_response

    def restore_history(self, messages: list[dict[str, Any]]) -> None:
        """Ignore persisted history for the stateless dummy agent.

        Parameters
        ----------
        messages : list[dict[str, Any]]
            Persisted messages. Ignored by this implementation.
        """

        del messages
        return None

    def accept(self, context: AgentContext) -> Iterable[AgentOutput]:
        """Synchronously bridge ``async_accept()`` for the stateless agent.

        Parameters
        ----------
        context : AgentContext
            Context payload forwarded from serving events.
        """

        yield from self.sync_iter_from_async(self.async_accept(context))

    async def async_accept(
        self,
        context: AgentContext,
    ) -> AsyncIterator[AgentOutput]:
        """Yield a canned response for generation-triggering contexts.

        Parameters
        ----------
        context : AgentContext
            Context payload forwarded from serving events.
        """

        context_type = str(context.get("type", "") or "")
        if context_type not in {"asr_final", "embedding"}:
            return
        yield self.default_response

    def clone(self) -> "Agent":
        """Create a fresh dummy agent with the same canned response.

        Returns
        -------
        Agent
            Cloned dummy agent instance.
        """

        return DummyAgent(self.default_response)
