from typing import Union
from .interfaces import Agent, AgentInput


class DummyAgent(Agent):
    """
    Dummy LLM model for testing purposes.

    This model ignores the input and returns a predefined test response.
    """

    def __init__(
        self,
        default_response: str = 'The term "psychology" can refer to the entirety of humans\' internal mental activities. It can also denote an organism\'s subjective reflection of the objective world, as well as the processes and phenomena related to mental activity, such as emotion, thinking, and behavior. In addition, "psychology" is often used to refer to the academic discipline that studies human psychological phenomena, mental functions, and behavior.',
    ):
        """
        Initialize the DummyLLM model.

        Args:
            default_response (str): The default response to return for any input
        """
        self.default_response = default_response

    def generate(self, input: Union[str, AgentInput]) -> str:
        """
        Generate response for the given input (dummy implementation).

        This method ignores the input and returns the predefined test response.

        Args:
            input: Input text or dict with content/context (ignored)

        Returns:
            str: The predefined test response
        """
        return self.default_response

    def clone(self) -> "Agent":
        return DummyAgent(self.default_response)
