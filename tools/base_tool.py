from abc import ABC, abstractmethod


class BaseTool(ABC):
    name: str
    category: str = "utility"
    summarizable: bool = False
    requires_confirmation: bool = False

    @abstractmethod
    def get_description(self) -> dict:
        """Return the tool schema for the Claude API."""
        ...

    @abstractmethod
    def execute(self, **kwargs) -> dict:
        """Run the tool and return a result dict with 'result' or 'error' key."""
        ...

    def validate_input(self, **kwargs) -> bool:
        """Validate parameters before execution. Override to add checks."""
        return True

    def get_confirmation_message(self, **kwargs) -> str:
        """Return a human-readable summary of the action about to be taken.

        Override on tools with requires_confirmation=True to show specific details.
        """
        return f"About to run '{self.name}' with: {kwargs}"

    def handle_error(self, error: Exception) -> dict:
        """Return a consistent error dict."""
        return {"error": str(error)}
