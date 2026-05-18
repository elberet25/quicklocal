from abc import ABC, abstractmethod


class BaseTool(ABC):
    name: str

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

    def handle_error(self, error: Exception) -> dict:
        """Return a consistent error dict."""
        return {"error": str(error)}
