import datetime

try:
    from tools.base_tool import BaseTool
except ImportError:
    from base_tool import BaseTool


class TimeTool(BaseTool):
    name = "get_current_time"

    def get_description(self) -> dict:
        return {
            "name": self.name,
            "description": (
                "Returns the current local date and time. "
                "Use this whenever the user asks what time or date it is."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }

    def execute(self, **kwargs) -> dict:
        try:
            now = datetime.datetime.now()
            return {"result": now.strftime("%A, %B %d %Y — %I:%M:%S %p")}
        except Exception as e:
            return self.handle_error(e)

    def validate_input(self, **kwargs) -> bool:
        return True
