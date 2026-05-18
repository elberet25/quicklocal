try:
    from tools.base_tool import BaseTool
except ImportError:
    from base_tool import BaseTool


class CalculatorTool(BaseTool):
    name = "calculate"

    def get_description(self) -> dict:
        return {
            "name": self.name,
            "description": (
                "Performs basic arithmetic: add, subtract, multiply, or divide two numbers. "
                "Use this whenever the user asks to calculate, compute, or do math."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The arithmetic operation to perform.",
                    },
                    "a": {"type": "number", "description": "The first number."},
                    "b": {"type": "number", "description": "The second number."},
                },
                "required": ["operation", "a", "b"],
            },
        }

    def execute(self, **kwargs) -> dict:
        try:
            operation = kwargs.get("operation", "")
            a = kwargs.get("a", 0)
            b = kwargs.get("b", 0)
            if operation == "add":
                return {"result": str(a + b)}
            elif operation == "subtract":
                return {"result": str(a - b)}
            elif operation == "multiply":
                return {"result": str(a * b)}
            elif operation == "divide":
                if b == 0:
                    return {"result": "Error: division by zero"}
                return {"result": str(a / b)}
            else:
                return {"result": f"Error: unknown operation '{operation}'"}
        except Exception as e:
            return self.handle_error(e)

    def validate_input(self, **kwargs) -> bool:
        return kwargs.get("operation") in ("add", "subtract", "multiply", "divide")
