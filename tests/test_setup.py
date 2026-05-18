import sys
print(f"✓ Python version: {sys.version}")
print(f"✓ Python path: {sys.executable}")

try:
    import anthropic
    print("✓ anthropic installed")
except ImportError:
    print("✗ anthropic not installed")

try:
    import dotenv
    print("✓ python-dotenv installed")
except ImportError:
    print("✗ python-dotenv not installed")

print("\n✅ All good! Ready to code.")
