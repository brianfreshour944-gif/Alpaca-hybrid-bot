#!/usr/bin/env python3
"""
Test script to demonstrate cohere library import and basic usage.
This shows how to import cohere without installing it in the current environment.
"""

# First, let's show what the import would look like
try:
    # This would work if cohere was installed
    import cohere
    print("✅ Cohere library imported successfully!")
    print(f"Cohere version: {getattr(cohere, '__version__', 'unknown')}")

    # Example of how to use cohere (would need API key)
    # co = cohere.Client('your-api-key')
    # response = co.generate(prompt="Hello, world!")
    # print(response)

except ImportError:
    print("⚠️ Cohere library not installed (this is expected in this environment)")
    print("\nTo install cohere, you would run:")
    print("pip install cohere")
    print("\nThen you can import it with:")
    print("import cohere")

print("\nCohere is an AI platform that provides:")
print("- Natural language processing")
print("- Text generation")
print("- Embeddings")
print("- Classification")
print("- And other AI capabilities")

print("\nYou would typically use it like this:")
print("""
import cohere
co = cohere.Client('your-api-key-here')
response = co.generate(
    model='command',
    prompt='Write a Python function to calculate Fibonacci numbers'
)
print(response.generations[0].text)
""")