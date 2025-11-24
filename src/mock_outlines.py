"""
Workaround script to disable outlines import in vLLM
Run this before starting the inference service
"""
import sys
import os

# Disable outlines by mocking the module
class MockOutlines:
    def __getattr__(self, name):
        return None

sys.modules['outlines'] = MockOutlines()
sys.modules['outlines.types'] = MockOutlines()
sys.modules['outlines.types.airports'] = MockOutlines()
sys.modules['pyairports'] = MockOutlines()
sys.modules['pyairports.airports'] = MockOutlines()

print("✓ Outlines/pyairports mocked successfully")
