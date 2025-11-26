"""
Workaround script to disable outlines import in vLLM
Run this before starting the inference service
"""
import sys
import os

# Disable outlines by mocking the module
class MockAirportsModule:
    """Mock for pyairports.airports module"""
    AIRPORT_LIST = []  # Empty list to make it iterable

    def __getattr__(self, name):
        return {}

class MockPyairportsModule:
    """Mock for pyairports module"""
    airports = MockAirportsModule()

    def __getattr__(self, name):
        return {}

class MockOutlines:
    def __getattr__(self, name):
        return {}

sys.modules['outlines'] = MockOutlines()
sys.modules['outlines.types'] = MockOutlines()
sys.modules['outlines.types.airports'] = MockOutlines()
sys.modules['pyairports'] = MockPyairportsModule()
sys.modules['pyairports.airports'] = MockAirportsModule()

print("✓ Outlines/pyairports mocked successfully")
