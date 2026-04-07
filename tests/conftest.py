"""
conftest.py — makes the project root available on sys.path
so that `from src.data.ingest import ...` works from any test.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))