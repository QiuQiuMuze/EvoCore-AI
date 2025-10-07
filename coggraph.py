"""Compatibility wrapper for the refactored CogGraph."""
from coggraph_components.core import CogGraph
from coggraph_components.task import TaskInjector

__all__ = ["CogGraph", "TaskInjector"]
