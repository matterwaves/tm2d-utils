"""Compatibility wrapper for the misspelled rotation_matricies module."""

from .rotation_matrices import get_cisTEM_rotation_matrix, get_rotation_matrix

__all__ = ["get_rotation_matrix", "get_cisTEM_rotation_matrix"]
