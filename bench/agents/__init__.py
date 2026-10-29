"""Harness subclasses that arms name by import path ("module.path:Class").

The bench runs them with `harbor run --agent-import-path`. Harbor then has to
import this package, so every Harbor child process and schema subprocess gets
the project root on PYTHONPATH (keys.child_env). An arm that uses one sets
`family` to the harness it extends, so reference figures, cap rules and
reports treat it as that harness.
"""
