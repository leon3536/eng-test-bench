"""Windows-safe file writes: retries on locked targets, write-once records,
append-only logs that survive a crash mid-line."""

from __future__ import annotations

import json
import pathlib
from collections.abc import Callable
from typing import Any
from unittest import mock

from tests.support import BenchTestCase

from bench import fsutil


def flaky_replace(failures: int) -> tuple[Callable[[pathlib.Path, Any], pathlib.Path], list[pathlib.Path]]:
    """Path.replace that raises PermissionError `failures` times, then works."""
    real = pathlib.Path.replace
    calls: list[pathlib.Path] = []

    def replace(self: pathlib.Path, target: Any) -> pathlib.Path:
        calls.append(self)
        if len(calls) <= failures:
            raise PermissionError(13, "The process cannot access the file", str(target))
        return real(self, target)

    return replace, calls


class AtomicWriteTests(BenchTestCase):
    def test_survives_two_permission_errors(self) -> None:
        path = self.tmpdir() / "meta.json"
        path.write_text('{"old": true}\n', encoding="utf-8")
        replace, calls = flaky_replace(2)
        with mock.patch.object(pathlib.Path, "replace", replace):
            ok = fsutil.atomic_write_json(path, {"title": "Ünïcode", "n": 1})
        self.assertTrue(ok)
        self.assertEqual(len(calls), 3)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"title": "Ünïcode", "n": 1})
        self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_gives_up_after_patience_and_keeps_old_file(self) -> None:
        path = self.tmpdir() / "queue.json"
        path.write_text('{"old": true}\n', encoding="utf-8")
        replace, calls = flaky_replace(10_000)
        with mock.patch.object(pathlib.Path, "replace", replace):
            ok = fsutil.atomic_write_json(path, {"new": True}, patience=0.3)
        self.assertFalse(ok)
        self.assertGreater(len(calls), 1)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"old": True})
        self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_other_errors_propagate_and_clean_up(self) -> None:
        path = self.tmpdir() / "x.json"

        def broken(self: pathlib.Path, target: Any) -> None:
            raise OSError(22, "Invalid argument")

        with mock.patch.object(pathlib.Path, "replace", broken):
            with self.assertRaises(OSError):
                fsutil.atomic_write_text(path, "text")
        self.assertEqual(list(path.parent.iterdir()), [])

    def test_unserializable_data_writes_nothing(self) -> None:
        path = self.tmpdir() / "x.json"
        with self.assertRaises(TypeError):
            fsutil.atomic_write_json(path, {"bad": object()})
        self.assertEqual(list(path.parent.iterdir()), [])

    def test_text_is_utf8_with_lf_endings(self) -> None:
        path = self.tmpdir() / "sub" / "a.yaml"
        self.assertTrue(fsutil.atomic_write_text(path, "a: 1\nb: «2»\n"))
        data = path.read_bytes()
        self.assertNotIn(b"\r\n", data)
        self.assertEqual(data.decode("utf-8"), "a: 1\nb: «2»\n")

    def test_indent_none_writes_one_line(self) -> None:
        path = self.tmpdir() / "a.json"
        fsutil.atomic_write_json(path, {"a": [1, 2]}, indent=None)
        self.assertEqual(path.read_text(encoding="utf-8"), '{"a": [1, 2]}\n')


class ExclusiveTests(BenchTestCase):
    def test_create_once_then_refuse(self) -> None:
        path = self.tmpdir() / "designs" / "d-x" / "v0001.json"
        fsutil.create_exclusive_json(path, {"version": 1, "title": "Ärm"})
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            fsutil.create_exclusive_json(path, {"version": 2})
        self.assertEqual(path.read_bytes(), before)
        self.assertNotIn(b"\r\n", before)
        self.assertEqual(json.loads(before.decode("utf-8"))["title"], "Ärm")

    def test_unserializable_leaves_no_file(self) -> None:
        path = self.tmpdir() / "v0001.json"
        with self.assertRaises(TypeError):
            fsutil.create_exclusive_json(path, {"bad": {1, 2}})
        self.assertFalse(path.exists())


class JsonlTests(BenchTestCase):
    def test_append_and_read(self) -> None:
        path = self.tmpdir() / "usage.jsonl"
        fsutil.append_jsonl(path, [{"a": 1}, {"b": "ü"}])
        fsutil.append_jsonl(path, [{"c": 3}])
        fsutil.append_jsonl(path, [])
        self.assertEqual(fsutil.read_jsonl(path), [{"a": 1}, {"b": "ü"}, {"c": 3}])
        self.assertNotIn(b"\r\n", path.read_bytes())

    def test_unicode_line_separators_round_trip(self) -> None:
        path = self.tmpdir() / "messages.jsonl"
        records = [{"text": "a b"}, {"text": "p q"}, {"text": "nel\x85x"}, {"text": "ok\r"}]
        fsutil.append_jsonl(path, records)
        self.assertEqual(fsutil.read_jsonl(path), records)
        self.assertEqual(path.read_bytes().count(b"\n"), len(records))

    def test_read_keeps_raw_separators_from_other_writers(self) -> None:
        path = self.tmpdir() / "messages.jsonl"
        raw = '{"text": "a b"}\n{"text": "nel\x85x"}\r\n{"text": "ok"}\n'
        path.write_bytes(raw.encode("utf-8"))
        self.assertEqual(
            fsutil.read_jsonl(path), [{"text": "a b"}, {"text": "nel\x85x"}, {"text": "ok"}]
        )

    def test_read_drops_truncated_last_line(self) -> None:
        path = self.tmpdir() / "messages.jsonl"
        path.write_bytes(b'{"a": 1}\n{"b": 2}\n{"c": "trunc')
        self.assertEqual(fsutil.read_jsonl(path), [{"a": 1}, {"b": 2}])

    def test_read_missing_file(self) -> None:
        self.assertEqual(fsutil.read_jsonl(self.tmpdir() / "none.jsonl"), [])

    def test_append_after_partial_line_starts_fresh(self) -> None:
        path = self.tmpdir() / "t.jsonl"
        path.write_bytes(b'{"a": 1}\n{"partial": ')
        fsutil.append_jsonl(path, [{"b": 2}])
        self.assertEqual(fsutil.read_jsonl(path), [{"a": 1}, {"b": 2}])

    def test_append_retries_a_locked_file(self) -> None:
        path = self.tmpdir() / "t.jsonl"
        real_open = open
        attempts = []

        def locked_twice(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            if "a" in mode:
                attempts.append(mode)
                if len(attempts) <= 2:
                    raise PermissionError(13, "locked", str(file))
            return real_open(file, mode, *args, **kwargs)

        with mock.patch("bench.fsutil.open", locked_twice, create=True):
            fsutil.append_jsonl(path, [{"x": 1}])
        self.assertEqual(len(attempts), 3)
        self.assertEqual(fsutil.read_jsonl(path), [{"x": 1}])

    def test_append_gives_up_with_permission_error(self) -> None:
        path = self.tmpdir() / "t.jsonl"

        def always_locked(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            raise PermissionError(13, "locked", str(file))

        with mock.patch("bench.fsutil.open", always_locked, create=True):
            with self.assertRaises(PermissionError):
                fsutil.append_jsonl(path, [{"x": 1}], patience=0.2)


class HashingTests(BenchTestCase):
    def test_canonical_json_ignores_key_order(self) -> None:
        a = fsutil.canonical_json({"b": 1, "a": {"y": [1, 2], "x": "é"}})
        b = fsutil.canonical_json({"a": {"x": "é", "y": [1, 2]}, "b": 1})
        self.assertEqual(a, b)
        self.assertEqual(a, '{"a":{"x":"é","y":[1,2]},"b":1}')

    def test_sha256_hex(self) -> None:
        self.assertEqual(fsutil.sha256_hex("abc"),
                         "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


class SegmentTests(BenchTestCase):
    def test_reserved_and_unsafe_names(self) -> None:
        for bad in ("", ".", "..", "con", "CON", "con.txt", "nul", "Aux", "prn", "com1", "LPT9",
                    "com0", "a/b", "a\\b", "a:b", "what?", "trailing.", "trailing ", "tab\there",
                    "x" * 256):
            with self.subTest(bad=bad):
                self.assertFalse(fsutil.is_safe_segment(bad))

    def test_ordinary_names(self) -> None:
        for good in ("design-1", "d-ds-pairs-block1-7f3a", "c-20261028-231502-ab12", "v0001.json",
                     "console", "com10", "lpt", "a.b"):
            with self.subTest(good=good):
                self.assertTrue(fsutil.is_safe_segment(good))

    def test_reserved_set(self) -> None:
        self.assertIn("com1", fsutil.WINDOWS_RESERVED)
        self.assertIn("lpt9", fsutil.WINDOWS_RESERVED)
        self.assertNotIn("con1", fsutil.WINDOWS_RESERVED)


if __name__ == "__main__":
    import unittest

    unittest.main()
