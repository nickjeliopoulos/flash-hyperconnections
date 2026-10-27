"""Exercise the local journal CLI without PyTorch or GPU dependencies."""

import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CLI = Path(__file__).with_name("forum")


class ForumTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.forum = self.root / ".agents" / "forum"
        self.env = dict(os.environ, KERNEL_FORUM_DIR=str(self.forum))

    def run_cli(self, *args, check=True, env=None, cwd=None):
        result = subprocess.run(
            [sys.executable, str(CLI), *args],
            cwd=cwd or self.root,
            env=self.env if env is None else env,
            text=True,
            capture_output=True,
        )
        if check:
            self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def test_roundtrip_and_correction(self):
        grounding = self.root / "goal.md"
        grounding.write_text("## Target\nReduce latency while preserving accuracy.\n", encoding="utf-8")
        self.run_cli("init", "--agent", "optimizer", "--grounding", str(grounding))
        body = self.root / "trial.md"
        body.write_text("Trial 001: Fréchet derivative ✓\nBaseline: 25 µs\n", encoding="utf-8")
        self.run_cli("append", "--agent", "optimizer", str(body))
        self.run_cli("append", "--agent", "optimizer", "--type", "correction", "-m", "Timing invalid.")
        before = (self.forum / "main.md").read_bytes()
        recent = self.run_cli("pull").stdout
        self.assertIn("Reduce latency", recent)
        self.assertIn("25 µs", recent)
        self.assertIn("(correction)", recent)
        entries = self.run_cli("list").stdout.splitlines()
        self.assertEqual(len(entries), 3)
        entry_id = entries[1].split("\t")[0]
        self.assertIn("Fréchet", self.run_cli("show", entry_id).stdout)
        self.assertIn("25 µs", self.run_cli("search", "BASELINE").stdout)
        self.assertNotIn("Trial 001", self.run_cli("recent", "--count", "0").stdout)
        self.assertEqual((self.forum / "main.md").read_bytes(), before)
        self.assertFalse((self.forum / ".git").exists())

    def test_concurrent_append(self):
        self.run_cli("init", "--agent", "optimizer")

        def append(i):
            self.run_cli("append", "--agent", f"worker-{i}", "-m", f"unique trial {i}")

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(append, range(24)))
        entries = self.run_cli("list").stdout.splitlines()
        self.assertEqual(len(entries), 25)
        self.assertEqual(len({entry.split("\t")[0] for entry in entries}), 25)
        messages = {entry.split("\t")[-1] for entry in entries}
        self.assertTrue({f"unique trial {i}" for i in range(24)} <= messages)

    def test_concurrent_init_does_not_overwrite(self):
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda _: self.run_cli("init", "--agent", "optimizer", check=False), range(6)))
        self.assertEqual(sum(result.returncode == 0 for result in results), 1)
        before = (self.forum / "main.md").read_bytes()
        self.assertNotEqual(self.run_cli("init", "--agent", "other", check=False).returncode, 0)
        self.assertEqual((self.forum / "main.md").read_bytes(), before)

    def test_rejects_empty_messages_and_reserved_markers(self):
        self.run_cli("init", "--agent", "optimizer")
        before = (self.forum / "main.md").read_bytes()
        for body in (" ", "x <<FORUM_ENTRY>>", "<<END_FORUM_ENTRY>>"):
            result = self.run_cli("append", "--agent", "optimizer", "-m", body, check=False)
            self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.forum / "main.md").read_bytes(), before)
        result = self.run_cli(
            "init", "--forum", "bad", "--agent", "optimizer", "--title", "<<FORUM_ENTRY>>", check=False
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.forum / "bad.md").exists())

    def test_root_discovery_and_local_only_storage(self):
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        nested = self.root / "work" / "nested"
        nested.mkdir(parents=True)
        env = dict(os.environ)
        env.pop("KERNEL_FORUM_DIR", None)
        self.run_cli("init", "--agent", "optimizer", "--forum", "../trial", cwd=nested, env=env)
        self.assertTrue((self.forum / "trial.md").exists())
        self.assertFalse((self.root / "trial.md").exists())
        self.assertFalse((nested / ".agents").exists())
        self.assertFalse((self.forum / ".git").exists())
        self.assertFalse((self.root / ".git" / "index").exists())


if __name__ == "__main__":
    unittest.main()
