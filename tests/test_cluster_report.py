"""The shareable report is an allowlist, not a redacted raw log."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("report", Path(__file__).resolve().parents[1] / "tools/fronk_report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class ReportTests(unittest.TestCase):
    def test_private_state_cannot_escape(self):
        state = {"secret": "pairing-secret", "text": "private conversation", "failure": "/private/path",
                 "nodes": {"alice-laptop": {"address": "10.11.12.13", "iface": "private-interface",
                                            "blocks": 3, "backend": "metal", "link": "thunderbolt",
                                            "cpu_model": "Apple M1 Pro", "mem_total_bytes": 16 * 2**30},
                           "head": {"blocks": 2, "hybrid": {"RAM": "1", "SSD": "1"}}},
                 "report": {"bottleneck": "alice-laptop", "toks": "15.61", "wall_s": "private-error"}}
        result = report.render(state)
        for secret in ("pairing-secret", "private conversation", "/private/path", "alice-laptop",
                       "10.11.12.13", "private-interface", "private-error"):
            self.assertNotIn(secret, result)
        for expected in ("Apple M1 Pro", "16.0", "1 / 1", "Node 1", "15.61"):
            self.assertIn(expected, result)

    def test_unknown_or_invalid_metrics_remain_unknown(self):
        for value in ("nan", "inf", "-2", {}, None):
            self.assertEqual(report.number(value), "unknown")
        self.assertIn("unknown", report.render({}))


if __name__ == "__main__":
    unittest.main()
