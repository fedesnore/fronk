"""Console configuration regressions; isolated HTTP server, no real head or nodes."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("fronk_web", os.environ.get("FRONK_WEB_UNDER_TEST", ROOT / "tools/fronk-web.py"))
web = importlib.util.module_from_spec(spec)
spec.loader.exec_module(web)


class ConsoleTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.conf = root / "fronk.conf"
        self.conf.write_text("state-dir state\nds4 tools/gpu-head/run.sh\nhead-blocks 9\nburst 1\n")
        (root / "state").mkdir()
        (root / "state/ds4-ctx").write_text("4096\n")
        self.root = root
        for name, value in (("ROOT", str(root)), ("CONF", str(self.conf)), ("FOLLOW_LOG", None),
                            ("STATE", dict(web.STATE, proc=None, lines=[], secret="", phase="IDLE"))):
            p = patch.object(web, name, value)
            p.start(); self.addCleanup(p.stop)
        p = patch.object(web, "start_head", return_value="head started")
        self.start = p.start(); self.addCleanup(p.stop)
        self.server = web.ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
        thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def request(self, path, body=None, token=None):
        data = json.dumps(body).encode() if body is not None else None
        headers = {"X-Fronk-Token": web.TOKEN if token is None else token}
        req = urllib.request.Request(f"http://127.0.0.1:{self.server.server_port}" + path, data, headers)
        try:
            response = urllib.request.urlopen(req, timeout=3)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.code, json.load(response)

    def test_start_preserves_omitted_placement(self):
        before = self.conf.read_bytes()
        self.assertEqual(self.request("/start", {"refresh": False})[0], 200)
        self.assertEqual(self.conf.read_bytes(), before)
        self.start.assert_called_once_with(False)

    def test_start_explicit_automatic_placement(self):
        self.assertEqual(self.request("/start", {"head_blocks": ""})[0], 200)
        self.assertNotIn("head-blocks", web.conf_read())

    def test_invalid_placement_does_not_start_or_change_config(self):
        before = self.conf.read_bytes()
        for value in (None, True, [], "21", "-1", "2.5", "9\nmodel bogus"):
            with self.subTest(value=value):
                self.assertEqual(self.request("/start", {"head_blocks": value})[0], 400)
                self.assertEqual(self.conf.read_bytes(), before)
        self.start.assert_not_called()

    def test_auth_required_for_configuration(self):
        before = self.conf.read_bytes()
        self.assertEqual(self.request("/start", {"head_blocks": ""}, "wrong")[0], 403)
        self.assertEqual(self.conf.read_bytes(), before)

    def test_context_contains_installed_4096(self):
        status, body = self.request("/api/context-size")
        self.assertEqual(status, 200)
        self.assertEqual(body["current"], 4096)
        self.assertIn(4096, [s["ctx"] for s in body["sizes"]])
        self.assertEqual(self.request("/api/context-size", {"ctx": 4096})[0], 200)

    def test_missing_context_uses_engine_default(self):
        (self.root / "state/ds4-ctx").unlink()
        self.assertEqual(self.request("/api/context-size")[1]["current"], 8192)
        web.conf_write({"ds4": "builtin"})
        self.assertEqual(self.request("/api/context-size")[1]["current"], 32768)

    def test_selftest_error_survives_generic_cycle_message(self):
        failure = "SELFTEST warm-up failed at layer 0; check compute and replies on node-1"
        web.STATE["lines"] = ["fronk: backend avx2 ready", failure, "fronk: cycle failed"]
        self.assertEqual(self.request("/state")[1]["failure"], failure)

    def test_cycle_message_is_reported_when_nothing_else_failed(self):
        last = "fronk: cycle failed; returning to CONNECTION on [m]"
        web.STATE["lines"] = ["fronk: backend avx2 ready", "fronk: cycle step at head.c:300 failed", last]
        self.assertEqual(self.request("/state")[1]["failure"], last)

    def test_unicast_fallback_is_not_a_failure(self):
        web.STATE["lines"] = ["fronk: backend avx2 ready",
                              "connection: no multicast reply from node-1 on en0; unicast on this interface"]
        self.assertEqual(self.request("/state")[1]["failure"], "")

    def test_settings_are_validated_before_any_write(self):
        before = self.conf.read_bytes()
        for updates in ({"burst": 5}, {"head-blocks": 21}, {"model": None}, {"model": "a b"},
                        {"ds4": "builtin#ignored"}, {"interfaces": "en0"}, {"interfaces": "239.7.7.7"},
                        {"state-dir": ""}, {"burst": 2, "secret": []}):
            with self.subTest(updates=updates):
                self.assertEqual(self.request("/settings", updates)[0], 400)
                self.assertEqual(self.conf.read_bytes(), before)
        self.assertEqual(self.request("/settings", {"burst": 2, "interfaces": "127.0.0.1"})[0], 200)
        self.assertEqual(web.conf_read()["burst"], "2")

    def test_settings_do_not_change_live_head(self):
        before = self.conf.read_bytes()
        with patch.object(web, "terminal_head_pid", return_value="123"):
            self.assertEqual(self.request("/settings", {"state-dir": "elsewhere"})[0], 409)
            self.assertEqual(self.request("/api/context-size", {"ctx": 8192})[0], 409)
        self.assertEqual(self.conf.read_bytes(), before)
        self.assertEqual((self.root / "state/ds4-ctx").read_text(), "4096\n")

    def test_config_comments_and_no_trailing_newline(self):
        self.conf.write_text("head-blocks 9 # saved placement\nburst 1")
        web.conf_write({"head-blocks": "8", "model": "/model.gguf"})
        self.assertEqual(web.conf_read(), {"head-blocks": "8", "burst": "1", "model": "/model.gguf"})
        self.assertEqual(self.conf.read_text().count("head-blocks"), 1)

    def test_failed_replace_keeps_config(self):
        before = self.conf.read_bytes()
        with patch.object(web.os, "replace", side_effect=OSError("fixture disk failure")):
            with self.assertRaises(OSError):
                web.conf_write({"head-blocks": "2"})
        self.assertEqual(self.conf.read_bytes(), before)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["fronk.conf", "state"])


if __name__ == "__main__":
    unittest.main()
