"""Fail before installation on invalid roster, checksum or unavailable services."""
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "tools/install/setup-cluster.sh"
PREREQUISITES = ROOT / "tools/install/ssh-prerequisites.sh"


class SSHSetupTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.env = dict(os.environ, HOME=str(self.root), PATH=str(self.bin) + os.pathsep + os.environ["PATH"])
        self.manager = self.root / "fronk-manager.sh"
        self.manager.write_text('#!/bin/bash\necho "unexpected manager execution" >&2\nexit 99\n__FRONK_PAYLOAD__\n')
        self.sums = self.root / "SHA256SUMS"
        self.sums.write_text(hashlib.sha256(self.manager.read_bytes()).hexdigest() + "  fronk-manager.sh\n")
        self.nodes = self.root / "nodes.txt"
        self.nodes.write_text("node-1 user@host /home/user/fronk\n")
        self.stub("ssh", 'echo "fixture: SSH unavailable" >&2; exit 91')

    def stub(self, name, body):
        path = self.bin / name
        path.write_text("#!/bin/bash\n" + body + "\n")
        path.chmod(0o755)

    def setup_command(self):
        return ["bash", str(SETUP), "--manager", str(self.manager), "--checksums", str(self.sums),
                "--head-prefix", str(self.root / "head"), "--nodes", str(self.nodes), "--through", "install"]

    def run_command(self, args):
        return subprocess.run(args, env=self.env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)

    def test_rejects_bad_checksum_before_ssh(self):
        self.sums.write_text("0" * 64 + "  fronk-manager.sh\n")
        result = self.run_command(self.setup_command())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("checksum mismatch", result.stdout)
        self.assertNotIn("Checking SSH", result.stdout)

    def test_source_manager_is_not_a_portable_download(self):
        self.manager.write_text("#!/bin/bash\nexit 0\n")
        self.sums.write_text(hashlib.sha256(self.manager.read_bytes()).hexdigest() + "  fronk-manager.sh\n")
        result = self.run_command(self.setup_command())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("portable release", result.stdout)
        self.assertNotIn("Checking SSH", result.stdout)

    def test_rejects_roster_before_any_install(self):
        for text in ("node-1 host /\n", "node-1 host /home/x/../fronk\n", "x" * 64 + " host /home/x/fronk\n",
                     "node-1 host /home/x/fronk\nnode-1 host2 /home/y/fronk\n",
                     "node-1 host /home/x/fronk\nnode-2 host /home/y/fronk\n"):
            with self.subTest(text=text):
                self.nodes.write_text(text)
                result = self.run_command(self.setup_command())
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("Checking SSH", result.stdout)
                self.assertNotIn("unexpected manager execution", result.stdout)

    def test_ssh_failure_does_not_install_head(self):
        result = self.run_command(self.setup_command())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SSH/prerequisites failed", result.stdout)
        self.assertNotIn("unexpected manager execution", result.stdout)

    def test_linux_service_preflight(self):
        self.stub("uname", "echo Linux")
        self.stub("systemctl", "exit 1")
        command = ["bash", str(PREREQUISITES), str(self.root / "node")]
        result = self.run_command(command)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("systemd user session", result.stdout)
        self.stub("systemctl", "exit 0")
        self.assertEqual(self.run_command(command).returncode, 0)
        service = self.root / ".config/systemd/user/fronk-node.service"
        service.parent.mkdir(parents=True)
        service.write_text("ExecStart=/home/old/fronk/src/fronk node\n")
        result = self.run_command(command)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("another prefix owns", result.stdout)

    def test_mac_requires_graphical_login(self):
        self.stub("uname", "echo Darwin")
        self.stub("xcode-select", "exit 0")
        self.stub("launchctl", "exit 1")
        command = ["bash", str(PREREQUISITES), str(self.root / "node")]
        result = self.run_command(command)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("log in graphically", result.stdout)
        # The manager itself must reject this before building or creating state,
        # even when it is used directly without the SSH wrapper.
        result = self.run_command(["bash", str(ROOT / "tools/install/fronk-install.sh"), "--yes",
                                   "--role", "node", "--name", "node-1", "--prefix", str(self.root / "node")])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("log in graphically", result.stdout)
        self.assertFalse((self.root / "node/state").exists())
        self.assertFalse((self.root / "node/src").exists())
        self.stub("launchctl", "exit 0")
        self.assertEqual(self.run_command(command).returncode, 0)


if __name__ == "__main__":
    unittest.main()
