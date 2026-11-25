"""CPU checks of historical build admission and ABI-qualified runtime discovery."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "scripts/check-macos-reader"


class MacReaderTests(unittest.TestCase):
    def test_historical_and_release_cfgs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = root / "build.jsonl"
            receipt.write_text("")
            def check(*args):
                return subprocess.run([sys.executable, CHECK, *args], cwd=root, capture_output=True)
            self.assertEqual(check("--historical").stdout.strip(), b"false")
            self.assertEqual(check("--historical", "--build", receipt).returncode, 0)
            self.assertNotEqual(check("--build", receipt).returncode, 0)
            native = root / "crates/vork-render/native/hwdecode/metal.m"
            native.parent.mkdir(parents=True)
            native.touch()
            self.assertEqual(check("--historical").stdout.strip(), b"true")
            self.assertNotEqual(check("--historical", "--build", receipt).returncode, 0)
            receipt.write_text("shader cache diagnostic\n" + json.dumps(dict(reason="build-script-executed", package_id="vork-render", cfgs=["vork_vtdecode"])) + "\n")
            self.assertEqual(check("--build", receipt).returncode, 0)

    def test_local_runtime_directory_is_excluded_from_distribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test = root / "loader.c"
            test.write_text(r'''
#include <stdio.h>
#include <string.h>
#define RTLD_NOW 1
#define RTLD_LOCAL 2
#define LIBAVFORMAT_VERSION_MAJOR 63
#define STR1(x) #x
#define STR(x) STR1(x)
#define VORK_LOCAL_AV_LIBDIR "/custom/lib"
static int seen;
static void *dlopen(const char *path, int flags) {
    if (strcmp(path, "/custom/lib/libavformat.63.dylib") != 0) return NULL;
    seen++;
    return &seen;
}
#include "mac_avformat.h"
int main(void) { open_mac_avformat(); return seen != EXPECTED; }
''')
            include = ROOT / "crates/vork-render/native/hwdecode"
            for flags in [["-DEXPECTED=1"], ["-DEXPECTED=0", "-DVORK_DIST_AVFORMAT_MAJOR=63"]]:
                exe = root / "loader"
                subprocess.run(["cc", "-Werror", "-I", include, *flags, test, "-o", exe], check=True)
                subprocess.run([exe], check=True)


if __name__ == "__main__":
    unittest.main()
