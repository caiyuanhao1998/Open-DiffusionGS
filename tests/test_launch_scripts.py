import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class LaunchScriptTests(unittest.TestCase):
    def test_scripts_select_the_right_stage_and_forward_overrides(self):
        root = Path(__file__).resolve().parents[1]
        cases = {
            "train_obj_stage1": ("--train", "diffusionGS_rel.yaml"),
            "train_obj_stage2": ("--train", "diffusionGS_rel_512.yaml"),
            "train_scene_stage1": ("--train", "diffusionGS_scene.yaml"),
            "train_scene_stage2": ("--train", "diffusionGS_scene_512.yaml"),
            "eval": ("--validate", "diffusionGS_scene_eval.yaml"),
            "eval_512": ("--validate", "diffusionGS_scene_eval_512.yaml"),
        }
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "torchrun"
            executable.write_text(f"#!{sys.executable}\nimport sys,json,os\n"
                                  "print(json.dumps({'args':sys.argv[1:],'marker':os.environ['AUDIT_ENV_MARKER']}))\n")
            executable.chmod(0o755)
            env = dict(os.environ, PATH=directory + os.pathsep + os.environ["PATH"],
                       CUDA_VISIBLE_DEVICES="2,3", NPROC_PER_NODE="2", AUDIT_ENV_MARKER="unchanged")
            for name, (mode, config) in cases.items():
                with self.subTest(script=name):
                    result = subprocess.run(["bash", str(root / "scripts" / f"{name}.sh"), "data.batch_size=1"],
                                            cwd=directory, env=env, check=True, capture_output=True, text=True)
                    output = json.loads(result.stdout)
                    self.assertIn("--nproc-per-node=2", output["args"])
                    self.assertIn(mode, output["args"])
                    self.assertIn("diffusionGS/configs/" + config, output["args"])
                    self.assertEqual(output["args"][-1], "data.batch_size=1")
                    self.assertEqual(output["marker"], "unchanged")


if __name__ == "__main__":
    unittest.main()
