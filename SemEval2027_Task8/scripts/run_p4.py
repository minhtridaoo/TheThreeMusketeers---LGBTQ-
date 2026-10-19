"""P4 train -> save -> fresh offline process reload -> validate predictions."""
import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sqe.checkpoint import write_json
from sqe.p4 import create_plan, train_suite, verify_suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "train", "verify", "package"])
    parser.add_argument("--mode", choices=["smoke", "dryrun", "final"], default="smoke")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    out = (args.output_dir or ROOT / "outputs/p4" / args.mode).resolve()
    if args.action == "plan":
        plan, train, probe = create_plan(args.mode)
        print(json.dumps({"mode": args.mode, "n_train": len(train), "n_probe": len(probe),
                          "components": [{"name": s["name"], "tasks": s["tasks"], "epochs": s["cfg"].get("epochs")} for s in plan["components"]],
                          "run_fingerprint": plan["run_fingerprint"]}, indent=2), flush=True)
    elif args.action == "train":
        try:
            train_suite(args.mode, out)
            env = os.environ.copy()
            env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HOME=str(out / "isolated_hf_cache"),
                       SQE_P4_FRESH_RELOAD="1", SQE_P4_PARENT_PID=str(os.getpid()))
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "verify", "--output-dir", str(out)],
                           env=env, check=True)
        except Exception as exc:
            write_json(out / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
            raise
    elif args.action == "verify":
        if os.environ.get("SQE_P4_FRESH_RELOAD") != "1":
            env = os.environ.copy()
            env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HOME=str(out / "isolated_hf_cache"),
                       SQE_P4_FRESH_RELOAD="1", SQE_P4_PARENT_PID=str(os.getpid()))
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "verify", "--output-dir", str(out)], env=env, check=True)
        else:
            assert int(os.environ["SQE_P4_PARENT_PID"]) != os.getpid()
            result = verify_suite(out)
            result.update(training_or_launcher_pid=int(os.environ["SQE_P4_PARENT_PID"]), reload_pid=os.getpid())
            write_json(out / "reload_report.json", result)
            write_json(out / "status.json", result)
    else:
        status = json.loads((out / "status.json").read_text(encoding="utf-8"))
        if status["status"] != "complete":
            raise RuntimeError("Only package a successfully verified P4 run")
        dest = args.archive or out.parent / f"sqe_p4_{status['mode']}.zip"
        dest.parent.mkdir(parents=True, exist_ok=True)
        included = [p for p in out.rglob("*") if p.is_file() and "isolated_hf_cache" not in p.parts and not any(".partial-" in part for part in p.parts)]
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for p in included:
                z.write(p, str(Path("SemEval2027_Task8/outputs/p4") / status["mode"] / p.relative_to(out)))
            for p in (ROOT / "src/sqe").glob("*.py"):
                z.write(p, str(Path("SemEval2027_Task8") / p.relative_to(ROOT)))
            for p in [Path(__file__), ROOT / "data/splits/folds_en.csv", ROOT / "outputs/submissions/recipe.json"]:
                z.write(p, str(Path("SemEval2027_Task8") / p.relative_to(ROOT)))
        print("OUTPUT:", dest, f"{dest.stat().st_size / 1024**3:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
