"""
LogCluster Test Runner.

Executes all unit tests in tests/ using standard Python unittest / assertion harness.
"""

import sys
import traceback
import tests.test_file_loader as t_loader
import tests.test_preprocessing as t_pre
import tests.test_pattern_grouping as t_group
import tests.test_rare_patterns as t_rare
import tests.test_results_exporter as t_exp
import tests.test_icons as t_icons
from pathlib import Path


def run_all_tests():
    modules = [
        ("File Loader Tests", t_loader),
        ("Preprocessing Tests", t_pre),
        ("Pattern Grouping Tests", t_group),
        ("Rare Pattern Detector Tests", t_rare),
        ("Results Exporter Tests", t_exp),
        ("Icon System Tests", t_icons),
    ]

    total_passed = 0
    total_failed = 0
    tmp_path = Path("./tests/tmp_test_dir")
    tmp_path.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("      Running LogCluster Test Suite               ")
    print("==================================================")

    for mod_name, mod in modules:
        print(f"\n[Suite] {mod_name}")
        for attr_name in dir(mod):
            if attr_name.startswith("test_"):
                fn = getattr(mod, attr_name)
                if callable(fn):
                    # Check if takes tmp_path
                    import inspect
                    sig = inspect.signature(fn)
                    try:
                        if "tmp_path" in sig.parameters:
                            fn(tmp_path)
                        else:
                            fn()
                        print(f"  PASS: {attr_name}")
                        total_passed += 1
                    except Exception as exc:
                        print(f"  FAIL: {attr_name} -> {exc}")
                        traceback.print_exc()
                        total_failed += 1

    # Clean up tmp_path
    import shutil
    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)

    print("\n==================================================")
    print(f"Results: {total_passed} passed, {total_failed} failed.")
    print("==================================================")

    if total_failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()
