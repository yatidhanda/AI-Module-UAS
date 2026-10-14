import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("run_pipeline", Path(__file__).parents[1] / "scripts" / "run_pipeline.py")
run_pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_pipeline)


def _previous_run(output_dir: Path) -> None:
    (output_dir / "evidence").mkdir(parents=True)
    (output_dir / "evidence" / "T-0001_frame_000000.jpg").write_bytes(b"jpg")
    for name in ("events.jsonl", "tracks.json", "summary.json", "track_states.jsonl", "annotated_demo.mp4"):
        (output_dir / name).write_text("old")
    (output_dir / "operator_actions.jsonl").write_text('{"action": "request_recheck"}\n')


def test_existing_run_is_not_mixed_without_overwrite(tmp_path):
    _previous_run(tmp_path)
    with pytest.raises(SystemExit) as exc:
        run_pipeline.prepare_output_dir(tmp_path, overwrite=False)
    assert exc.value.code == 2
    assert (tmp_path / "events.jsonl").read_text() == "old"


def test_overwrite_removes_generated_files_but_keeps_operator_log(tmp_path):
    _previous_run(tmp_path)
    run_pipeline.prepare_output_dir(tmp_path, overwrite=True)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["operator_actions.jsonl"]


def test_fresh_folder_needs_no_overwrite(tmp_path):
    run_pipeline.prepare_output_dir(tmp_path / "new", overwrite=False)
