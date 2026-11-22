import pytest

from evaluation.cross_benchmark.vbvr_pro.scoring import (
    resolve_vbvr_candidate,
    resolve_vbvr_work,
)


def sample(sample_id: str) -> dict:
    split, task, instance = sample_id.split("/")
    return {"sample_id": sample_id, "task": task, "split": split, "instance": instance}


def test_resolve_vbvr_work_supports_prediction_webm_and_mp4_layouts(tmp_path):
    rows = [sample("In_Domain/G-1/00000"), sample("Out_of_Domain/O-2/00001")]
    webm = tmp_path / "In_Domain/G-1/00000/prediction.webm"
    webm.parent.mkdir(parents=True)
    webm.write_bytes(b"webm")
    mp4 = tmp_path / "Out_of_Domain/O-2/00001.mp4"
    mp4.parent.mkdir(parents=True)
    mp4.write_bytes(b"mp4")

    work = resolve_vbvr_work(rows, tmp_path)
    assert [row["sample_id"] for row, _ in work] == [
        "In_Domain/G-1/00000",
        "Out_of_Domain/O-2/00001",
    ]
    assert [path for _, path in work] == [webm, mp4]

    with pytest.raises(FileNotFoundError):
        resolve_vbvr_candidate(tmp_path, "In_Domain/G-1/00002")
    with pytest.raises(ValueError, match="duplicate"):
        resolve_vbvr_work(rows + rows[:1], tmp_path)
