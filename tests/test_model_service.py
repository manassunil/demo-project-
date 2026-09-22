import os
from pathlib import Path
import subprocess
import sys

import pytest

from backend.services.model_service import MODEL_DIR, predict_depression


@pytest.mark.parametrize(
    ("text", "expected_label"),
    [
        ("I have been diagnosed with depression and feel hopeless.", 1),
        ("I am happy and looking forward to the weekend.", 0),
    ],
)
def test_representative_predictions(text, expected_label):
    label, probability = predict_depression(text)

    assert label == expected_label
    assert label in (0, 1)
    assert 0.0 <= probability <= 1.0


def test_combined_model_directory_exists():
    assert MODEL_DIR.name == "combined_distilbert"
    assert MODEL_DIR.is_dir()


def test_model_service_loads_from_backend_directory():
    project_root = Path(__file__).resolve().parents[1]
    backend_dir = project_root / "backend"
    script = (
        "from services.model_service import MODEL_DIR; "
        "assert MODEL_DIR.is_dir(); "
        "assert MODEL_DIR.name == 'combined_distilbert'"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(backend_dir)

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=backend_dir,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
