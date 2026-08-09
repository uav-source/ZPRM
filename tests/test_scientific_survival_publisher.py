from __future__ import annotations

from pathlib import Path

from phase_a_harness.full_synthetic_publisher import _direction_figure


def test_direction_figure_uses_matplotlib_382_compatible_boxplot_labels(
    tmp_path: Path,
) -> None:
    output = tmp_path / "direction.png"
    rows = [
        {
            "backend": backend,
            "translation_direction_concentration": value,
        }
        for backend, values in (
            ("Open3D", (0.2, 0.4, 0.6)),
            ("PCL", (0.3, 0.5, 0.7)),
        )
        for value in values
    ]

    _direction_figure(output, rows)

    data = output.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(data) > 100
