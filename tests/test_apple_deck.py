"""Humo del deck Apple: que los cinco gráficos y el .pptx se construyan desde las
salidas de ``run_apple_study.py`` sin romper. Protege el generador de cambios en las
columnas de los CSV, que es como se rompería en la práctica."""

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
pytest.importorskip("pptx")

from generate_apple_deck import BIN_EDGES, build  # noqa: E402
from src import apple_deck_charts as ac  # noqa: E402


@pytest.fixture
def study_outputs(tmp_path):
    """Salidas mínimas del estudio, con las mismas columnas que escribe run_apple_study."""
    bins = np.arange(1.0, 21.0)
    pd.DataFrame(
        {"bin": bins, "b2_pct": np.linspace(40, 1, 20), "exposure_ratio": 6.5, "booked_eur": np.linspace(1e4, 9e6, 20)}
    ).to_csv(tmp_path / "curva_riesgo_online.csv", index=False)
    pd.DataFrame(
        {
            "segmento": ["new", "known_ab", "TOTAL"],
            "demanda_n": [100, 50, 150],
            "demanda_eur": [1e6, 5e5, 1.5e6],
            "booked_n": [20, 10, 30],
            "produccion_eur": [2e5, 1e5, 3e5],
            "ta_pct": [20.0, 20.0, 20.0],
            "riesgo_real_pct": [5.0, 1.2, 4.2],
        }
    ).to_csv(tmp_path / "p1_online_actual.csv", index=False)
    pd.DataFrame(
        {
            "segmento": ["new", "TOTAL"],
            "corte_efx": [47.0, np.nan],
            "demanda_eur": [1e6, 1e6],
            "pct_demanda_aceptada": [70.0, 70.0],
            "produccion_est_eur": [5e5, 5e5],
            "ta_efectiva_pct": [57.0, 57.0],
            "riesgo_imputado_pct": [3.1, 3.1],
            "ta_actual_pct": [47.6, 47.6],
            "riesgo_imputado_actual_pct": [3.75, 3.75],
        }
    ).to_csv(tmp_path / "p2_tienda_parrilla_online.csv", index=False)
    pd.DataFrame(
        {
            "corte_tramo": bins.astype(int),
            "corte_score": [BIN_EDGES[int(b) - 1] for b in bins],
            "riesgo_total_pct": np.linspace(8, 1.3, 20),
            "ta_apple_pct": np.linspace(100, 6, 20),
            "produccion_mensual_eur": np.linspace(1.6e7, 8e5, 20),
            "ta_online_pct": np.linspace(100, 5, 20),
            "ta_tienda_pct": np.linspace(100, 11, 20),
        }
    ).to_csv(tmp_path / "p3_escalera_escenarios.csv", index=False)
    pd.DataFrame(
        {
            "objetivo_pct": [2.5, 3.0],
            "corte_tramo": [15, 13],
            "corte_score": [68.0, 56.0],
            "riesgo_logrado_pct": [2.3, 2.8],
            "ta_apple_pct": [42.6, 63.7],
            "produccion_mensual_eur": [8.7e6, 1.27e7],
            "nota": ["", ""],
        }
    ).to_csv(tmp_path / "p3_escenarios_objetivo.csv", index=False)
    pd.DataFrame(
        {"mes": range(1, 13), "indice": [0.94, 0.81, 0.86, 0.74, 0.71, 0.63, 0.67, 0.53, 1.94, 1.68, 1.21, 1.28]}
    ).to_csv(tmp_path / "indice_estacional.csv", index=False)
    return tmp_path


def test_deck_builds_from_study_outputs(study_outputs, tmp_path):
    out = build(study_outputs, tmp_path / "deck" / "Estudio.pptx")
    assert out.exists() and out.stat().st_size > 0
    from pptx import Presentation

    assert len(Presentation(str(out)).slides._sldIdLst) == 9
    # 5 gráficos de slide + la variante compacta de tienda y el desglose por segmento
    assert len(list((tmp_path / "deck" / "images").glob("*.png"))) == 7


def test_spanish_decimal_format():
    # el deck se presenta en español: coma decimal, no punto
    assert ac.es(3.128, 2) == "3,13"
    assert ac.es(42.6, 1, "%") == "42,6%"


def test_charts_render_without_labels_overflowing_axes(study_outputs):
    """Las anotaciones deben caer dentro de los ejes: fuera se recortan al exportar."""
    curve = pd.read_csv(study_outputs / "curva_riesgo_online.csv")
    fig = ac.chart_risk_curve(curve, BIN_EDGES)
    ax = fig.axes[0]
    ymin, ymax = ax.get_ylim()
    for child in ax.texts:
        assert ymin <= child.get_position()[1] <= ymax, f"anotación fuera del eje: {child.get_text()!r}"
    matplotlib.pyplot.close(fig)


def test_segment_breakdown_excludes_always_rejected(study_outputs):
    """known_g no puede dibujarse: no tiene ni aceptación ni riesgo que comparar."""
    p2 = pd.read_csv(study_outputs / "p2_tienda_parrilla_online.csv")
    p2 = pd.concat(
        [
            p2,
            pd.DataFrame(
                [
                    {
                        "segmento": "known_g",
                        "corte_efx": np.inf,
                        "demanda_eur": 4e4,
                        "pct_demanda_aceptada": 0.0,
                        "produccion_est_eur": 0.0,
                        "ta_efectiva_pct": 0.0,
                        "riesgo_imputado_pct": np.nan,
                        "ta_actual_pct": 0.0,
                        "riesgo_imputado_actual_pct": np.nan,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    fig = ac.chart_stores_by_segment(p2)
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert "known_g" not in labels and "TOTAL" not in labels
    matplotlib.pyplot.close(fig)
