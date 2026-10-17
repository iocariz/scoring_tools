"""Cross-check every figure in the Apple e-mail draft against the study outputs.

The e-mail (``reports/correo_jenny_resultados_apple.md``) is hand-written markdown
whose numbers are copied from ``output/apple_study/`` (``run_apple_study.py``). Nothing
regenerates it, so after every re-run of the study this script is the check that the
draft still matches the data: it recomputes each table cell and each quoted prose
figure from the CSVs + ``periodos.json``, formats it the way the e-mail writes it
(Spanish decimal comma, same rounding) and looks for it in the right row / sentence.

What is covered (the e-mail body, up to "Notas para Iñigo"):

* "Dónde acaba cada euro" — the six demand states × three columns, the system-approval
  line (sum of the first three states) and the take-up line ("de lo aprobado, contrata").
* Point 1 — Online / Stores / Total rows and the per-group risks quoted in the prose.
* Point 2 — the stores table on the **channel take-up basis** (PR #258): the booking rate
  under the grid is the ``p2b`` production (kept + swap-in) over demand, not the ``p2``
  TOTAL, whose take-up goes per letter group (see CLAUDE.md, Apple EFX study).
* The swap breakdown table (``p2b``), its "accepted by the grid" / "today" summary rows and
  the window → run-rate bridge sentence (÷ effective months).
* The channel-blend table (``p3b``) and the 31 % / 55 % sentence.
* Scenarios — the Apple-total table and the per-channel table (displaced-grid rows), the
  per-group grids (``p4``), and the headline / prose figures (+pp, +M€, grid fidelity,
  volume effect, the 23 contracts booked without an Equifax call).

Usage:
    uv run python scripts/check_apple_email.py
    uv run python scripts/check_apple_email.py --study-dir output/apple_study --email reports/correo_jenny_resultados_apple.md

Exit code 0 when every figure matches, 1 otherwise (each mismatch is printed with the
e-mail cell and the value the data gives).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

DEFAULT_STUDY_DIR = Path("output/apple_study")
DEFAULT_EMAIL = Path("reports/correo_jenny_resultados_apple.md")

# Rows of the "Dónde acaba cada euro" table: (label in the e-mail, state in p0).
_DEMAND_ROWS = (
    ("contratado", "contratado"),
    ("aprobado y cancelado", "aprobado, cancelado (no finalizada)"),
    ("aprobado y rechazado", "aprobado, rechazado después"),
    ("rechazado por score", "rechazado por score"),
    ("rechazado por otras", "rechazado por otras reglas"),
    ("estudio ligero", "rechazada en estudio ligero (sin llamada a Equifax)"),
)
_DEMAND_CHANNELS = ("online", "tienda_ventana", "tienda_score")
# Rows of the swap table: (label in the e-mail, category in p2b).
_SWAP_ROWS = (
    ("se mantiene", "se mantiene: aprobada hoy, por encima del corte"),
    ("swap-in", "swap-in: rechazada hoy por score, por encima del corte"),
    ("swap-out", "swap-out: aprobada hoy, por debajo del corte"),
    ("rechazada por los dos", "rechazada por los dos scores: hoy por el interno, por debajo del corte EFX"),
    ("intacta: rechazada por otras", "intacta: rechazada por otras reglas"),
)
_LIGHT_CATEGORY = "intacta: rechazada en estudio ligero (sin llamada a Equifax)"
# Scenario rows: (label in the e-mail, row in p3). The e-mail quotes the displaced grid.
_SCENARIO_ROWS = (
    ("Hoy, medido en la ventana", "Hoy (medido)"),
    ("Hoy, con la demanda actual", "Hoy, con la demanda actual de tienda"),
    ("2,5%", "2.5% · parrilla desplazada"),
    ("3,0%", "3% · parrilla desplazada"),
    ("3,5%", "3.5% · parrilla desplazada"),
    ("4,0%", "4% · parrilla desplazada"),
)
_GRID_ROWS = (("New", "New"), ("Inactive", "Inactive"), ("A-C", "A-C"), ("D-F", "D-F"), (">=G", "≥G"))
_GRID_COLUMNS = ("Parrilla actual", "2.5% · parrilla", "3% · parrilla", "3.5% · parrilla", "4% · parrilla")


def es(x: float, nd: int = 1) -> str:
    """Format a number the way the e-mail writes it (Spanish decimal comma)."""
    return f"{x:.{nd}f}".replace(".", ",")


def _pct(x: float, nd: int = 1) -> str:
    return es(x, nd) + "%"


def _meur(x: float) -> str:
    return es(x / 1e6) + " M€"


class EmailChecker:
    """Recompute the e-mail's figures from the study outputs and compare them."""

    def __init__(self, study_dir: Path, email_path: Path) -> None:
        text = email_path.read_text(encoding="utf-8")
        self.body = text.split("## Notas para Iñigo")[0]
        self.table_lines = [line for line in self.body.splitlines() if line.startswith("|")]
        read = lambda name: pd.read_csv(study_dir / name)  # noqa: E731
        self.p0 = read("p0_demanda_por_estado.csv")
        self.p1 = read("p1_online_actual.csv").set_index("segmento")
        self.p1b = read("p1b_tienda_actual.csv").set_index("segmento")
        self.p2 = read("p2_tienda_parrilla_online.csv").set_index("segmento")
        self.p2b = read("p2b_tienda_swap.csv")
        self.p3 = read("p3_escenarios_objetivo.csv").set_index("escenario")
        self.p3b = read("p3b_mezcla_canales.csv").set_index("canal")
        self.p4 = read("p4_parrillas_escenarios.csv").set_index("grupo")
        self.per = json.loads((study_dir / "periodos.json").read_text(encoding="utf-8"))
        self.fails: list[str] = []

    # ---- markdown helpers -------------------------------------------------------------
    @staticmethod
    def _cells(line: str) -> list[str]:
        return [c.strip().replace("*", "") for c in line.strip().strip("|").split("|")]

    def rows(self, label: str, exact: bool = False) -> list[list[str]]:
        """Every table row whose first cell contains (or equals) *label*."""
        out = []
        for line in self.table_lines:
            first = self._cells(line)[0]
            if (first == label) if exact else (label in first):
                out.append(self._cells(line))
        return out

    def row(self, label: str, exact: bool = False, width: int | None = None) -> list[str]:
        """The first matching row (optionally with *width* cells), without its label cell."""
        for cells in self.rows(label, exact):
            if width is None or len(cells) == width:
                return cells[1:]
        raise KeyError(f"no table row labelled {label!r}" + (f" with {width} cells" if width else ""))

    def check_row(self, name: str, cells: list[str], expected: list[str | None]) -> None:
        got = [re.sub(r"\s", "", c) for c in cells]
        for i, e in enumerate(expected):
            if e is None:
                continue
            if i >= len(got) or re.sub(r"\s", "", e) not in got[i]:
                found = cells[i] if i < len(cells) else "<missing>"
                self.fails.append(f"{name} col{i}: email={found!r} expected≈{e!r}")

    def check_prose(self, name: str, snippet: str) -> None:
        if snippet not in self.body:
            self.fails.append(f"{name}: prose does not contain {snippet!r}")

    # ---- sections -----------------------------------------------------------------------
    def demand_breakdown(self) -> None:
        states = {ch: self.p0[self.p0.canal == ch].set_index("estado")["pct_demanda"] for ch in _DEMAND_CHANNELS}
        for label, key in _DEMAND_ROWS:
            self.check_row(f"demanda {label}", self.row(label), [_pct(states[ch][key]) for ch in _DEMAND_CHANNELS])
        approved = {ch: sum(states[ch][key] for _, key in _DEMAND_ROWS[:3]) for ch in _DEMAND_CHANNELS}
        self.check_row(
            "demanda aprobación",
            self.row("aprobación del sistema (OK + RV"),
            [_pct(approved[ch]) for ch in _DEMAND_CHANNELS],
        )
        self.check_row(
            "demanda take-up",
            self.row("de lo aprobado"),
            [_pct(100 * states[ch]["contratado"] / approved[ch], 0) for ch in _DEMAND_CHANNELS],
        )

    def point_1(self) -> None:
        for label, frame in (("Apple Online", self.p1), ("Apple Stores", self.p1b)):
            t = frame.loc["TOTAL"]
            self.check_row(
                f"P1 {label}",
                self.row(label),
                [_pct(t.aprobacion_sistema_pct), _pct(t.ta_pct), _pct(t.riesgo_real_pct, 2)],
            )
        h = self.p3.loc["Hoy (medido)"]
        self.check_row(
            "P1 Apple Total",
            self.row("Apple Total"),
            [_pct(h.ta_score_pct), _pct(h.ta_efectiva_pct), _pct(h.riesgo_total_pct, 2)],
        )
        for seg in ("New", "A-C", "Inactive"):
            self.check_prose(f"P1 riesgo {seg}", f"`{seg}` ({_pct(self.p1.loc[seg, 'riesgo_real_pct'], 2)})")

    def point_2_and_swap(self) -> None:
        t2 = self.p2.loc["TOTAL"]
        stores = self.p3b.loc["tienda"]
        demand = t2.demanda_eur
        swap = self.p2b.set_index("categoria")
        # Channel-basis production under the grid = kept + swap-in (PR #258).
        prod_grid = self.p2b.produccion_est_eur.iloc[:2].sum()
        self.check_row(
            "P2 aprobación",
            self.row("Aprobación del sistema", exact=True),
            [_pct(t2.aprobacion_sistema_actual_pct), _pct(t2.pct_demanda_aceptada)],
        )
        self.check_row(
            "P2 contratación",
            self.row("Tasa de contratación", exact=True),
            [_pct(t2.ta_actual_pct), _pct(100 * prod_grid / demand)],
        )
        self.check_row(
            "P2 riesgo",
            self.row("Riesgo", exact=True),
            [_pct(t2.riesgo_imputado_actual_pct, 2), _pct(t2.riesgo_imputado_pct, 2)],
        )
        self.check_prose("P2 riesgo imputado sin ancla", _pct(self.per["riesgo_tienda_imputado_pct"], 2))
        self.check_prose("P2 factor de nivel", es(self.per["factor_nivel_tienda"], 3))
        self.check_prose("P2 take-up ok", f"aprobada {es(self.per['take_up_tienda_ok_pct'], 0)}%")
        self.check_prose("P2 take-up rv", f"revisión {es(self.per['take_up_tienda_rv_pct'], 0)}%")

        for label, key in _SWAP_ROWS:
            r = swap.loc[key]
            self.check_row(
                f"swap {label}",
                self.row(label),
                [_pct(r.pct_demanda), _meur(r.produccion_est_eur), _pct(r.riesgo_imputado_pct)],
            )
        self.check_row(
            "swap estudio ligero",
            self.row("intacta: rechazada en estudio ligero"),
            [_pct(swap.loc[_LIGHT_CATEGORY, "pct_demanda"]), "—", "—"],
        )
        self.check_row(
            "swap aceptado por la parrilla",
            self.row("aceptado por la parrilla"),
            [
                _pct(t2.pct_demanda_aceptada),
                f"{_meur(prod_grid)} ({_pct(100 * prod_grid / demand)})",
                _pct(stores.riesgo_pct, 2),
            ],
        )
        booked = self.p0[(self.p0.canal == "tienda_score") & (self.p0.estado == "contratado")].eur.item()
        self.check_row(
            "swap hoy",
            self.row("hoy, con el score interno"),
            [
                _pct(t2.aprobacion_sistema_actual_pct),
                f"{_meur(booked)} ({_pct(t2.ta_actual_pct)})",
                _pct(t2.riesgo_imputado_actual_pct, 2),
            ],
        )
        months = stores.meses_efectivos
        self.check_prose(
            "puente ventana→run-rate",
            f"{_meur(prod_grid)} aceptados, divididos por los {es(months, 2)} meses efectivos",
        )
        self.check_prose("puente producción/mes", f"los {_meur(stores.produccion_mensual_eur)}/mes")
        self.check_prose(
            "puente demanda/mes", f"los {_meur(demand)} de demanda, los {_meur(stores.demanda_mensual_eur)}/mes"
        )

    def channel_blend(self) -> None:
        on, st, tot = (self.p3b.loc[c] for c in ("online", "tienda", "total"))
        self.check_row(
            "mezcla meses", self.row("meses efectivos"), [es(on.meses_efectivos, 2), es(st.meses_efectivos, 2), "—"]
        )
        self.check_row(
            "mezcla demanda", self.row("demanda", exact=True), [_meur(x.demanda_mensual_eur) for x in (on, st, tot)]
        )
        self.check_row(
            "mezcla producción",
            self.row("producción esperada"),
            [_meur(x.produccion_mensual_eur) for x in (on, st, tot)],
        )
        self.check_row(
            "mezcla peso",
            self.row("peso en la exposición"),
            [_pct(on.peso_exposicion_pct, 0), _pct(st.peso_exposicion_pct, 0), "100%"],
        )
        self.check_row("mezcla riesgo", self.row("riesgo (b2)"), [_pct(x.riesgo_pct, 2) for x in (on, st, tot)])
        self.check_prose(
            "mezcla pesos (prosa)",
            f"Tienda es el {es(self.per['peso_tienda_propio_pct'], 0)}% de la demanda pero el "
            f"{es(st.peso_exposicion_pct, 0)}% de la exposición",
        )

    def scenarios(self) -> None:
        for label, key in _SCENARIO_ROWS:
            r = self.p3.loc[key]
            if "Hoy" in label:
                grid = "vigente"
            elif label == "4,0%":
                grid = "sin cambios"
            else:
                grid = r.corte
            self.check_row(
                f"escenario total {label}",
                self.row(label, width=6),
                [
                    _pct(r.ta_score_pct),
                    _pct(r.ta_efectiva_pct),
                    _meur(r.produccion_mensual_eur) + "/mes",
                    _pct(r.riesgo_total_pct, 2),
                    grid,
                ],
            )
            self.check_row(
                f"escenario por canal {label}",
                self.row(label, width=9),
                [
                    _pct(r.online_aprobacion_sistema_pct),
                    _pct(r.online_tasa_contratacion_pct),
                    _meur(r.online_produccion_mensual_eur),
                    _pct(r.online_riesgo_pct, 2),
                    _pct(r.tienda_aprobacion_sistema_pct),
                    _pct(r.tienda_tasa_contratacion_pct),
                    _meur(r.tienda_produccion_mensual_eur),
                    _pct(r.tienda_riesgo_pct, 2),
                ],
            )
        for key, label in _GRID_ROWS:
            self.check_row(f"parrilla {key}", self.row(label, width=6), [self.p4.loc[key, c] for c in _GRID_COLUMNS])

    def headline(self) -> None:
        base = self.p3.loc["Hoy, con la demanda actual de tienda"]
        grid = self.p3.loc["Parrilla Online en ambos canales"]
        meas = self.p3.loc["Hoy (medido)"]
        grid_effect = (grid.produccion_mensual_eur - base.produccion_mensual_eur) / 1e6
        volume = (base.produccion_mensual_eur - meas.produccion_mensual_eur) / 1e6
        for name, snippet in (
            ("titular pp", f"+{es(grid.ta_efectiva_pct - base.ta_efectiva_pct)} puntos de tasa de contratación"),
            ("titular M€", f"+{es(grid_effect)} M€ de producción"),
            ("titular riesgo", f"({_pct(grid.riesgo_total_pct, 2)} frente a {_pct(base.riesgo_total_pct, 2)})"),
            (
                "fidelidad parrilla",
                f"acepta el {_pct(self.per['online_parrilla_pct'])} de la demanda; el motor aprobó el "
                f"{_pct(self.per['online_aprobado_real_pct'])}",
            ),
            ("efecto parrilla", f"+{es(grid_effect)} M€ por la parrilla"),
            ("efecto volumen", f"+{es(volume)} M€ porque"),
            ("demanda tienda vs ventana", f"un {es(self.per['demanda_tienda_vs_ventana_pct'], 0)}% por encima"),
            ("contratos sin EFX", f"{self.per['online_contratados_sin_efx_n']} contratos aprobados sin llamada"),
            (
                "contratos sin EFX (peso y mora)",
                f"el {_pct(self.per['online_contratados_sin_efx_produccion_pct'], 2)} de la producción, con un "
                f"{_pct(self.per['online_contratados_sin_efx_riesgo_pct'])} de mora",
            ),
        ):
            self.check_prose(name, snippet)

    def run(self) -> list[str]:
        for section in (
            self.demand_breakdown,
            self.point_1,
            self.point_2_and_swap,
            self.channel_blend,
            self.scenarios,
            self.headline,
        ):
            try:
                section()
            except KeyError as exc:  # a renamed / missing table row is a failure, not a crash
                self.fails.append(f"{section.__name__}: {exc}")
        return self.fails


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR, help="run_apple_study.py output dir")
    parser.add_argument("--email", type=Path, default=DEFAULT_EMAIL, help="e-mail draft (markdown)")
    args = parser.parse_args(argv)
    fails = EmailChecker(args.study_dir, args.email).run()
    if fails:
        print(f"FAIL: {len(fails)} figure(s) in {args.email} do not match {args.study_dir}")
        print("\n".join(f"  - {f}" for f in fails))
        return 1
    print(f"OK: every table cell and prose figure in {args.email} matches {args.study_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
