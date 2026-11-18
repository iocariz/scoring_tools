"""Deck de apoyo a la respuesta a Riesgos sobre el estudio Apple (dudas del 18-11-2026).

Acompaña a ``reports/correo_bernard_dudas_apple.md``: distribución por score, deriva del
riesgo, tiendas con CESP e ingresos × score. En castellano. Lee las tablas ``b1``–``b4`` que
escribe ``run_apple_study.py``; toda la lógica está en :mod:`src.studies.apple.followup_deck`.

Uso:
    uv run python run_apple_study.py && uv run python generate_apple_followup_deck.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from src.studies.apple.followup_deck import build


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="output/apple_study")
    p.add_argument("--output", default="output/apple_deck/Apple_EFX_Dudas_Riesgos.pptx")
    args = p.parse_args(argv)
    data_dir = Path(args.data_dir)
    if not (data_dir / "b1_tienda_distribucion_score.csv").exists():
        logger.error(f"Faltan las tablas de seguimiento en {data_dir}. Antes: uv run python run_apple_study.py")
        return 1
    path = build(data_dir, Path(args.output))
    logger.info(f"Deck escrito: {path}")
    print(f"\nDeck: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
