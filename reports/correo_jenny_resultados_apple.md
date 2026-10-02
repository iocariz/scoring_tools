# Borrador de correo — Resultados del estudio Apple

> Generado el 01-10-2026 a partir de `output/apple_study/` (corrida con la parrilla vigente,
> ancla de nivel automática y reject inference). Si se vuelve a correr el estudio, revisar que
> las cifras siguen cuadrando antes de enviar: salen de `p1_online_actual.csv`,
> `p1b_tienda_actual.csv`, `p2_tienda_parrilla_online.csv`, `p3_escenarios_objetivo.csv` y
> `p4_parrillas_escenarios.csv`.
>
> **Corregido el 01-10-2026 (tarde):** el take-up se medía solo sobre las solicitudes `ok` y se
> aplicaba también a las `rv` (revisión, ~5% de conversión) que la parrilla acepta; Online salía un
> 12% hinchado y el titular era +4,0 pp / +2,6 M€. Ahora va por decisión del motor. Los cortes de
> los escenarios no cambian.

**Asunto:** Apple — Score Equifax en tienda: resultados y escenarios
**Adjunto:** `output/apple_deck/Estudio_Apple_EFX.pptx`

---

Hola Jenny,

Ya está el estudio. Te resumo primero la conclusión y debajo los tres puntos con el detalle.

**En una línea: al mismo nivel de riesgo que hoy (3,99%), extender la parrilla de Equifax de Online a tienda nos daría +1,9 puntos de tasa de aceptación y unos +1,5 M€ de producción al mes.** Y el objetivo de riesgo del 4% ya lo cumplimos hoy: no requiere ningún cambio.

---

## 1. Situación actual, medida

Ventana de originación marzo-2025 a febrero-2026, que es la más reciente con los seis meses de comportamiento cumplidos.

| | TA | Riesgo |
|---|---:|---:|
| Apple Online | 20,9% | 4,18% |
| Apple Stores | 58,7% | 3,74% |
| **Apple Total** | **29,5%** | **3,99%** |

Tienda acepta casi el triple que Online y arriesga menos. En Online el riesgo se concentra en `New` (5,08%) frente a `A-C` (1,32%) o `Inactive` (0,81%).

## 2. Tienda con la parrilla de Online

Sobre agosto-2026, que es el único mes con score Equifax en tienda:

| | hoy | con la parrilla de Online |
|---|---:|---:|
| Tasa de aceptación | 47,6% | **62,5%** |
| Riesgo | 3,74% | **3,66%** |

Sube la aceptación y baja el riesgo a la vez. El riesgo de tienda es **estimado**, no medido —no habrá medición real hasta febrero de 2027—, pero su **nivel** está calibrado contra la morosidad que tienda ha tenido de verdad: la curva de Online predecía 3,79% y lo observado es 3,74%, un 1,2% de diferencia. Lo que seguimos asumiendo es que el score ordena en tienda igual que en e-commerce.

## 3. Escenarios de riesgo global Apple

| objetivo | TA Apple | producción | parrilla |
|---|---:|---:|---|
| Hoy | 29,5% | 15,3 M€/mes | vigente |
| 2,5% | 19,8% | 10,6 M€/mes | +38 puntos |
| 3,0% | 25,4% | 13,6 M€/mes | +25 puntos |
| 3,5% | 29,3% | 15,7 M€/mes | +13 puntos |
| **4,0%** | **31,4%** | **16,8 M€/mes** | **sin cambios** |

Las parrillas concretas (umbral de rechazo por grupo, se acepta por encima):

| grupo | hoy | 2,5% | 3% | 3,5% | 4% |
|---|---|---|---|---|---|
| New | > 27 | > 65 | > 52 | > 40 | > 27 |
| Inactive | > 22 | > 60 | > 47 | > 35 | > 22 |
| A-C | > 16 | > 54 | > 41 | > 29 | > 16 |
| D-F | > 22 | > 60 | > 47 | > 35 | > 22 |
| ≥G | rechazo | rechazo | rechazo | rechazo | rechazo |

---

## Cuatro cosas que conviene tener presentes

**El riesgo de tienda es estimado.** El nivel está calibrado contra dato real, pero el poder discriminante del score en tienda es un supuesto hasta que maduren las operaciones de agosto.

**La ganancia es toda de tienda.** La parrilla vigente calca lo que el motor decide hoy en Online (acepta el 51,8% de la demanda; el motor aprobó el 51,9%), y cada solicitud aceptada convierte a la tasa de su decisión real —la misma tasa de financiación que usamos en el proceso general—, así que Online modelado y medido coinciden. Los 16,8 frente a 15,3 M€/mes son tienda, con un matiz de base: tienda se modela sobre agosto-2026, el único mes con score, y el "hoy" sobre la ventana completa.

**Tienda lleva un año endureciendo por su cuenta**: su TA ha pasado del 58,7% (marzo-2025 a febrero-2026) al 47,6% en agosto-2026. La base contra la que comparamos se está moviendo.

**El perímetro es Apple Propio**, una sola cadena, sin APRS.

## Siguiente paso

Decidme qué objetivo de riesgo queréis y si preferís un corte único para todo Apple —más simple de operar— o desplazar la parrilla por grupo, que da algo más de aceptación al mismo riesgo. Con eso lanzo la corrida definitiva y preparo la propuesta de implementación.

Tienes la presentación adjunta con el detalle y la metodología.

Un saludo,
Iñigo

---

## Notas para Iñigo (no enviar)

* Los escenarios de la tabla van con la **parrilla desplazada**, no con el corte único, porque dan más TA al mismo riesgo. Las dos opciones están en el deck (slide de parrillas).
* El objetivo del 4% va en negrita como "sin cambios" por ser el hallazgo más accionable y el contrario de lo que probablemente esperan. Quitar la negrita si se prefiere menos relieve.
* Los cuatro avisos quitan brillo al titular, pero son las preguntas que saldrán en la sala: mejor que salgan de ti.
* El take-up va **por decisión del motor** (`ok` 44% / `rv` 5% en Online; las que entran nuevas por score convierten como `ok`), que es la `tasa_fin` del pipeline a nivel solicitud. Si alguien pregunta por la versión agregada `contratado / (ok+rv)` del proceso general: da 16,8 M€/mes también (16,76).
* La TA de los escenarios es **efectiva** (producción / demanda), la misma base que la del punto 1. El deck publica además la TA "de score" (demanda que supera el corte), que es mucho más alta y no es comparable con el 29,5% de hoy.
