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
>
> **Actualizado el 14-10-2026:** extracto con septiembre de tienda (ventana ago-sep 2026; antes
> solo agosto) y corrección del ancla de nivel, que se aplicaba al peso de tienda y no a su riesgo
> (tienda entraba en los escenarios sin anclar; con el factor de agosto, 0,99, no se notaba; con el
> de ago-sep, 0,95, valía 0,09 pp). La demanda de tienda en ago-sep va un 40% por encima de la
> ventana de "hoy", así que el titular se da sobre la misma base (fila nueva en `p3`): +4,1 pp y
> +2,3 M€/mes; frente a lo medido en la ventana, +3,2 pp y +3,2 M€.

**Asunto:** Apple — Score Equifax en tienda: resultados y escenarios
**Adjunto:** `output/apple_deck/Estudio_Apple_EFX.pptx`

---

Hola Jenny,

Ya está el estudio, actualizado con septiembre de tienda. Te resumo primero la conclusión y debajo los tres puntos con el detalle.

**En una línea: extender la parrilla de Equifax de Online a tienda nos daría +4,1 puntos de tasa de aceptación y unos +2,3 M€ de producción al mes, con el riesgo global algo por debajo del de hoy (3,85% frente a 3,97%).** Y el objetivo de riesgo del 4% ya lo cumplimos hoy: no requiere ningún cambio.

---

## 1. Situación actual, medida

Ventana de originación marzo-2025 a febrero-2026 (doce meses con los seis de comportamiento cumplidos).

| | TA | Riesgo |
|---|---:|---:|
| Apple Online | 20,9% | 4,18% |
| Apple Stores | 58,7% | 3,74% |
| **Apple Total** | **29,5%** | **3,99%** |

Tienda acepta casi el triple que Online y arriesga menos. En Online el riesgo se concentra en `New` (5,07%) frente a `A-C` (1,32%) o `Inactive` (0,81%).

## 2. Tienda con la parrilla de Online

Sobre agosto y septiembre de 2026, los dos meses con score Equifax en tienda:

| | hoy | con la parrilla de Online |
|---|---:|---:|
| Tasa de aceptación | 47,2% | **61,2%** |
| Riesgo | 3,74% | **3,51%** |

Sube la aceptación y baja el riesgo a la vez. El riesgo de tienda es **estimado**, no medido —no habrá medición real hasta marzo de 2027—, pero su **nivel** está calibrado contra la morosidad que tienda ha tenido de verdad: la curva de Online predecía 3,93% para la cartera que tienda contrató y lo observado es 3,74%, un 5% de diferencia que se corrige. Lo que seguimos asumiendo es que el score ordena en tienda igual que en e-commerce.

## 3. Escenarios de riesgo global Apple

| objetivo | TA Apple | producción | parrilla |
|---|---:|---:|---|
| Hoy, medido en la ventana | 29,5% | 15,3 M€/mes | vigente |
| Hoy, con la demanda actual de tienda | 28,5% | 16,2 M€/mes | vigente |
| 2,5% | 21,7% | 12,3 M€/mes | +36 puntos |
| 3,0% | 27,5% | 15,6 M€/mes | +23 puntos |
| 3,5% | 31,2% | 17,7 M€/mes | +10 puntos |
| **4,0%** | **32,7%** | **18,5 M€/mes** | **sin cambios** |

Las parrillas concretas (umbral de rechazo por grupo, se acepta por encima):

| grupo | hoy | 2,5% | 3% | 3,5% | 4% |
|---|---|---|---|---|---|
| New | > 27 | > 63 | > 50 | > 37 | > 27 |
| Inactive | > 22 | > 58 | > 45 | > 32 | > 22 |
| A-C | > 16 | > 52 | > 39 | > 26 | > 16 |
| D-F | > 22 | > 58 | > 45 | > 32 | > 22 |
| ≥G | rechazo | rechazo | rechazo | rechazo | rechazo |

---

## Cuatro cosas que conviene tener presentes

**El riesgo de tienda es estimado.** El nivel está calibrado contra dato real, pero el poder discriminante del score en tienda es un supuesto hasta que maduren las operaciones de agosto y septiembre.

**Parte de la mejora frente a lo medido es volumen, no parrilla.** La parrilla vigente calca lo que el motor decide hoy en Online (acepta el 51,8% de la demanda; el motor aprobó el 51,9%), y cada solicitud aceptada convierte a la tasa de su decisión real —la misma tasa de financiación que usamos en el proceso general—, así que Online modelado y medido coinciden. Los 18,5 frente a 15,3 M€/mes son tienda, pero en dos partes: +2,3 M€ por la parrilla, sobre la misma demanda, y +0,9 M€ porque la demanda de tienda en agosto-septiembre va un 40% por encima de la media de la ventana de "hoy". Por eso el titular va sobre la misma base.

**Tienda lleva un año endureciendo por su cuenta**: su TA ha pasado del 58,7% (marzo-2025 a febrero-2026) al 47,2% en agosto-septiembre de 2026, mientras su demanda crece. La base contra la que comparamos se está moviendo.

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
* La fila "Hoy, con la demanda actual de tienda" es la política vigente evaluada sobre la misma demanda que los escenarios (Online en su ventana, tienda en ago-sep 2026): su diferencia con la fila siguiente es el efecto puro de la parrilla. Si a Jenny le sobra una fila de "hoy", quitar la primera y dejar esta, que es la comparable.
* El take-up va **por decisión del motor** (`ok` 44% / `rv` 5% en Online; las que entran nuevas por score convierten como `ok`), que es la `tasa_fin` del pipeline a nivel solicitud. En la corrida de agosto, la versión agregada `contratado / (ok+rv)` del proceso general daba lo mismo (16,76 frente a 16,82 M€/mes).
* La TA de los escenarios es **efectiva** (producción / demanda), la misma base que la del punto 1. El deck publica además la TA "de score" (demanda que supera el corte), que es mucho más alta y no es comparable con el 29,5% de hoy.
* La ventana de Online se mantiene en marzo-2025 a febrero-2026 aunque marzo-2026 ya está maduro en el extracto del 14-10. Moverla a abril-2025 → marzo-2026 no cambia Online (20,9% / 4,18%) y deja la parrilla vigente en 3,84%: mismo mensaje. Es una decisión aparte de añadir septiembre; de moverla, en la corrida definitiva.
