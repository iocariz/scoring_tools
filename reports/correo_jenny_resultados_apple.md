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
>
> **14-10-2026 (tarde):** sección "Dos definiciones" a petición de Jenny (definición de la TA y tratamiento de la
> demanda no finalizada); la tabla sale de `p0_demanda_por_estado.csv`. El deck gana la slide 3, "How acceptance
> is measured".
>
> **14-10-2026 (noche):** sección "Cómo se ha tratado la demanda de tienda de agosto-septiembre" (Jenny quiere
> profundizar); tablas desde `p2b_tienda_swap.csv` y `p3b_mezcla_canales.csv`. El deck gana la slide 10,
> "Store demand Aug–Sep 2026: from applications to risk" (15 slides).

**Asunto:** Apple — Score Equifax en tienda: resultados y escenarios
**Adjunto:** `output/apple_deck/Apple_EFX_Study.pptx` (en inglés)

---

Hola Jenny,

Ya está el estudio, actualizado con septiembre de tienda. Te resumo primero la conclusión y debajo los tres puntos con el detalle.

**En una línea: extender la parrilla de Equifax de Online a tienda nos daría +4,1 puntos de tasa de aceptación y unos +2,3 M€ de producción al mes, con el riesgo global algo por debajo del de hoy (3,85% frente a 3,97%).** Y el objetivo de riesgo del 4% ya lo cumplimos hoy: no requiere ningún cambio.

---

## Dos definiciones antes de las cifras

**Tasa de aceptación (TA).** En todo el estudio es la **efectiva**: euros contratados / euros solicitados, sobre toda la demanda que entra —rechazada, cancelada y contratada—. No es la tasa de aprobación del motor, que es más alta: entre aprobar y contratar el cliente puede no finalizar. En el deck aparece además una "TA de score" (demanda que supera el corte), comparable con la aprobación del motor y no con la TA.

**Demanda no finalizada.** La que el motor aprueba y el cliente cancela. Cuenta en el denominador y no en el numerador, igual que un rechazo. En los escenarios, cada solicitud que pasa el corte convierte a la tasa real de contratación de su decisión, así que el modelo asume que se cancela igual que hoy: una parrilla más laxa no recupera cancelaciones. En la corrección por selección (reject inference) no cuenta ni como aceptada ni como rechazada, de modo que no infla el recargo. Lo mismo vale para la aprobada y rechazada en una fase posterior.

Dónde acaba cada euro solicitado:

| | Online mar-25 → feb-26 | Tienda mar-25 → feb-26 | Tienda ago-sep 26 |
|---|---:|---:|---:|
| contratado (= TA) | 20,9% | 58,7% | 47,2% |
| aprobado y cancelado (**no finalizada**) | 27,9% | 7,0% | 9,6% |
| aprobado y rechazado después | 3,1% | 2,6% | 2,0% |
| rechazado por score | 6,9% | 11,3% | 21,1% |
| rechazado por otras reglas | 41,2% | 20,3% | 20,1% |
| **total solicitado** | 100% | 100% | 100% |
| aprobación del motor (tres primeras filas) | 51,9% | 68,3% | 58,8% |
| de lo aprobado, contrata | 40% | 86% | 80% |

Dos lecturas: en Online más de la mitad de lo aprobado no se finaliza y en tienda uno de cada seis, y ahí está la razón de fondo de que tienda acepte casi el triple con una aprobación parecida (51,9% frente a 58,8%). Y la caída de la TA de tienda del 58,7% al 47,2% es casi toda rechazo por score (11,3% → 21,1% de la demanda), no cancelaciones: el score interno ha endurecido.

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

## Cómo se ha tratado la demanda de tienda de agosto-septiembre

Es la única demanda de tienda con score Equifax, y los escenarios de tienda y del total salen de ella. Paso a paso:

1. **Score y tramo.** A cada solicitud se le asigna su score EFX y uno de los 20 tramos de la curva de Online.
2. **Parrilla de Online.** Se acepta si supera el umbral de rechazo de su grupo y no venía rechazada por otra regla: solo se cambia la regla de score, las demás reglas del motor se mantienen.
3. **Producción esperada.** Euros solicitados × take-up real de su decisión del motor (en tienda: aprobada 82%, en revisión 18%); se asume que se cancela igual que hoy.
4. **Riesgo imputado.** Cada euro aceptado toma el riesgo de su tramo en la curva de Online (ya corregida por selección), ponderado por la exposición H6 esperada; el nivel se reescala por 0,952 para que la cartera que tienda contrató reproduzca su morosidad real (3,74%).
5. **Total Apple.** Online y tienda se llevan a run-rate mensual con su propia estacionalidad y se mezclan ponderando por exposición esperada, que es la base del b2.

Qué hace la parrilla con cada euro:

| cada euro de la demanda de tienda ago-sep 26 | % demanda | producción esperada (total ago-sep) | riesgo imputado |
|---|---:|---:|---:|
| se mantiene: aprobada hoy y por encima del corte EFX | 56,8% | 10,9 M€ | 3,3% |
| **swap-in**: rechazada hoy por el score interno, por encima del corte EFX | 19,3% | 3,7 M€ | 4,1% |
| **swap-out**: aprobada hoy, por debajo del corte EFX | 2,0% | 0,4 M€ | 47,6% |
| **rechazada por los dos scores**: hoy por el score interno y por debajo del corte EFX | 1,8% | 0,4 M€ | 41,6% |
| intacta: rechazada por otras reglas (no se tocan) | 20,1% | 3,9 M€ | 8,0% |
| **aceptado por la parrilla** (las dos primeras) | **76,0%** | **14,6 M€** | **3,51%** |
| la cartera contratada hoy, valorada igual | 47,2% | 11,3 M€ | 3,74% |

Dos cosas que se ven ahí. El swap-out es pequeño (2,0% de la demanda) pero carga un 48% de riesgo imputado (score EFX mediano 19): son las aprobaciones más débiles del score interno, y la parrilla las quita. El swap-in trae un 19,3% de demanda que hoy se rechaza, a un 4,1%. De ahí que suban aceptación y baje riesgo a la vez.

Ojo con las bases: esta tabla son totales de los dos meses y la siguiente es run-rate mensual. Los 14,6 M€ aceptados, divididos por los 1,45 meses efectivos de la ventana, dan los 10,1 M€/mes de abajo (y los 23,9 M€ de demanda, los 16,5 M€/mes). No se divide por 2 porque agosto y septiembre son meses flojos de tienda (índices 0,76 y 0,69): en un mes medio tienda pide y contrata más que en esos dos.

Y cómo se mezcla con Online:

| run-rate mensual | Online | Tienda | Apple |
|---|---:|---:|---:|
| meses efectivos de la ventana | 12,00 | 1,45 | — |
| demanda | 40,1 M€ | 16,5 M€ | 56,6 M€ |
| producción esperada | 8,4 M€ | 10,1 M€ | 18,5 M€ |
| peso en la exposición H6 esperada | 45% | 55% | 100% |
| riesgo (b2) | 4,26% | 3,51% | **3,85%** |

Tienda es el 29% de la demanda pero el 55% de la exposición, porque convierte el 61% de lo que se le pide y Online el 21%: en el riesgo total pesa más de la mitad. El 4,26% de Online lleva el recargo de la corrección por selección también sobre su propia cartera (su realizado es el 4,18%), como se explica en la slide de escenarios.

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

Tienes la presentación adjunta (en inglés) con el detalle y la metodología.

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
* El 3,85% de la parrilla lleva el recargo de reject inference también sobre la cartera ya contratada de Online (+0,08 pp sobre su 4,18% realizado); las dos filas de "hoy" van a realizado. Like-for-like con el 3,97%, la parrilla daría 3,76% (−0,21 pp en vez de −0,12). Se deja el conservador; el deck lo explica en la slide de la frontera.
* La ventana de Online se mantiene en marzo-2025 a febrero-2026 aunque marzo-2026 ya está maduro en el extracto del 14-10. Moverla a abril-2025 → marzo-2026 no cambia Online (20,9% / 4,18%) y deja la parrilla vigente en 3,84%: mismo mensaje. Es una decisión aparte de añadir septiembre; de moverla, en la corrida definitiva.
