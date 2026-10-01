# Borrador de correo — Resultados del estudio Apple

> Generado el 01-10-2026 a partir de `output/apple_study/` (corrida con la parrilla vigente,
> ancla de nivel automática y reject inference). Si se vuelve a correr el estudio, revisar que
> las cifras siguen cuadrando antes de enviar: salen de `p1_online_actual.csv`,
> `p1b_tienda_actual.csv`, `p2_tienda_parrilla_online.csv`, `p3_escenarios_objetivo.csv` y
> `p4_parrillas_escenarios.csv`.

**Asunto:** Apple — Score Equifax en tienda: resultados y escenarios
**Adjunto:** `output/apple_deck/Estudio_Apple_EFX.pptx`

---

Hola Jenny,

Ya está el estudio. Te resumo primero la conclusión y debajo los tres puntos con el detalle.

**En una línea: al mismo nivel de riesgo que hoy (3,99%), extender la parrilla de Equifax de Online a tienda nos daría +4,0 puntos de tasa de aceptación y unos +2,6 M€ de producción al mes.** Y el objetivo de riesgo del 4% ya lo cumplimos hoy: no requiere ningún cambio.

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
| Tasa de aceptación | 47,6% | **63,5%** |
| Riesgo | 3,74% | **3,66%** |

Sube la aceptación y baja el riesgo a la vez. El riesgo de tienda es **estimado**, no medido —no habrá medición real hasta febrero de 2027—, pero su **nivel** está calibrado contra la morosidad que tienda ha tenido de verdad: la curva de Online predecía 3,79% y lo observado es 3,74%, un 1,2% de diferencia. Lo que seguimos asumiendo es que el score ordena en tienda igual que en e-commerce.

## 3. Escenarios de riesgo global Apple

| objetivo | TA Apple | producción | parrilla |
|---|---:|---:|---|
| Hoy | 29,5% | 15,3 M€/mes | vigente |
| 2,5% | 20,9% | 11,2 M€/mes | +38 puntos |
| 3,0% | 27,0% | 14,5 M€/mes | +25 puntos |
| 3,5% | 31,2% | 16,7 M€/mes | +13 puntos |
| **4,0%** | **33,4%** | **17,9 M€/mes** | **sin cambios** |

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

**El modelo produce más de lo que de hecho producimos** (17,9 frente a 15,3 M€/mes). No es un error: la parrilla vigente aprueba el 51,8% de la demanda de Online mientras el motor aprobó el 46,4% en esa misma ventana. Es 1,12x más laxa porque Online ha ido aflojando — del 40,7% de aprobación en 2025Q1 al 50,3% en 2025Q4. Parte de la ganancia que ves ya está en la parrilla de hoy, no en llevarla a tienda.

**Tienda lleva un año endureciendo por su cuenta**: su TA ha pasado del 59% en 2025 al 47,6% en agosto-2026. La base contra la que comparamos se está moviendo.

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
* La TA de los escenarios es **efectiva** (producción / demanda), la misma base que la del punto 1. El deck publica además la TA "de score" (demanda que supera el corte), que es mucho más alta y no es comparable con el 29,5% de hoy.
