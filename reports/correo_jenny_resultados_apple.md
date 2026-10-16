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
> ventana de "hoy", así que el titular se da sobre la misma base (fila nueva en `p3`).
>
> **14-10-2026 (tarde y noche):** secciones "Dos definiciones" y "Cómo se ha tratado la demanda de tienda" a
> petición de Jenny; tablas desde `p0_demanda_por_estado.csv`, `p2b_tienda_swap.csv` y `p3b_mezcla_canales.csv`.
> El deck gana las slides 3 y 10 (15 slides).
>
> **Actualizado el 16-10-2026:** extracto con las solicitudes **rechazadas en estudio ligero** (marca
> `tipotran = est1`: no pasan al estudio completo, no se llama a la API de Equifax, sin score; el extracto
> trae un 0 centinela). Se tratan como demanda —cuentan en el denominador de la TA, como en Online, que
> siempre llevó una población equivalente sin marca— pero nunca se puntúan ni se les imputa riesgo. Están
> en tienda desde julio de 2025 y en Online desde mayo de 2026 (fuera de su ventana). Efecto: el riesgo no cambia en
> nada; bajan las TA de tienda y del total por el denominador (titular +4,1 → +4,0 pp; +2,3 M€/mes igual).
> El índice estacional se calcula sin ellas para no inflar la segunda mitad de 2025.
>
> **16-10-2026 (tarde):** indicadores con nombre: *aprobación del sistema* (OK+RV / demanda) y *tasa de contratación*
> (contratado / demanda), y todo el total Apple desglosado en e-commerce / tienda / total (tablas nuevas en el punto 3;
> slide 8 del deck, 16 slides).
>
> **16-10-2026 (noche), verificación:** el estudio reproduce byte a byte y todas las cifras de este correo cuadran con
> los CSV. Nota al pie nueva en la tabla por canal: el riesgo de Online en la fila "hoy, con la demanda actual de
> tienda" es 4,15 % y no el 4,18 % medido porque 23 contratos aprobados **sin llamada a Equifax** (sin score, 88,6 % de
> mora) no tienen tramo y quedan fuera de la curva (`online_contratados_sin_efx_*` en `periodos.json`; slide 8).
>
> **16-10-2026 (noche), tasa de contratación de tienda en una sola base:** el punto 2 y la fila "aceptado por la parrilla"
> decían 55,7 % (total de `p2_tienda_parrilla_online.csv`, take-up `ok` por grupo de letra: 14,61 M€) mientras la tabla por canal,
> los 14,6 M€ del swap y el titular van con el take-up del canal (`p2b`/`p3b`/`p3`: 14,64 M€, 55,9 %). El correo usa ahora la base del
> canal en todas partes; el código no cambia (0,1 pp, 30 k€ en dos meses; el take-up por grupo sigue en las filas por grupo de `p2`).

**Asunto:** Apple — Score Equifax en tienda: resultados y escenarios
**Adjunto:** `output/apple_deck/Apple_EFX_Study.pptx` (en inglés)

---

Hola Jenny,

Ya está el estudio, actualizado con septiembre de tienda y con las solicitudes rechazadas en estudio ligero en la base. Te resumo primero la conclusión y debajo los tres puntos con el detalle.

**En una línea: extender la parrilla de Equifax de Online a tienda nos daría +4,0 puntos de tasa de contratación y unos +2,3 M€ de producción al mes, con el riesgo global algo por debajo del de hoy (3,85% frente a 3,95%).** Y el objetivo de riesgo del 4% ya lo cumplimos hoy: no requiere ningún cambio.

---

## Tres definiciones antes de las cifras

**Aprobación del sistema.** (OK + RV) / demanda solicitada, en euros: lo que el motor deja pasar antes de que el cliente decida. Bajo una parrilla se simula como la demanda que supera el corte y no viene rechazada por otra regla, así que es comparable entre hoy y cualquier escenario.

**Tasa de contratación.** Euros contratados / euros solicitados, sobre toda la demanda que entra —rechazada, cancelada y contratada—. Es la que usan el punto 1, los escenarios y el titular, y las dos se encadenan con el take-up: tasa de contratación = aprobación del sistema × take-up (de lo aprobado, lo que se contrata).

**Demanda no finalizada.** La que el motor aprueba y el cliente cancela. Cuenta en el denominador y no en el numerador, igual que un rechazo. En los escenarios, cada solicitud que pasa el corte convierte a la tasa real de contratación de su decisión, así que el modelo asume que se cancela igual que hoy: una parrilla más laxa no recupera cancelaciones. En la corrección por selección (reject inference) no cuenta ni como aceptada ni como rechazada, de modo que no infla el recargo. Lo mismo vale para la aprobada y rechazada en una fase posterior.

**Rechazada en estudio ligero.** La que se rechaza antes del estudio completo: no se llama a la API de Equifax y no tiene score. Cuenta en el denominador como cualquier rechazo, pero nunca es candidata a entrar por la parrilla ni se le imputa riesgo. En tienda está en la base desde julio de 2025 —la ventana de "hoy" la lleva en 8 de sus 12 meses, así que su TA es una mezcla— y Online siempre ha llevado una población equivalente (4,1% de su demanda).

Dónde acaba cada euro solicitado:

| | Online mar-25 → feb-26 | Tienda mar-25 → feb-26 | Tienda ago-sep 26 |
|---|---:|---:|---:|
| contratado (= tasa de contratación) | 20,9% | 55,1% | 43,0% |
| aprobado y cancelado (**no finalizada**) | 27,9% | 6,6% | 8,7% |
| aprobado y rechazado después | 3,1% | 2,5% | 1,8% |
| rechazado por score | 6,8% | 10,6% | 19,2% |
| rechazado por otras reglas | 37,2% | 19,1% | 18,3% |
| **rechazada en estudio ligero** (sin llamada a Equifax) | 4,1% | 6,2% | 9,0% |
| **total solicitado** | 100% | 100% | 100% |
| **aprobación del sistema** (OK + RV, tres primeras filas) | 51,9% | 64,1% | 53,5% |
| de lo aprobado, contrata | 40% | 86% | 80% |

Dos lecturas: en Online más de la mitad de lo aprobado no se finaliza y en tienda uno de cada seis, y ahí está la razón de fondo de que tienda contrate el doble con una aprobación del sistema parecida (51,9% frente a 53,5%). Y la caída de la tasa de contratación de tienda del 55,1% al 43,0% es sobre todo rechazo por score (10,6% → 19,2% de la demanda), no cancelaciones: el score interno ha endurecido.

## 1. Situación actual, medida

Ventana de originación marzo-2025 a febrero-2026 (doce meses con los seis de comportamiento cumplidos).

| | Aprobación del sistema | Tasa de contratación | Riesgo |
|---|---:|---:|---:|
| Apple Online | 51,9% | 20,9% | 4,18% |
| Apple Stores | 64,1% | 55,1% | 3,74% |
| **Apple Total** | **54,8%** | **29,0%** | **3,99%** |

Tienda contrata más del doble que Online y arriesga menos. En Online el riesgo se concentra en `New` (5,07%) frente a `A-C` (1,32%) o `Inactive` (0,81%).

## 2. Tienda con la parrilla de Online

Sobre agosto y septiembre de 2026, los dos meses con score Equifax en tienda:

| | hoy | con la parrilla de Online |
|---|---:|---:|
| Aprobación del sistema | 53,5% | **69,2%** |
| Tasa de contratación | 43,0% | **55,9%** |
| Riesgo | 3,74% | **3,52%** |

Sube la contratación y baja el riesgo a la vez. El riesgo de tienda es **estimado**, no medido —no habrá medición real hasta marzo de 2027—, pero su **nivel** está calibrado contra la morosidad que tienda ha tenido de verdad: la curva de Online predecía 3,92% para la cartera que tienda contrató y lo observado es 3,74%, un 5% de diferencia que se corrige. Lo que seguimos asumiendo es que el score ordena en tienda igual que en e-commerce.

## Cómo se ha tratado la demanda de tienda de agosto-septiembre

Es la única demanda de tienda con score Equifax, y los escenarios de tienda y del total salen de ella. Paso a paso:

1. **Score y tramo.** A cada solicitud se le asigna su score EFX y uno de los 20 tramos de la curva de Online. Las rechazadas en estudio ligero no tienen score y se quedan fuera de este paso.
2. **Parrilla de Online.** Se acepta si supera el umbral de rechazo de su grupo y no venía rechazada por otra regla: solo se cambia la regla de score, las demás reglas del motor se mantienen.
3. **Producción esperada.** Euros solicitados × take-up real de su decisión del motor (en tienda: aprobada 82%, en revisión 18%); se asume que se cancela igual que hoy.
4. **Riesgo imputado.** Cada euro aceptado toma el riesgo de su tramo en la curva de Online (ya corregida por selección), ponderado por la exposición H6 esperada; el nivel se reescala por 0,953 para que la cartera que tienda contrató reproduzca su morosidad real (3,74%).
5. **Total Apple.** Online y tienda se llevan a run-rate mensual con su propia estacionalidad y se mezclan ponderando por exposición esperada, que es la base del b2.

Qué hace la parrilla con cada euro:

| cada euro de la demanda de tienda ago-sep 26 | % demanda | producción esperada (total ago-sep) | riesgo imputado |
|---|---:|---:|---:|
| se mantiene: aprobada hoy y por encima del corte EFX | 51,7% | 10,9 M€ | 3,3% |
| **swap-in**: rechazada hoy por el score interno, por encima del corte EFX | 17,5% | 3,7 M€ | 4,1% |
| **swap-out**: aprobada hoy, por debajo del corte EFX | 1,8% | 0,4 M€ | 46,7% |
| **rechazada por los dos scores**: hoy por el score interno y por debajo del corte EFX | 1,7% | 0,4 M€ | 41,1% |
| intacta: rechazada por otras reglas (no se tocan) | 18,3% | 3,9 M€ | 7,9% |
| intacta: **rechazada en estudio ligero** (sin llamada a Equifax, sin score) | 9,0% | — | — |
| **aceptado por la parrilla** (las dos primeras) | **69,2%** | **14,6 M€ (55,9%)** | **3,52%** |
| hoy, con el score interno: aprobado por el motor → contratado | 53,5% | 11,3 M€ (43,0%) | 3,74% |

La última fila es lo que tienda contrató de verdad en agosto-septiembre con su score interno: 11,3 M€, el 43,0% de la demanda; valorada con la misma curva y el ancla da su 3,74% realizado. En las dos filas resumen, el porcentaje es la demanda que pasa la decisión (el motor hoy, 53,5%; la parrilla, 69,2%) y el paréntesis de la producción es la tasa de aceptación efectiva.

Dos cosas que se ven ahí. El swap-out es pequeño (1,8% de la demanda) pero carga un 47% de riesgo imputado (score EFX mediano 19): son las aprobaciones más débiles del score interno, y la parrilla las quita. El swap-in trae un 17,5% de demanda que hoy se rechaza, a un 4,1%. De ahí que suban aceptación y baje riesgo a la vez. Las rechazadas en estudio ligero (9,0% de la demanda) no llegan a la parrilla: sin llamada a Equifax no hay score.

Ojo con las bases: esta tabla son totales de los dos meses y la siguiente es run-rate mensual. Los 14,6 M€ aceptados, divididos por los 1,45 meses efectivos de la ventana, dan los 10,1 M€/mes de abajo (y los 26,2 M€ de demanda, los 18,1 M€/mes). No se divide por 2 porque agosto y septiembre son meses flojos de tienda: en un mes medio tienda pide y contrata más que en esos dos.

Y cómo se mezcla con Online:

| run-rate mensual | Online | Tienda | Apple |
|---|---:|---:|---:|
| meses efectivos de la ventana | 12,00 | 1,45 | — |
| demanda | 40,1 M€ | 18,1 M€ | 58,2 M€ |
| producción esperada | 8,4 M€ | 10,1 M€ | 18,5 M€ |
| peso en la exposición H6 esperada | 45% | 55% | 100% |
| riesgo (b2) | 4,26% | 3,52% | **3,85%** |

Tienda es el 31% de la demanda pero el 55% de la exposición, porque convierte el 56% de lo que se le pide y Online el 21%: en el riesgo total pesa más de la mitad. El 4,26% de Online lleva el recargo de la corrección por selección también sobre su propia cartera (su realizado es el 4,18%), como se explica en la slide de escenarios.

## 3. Escenarios de riesgo global Apple

Apple en total (Online + tienda, run-rate mensual):

| objetivo | aprobación del sistema | tasa de contratación | producción | riesgo | parrilla |
|---|---:|---:|---:|---:|---|
| Hoy, medido en la ventana | 54,8% | 29,0% | 15,3 M€/mes | 3,99% | vigente |
| Hoy, con la demanda actual de tienda | 52,4% | 27,7% | 16,2 M€/mes | 3,95% | vigente |
| 2,5% | 37,2% | 21,1% | 12,3 M€/mes | 2,48% | +36 puntos |
| 3,0% | 47,5% | 26,7% | 15,6 M€/mes | 2,97% | +23 puntos |
| 3,5% | 54,4% | 30,4% | 17,7 M€/mes | 3,48% | +10 puntos |
| **4,0%** | **57,2%** | **31,7%** | **18,5 M€/mes** | **3,85%** | **sin cambios** |

Y separado por canal:

| escenario | e-commerce: aprobación | contratación | producción/mes | riesgo | tienda: aprobación | contratación | producción/mes | riesgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hoy, medido en la ventana | 51,9% | 20,9% | 8,4 M€ | 4,18% | 64,1% | 55,1% | 6,9 M€ | 3,74% |
| Hoy, con la demanda actual de tienda | 51,9% | 20,9% | 8,4 M€ | 4,15% | 53,5% | 43,0% | 7,8 M€ | 3,74% |
| 2,5% | 32,1% | 13,0% | 5,2 M€ | 2,61% | 48,4% | 39,1% | 7,1 M€ | 2,40% |
| 3,0% | 41,8% | 16,9% | 6,8 M€ | 3,14% | 60,1% | 48,5% | 8,8 M€ | 2,83% |
| 3,5% | 48,8% | 19,7% | 7,9 M€ | 3,75% | 67,0% | 54,1% | 9,8 M€ | 3,27% |
| **4,0% (sin cambios)** | **51,8%** | **20,9%** | **8,4 M€** | **4,26%** | **69,2%** | **55,9%** | **10,1 M€** | **3,52%** |

En tienda, la fila "hoy, medido en la ventana" es marzo-2025 a febrero-2026 con el score interno; el resto de filas son agosto-septiembre de 2026, lo único con score Equifax. Online es siempre marzo-2025 a febrero-2026. El riesgo de Online en las filas de parrilla lleva el recargo de la corrección por selección sobre su propia cartera (realizado 4,18%). El 4,15% de Online en la fila "hoy, con la demanda actual de tienda" va sobre la cartera con score: 23 contratos aprobados sin llamada a Equifax (sin score; el 0,03% de la producción, con un 88,6% de mora) entran en el 4,18% medido pero no tienen tramo, así que quedan fuera de la curva y de todas las filas valoradas por tramo; la parrilla nunca podría aceptarlos.

Las parrillas concretas (umbral de rechazo por grupo, se acepta por encima):

| grupo | hoy | 2,5% | 3% | 3,5% | 4% |
|---|---|---|---|---|---|
| New | > 27 | > 63 | > 50 | > 37 | > 27 |
| Inactive | > 22 | > 58 | > 45 | > 32 | > 22 |
| A-C | > 16 | > 52 | > 39 | > 26 | > 16 |
| D-F | > 22 | > 58 | > 45 | > 32 | > 22 |
| ≥G | rechazo | rechazo | rechazo | rechazo | rechazo |

---

## Cinco cosas que conviene tener presentes

**El riesgo de tienda es estimado.** El nivel está calibrado contra dato real, pero el poder discriminante del score en tienda es un supuesto hasta que maduren las operaciones de agosto y septiembre.

**Parte de la mejora frente a lo medido es volumen, no parrilla.** La parrilla vigente calca lo que el motor decide hoy en Online (acepta el 51,8% de la demanda; el motor aprobó el 51,9%), y cada solicitud aceptada convierte a la tasa de su decisión real —la misma tasa de financiación que usamos en el proceso general—, así que Online modelado y medido coinciden. Los 18,5 frente a 15,3 M€/mes son tienda, pero en dos partes: +2,3 M€ por la parrilla, sobre la misma demanda, y +0,9 M€ porque la demanda de tienda en agosto-septiembre va un 40% por encima de la media de la ventana de "hoy". Por eso el titular va sobre la misma base.

**Tienda lleva un año endureciendo por su cuenta**: su tasa de contratación ha pasado del 55,1% (marzo-2025 a febrero-2026) al 43,0% en agosto-septiembre de 2026, mientras su demanda crece. La base contra la que comparamos se está moviendo.

**La base de tienda incluye desde julio de 2025 las rechazadas en estudio ligero.** Son el 9,0% de su demanda de agosto-septiembre. Bajan la tasa de contratación de tienda y del total pero no tocan el riesgo: no llegan a la parrilla. Como la ventana de "hoy" solo las lleva en 8 de sus 12 meses, el 55,1% de tienda es una mezcla de bases; si se puede extraer su histórico completo, conviene rehacer esa cifra.

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
* Los cinco avisos quitan brillo al titular, pero son las preguntas que saldrán en la sala: mejor que salgan de ti.
* La fila "Hoy, con la demanda actual de tienda" es la política vigente evaluada sobre la misma demanda que los escenarios (Online en su ventana, tienda en ago-sep 2026): su diferencia con la fila siguiente es el efecto puro de la parrilla. Frente a lo medido en la ventana, la parrilla da +2,7 pp y +3,2 M€, de lo que +0,9 M€ es volumen.
* El take-up va **por decisión del motor** (`ok` 44% / `rv` 6% en Online; las que entran nuevas por score convierten como `ok`), que es la `tasa_fin` del pipeline a nivel solicitud.
* Nombres (16-10): "aprobación del sistema" = (OK + RV) / demanda, medida hoy y simulada bajo cada parrilla; "tasa de contratación" = contratado / demanda. Negocio llama a la segunda *financing rate*; aquí se evita porque en el proceso general "tasa de financiación" (`tasa_fin`) es contratado / aprobado, o sea el take-up, y chocarían. Decidido (16-10): se queda *booking rate* / tasa de contratación. El deck llamaba antes "TA de score" a la aprobación simulada y "TA efectiva" a la contratación.
* El 3,85% de la parrilla lleva el recargo de reject inference también sobre la cartera ya contratada de Online (+0,08 pp sobre su 4,18% realizado); las dos filas de "hoy" van a realizado. Like-for-like con el 3,95%, la parrilla daría 3,76% (-0,19 pp en vez de -0,10). Se deja el conservador; el deck lo explica en la slide de la frontera.
* Estudio ligero: pedir si el histórico de estas rechazadas existe antes de julio de 2025 en tienda y antes de mayo de 2026 en Online. Con la historia completa, la ventana de "hoy" de tienda dejaría de ser una mezcla y el índice estacional podría calcularse con toda la demanda. Hoy el índice va sin ellas (`tipotran = est1`), para no inflar la segunda mitad de 2025.
* El total de `p2_tienda_parrilla_online.csv` da 55,7% de contratación de tienda bajo la parrilla porque su take-up va por grupo de
  letra (para que ningún grupo parezca perder aceptación); el correo va con la base del canal (55,9%), que es la del titular, los
  escenarios y los 14,6 M€ de la slide 11. Diferencia 0,1 pp / 30 k€ en dos meses.
* Los 23 contratos de Online aprobados **sin llamada a Equifax** (marzo-mayo de 2025, 22 `ok` + 1 `rv`, 32,6 k€) tienen un 88,6% de mora: una anomalía que conviene enseñar a negocio y preguntar por qué se aprobaron sin score. Son el 0,03% de la producción, pero mueven el riesgo de Online 0,03 pp (4,18 → 4,15) y explican por qué el like-for-like pasó de -0,21 a -0,19 pp al excluirlos del tramo 1 (antes del 16-10 el centinela 0 caía en el tramo más bajo).
* La ventana de Online se mantiene en marzo-2025 a febrero-2026 aunque marzo-2026 ya está maduro. Moverla no cambia Online (20,9% / 4,18%); de moverla, en la corrida definitiva. Ojo: Online lleva estudio ligero desde mayo de 2026 (17-23% de su demanda), así que mover la ventana hasta ahí bajaría su TA por el denominador.
