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
>
> **Actualizado el 18-11-2026:** extracto del 17-11 con **octubre de tienda** (ventana ago-oct 2026, tres meses con score;
> Online sigue hasta septiembre y su ventana no cambia). El factor de nivel pasa de 0,953 a 0,956; el titular, de +4,0 a
> +3,9 pp de contratación con los mismos +2,3 M€/mes; el riesgo de la parrilla sigue en 3,85% y los cortes de los escenarios
> no se mueven. La demanda de tienda va ya un 43% por encima de la ventana (octubre es un mes fuerte). Las dudas de Bernard
> (distribución por score, deriva del riesgo, CESP, ingresos) se responden aparte: `reports/correo_bernard_dudas_apple.md`.
>
> **18-11-2026 (tarde), revisión de la metodología:** dos cambios de base en el total Apple. (1) Online entra con su
> **volumen actual** (demanda por mes efectivo de mar-sep 2026, +23% sobre la ventana), no con el de hace un año: antes se
> mezclaba con el volumen actual de tienda y tienda pesaba el 55% de la exposición (ahora el 50%). (2) El riesgo de Online
> lleva un **factor de deriva 1,15** (índice H6 a igual score de las cohortes de 2026, `b2_deriva_riesgo.csv`). Titular:
> +3,4 pp / +2,3 M€/mes, riesgo 4,20% frente a 4,32% hoy; **el 4% ya no se cumple sin cambios: pide +5 puntos**. El mensaje
> sobre el riesgo de tienda pasa de "baja" a "parecido" (la bajada depende de los tramos 1-5 de la curva). Reproducir lo
> anterior: `--online-risk-factor 1.0 --online-window-volume`.

**Asunto:** Apple — Score Equifax en tienda: resultados y escenarios
**Adjunto:** `output/apple_deck/Apple_EFX_Study.pptx` (en inglés)

---

Hola Jenny,

Ya está el estudio, actualizado con octubre de tienda (ya son tres meses con score Equifax) y con dos cambios de base que explico abajo: Online entra en el total con su volumen actual y con su riesgo reciente. Te resumo primero la conclusión y debajo los tres puntos con el detalle.

**En una línea: extender la parrilla de Equifax de Online a tienda nos daría +3,4 puntos de tasa de contratación y unos +2,3 M€ de producción al mes, con un riesgo global parecido al de hoy (4,20% frente a 4,32%).** Lo que cambia respecto a la versión anterior es el objetivo del 4%: con el riesgo reciente de Online ya no se cumple sin tocar nada, ni hoy ni con la parrilla; pide subir los cortes 5 puntos, y aun así quedan +1,9 M€/mes sobre lo de hoy.

---

## Tres definiciones antes de las cifras

**Aprobación del sistema.** (OK + RV) / demanda solicitada, en euros: lo que el motor deja pasar antes de que el cliente decida. Bajo una parrilla se simula como la demanda que supera el corte y no viene rechazada por otra regla, así que es comparable entre hoy y cualquier escenario.

**Tasa de contratación.** Euros contratados / euros solicitados, sobre toda la demanda que entra —rechazada, cancelada y contratada—. Es la que usan el punto 1, los escenarios y el titular, y las dos se encadenan con el take-up: tasa de contratación = aprobación del sistema × take-up (de lo aprobado, lo que se contrata).

**Demanda no finalizada.** La que el motor aprueba y el cliente cancela. Cuenta en el denominador y no en el numerador, igual que un rechazo. En los escenarios, cada solicitud que pasa el corte convierte a la tasa real de contratación de su decisión, así que el modelo asume que se cancela igual que hoy: una parrilla más laxa no recupera cancelaciones. En la corrección por selección (reject inference) no cuenta ni como aceptada ni como rechazada, de modo que no infla el recargo. Lo mismo vale para la aprobada y rechazada en una fase posterior.

**Rechazada en estudio ligero.** La que se rechaza antes del estudio completo: no se llama a la API de Equifax y no tiene score. Cuenta en el denominador como cualquier rechazo, pero nunca es candidata a entrar por la parrilla ni se le imputa riesgo. En tienda está en la base desde julio de 2025 —la ventana de "hoy" la lleva en 8 de sus 12 meses, así que su TA es una mezcla— y Online siempre ha llevado una población equivalente (4,1% de su demanda).

Dónde acaba cada euro solicitado:

| | Online mar-25 → feb-26 | Tienda mar-25 → feb-26 | Tienda ago-oct 26 |
|---|---:|---:|---:|
| contratado (= tasa de contratación) | 20,9% | 55,1% | 43,7% |
| aprobado y cancelado (**no finalizada**) | 27,9% | 6,6% | 8,4% |
| aprobado y rechazado después | 3,1% | 2,5% | 1,8% |
| rechazado por score | 6,8% | 10,6% | 18,5% |
| rechazado por otras reglas | 37,2% | 19,1% | 19,0% |
| **rechazada en estudio ligero** (sin llamada a Equifax) | 4,1% | 6,2% | 8,6% |
| **total solicitado** | 100% | 100% | 100% |
| **aprobación del sistema** (OK + RV, tres primeras filas) | 51,9% | 64,1% | 53,9% |
| de lo aprobado, contrata | 40% | 86% | 81% |

Dos lecturas: en Online más de la mitad de lo aprobado no se finaliza y en tienda uno de cada seis, y ahí está la razón de fondo de que tienda contrate el doble con una aprobación del sistema parecida (51,9% frente a 53,9%). Y la caída de la tasa de contratación de tienda del 55,1% al 43,7% es sobre todo rechazo por score (10,6% → 18,5% de la demanda), no cancelaciones: el score interno ha endurecido.

## 1. Situación actual, medida

Ventana de originación marzo-2025 a febrero-2026 (doce meses con los seis de comportamiento cumplidos).

| | Aprobación del sistema | Tasa de contratación | Riesgo |
|---|---:|---:|---:|
| Apple Online | 51,9% | 20,9% | 4,18% |
| Apple Stores | 64,1% | 55,1% | 3,74% |
| **Apple Total** | **54,8%** | **29,0%** | **3,99%** |

Tienda contrata más del doble que Online y arriesga menos. En Online el riesgo se concentra en `New` (5,07%) frente a `A-C` (1,32%) o `Inactive` (0,81%).

## 2. Tienda con la parrilla de Online

Sobre agosto, septiembre y octubre de 2026, los tres meses con score Equifax en tienda:

| | hoy | con la parrilla de Online |
|---|---:|---:|
| Aprobación del sistema | 53,9% | **68,9%** |
| Tasa de contratación | 43,7% | **56,2%** |
| Riesgo | 3,74% | **3,52%** |

Sube la contratación con un riesgo parecido. El riesgo de tienda es **estimado**, no medido —no habrá medición real hasta marzo de 2027—, pero su **nivel** está calibrado contra la morosidad que tienda ha tenido de verdad: la curva de Online predecía 3,91% para la cartera que tienda contrató y lo observado es 3,74%, algo menos de un 5% de diferencia que se corrige. Lo que seguimos asumiendo es que el score ordena en tienda igual que en e-commerce. Y la bajada del 3,74% al 3,52% hay que leerla como "parecido", no como mejora: sale entera de quitar el swap-out, cuyo riesgo se imputa con los tramos más bajos de la curva, que son los de menos muestra.

## Cómo se ha tratado la demanda de tienda de agosto-octubre

Es la única demanda de tienda con score Equifax, y los escenarios de tienda y del total salen de ella. Paso a paso:

1. **Score y tramo.** A cada solicitud se le asigna su score EFX y uno de los 20 tramos de la curva de Online. Las rechazadas en estudio ligero no tienen score y se quedan fuera de este paso.
2. **Parrilla de Online.** Se acepta si supera el umbral de rechazo de su grupo y no venía rechazada por otra regla: solo se cambia la regla de score, las demás reglas del motor se mantienen.
3. **Producción esperada.** Euros solicitados × take-up real de su decisión del motor (en tienda: aprobada 82%, en revisión 18%); se asume que se cancela igual que hoy.
4. **Riesgo imputado.** Cada euro aceptado toma el riesgo de su tramo en la curva de Online (ya corregida por selección), ponderado por la exposición H6 esperada; el nivel se reescala por 0,956 para que la cartera que tienda contrató reproduzca su morosidad real (3,74%).
5. **Total Apple.** Online y tienda se llevan a run-rate mensual con su propia estacionalidad y se mezclan ponderando por exposición esperada, que es la base del b2. Los dos canales entran con su volumen actual: tienda con agosto-octubre de 2026 y Online con el de marzo-septiembre de 2026, que va un 23% por encima del de su ventana (sus tasas y su curva siguen saliendo de la ventana madura). Y el riesgo de Online se multiplica por 1,15: a igual score, sus cohortes de 2026 llevan un 15% más de riesgo a seis meses que la ventana con la que se estima la curva.

Qué hace la parrilla con cada euro:

| cada euro de la demanda de tienda ago-oct 26 | % demanda | producción esperada (total ago-oct) | riesgo imputado |
|---|---:|---:|---:|
| se mantiene: aprobada hoy y por encima del corte EFX | 52,1% | 20,2 M€ | 3,3% |
| **swap-in**: rechazada hoy por el score interno, por encima del corte EFX | 16,8% | 6,6 M€ | 4,1% |
| **swap-out**: aprobada hoy, por debajo del corte EFX | 1,8% | 0,7 M€ | 45,5% |
| **rechazada por los dos scores**: hoy por el score interno y por debajo del corte EFX | 1,7% | 0,7 M€ | 40,2% |
| intacta: rechazada por otras reglas (no se tocan) | 19,0% | 7,5 M€ | 7,9% |
| intacta: **rechazada en estudio ligero** (sin llamada a Equifax, sin score) | 8,6% | — | — |
| **aceptado por la parrilla** (las dos primeras) | **68,9%** | **26,9 M€ (56,2%)** | **3,52%** |
| hoy, con el score interno: aprobado por el motor → contratado | 53,9% | 20,9 M€ (43,7%) | 3,74% |

La última fila es lo que tienda contrató de verdad en agosto-octubre con su score interno: 20,9 M€, el 43,7% de la demanda; valorada con la misma curva y el ancla da su 3,74% realizado. En las dos filas resumen, el porcentaje es la demanda que pasa la decisión (el motor hoy, 53,9%; la parrilla, 68,9%) y el paréntesis de la producción es la tasa de aceptación efectiva.

Dos cosas que se ven ahí. El swap-out es pequeño (1,8% de la demanda) pero carga un 46% de riesgo imputado (score EFX mediano 19): son las aprobaciones más débiles del score interno, y la parrilla las quita. El swap-in trae un 16,8% de demanda que hoy se rechaza, a un 4,1%. De ahí que suba la aceptación sin que suba el riesgo. Las rechazadas en estudio ligero (8,6% de la demanda) no llegan a la parrilla: sin llamada a Equifax no hay score.

Ojo con las bases: esta tabla son totales de los tres meses y la siguiente es run-rate mensual. Los 26,9 M€ aceptados, divididos por los 2,58 meses efectivos de la ventana, dan los 10,4 M€/mes de abajo (y los 47,8 M€ de demanda, los 18,5 M€/mes). No se divide por 3 porque agosto y septiembre son meses flojos de tienda (octubre ya está por encima de la media): en un mes medio tienda pide y contrata más que en el promedio de esos tres.

Y cómo se mezcla con Online:

| run-rate mensual | Online | Tienda | Apple |
|---|---:|---:|---:|
| meses efectivos de la ventana | 9,77 | 2,58 | — |
| demanda | 49,3 M€ | 18,5 M€ | 67,8 M€ |
| producción esperada | 10,3 M€ | 10,4 M€ | 20,7 M€ |
| peso en la exposición H6 esperada | 50% | 50% | 100% |
| riesgo (b2) | 4,89% | 3,52% | **4,20%** |

Tienda es el 27% de la demanda pero el 50% de la exposición, porque convierte el 56% de lo que se le pide y Online el 21%: en el riesgo total pesa la mitad. Los 9,77 meses de Online son sus 12 de ventana llevados al volumen actual (un 23% más de demanda por mes). El 4,89% de Online es su curva con el recargo de la corrección por selección (4,25%; su realizado es el 4,18%) multiplicada por el 1,15 de deriva.

## 3. Escenarios de riesgo global Apple

Apple en total (Online + tienda, run-rate mensual):

| objetivo | aprobación del sistema | tasa de contratación | producción | riesgo | parrilla |
|---|---:|---:|---:|---:|---|
| Hoy, medido en la ventana | 54,8% | 29,0% | 15,3 M€/mes | 3,99% | vigente |
| Hoy, con la demanda actual | 52,4% | 27,1% | 18,4 M€/mes | 4,32% | vigente |
| Parrilla de Online en ambos canales | 56,5% | 30,5% | 20,7 M€/mes | 4,20% | vigente |
| 2,5% | 32,2% | 17,9% | 12,1 M€/mes | 2,49% | +41 puntos |
| 3,0% | 43,1% | 23,7% | 16,1 M€/mes | 2,97% | +28 puntos |
| 3,5% | 50,5% | 27,5% | 18,7 M€/mes | 3,47% | +17 puntos |
| **4,0%** | **55,2%** | **29,9%** | **20,3 M€/mes** | **3,97%** | **+5 puntos** |

Y separado por canal:

| escenario | e-commerce: aprobación | contratación | producción/mes | riesgo | tienda: aprobación | contratación | producción/mes | riesgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hoy, medido en la ventana | 51,9% | 20,9% | 8,4 M€ | 4,18% | 64,1% | 55,1% | 6,9 M€ | 3,74% |
| Hoy, con la demanda actual | 51,9% | 20,9% | 10,3 M€ | 4,77% | 53,9% | 43,7% | 8,1 M€ | 3,74% |
| Parrilla de Online en ambos canales | 51,8% | 20,9% | 10,3 M€ | 4,89% | 68,9% | 56,2% | 10,4 M€ | 3,52% |
| 2,5% | 28,1% | 11,4% | 5,6 M€ | 2,77% | 43,3% | 35,3% | 6,5 M€ | 2,24% |
| 3,0% | 38,2% | 15,4% | 7,6 M€ | 3,34% | 55,9% | 45,6% | 8,4 M€ | 2,65% |
| 3,5% | 45,6% | 18,4% | 9,1 M€ | 3,93% | 63,6% | 51,9% | 9,6 M€ | 3,03% |
| **4,0% (+5 puntos)** | **50,4%** | **20,3%** | **10,0 M€** | **4,56%** | **68,1%** | **55,5%** | **10,3 M€** | **3,39%** |

En tienda, la fila "hoy, medido en la ventana" es marzo-2025 a febrero-2026 con el score interno; el resto de filas son agosto-octubre de 2026, lo único con score Equifax. En Online las tasas son siempre las de marzo-2025 a febrero-2026; fuera de la fila medida, su producción va al volumen actual y su riesgo lleva el 1,15 de deriva, y en las filas de parrilla además el recargo de la corrección por selección sobre su propia cartera (realizado 4,18%). El 4,77% de Online en la fila "hoy, con la demanda actual" es el 4,14% de su cartera con score por 1,15: 23 contratos aprobados sin llamada a Equifax (sin score; el 0,03% de la producción, con un 88,6% de mora) entran en el 4,18% medido pero no tienen tramo, así que quedan fuera de la curva y de todas las filas valoradas por tramo; la parrilla nunca podría aceptarlos.

Las parrillas concretas (umbral de rechazo por grupo, se acepta por encima):

| grupo | hoy | 2,5% | 3% | 3,5% | 4% |
|---|---|---|---|---|---|
| New | > 27 | > 68 | > 55 | > 44 | > 32 |
| Inactive | > 22 | > 63 | > 50 | > 39 | > 27 |
| A-C | > 16 | > 57 | > 44 | > 33 | > 21 |
| D-F | > 22 | > 63 | > 50 | > 39 | > 27 |
| ≥G | rechazo | rechazo | rechazo | rechazo | rechazo |

---

## Seis cosas que conviene tener presentes

**El riesgo de tienda es estimado.** El nivel está calibrado contra dato real, pero el poder discriminante del score en tienda es un supuesto hasta que maduren las operaciones de agosto a octubre. Además, que el riesgo de tienda baje con la parrilla depende de los tramos de score más bajos de la curva (582 contratos de Online): si su riesgo real fuera la mitad del estimado, tienda quedaría igual que hoy. Lo sólido es más contratación a riesgo parecido.

**El riesgo de Online ha subido a igual score, y el estudio lo recoge con un factor.** La curva se estima sobre marzo-2025 a febrero-2026 y no lleva tendencia. Las cohortes de Online de 2026 van un 15% por encima a seis meses, y a tres meses el segundo trimestre va un 33% por encima, así que el 1,15 es el caso central y no el pesimista: con un 1,30 la parrilla vigente daría 4,52% y el 4% pediría 12 puntos de corte en vez de 5. En tienda el dato disponible, hasta julio, no muestra esa subida y no se le aplica factor.

**La mayor parte de la diferencia con lo medido es volumen, no parrilla.** La parrilla vigente calca lo que el motor decide hoy en Online (acepta el 51,8% de la demanda; el motor aprobó el 51,9%), y cada solicitud aceptada convierte a la tasa de su decisión real —la misma tasa de financiación que usamos en el proceso general—, así que las tasas de Online modeladas y medidas coinciden. Los 20,7 frente a 15,3 M€/mes son +2,3 M€ por la parrilla en tienda, sobre la misma demanda, y +3,1 M€ porque los dos canales van hoy por encima de la ventana de "hoy": la demanda de tienda en agosto-octubre un 43% por encima y la de Online desde marzo de 2026 un 23%. Por eso el titular va sobre la misma base.

**Tienda lleva un año endureciendo por su cuenta**: su tasa de contratación ha pasado del 55,1% (marzo-2025 a febrero-2026) al 43,7% en agosto-octubre de 2026, mientras su demanda crece. La base contra la que comparamos se está moviendo.

**La base de tienda incluye desde julio de 2025 las rechazadas en estudio ligero.** Son el 8,6% de su demanda de agosto-octubre. Bajan la tasa de contratación de tienda y del total pero no tocan el riesgo: no llegan a la parrilla. Como la ventana de "hoy" solo las lleva en 8 de sus 12 meses, el 55,1% de tienda es una mezcla de bases; si se puede extraer su histórico completo, conviene rehacer esa cifra.

**El perímetro es Apple Propio**, una sola cadena, sin APRS.

## Siguiente paso

Decidme qué objetivo de riesgo queréis y si preferís un corte único para todo Apple —más simple de operar— o desplazar la parrilla por grupo, que da algo más de aceptación al mismo riesgo. Con eso lanzo la corrida definitiva y preparo la propuesta de implementación.

Tienes la presentación adjunta (en inglés) con el detalle y la metodología.

Un saludo,
Iñigo

---

## Notas para Iñigo (no enviar)

* Los escenarios de la tabla van con la **parrilla desplazada**, no con el corte único, porque dan más TA al mismo riesgo. Las dos opciones están en el deck (slide de parrillas).
* El objetivo del 4% va en negrita por ser el hallazgo más accionable. Hasta el 18-11 decía "sin cambios": con Online a volumen actual y riesgo × 1,15 ya no se cumple sin mover la parrilla (+5 puntos). Sin el factor (`--online-risk-factor 1.0`) la parrilla vigente da 3,88% y sí se cumple.
* Los seis avisos quitan brillo al titular, pero son las preguntas que saldrán en la sala: mejor que salgan de ti.
* La fila "Hoy, con la demanda actual" es la política vigente evaluada sobre la misma base que los escenarios (Online con las tasas de su ventana, su volumen de mar-sep 2026 y su riesgo × 1,15; tienda en ago-oct 2026): su diferencia con la fila siguiente es el efecto puro de la parrilla. Frente a lo medido en la ventana, la parrilla da +1,5 pp y +5,4 M€, de lo que +3,1 M€ es volumen.
* El take-up va **por decisión del motor** (`ok` 44% / `rv` 6% en Online; las que entran nuevas por score convierten como `ok`), que es la `tasa_fin` del pipeline a nivel solicitud.
* Nombres (16-10): "aprobación del sistema" = (OK + RV) / demanda, medida hoy y simulada bajo cada parrilla; "tasa de contratación" = contratado / demanda. Negocio llama a la segunda *financing rate*; aquí se evita porque en el proceso general "tasa de financiación" (`tasa_fin`) es contratado / aprobado, o sea el take-up, y chocarían. Decidido (16-10): se queda *booking rate* / tasa de contratación. El deck llamaba antes "TA de score" a la aprobación simulada y "TA efectiva" a la contratación.
* El 4,20% de la parrilla lleva el recargo de reject inference también sobre la cartera ya contratada de Online (+0,08 pp sobre su 4,18% realizado, antes de la deriva). Like-for-like con el 4,32%, la parrilla daría 4,10% (-0,22 pp en vez de -0,12). Se deja el conservador; el deck lo explica en la slide de la frontera.
* Estudio ligero: pedir si el histórico de estas rechazadas existe antes de julio de 2025 en tienda y antes de mayo de 2026 en Online. Con la historia completa, la ventana de "hoy" de tienda dejaría de ser una mezcla y el índice estacional podría calcularse con toda la demanda. Hoy el índice va sin ellas (`tipotran = est1`), para no inflar la segunda mitad de 2025.
* El total de `p2_tienda_parrilla_online.csv` da 56,1% de contratación de tienda bajo la parrilla porque su take-up va por grupo de
  letra (para que ningún grupo parezca perder aceptación); el correo va con la base del canal (56,2%), que es la del titular, los
  escenarios y los 26,9 M€ de la slide 11. Diferencia 0,1 pp / 45 k€ en tres meses.
* Los 23 contratos de Online aprobados **sin llamada a Equifax** (marzo-mayo de 2025, 22 `ok` + 1 `rv`, 32,6 k€) tienen un 88,6% de mora: una anomalía que conviene enseñar a negocio y preguntar por qué se aprobaron sin score. Son el 0,03% de la producción, pero mueven el riesgo realizado de Online 0,03 pp (4,18 → 4,14, antes de la deriva) y explican por qué el like-for-like pasó de -0,21 a -0,19 pp al excluirlos del tramo 1 (antes del 16-10 el centinela 0 caía en el tramo más bajo).
* La ventana de Online se mantiene en marzo-2025 a febrero-2026 aunque marzo-2026 ya está maduro. Moverla no cambia Online (20,9% / 4,18%); de moverla, en la corrida definitiva. Ojo: Online lleva estudio ligero desde mayo de 2026 (17-23% de su demanda), así que mover la ventana hasta ahí bajaría su TA por el denominador.
