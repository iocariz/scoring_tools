# Petición de datos — Extensión del score Equifax a Apple Stores

| | |
|---|---|
| **Solicitante** | Iñigo Lopez de Ocariz — Riesgos |
| **Fecha** | 30-09-2026 |
| **Destinatario** | Equipo de Datos / Equifax |
| **Objetivo de negocio** | Evaluar TA y riesgo de aplicar el score Equifax (V3) de Apple Online a Apple Stores, y construir escenarios de riesgo global Apple al 2,5% / 3% / 3,5% / 4%. El riesgo de tienda será **estimado** a partir de Apple Online (ver §2.4) |
| **Perímetro** | Apple Propio (no APRS), canales E-Commerce y Off-Line |

---

## 1. Por qué se pide

La extracción actual (`demanda_apple.sas7bdat`, 796.439 filas) permite resolver el análisis de Apple Online, pero **no** el de Apple Stores, por dos motivos:

1. **Las solicitudes de tienda no traen score externo.** En las 9.080 filas Off-Line, `ext_business_name` viene vacío y `rf_business_name` indica `A-Score DISTRIB Retail Traditional CTLM`, mientras que en las 787.359 filas E-Commerce indica `Equifax Risk Score V3 - Retail E-COMM CTLM`. Está pendiente de confirmar si es una limitación del etiquetado del feed Off-Line o si realmente no hay score Equifax en esas solicitudes (ver §5, punto 1).

2. **Los dos canales no comparten ningún mes.** Online llega hasta feb-2026 y tienda solo tiene jul y ago de 2026. No hay un solo mes con ambos canales, así que el "riesgo total Apple" hoy solo puede calcularse normalizando a run-rate mensual, comparando una media de 21 meses de Online contra jul/ago de tienda — que además es temporada alta de Apple.

3. **Las operaciones de tienda no tienen comportamiento observable.** Los 4.550 contratos de jul/ago-2026 tienen `todu_amt_pile_H6` y `todu_amt_pile_H3` a cero: entre 0 y 2 meses de antigüedad. El riesgo de tienda hoy solo puede imputarse desde la curva de Online, no medirse.

---

## 2. Qué se pide

### Bloque A — Refresco de Apple Online hasta ago-2026 · PRIORIDAD ALTA

Mismo perímetro, campos y formato que la extracción actual, extendiendo la ventana hasta **2026-08-31**.

**Desbloquea:** que ambos canales compartan periodo, y con ello un riesgo y un mix de producción Apple Total homogéneos, sin normalizaciones ni supuestos de estacionalidad.

### Bloque B — Apple Stores con score Equifax V3, jul y ago 2026 · CONDICIONADO a §5.1

Solo si la confirmación de §5.1 indica que el `risk_score_rf` entregado para tienda **no** es el score Equifax. En ese caso: las mismas 9.080 solicitudes ya entregadas, añadiendo el **valor del Equifax Risk Score V3 a fecha de solicitud**.

**Desbloquea:** la TA de tienda bajo la parrilla de Online (punto 2 de la petición de negocio).
**No desbloquea:** el riesgo de tienda — esas cohortes no tendrán H6 hasta ~feb-2027.

Si el score ya es el correcto, este bloque decae y no hay que pedir nada.

### Bloque C — Histórico de comportamiento de Apple Stores · PRIORIDAD ALTA

Contratos de Apple Stores de un periodo **ya maduro** (orientativamente 2024 y 2025), con sus resultados de morosidad. **No hace falta ningún score**, ni interno ni externo: solo comportamiento y exposición.

Campos mínimos: `authorization_id`, `mis_Date`, `oa_amt_h0`, `todu_30ever_H6`, `todu_amt_pile_H6` y, si están disponibles, sus equivalentes H3 y los campos `h_*`.

**Además, y por separado: un agregado mensual de demanda de tienda** para el mismo periodo — una tabla `mes → nº de solicitudes, demanda €`. No necesita detalle por operación, vale un agregado. Sirve para estimar la **estacionalidad propia de tienda** (ver A.6): hoy se le está aplicando la de Online, y el lanzamiento de iPhone es un fenómeno de tienda física, así que tienda podría ser más estacional. Como el único mes válido de tienda es agosto —el más bajo del año— ese supuesto decide si tienda pesa el 31% de Apple o bastante más.

**Para qué.** Descartada la re-puntuación (ver §2.4), el riesgo de tienda tiene que **estimarse** a partir de la curva de Apple Online. Ese enfoque tiene dos componentes distintos:

- la **forma** de la curva — cuánto separa el score EFX a buenos de malos — que se toma de Online y es el supuesto inevitable;
- el **nivel** — la morosidad media de la cartera — que hoy también se estaría tomando de Online, y no hay ninguna razón para que tienda tenga el mismo nivel que e-commerce.

Con este bloque el nivel deja de suponerse: se reescala la curva de Online para que reproduzca por construcción la morosidad que tienda realmente tiene con su política actual. Así el único supuesto que queda es el poder discriminante del score, no la altura de la curva.

**Cuánto importa.** Con la curva de Online sin anclar, el riesgo estimado de tienda es 2,67% y el corte necesario para un objetivo global del 2,5% es el tramo ≥13. Si el nivel real de tienda fuese 1,5 veces ese valor, el corte pasa a ≥15 — dos tramos más duros y una caída de TA muy relevante. Es la incertidumbre que más mueve los escenarios y la más fácil de cerrar, porque el dato es interno.

---

## 2.4 Vía descartada: re-puntuación a V3 del histórico de tienda

Se valoró pedir a Equifax la re-puntuación con V3 de las solicitudes de tienda del periodo en que el score estuvo activado. Habría dado riesgo **medido** en tienda, ya que esas cohortes están maduras, y existía precedente interno: en Apple Online las ~450.000 solicitudes de ene-2024 a may-2025 se decidieron con la versión antigua del score y en la extracción vienen con el valor V3 (`rf_business_name` = `Equifax Risk Score V3` en el 100% de las filas, sin salto en la distribución en la migración de jun-2025).

**Confirmado que no es viable.** El histórico de tienda corresponde a la versión antigua del score y no puede re-puntuarse.

En consecuencia, el riesgo de Apple Stores de este estudio es **estimado, no medido**, y así debe figurar en cualquier presentación de resultados. La medición real no estará disponible hasta ~feb-2027 (H6 de las cohortes de ago-2026); el H3 llegaría hacia nov-2026.

---

## 2.5 Formato de entrega: dos ficheros

| Fichero | Contenido | Bloques | Población | Periodo |
|---|---|---|---|---|
| **1 — Demanda Apple** | Todas las solicitudes, ambos canales, con score EFX y `segment_cutoff_1` informado | A + B | Solicitudes (contratadas, denegadas y canceladas) | ene-2024 → ago-2026 |
| **2 — Comportamiento tienda** | Contratos de Apple Stores con morosidad realizada, sin score | C | Solo contratos | periodo maduro (orient. 2024-2025) |

**El fichero 1 sustituye a la extracción actual**, no se añade a ella: así no hay que reconciliar solapes.

**El fichero 2 debe ir separado.** Es población de solo contratos y sin score: si sus filas se mezclasen con las del fichero 1 contaminarían el aprendizaje de tramos y falsearían cualquier tasa de aceptación, porque un fichero de solo contratados tiene una TA aparente del 100%. Separados, el riesgo de mezcla desaparece.

---

## 3. Requisitos transversales

1. **Nivel solicitud, no contrato.** Imprescindible: sin las denegadas no se puede calcular la tasa de aceptación ni aplicar reject inference.
2. **Canal declarado** en `segment_cutoff_1` (`E-Commerce` / `Off-Line`) en todas las filas de todos los bloques. Hoy la separación entre canales depende de que sus ventanas de fechas no se solapen; en cuanto Online se refresque a ago-2026 esa separación desaparece y debe poder hacerse por canal.
3. **Mismos nombres de campo y formato** que la extracción actual, para poder integrar sin reprocesar.
4. **Score a fecha de solicitud**, no a fecha de extracción.

---

## 4. Campos solicitados

| Campo | Para qué |
|---|---|
| `authorization_id`, `account_id` | Clave de cruce |
| `mis_Date`, `Entry_date` | Cohorte y ventana de maduración |
| `segment_cutoff_1` | **Canal (E-Commerce / Off-Line)** — ver §3.2 |
| `segment_cut_off`, `SCRV_customer_init`, `SCRV_customer_Group_init` | Segmentación (inactive / new / known_ab / known_cd / known_ef / known_g) |
| `risk_score_rf` | **Score Equifax V3 a fecha de solicitud** — campo central de la petición |
| `ext_business_name`, `rf_business_name` | Trazabilidad de qué versión de score se usó y cuál se entrega |
| `status_name`, `SE_Decision_id`, `reject_reason` | TA, tasa de rechazo de sistema, reject inference |
| `oa_amt`, `oa_amt_h0` | Demanda y producción |
| `todu_30ever_H3`, `todu_amt_pile_H3` | Riesgo a 3 meses |
| `todu_30ever_H6`, `todu_amt_pile_H6` | **Riesgo a 6 meses — métrica objetivo (b2)** |
| `h_num_H3`, `h_den_H3`, `h_num_H6`, `h_den_H6` | Indicador de riesgo armonizado (HRI) |
| `early_bad` | Métrica de contraste |
| `product_type_1`, `product_type_2`, `product_type_3` | Tipo de producto |
| `CHAINE`, `vendedor_cadena_top_name`, `ext_business_name` | Perímetro Apple Propio vs APRS |
| `fuera_norma`, `fraud_flag`, `nature_holder` | Filtros estándar del pipeline |
| `income_T1_m`, `income_T1T2_m` | Comparación de poblaciones entre canales |
| `acct_booked_H0` | Marca de contratación |

---

## 5. Puntos a confirmar

1. **¿`risk_score_rf` en las filas Off-Line es el score Equifax o el A-Score interno?** En esas 9.080 filas coincide al 100% con `SCRPLUST1` y `rf_business_name` apunta al A-Score DISTRIB. Es la confirmación más urgente: si ya fuese Equifax, el Bloque B no haría falta.
2. **¿`CHAINE` = 9108994 corresponde a Apple Propio y los otros cinco valores a APRS?** Necesario para aplicar el perímetro "Apple Propio, no APRS" que pide negocio. Hoy las filas Off-Line son todas de esa cadena.
3. ~~**¿El segmento `known_g` entra en "Apple Total"?**~~ · **RESUELTO** — ver A.3.4: entra como demanda, se rechaza siempre.
4. **¿Se activó el score Equifax en tienda en agosto-2026?** El extracto trae también julio, pero con el **9% del volumen** que le correspondería por estacionalidad (ratio jul/ago observado 0,11 frente a 1,26 esperado). Se está tratando como rollout parcial y excluyendo del cálculo; conviene confirmarlo y, si hubo días concretos de activación, indicarlos.
5. **¿Qué estacionalidad tiene la demanda de tienda?** Hoy se le aplica la de Online por falta de histórico (ver A.6). El agregado mensual del Bloque C lo resuelve.

---

## 6. Validaciones que se harán al recibir los datos

- Cobertura del score por canal y mes = 100%, rango 0-99.
- `ext_business_name` / `rf_business_name` informados en todas las filas, incluidas las Off-Line.
- Presencia de solicitudes denegadas con `reject_reason` informado.
- Ningún mes con un solo canal en el periodo en que ambos deberían existir.
- H6 realizado (`todu_amt_pile_H6` > 0) en las cohortes con 6 o más meses de antigüedad.
- Continuidad de volúmenes contra la extracción actual en el periodo solapado.

---

## 7. Impacto en plazos

| Entregable | Depende de | Plazo desde recepción |
|---|---|---|
| TA y riesgo real de Apple Online | Nada — ya disponible | 1-2 días |
| TA de Apple Stores con la parrilla de Online | Bloque B | 1 día |
| Escenarios con el nivel de tienda anclado a su morosidad real | **Bloque C** | 2-3 días |
| Total Apple sin normalizar por estacionalidad | Bloque A | incluido en lo anterior |
| Escenarios con nivel de tienda tomado de Online (sin anclar) | Nada | 2-3 días |

El riesgo de tienda será estimado en todos los escenarios (ver §2.4). El Bloque C no lo convierte en medido, pero elimina el supuesto sobre el nivel, que es el que más mueve los cortes.

---

## Anexo — Parrilla actual de Apple Online (input de política, no de extracción)

Este anexo **no es una petición al equipo de datos**: es el input de negocio necesario para responder al punto 2 de la petición ("TA y riesgo de Apple Stores si aplicamos la parrilla existente de Apple Online"). Sin él, el cálculo se hace con los cortes que el optimizador propone, que no son la política viva.

### A.1 Cortes vigentes a cumplimentar

| `segment_cut_off` | Corte EFX actual (0-99) | Operador | ¿Vigente desde? |
|---|---|---|---|
| inactive | | ≥ / > | |
| new | | | |
| known_ab | | | |
| known_cd | | | |
| known_ef | | | |
| known_g | | | |

En **escala de score 0-99**, no en número de tramo: los tramos son una construcción del modelo y no tienen por qué coincidir con los cortes reales. Si algún segmento no tiene corte de score, indicarlo explícitamente.

### A.2 Por qué el corte de score no basta para calcular el impacto

Reparto de la demanda en € observado en la ventana de análisis:

| Sobre la demanda € | Apple Online | Apple Stores |
|---|---:|---:|
| Decisión OK del sistema | 44,9% | 58,1% |
| Finalmente contratado | 20,3% | 47,6% |
| Take-up sobre lo aprobado | 45,2% | **81,9%** |
| Rechazo por **09-SCORE** | **8,3%** | **21,7%** |
| 04-CUSTOMER PROFILE | 15,2% | 10,5% |
| 01-CUSTOMER RISK | 8,5% | **0,0%** |
| 02-BUREAUX | 7,8% | **0,0%** |
| 03-BUDGET | 5,7% | 4,1% |
| 05-FRAUD | 1,8% | **0,0%** |

Tres consecuencias:

1. **El score explica solo 8,3 de los ~55 puntos de rechazo de Online.** El grueso son perfil de cliente, riesgo de cliente, bureaux y presupuesto. Cambiar la parrilla mueve una palanca, no la política.
2. **Tienda ya rechaza más por score que Online** (21,7% frente a 8,3%), con su score interno. El cambio es una **sustitución** de una regla por otra, y el impacto es el diferencial entre ambas, no el corte en bruto.
3. **Online aplica ~18 puntos de reglas que tienda hoy no tiene** (customer risk, bureaux y fraude están a 0,0% en tienda).

### A.3 Cuestiones a cerrar con negocio

1. **¿La parrilla tiene una sola dimensión?** Si cruza con importe, ingresos o tipo de producto, hace falta la matriz completa, no una lista de cortes.
2. ~~**¿Se importa a tienda solo el corte de score, o también las reglas de customer risk / bureaux / fraude?**~~ · **RESUELTO (30-09-2026): solo el corte de score.** El resto de reglas de tienda se mantienen como están. El impacto se modela por tanto como swap de la regla de score (ver A.4), y los ~18 puntos de demanda que Online filtra por customer risk / bureaux / fraude **no** se trasladan a tienda.
3. ~~**¿Los cortes han cambiado durante la ventana de observación?**~~ · **RESUELTO (30-09-2026): ha habido cambios de parrilla, pero solo interesa la última.** Se aplicará la parrilla vigente. La parrilla tiene **una sola dimensión** (un corte de EFX por `segment_cut_off`), sin cruce con importe, ingresos ni producto.
4. ~~**¿`known_g` entra en el perímetro?**~~ · **RESUELTO (30-09-2026): `known_g` se rechaza siempre.** Entra en el perímetro como demanda —cuenta en el denominador de la TA— pero nunca aporta producción ni exposición, así que no afecta al riesgo. El dato lo confirma: 0 contratos sobre 23,9 M€ de demanda Online y 0,05 M€ en tienda. Incluirlo baja la TA de Online del 21,6% al **20,7%**.

5. **Ventana de estimación** · **DECIDIDO (30-09-2026): 12 meses.** La curva de riesgo y el riesgo actual de Online se estiman sobre los últimos 12 meses de originación, no sobre los 21 disponibles. Motivo: la curva se ha desplazado al alza entre las dos mitades del periodo (riesgo global 3,86% en 21 meses frente a 4,21% en 12), y la decisión es prospectiva. Coste: cada objetivo de riesgo exige unos dos tramos más de dureza y ~15 puntos de TA frente a la ventana larga. **Debe figurar explícitamente en la presentación**, no heredarse de un default.

   Nota: refrescar la extracción hasta ago-2026 **no alarga la curva**. La limita la madurez, no la fecha de extracción: con datos a ago-2026 las cohortes con H6 realizado siguen acabando en feb-2026.

### A.4 Cómo se usará

El impacto se calculará como **swap de la regla de score**, manteniendo el resto de reglas de tienda tal como se observan: una solicitud se rechaza si no supera el corte EFX **o** si venía rechazada por un motivo distinto de `09-SCORE`. Es el mismo criterio que aplica el pipeline en su métrica de *System Rejection Rate*, de modo que los números del estudio y los del motor de decisión son comparables.

### A.5 Limitación a declarar en cualquier presentación

El riesgo de tienda, tanto el de la política actual como el de los escenarios, se calcula imputando la curva de riesgo de Apple Online por tramo EFX. Eso implica que la comparación entre el score interno y EFX se hace **con la vara de medir de EFX**: el resultado es condicional ("si EFX ordena en tienda como ordena en Online, seleccionaría mejor"), no una comparación empírica entre ambos scores.

Una comparación real exige resultados observados en tienda bajo score EFX, que no estarán disponibles hasta ~feb-2027 (H3 hacia nov-2026). No debe presentarse como evidencia de que EFX supera al score interno en tienda.

### A.6 Anualización y estacionalidad · DECIDIDO (30-09-2026)

Apple es extremadamente estacional: índice de demanda de Online entre **0,53 en agosto** (el mes más bajo del año) y **1,94 en septiembre**, un factor 3,6x por el lanzamiento de iPhone. Como el único mes válido de tienda es agosto, comparar su volumen crudo contra la media anual de Online la infravaloraba a un tercio de su tamaño real.

**Mecanismo adoptado:** el divisor del run-rate deja de ser el número de meses de calendario y pasa a ser **meses efectivos** = suma del índice estacional de los meses cubiertos, prorrateando los meses parciales por días. Una ventana natural completa suma 12,00 y no altera nada (Online no se toca); un único agosto suma 0,53 y corrige el run-rate.

**Efecto en los resultados:** el peso de tienda en la demanda Apple pasa del 12% al **31%**, y como tienda es menos arriesgada y transforma mucho más, la producción de cada escenario casi se duplica y la TA sube entre 3 y 10 puntos.

**Supuesto pendiente:** el índice se estima sobre **Online**. Si tienda es más estacional —plausible, porque las colas de lanzamiento son un fenómeno de tienda física— su peso real sería aún mayor que el 31%. El agregado mensual solicitado en el Bloque C lo cierra.

**Julio-2026 queda excluido** por rollout parcial (9% del volumen esperado). El script lo detecta comparando cada mes contra su perfil estacional, en vez de depender de que alguien recuerde excluirlo.
