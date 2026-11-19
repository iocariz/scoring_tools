# Borrador de correo — Respuesta a las dudas de Bernard (estudio Apple)

> Generado el 18-11-2026 y rehecho el 19-11 con el extracto de ese día, que trae la marca CESP redefinida
> (tienda con score Equifax en agosto, septiembre y octubre de 2026). Todas las cifras salen de `output/apple_study/b1_*.csv` … `b4_*.csv`, que escribe
> `run_apple_study.py` (funciones en `src/studies/apple/followup.py`). Este correo no tiene verificador
> automático: si se vuelve a correr el estudio, repasar las cifras contra esos CSV antes de enviar.

**Asunto:** RE: Apple — Score Equifax en tienda: dudas

---

Hola Bernard,

Te contesto punto por punto, ya con octubre de tienda incluido (tres meses con score Equifax: agosto a octubre de 2026).

**En corto:**

1. La demanda de tienda está cargada hacia los scores altos y no tiene ningún punto de concentración.
2. La curva no lleva tendencia. En Online el riesgo a igual score va un 15% por encima de la ventana del estudio (30% en las cohortes más recientes), y ya lo he factorizado: con ese 15% el objetivo del 4% deja de cumplirse sin cambios y pide 5 puntos más de corte. En tienda el dato que tengo, hasta julio, todavía no muestra la subida.
3. Las operaciones con CESP son ya solo una cuarta parte de la demanda de tienda y puntúan mejor en Equifax que el resto, pero su riesgo se ha acercado al de las demás y su ingreso declarado no ordena el riesgo. La parrilla no cambia su peso en lo aceptado.
4. Los ingresos altos con score bajo existen pero son pocos, unas 30 solicitudes al mes que hoy se aprueban. En Online, los que hay tienen cuatro veces el riesgo de su tramo de ingresos.

## 1. Distribución de la demanda y la producción de tienda por score

Tienda, agosto-octubre de 2026 (47,8 M€ de demanda; 43,7 M€ con score), frente a Online en su ventana:

| score EFX | tienda: % demanda | tienda: % producción | tienda: contratación hoy | Online: % demanda | Online: % producción |
|---|---:|---:|---:|---:|---:|
| 1-27 | 5,5% | 3,3% | 28% | 17,3% | 1,5% |
| 28-51 | 16,0% | 12,5% | 38% | 21,1% | 20,2% |
| 52-73 | 31,5% | 28,2% | 43% | 28,9% | 33,0% |
| 74-99 | 47,1% | 55,9% | 57% | 32,8% | 45,3% |

- **Tienda puntúa bastante mejor que Online**: score mediano 72-73 frente a 61, y casi la mitad de la demanda está por encima de 74. Solo el 5% cae por debajo de los cortes de la parrilla, frente al 17% en Online.
- **No hay punto de concentración.** La distribución crece de forma suave hasta el tramo más alto; el valor de score más repetido (99) es el 2,8% de la demanda, y no hay acumulación junto a los cortes. Los tres meses son prácticamente iguales (mediana 73 / 72 / 72).
- La producción está algo más concentrada arriba que la demanda porque el score interno de hoy ya aprueba más cuanto mejor es el score Equifax (contratación del 28% por debajo de 27 y del 71% en el tramo 95-99).
- Lo que sí llama la atención: por debajo de 27, tienda contrata hoy el 28% de lo que le piden y Online el 2%. Ese es el swap-out de la parrilla: 1,8% de la demanda de tienda, con un riesgo imputado del 46%.

El detalle por los 20 tramos está en `b1_tienda_distribucion_score.csv`.

## 2. Evolución reciente del riesgo y deterioro del perfil

**Qué hacía la simulación.** La curva de riesgo por tramo de score se estima sobre la cartera de Online originada de marzo-2025 a febrero-2026, y el nivel de tienda se ancla a su morosidad real de esa misma ventana (3,74%). No llevaba ningún factor de tendencia: asumía que cada tramo de score se comporta como en esa ventana. A raíz de tu pregunta lo he medido y lo he incorporado.

**Qué parte del deterioro sí recoge.** El cambio de perfil que el score ve. Los escenarios aplican la curva a la demanda actual, así que si la población empeora en score, cae más demanda bajo el corte. Pero eso no es lo que está pasando: en Online el score mediano de la demanda ha subido de 58 a 65 entre el primer trimestre de 2025 y el tercero de 2026, y la demanda bajo el corte ha bajado del 18% al 11%.

**Qué parte no recoge.** La subida del riesgo a igual score. La he medido comparando el riesgo realizado de cada trimestre con el que le daría la curva de la ventana a esos mismos contratos (índice 1,00 = se comporta como la ventana):

| trimestre de originación | Online H6 | Online H3 | Tienda H6 | Tienda H3 |
|---|---:|---:|---:|---:|
| 2025 T2 | 1,04 | 1,03 | 1,03 | 1,07 |
| 2025 T3 | 1,04 | 0,96 | 1,37 | 1,40 |
| 2025 T4 | 0,94 | 0,93 | 0,91 | 0,90 |
| 2026 T1 | 1,14 | 1,15 | 0,80 | 0,79 |
| 2026 T2 | 1,16 (solo abril) | 1,33 | 0,71 (solo abril) | 0,70 |
| 2026 T3 | — | 1,49 (solo julio) | — | 1,18 (solo julio) |

En tienda, que no tiene score histórico, el esperado va por grupo de cliente (New / A-C / D-F / Inactive).

- **Online: sí, hay que factorizarlo.** A igual score, las cohortes de 2026 llevan un 15% más de riesgo a seis meses que la ventana, y a tres meses el segundo trimestre va un 33% por encima (un año antes, el mismo trimestre estaba en 1,03, así que no es estacionalidad).
- **Tienda: en mi dato no se ve todavía.** Las cohortes de enero a junio de 2026 van por debajo de la ventana (2,5-2,9% frente a 3,74%), de modo que el ancla actual es conservadora. Julio repunta (H3 del 2,60%), pero queda por debajo de julio de 2025 (2,80%). Si la subida que veis es de agosto en adelante, este extracto aún no trae su comportamiento.

**Qué he cambiado en los escenarios.** Dos cosas, las dos en Online:

- Su riesgo se multiplica por 1,15 en todas las filas simuladas (caso central). Tienda no lleva factor.
- Entra en el total Apple con su volumen actual (marzo-septiembre de 2026, un 23% por encima del de su ventana), igual que tienda entra con el de agosto-octubre. Antes se mezclaba el volumen de Online de hace un año con el actual de tienda, y tienda pesaba de más en el riesgo total.

Con eso, el total Apple queda así:

| | aprobación del sistema | producción | riesgo |
|---|---:|---:|---:|
| Hoy, con la demanda actual | 52,4% | 18,4 M€/mes | 4,32% |
| Parrilla de Online en ambos canales | 56,5% | 20,7 M€/mes | 4,20% |
| Objetivo 4%: parrilla +5 puntos | 55,2% | 20,3 M€/mes | 3,97% |

Y la sensibilidad al factor, que es lo que preguntabas:

| riesgo real frente a la curva | riesgo con la parrilla vigente | para cumplir el 4% | producción |
|---|---:|---|---:|
| como la ventana (sin factor) | 3,88% | sin cambios | 20,7 M€/mes |
| **Online +15% (caso central)** | **4,20%** | **parrilla +5 puntos** | **20,3 M€/mes** |
| Online +30% | 4,52% | parrilla +12 puntos | 19,5 M€/mes |
| Online y tienda +15% | 4,47% | parrilla +11 puntos | 19,6 M€/mes |
| Online y tienda +30% | 5,05% | parrilla +21 puntos | 17,8 M€/mes |

Es decir: la conclusión anterior de que el 4% se cumplía sin tocar nada dependía de que el riesgo no hubiera subido. Con el +15% que ya se mide en Online hacen falta 5 puntos más de corte, a un coste pequeño (0,4 M€/mes). El +30% lo dejo como estrés, y tienda sin factor mientras no tengamos el dato; si preferís prudencia, la fila con los dos canales al +15% da la referencia.

## 3. Operaciones con CESP

Las he separado con la marca del extracto, que va por operación: CESP son las solicitudes a las que se aplicó la condición especial, sin verificación de ingresos. El comercio puede tener CESP y el cliente optar por aportar la nómina; esas solicitudes cuentan como no CESP.

| tienda | CESP | no CESP |
|---|---:|---:|
| % de la demanda, mar-25 → feb-26 | 43% | 57% |
| % de la demanda, ago-oct 26 | 26% | 74% |
| aprobación del sistema, ago-oct 26 | 54% | 54% |
| rechazo por score interno, ago-oct 26 | 18% | 19% |
| riesgo H6, mar-25 → feb-26 | 3,18% | 4,15% |
| score EFX mediano, ago-oct 26 | 76 | 71 |
| % bajo el corte EFX, ago-oct 26 | 3,4% | 5,9% |

- **Cada vez son menos.** Las operaciones con CESP eran el 61% de la demanda de tienda a principios de 2024, el 43% en la ventana del estudio y el 20% en octubre de 2026.
- **El score Equifax las ve mejores que al resto**, no peores: mediana de 76 frente a 71, y el 3,4% bajo el corte frente al 5,9%. En el tramo más alto está el 14,0% de su demanda frente al 9,4% de la no CESP (`b3_tienda_cesp_score.csv`). El score interno de hoy las trata igual que a las demás (misma aprobación, mismo rechazo por score).
- **Su riesgo relativo ha empeorado.** Hasta mediados de 2025 tenían en torno a la mitad del riesgo de las no CESP (H6 de 2024: 1,5-3,3% frente a 3,6-5,5%). Desde entonces la diferencia se ha estrechado y se cruza dos veces: 5,47% frente a 5,06% en 2025 T3 y 2,64% frente a 2,47% en abril de 2026 (en 2025 T4 y 2026 T1 siguen por debajo: 2,98% frente a 3,69% y 2,35% frente a 3,28%); a tres meses, 1,99% frente a 1,22% en 2026 T2 y 2,92% frente a 2,42% en julio. En la ventana completa siguen por debajo (3,18% frente a 4,15%), así que el nivel medio todavía no lo refleja. Como su peso cae a la vez, los trimestres no comparan la misma población.
- **Los ingresos declarados no ordenan el riesgo en CESP**, y sí en el resto:

| riesgo H6 de tienda por ingresos declarados (mar-25 → feb-26) | CESP | no CESP |
|---|---:|---:|
| < 1.000 € | 2,17% | 5,07% |
| 1.000-1.499 € | 2,19% | 5,11% |
| 1.500-1.999 € | 2,28% | 4,04% |
| 2.000-2.999 € | 5,67% | 2,40% |
| ≥ 3.000 € | 3,47% | 1,11% |

  En no CESP el riesgo baja con el ingreso, como cabe esperar. Con CESP los ingresos altos son los peores: como el ingreso no se verifica, es la huella de una situación laboral declarada que no es cierta. Encaja con lo que dices, y con que el score no lo vea: puntúan mejor y aun así su riesgo reciente es parecido al del resto.

- **Efecto sobre el estudio: la parrilla no cambia el peso de CESP.** El swap-in es el 17,3% de la demanda CESP y el 16,7% de la no CESP, y las operaciones con CESP son el 26% de lo que se mantiene aprobado, el 26% del swap-in y el 26% de lo aceptado con la parrilla. Lo que sí queda corto es su riesgo: a ese swap-in le imputo un 3,9% con la curva de Online por score, y para una población que el score ve mejor de lo que luego se comporta esa cifra es optimista.

**Cómo integrarlo.** Dos opciones, que puedo simular:

1. Aplicar la parrilla a todas las operaciones con un nivel de riesgo propio para CESP, calibrado con su morosidad reciente en vez de con la de la ventana. Es la que prefiero, porque la parrilla no altera la mezcla.
2. Aplicarla solo a no CESP y mantener para CESP la regla actual (score interno más el check de Seguridad Social) hasta tener su comportamiento con score.

## 4. Ingresos altos con score Equifax bajo

Tienda, agosto-octubre de 2026, por ingresos declarados del primer titular (sin los clientes ≥G, que se rechazan por letra):

| ingresos | demanda | score mediano | score ≤ 27 | bajo el corte EFX | de ellos, aprobados hoy (el caso del vendedor) |
|---|---:|---:|---:|---:|---:|
| < 1.000 € | 4,3 M€ | 72 | 5,2% | 5,1% | 0,4% (11 solicitudes) |
| 1.000-1.499 € | 16,5 M€ | 70 | 6,7% | 6,4% | 2,3% (246) |
| 1.500-1.999 € | 12,1 M€ | 71 | 6,1% | 5,9% | 2,4% (183) |
| 2.000-2.999 € | 7,5 M€ | 81 | 3,3% | 3,1% | 1,5% (68) |
| ≥ 3.000 € | 3,2 M€ | 80 | 2,3% | 2,2% | 1,4% (26) |

- **El score ya premia el ingreso**: mediana de 80-81 por encima de 2.000 € frente a 70-72 por debajo, y la mitad de rechazo por score (2-3% frente a 6%).
- **El caso que temes existe pero es pequeño.** Con ingresos de 2.000 € o más, la parrilla rechazaría por score el 2-3% de la demanda; y los que hoy se aprueban y pasarían a rechazarse son 94 solicitudes en tres meses, unas 30 al mes (el 0,3% de la demanda de tienda). Separando por CESP el porcentaje es parecido: por encima de 3.000 €, 2,0% bajo el corte en CESP y 2,3% en no CESP (`b4_ingresos_score.csv`).
- **Qué riesgo tienen.** Solo se puede medir en Online, que tiene score, ingresos y comportamiento:

| riesgo H6 Online (mar-25 → feb-26) | score 1-27 | 28-50 | 51-75 | 76-99 | total del tramo |
|---|---:|---:|---:|---:|---:|
| < 1.000 € | 31,7% | 12,2% | 6,2% | 3,1% | 6,7% |
| 1.000-1.499 € | 19,1% | 9,2% | 4,4% | 3,2% | 5,3% |
| 1.500-1.999 € | 14,1% | 7,4% | 3,7% | 2,0% | 3,9% |
| 2.000-2.999 € | 8,1% | 4,0% | 2,5% | 1,2% | 2,1% |
| ≥ 3.000 € | 0,0% (38 contratos) | 4,7% | 1,9% | 0,3% | 1,4% |

  El ingreso alto atenúa el riesgo, pero el score sigue ordenando dentro de cada tramo de ingresos. Un cliente de 2.000-2.999 € con score por debajo de 27 tiene un 8,1% de riesgo: la mitad que un ingreso medio con ese score, pero cuatro veces la media de su tramo y el doble del objetivo. Por encima de 3.000 € no hay muestra (38 contratos).

**Mi lectura.** Una excepción por ingreso alto costaría poco volumen de riesgo (unas 30 solicitudes al mes), así que es defendible por la razón comercial que das: un corte algo más bajo a partir de 2.000 o 3.000 €, no una exención. Pero con una condición que sale del punto 3: en CESP el ingreso declarado no es fiable (los CESP de 2.000 € o más tienen un 5,7% y un 3,5% de riesgo), así que la excepción debería limitarse a las operaciones sin CESP, es decir, a quien aporta la nómina.

## Qué propongo

1. Presentar los escenarios con el factor de deriva de Online ya incorporado (+15% central, +30% como estrés): dime si el 15% te parece el nivel adecuado.
2. Simular la parrilla para todos con un nivel de riesgo propio para CESP, y compararla con dejar CESP en la regla actual.
3. Simular un corte reducido para ingresos altos no CESP y ver cuánta demanda recupera y a qué riesgo.

Dime si te encaja y lo preparo.

Un saludo,
Iñigo

---

## Notas para Iñigo (no enviar)

* `cespa = s` = **la CESP (condición especial) se aplicó a la operación**: sin verificación de ingresos. El comercio puede tener CESP y el cliente optar por aportar nómina, y entonces es `n` (Iñigo, 19-11). El extracto del 17-11 traía la marca por punto de venta (192.903 filas `s`); el del 19-11 trae 165.032 y cambia el punto 3 entero: antes "Equifax no los distingue" y "la parrilla deja entrar más CESP" (51% del swap-in), ahora puntúan mejor y la parrilla es neutra (26%). También desaparece la duda de la aprobación más baja en CESP (54% en los dos grupos en ago-oct). El resto del correo no cambia: el extracto nuevo solo quita 21 filas duplicadas.
* Quien va por CESP se autoselecciona (no aporta nómina) y su peso cae del 61% al 20% de la demanda: la convergencia del riesgo puede ser en parte cambio de composición. No se puede medir a igual score porque tienda no tiene score histórico.
* Hay 31 filas de Online con `cespa = s` en el extracto nuevo; el análisis solo usa tienda.
* Bernard dice que el riesgo de tienda sube; en el extracto (H6 hasta abril-2026, H3 hasta julio-2026) las cohortes de 2026 de tienda van por debajo de la ventana. Puede estar viendo indicadores más tempranos o cohortes posteriores a julio. Conviene preguntarle qué serie mira antes de discutir el factor de tienda.
* El índice de deriva de 2024 sale en 0,6 en Online (no está en el correo ni en el CSV, que empieza en 2025): el score v3 entra en producción en junio de 2025 y `risk_score_rf` anterior es retro-puntuado, así que no lo compararía.
* Desde el 18-11 (tarde) el factor 1,15 de Online y su volumen actual SON el caso central: `p3_escenarios_objetivo.csv`, el deck y el correo de Jenny ya los llevan (`--online-risk-factor`, `--online-window-volume` para volver atrás). La tabla de sensibilidad (`drift_sensitivity`) da factores absolutos frente a la curva, descontando el 1,15 ya aplicado.
* Revisión de la metodología (18-11): además de la deriva y el volumen, la bajada del riesgo de tienda con la parrilla (3,74 → 3,52%) depende de los tramos 1-5 de la curva (582 contratos de Online, 467 de marzo-mayo de 2025, antes del score v3); con su riesgo a la mitad, tienda quedaría en 3,77%. Por eso el mensaje es "riesgo parecido". No está en este correo: decidir si se le cuenta a Bernard.
* La tabla de riesgo por ingresos de Online usa la ventana (pocos contratos con score ≤ 27: 106 en 2.000-2.999 € y 38 por encima de 3.000 €). En toda la historia madura de Online hay 391 y 138, con 10,5% y 6,8% de riesgo: mismo mensaje, más muestra. No está en los CSV.
* El titular del estudio con octubre y la base nueva: +3,4 pp de contratación y +2,3 M€/mes, riesgo 4,20% frente a 4,32% hoy; el 4% pide +5 puntos. El correo de Jenny ya está actualizado y cuadra con `scripts/check_apple_email.py`.
