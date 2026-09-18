"""Lógica de decisión del presupuesto de latencia — sin torch, sin audio, puro cálculo.

Este módulo contiene SOLO la lógica que decide si una configuración cabe en el
presupuesto de latencia. Al no importar torch ni audio, se puede probar en
milisegundos y sin GPU. Eso es lo que hace testeable al código.
"""


def cabe_en_presupuesto(etapas: dict[str, float], techo_ms: float) -> bool:
    """True si la suma de latencias de todas las etapas no supera el techo.

    El techo es INCLUSIVO: 950 ms exactos contra techo de 950 ms = cabe.
    """
    return sum(etapas.values()) <= techo_ms


def etapa_mas_lenta(etapas: dict[str, float]) -> str:
    """Nombre de la etapa con mayor latencia.

    Excepciones:
        ValueError: si el dict está vacío.
    """
    if not etapas:
        raise ValueError("etapas vacío: no hay etapa más lenta")
    return max(etapas, key=etapas.__getitem__)


def margen_restante(etapas: dict[str, float], techo_ms: float) -> float:
    """Milisegundos que quedan por debajo del techo (negativo = se pasa)."""
    return techo_ms - sum(etapas.values())


def degradar_configuracion(
    etapas: dict[str, float],
    techo_ms: float,
    orden_degradacion: list[str],
) -> dict[str, float]:
    """Reduce calidad de etapas (en orden) hasta caber en el techo.

    No modifica el dict original. Devuelve una copia con valores reducidos.
    La reducción es un placeholder: en el Paso 3 real esto cambiará
    tamaño de modelo, beam size, etc.

    Itera en ciclos sobre orden_degradacion hasta caber o hasta que no quede
    margen de mejora. Si el techo es imposible, devuelve el mejor esfuerzo
    (caller debe verificar con cabe_en_presupuesto).

    Excepciones:
        ValueError: si algún nombre en orden_degradacion no existe en etapas.
    """
    for nombre in orden_degradacion:
        if nombre not in etapas:
            raise ValueError(f"etapa desconocida en orden_degradacion: {nombre!r}")

    resultado = etapas.copy()
    # Itera en ciclos completos para converger; cada etapa puede degradarse varias veces
    # tope arbitrario, no bug si cambia a 11
    max_ciclos = 10  # pragma: no mutate
    for _ in range(max_ciclos):
        if cabe_en_presupuesto(resultado, techo_ms):
            break
        for nombre in orden_degradacion:
            if cabe_en_presupuesto(resultado, techo_ms):
                break
            # reduce 20% la latencia simulando modelo más chico
            resultado[nombre] *= 0.8
    return resultado
