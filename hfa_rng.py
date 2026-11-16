"""Eigener Zufallsgenerator (SplitMix64) für die Zufallssysteme in Experiment 4: gleiche Zufalls-Seeds geben auf jeder Python-Version und Plattform dieselben Folgen
(bitgleich, auch bei den Normalzahlen).

Herkunft: Kopie der Vorlage `tools/templates/rng_splitmix64_exact.py` (Website-Repo), nur mit Präfix und gekürzt (`random`, `normal`, `uniform`; ohne `expovariate`, `choice`,
`shuffle`). Python-`hash()` und `random` sind für feste Zufalls-Seeds nicht geeignet (pro Prozess gesalzen bzw. nicht versionsstabil).

Plattformfest: `math.log` und `math.cos` kommen aus der C-Bibliothek des Systems (Windows: MSVC, Linux: glibc) und weichen in einzelnen Fällen in der letzten Binärstelle
voneinander ab (gemessen in der Vorlage: 139 von 5.000 Normalzahlen). Darum rechnen `exact_log` und `exact_cos` mit `decimal` (reine Ganzzahlarithmetik) und runden einmal auf
double. `math.sqrt` ist nach IEEE 754 exakt gerundet und darf bleiben."""

import decimal
import math

MASK = (1 << 64) - 1
_CTX = decimal.Context(prec=45)


def exact_log(x: float) -> float:
    """Natürlicher Logarithmus, für x > 0, auf jeder Plattform bitgleich (praktisch auf das nächste double gerundet)."""
    return float(_CTX.ln(decimal.Decimal(x)))


def exact_cos(x: float) -> float:
    """Kosinus für |x| bis etwa 10, auf jeder Plattform bitgleich (Taylorreihe in 45 Stellen, dann einmal auf double gerundet)."""
    d = decimal.Decimal(x)
    x2 = _CTX.multiply(d, d)
    term = decimal.Decimal(1)
    total = decimal.Decimal(1)
    k = 0
    limit = decimal.Decimal(10) ** -48
    while True:
        k += 2
        term = _CTX.divide(_CTX.multiply(term, x2), k * (k - 1))
        total = _CTX.subtract(total, term) if (k // 2) % 2 == 1 else _CTX.add(total, term)
        if term < limit:
            break
    return float(total)


class SplitMix64:
    def __init__(self, seed: int):
        self.state = seed & MASK

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def random(self) -> float:
        """Gleichverteilt in [0, 1) mit 53 Bit."""
        return (self.next_u64() >> 11) / float(1 << 53)

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.random()

    def normal(self) -> float:
        """Standardnormalverteilt (Box-Muller)."""
        u1 = max(self.random(), 1e-12)
        u2 = self.random()
        return math.sqrt(-2.0 * exact_log(u1)) * exact_cos(2.0 * math.pi * u2)
