# Die Falle des endlichen Horizonts – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-horizont-falle-demo.streamlit.app/)**

Diese Demo gehört zum Themenfeld „Regelung (von LQR bis MPC)“ auf der Website „Sebastian Hanisch – Operations Research und Machine Learning“. Sie baut auf [`lqr-riccati-demo`](https://github.com/sebastian-hanisch/lqr-riccati-demo) auf, die den Gewinn K des unendlichen Horizonts liefert (dieser Gewinn sagt nicht, was ein Plan über endlich viele Schritte daraus macht, wenn man ihn in jedem Schritt neu rechnet). Vehikel sind Kalmans Gegenbeispiel zum endlichen Horizont, der Wagen (Doppelintegrator, Abtastzeit 1), ein Skalar von Hand und als Zusatz das linearisierte Vier-Tank-System als **eigene modellbasierte Entsprechung** zu einem Beispiel von Berberich u. a. Die Demo misst in einem linearen Modell **ohne Beschränkungen und ohne Störung**, wann das Gesetz **u = −K_N x** eines Plans über N Schritte stabil ist. Alle Zahlen sind eigene Nachrechnung in diesem Modell, wo eine Quelle gelesen ist, steht sie bei den Formeln und unter „Quellenstand“.

**Namensfalle:** „Kalmans Gegenbeispiel“ (Kalman 1960, bei Rawlings/Mayne/Diehl in 1.3.4 zitiert: Optimalität ohne Stabilität) hat inhaltlich nichts mit dem Kalman-**Filter** zu tun (gleicher Namensgeber, andere Aufgabe); die Demo zum Filter ist [`kalman-filter-demo`](https://github.com/sebastian-hanisch/kalman-filter-demo) (Beobachter ohne Stellgröße).

## Warum dieses Problem

Wer in jedem Schritt über einen endlichen Horizont N neu optimiert und nur den ersten Stellwert anwendet (MPC ohne Beschränkungen), hält den Plan für optimal. Optimal über N Schritte heißt aber nicht, dass der **wiederholte** Plan stabil ist. Die Kernfragen der Demo: Wie lang muss der Horizont sein, damit das Gesetz stabil ist? Was ändern die Terminalkosten? Was kostet ein Horizont, der stabil, aber kurz ist?

## Modell

Strecke x⁺ = A x + B u, Kosten über N Schritte Σ (x′Qx + u′Ru) + x_N′ P_f x_N, **Gesetz u = −K x** (K positiv bei Rückführung), **kein Faktor ½**. Das Gesetz u(x), ρ und die Kostenmatrix hängen von beiden Konventionen nicht ab; nur das Vorzeichen von K (u = −K x gegen u = K x) und der Wert der Kosten (Faktor ½) ändern sich. Andere Quellen schreiben u = K x mit negativem K (Rawlings/Mayne/Diehl, Borrelli/Bemporad/Morari) oder haben einen Faktor ½. Vorzeichen und Faktor ½ sind dieselben wie in [`lqr-riccati-demo`](https://github.com/sebastian-hanisch/lqr-riccati-demo); die Nummerierung unterscheidet sich: Hier ist K_N der Gewinn der ersten Stufe eines Horizonts der Länge N, dort K_k der Gewinn der Stufe k bei festem Endzeitpunkt (der Gewinn der letzten Stufe heißt dort K_{N−1}, hier K_1). K_N ist der Gewinn der ersten Stufe des Horizonts N, gerechnet rückwärts (Riccati-Rekursion, Π₀ = P_f); die **Pole** sind die Eigenwerte von A − B K_N, **ρ** ist der Betrag des größten Pols (dieselbe Größe wie das ρ in den Demos zur Rückkopplung, siehe [`lager-lieferzeit-demo`](https://github.com/sebastian-hanisch/lager-lieferzeit-demo) und [`meldeverzug-demo`](https://github.com/sebastian-hanisch/meldeverzug-demo)). **Stabil** heißt ρ < 1 (Band 10⁻⁹), **grenzstabil** ρ = 1, **instabil** ρ > 1. Formeln stehen im 📐-Expander der App.

Strecken: **Kalmans Gegenbeispiel** (Rawlings/Mayne/Diehl, 1.3.4, S. 21 bis 22: A = [[4/3, −2/3], [1, 0]], B = (1, 0)′, Ausgang C = (−c, 1) mit c = 2/3, Q = C′C + ε·I, ε = 10⁻³, R = 10⁻³, P_f = Q); die Nullstelle der Strecke liegt bei 1/c, die Familie über c ist eigene Wahl. **Wagen**: A = [[1, 1], [0, 1]], B = (0, 1)′, Q = I. **Vier-Tank**: Matrizen A, B, C aus der Textextraktion von Berberich u. a. (arXiv 1906.04679, Abschnitt V, Seite 14), Q = 3·C′C, R·I; die Zeilenfolge von B ist nicht gegen das Seitenbild geprüft.

## Methodik

- **Kern** (`hfa_dynamics.py`, numpy und scipy): Riccati-Rekursion, ρ aus den Eigenwerten, DARE (SciPy), Kosten der unendlichen Zukunft über die Lyapunov-Gleichung (schlechtester Anfangszustand = größter verallgemeinerter Eigenwert gegen P_∞), harte Terminalbedingung x(N) = 0 über das KKT-System, zeitvarianter Ablauf K_N, …, K_1.
- **Exakt** (`hfa_exact.py`, `fractions`): dieselbe Rekursion in Brüchen für die Handbeispiele und die Poltabelle am Wagen; was „exakt“ heißt, ist in Brüchen gerechnet.
- **Zweite Rechenwege** (nur in den Tests): Batch-Lösung (Normalgleichungen) in Gleitkomma und in Brüchen, numerische Minimierung über alle Stellfolgen mit analytischem Gradienten, ρ aus Spur und Determinante, DARE-Fixpunkt, Simulation.
- **Zufall** (nur Experiment 4): ganzzahliger SplitMix64 (`hfa_rng.py`, `normal()` über `decimal`, plattformfest). Die Zählungen über Zufallssysteme sind **vorgerechnet** (`tools/make_frozen.py` erzeugt `data/hfa_frozen.json`, rund 3 Minuten); die Karten und alle übrigen Messreihen rechnet die App live (unter 2 Sekunden je Eingabe).

Zufallssysteme: n = 2, ein Eingang, A und B standardnormal, Q = q·I mit q = 10^U(−3, 0), R = r = 10^U(−3, 1); gültig sind steuerbare Systeme (|det [B, AB]| ≥ 10⁻³) mit lösbarer Riccati-Gleichung. Die Familie ist eine Wahl, keine Aussage über „typische“ Systeme.

## Befunde (gemessen, keine Behauptungen)

Alles gilt im linearen Modell ohne Beschränkungen und ohne Störung, mit dem Gesetz u = −K_N x in jedem Schritt wiederholt. Eine Spalte „Test“ nennt, wo die Zahl geprüft ist; die vollständige Zuordnung steht in der Tabelle „Zahl → Test“ unten.

| Frage | Befund |
|---|---|
| **B1** Kalmans Gegenbeispiel, P_f = Q (c = 2/3, ε = R = 10⁻³) | ρ für N = 1 bis 12: 1,495; 1,484; 1,458; 1,405; 1,307; 1,158; 0,989; 0,846; 0,756; 0,707; 0,684; 0,673; für N → ∞: 0,664; offene Strecke 0,816. Instabil sind N = 1 bis 6, kleinstes stabiles N = 7; von N = 13 bis 40 ist kein Horizont instabil (0 von 28). Abstand zum Grenzwert: 1,3·10⁻⁵ (N = 20), 3,5·10⁻⁹ (N = 30), 9,7·10⁻¹³ (N = 40). Die Quelle (S. 21 bis 22) nennt N = 5 (1,307) und N = 7 (0,989); hier bestätigt. |
| **B2** Terminalgewicht P_f = P_∞ (Riccati-Lösung) | K_N − K_∞ höchstens 3,3·10⁻¹⁶ für N = 1 bis 40 (SciPy, Gleitkomma), ρ = 0,66404 für jedes gemessene N; die Kostenmatrix nach 40 Schritten weicht um 4,4·10⁻¹⁶ von P_∞ ab. **Exakt** (Brüche): Wagen mit P = [[2, 2], [2, 4]] (daraus Q = [[4/5, 2/5], [2/5, 6/5]], R = 1): K_N = (2/5, 6/5) und Π_N = P für N = 1 bis 30; die Batch-Lösung gibt dasselbe K_N für N = 1 bis 6; mit P_f = Q statt P ist K_N ≠ K in 6 von 6 Horizonten. Rawlings (2.5.1, S. 132) schreibt das Ergebnis „K_N unabhängig von N“ für den unbeschränkten Fall; hier eigene Nachrechnung. |
| **B3** Zufallssysteme (Zufalls-Seeds 1 bis 20, je 6 000 Ziehungen) | 119 624 gültige Systeme; bei P_f = P_∞ weicht K_N (N ≤ 10) relativ höchstens um 2,9·10⁻⁶ von K_∞ ab (Median 8,2·10⁻¹⁶; reines Rundungsrauschen, das von der Form der Rekursion abhängt: eine andere Form der Riccati-Rekursion gab für dasselbe System 4,5·10⁻⁹; die größten Abweichungen haben schlecht konditionierte Systeme). Offen stabile Strecken: 40 727; bei P_f = Q (Q = q·I) ist N = 1 instabil in 1 238 (3,0 %; je Zufalls-Seed 2,5 % bis 3,5 %). Zufalls-Seed 12: 67 von 2 047 (3,3 %). Irgendein N von 1 bis 10 ist instabil in 1 247 Strecken, also in 9 mehr als bei N = 1 allein (in 8 der 20 Zufalls-Seeds). |
| **B4** andere Terminalgewichte (Kalman, instabile N bis 150) | P_f = 0: N = 2 bis 7 (N = 1 ist „stabil“, weil K_1 = 0 ist und die offene Strecke bleibt); P_f = Q: N = 1 bis 6; 10·Q: N = 1 bis 4; 1 000·Q: nur N = 1; P_f = I, 100·I, 0,5·P_∞, 0,9·P_∞, 1,1·P_∞, 2·P_∞, 1 000·P_∞: keines. Die Falle verschiebt sich, sie ist keine Einbahnstraße. **Nicht monoton:** bei P_f = 0,5·P_∞ ist ρ(1) = 0,664, aber ρ(2) = 0,920. Verschiebung: der Horizont N mit P_f = 0 gibt dasselbe Gesetz wie der Horizont N − 1 mit P_f = Q (exakt in Brüchen am Wagen bis N = 4, in Gleitkomma an 30 Zufallssystemen). |
| **B5** harte Terminalbedingung x(N) = 0 (Grenzfall P_f → ∞, Kalman) | ρ = 0 bei N = 2 (Deadbeat: K = (4/3, −2/3), die erste Zeile von A, exakt in Brüchen), danach 0,460; 0,585; 0,631; 0,650; 0,658; 0,661; 0,663; 0,664; 0,664; 0,664 (N = 3 bis 12). Für N = 3 bis 12 stimmen die Werte mit P_f = 10⁶·I auf fünf Stellen überein. |
| **B6** Preis eines kurzen, stabilen Horizonts (Kosten der unendlichen Zukunft im schlechtesten Anfangszustand ÷ optimale Kosten) | Kalman, P_f = Q: N = 7: 9,31 (stabil, ρ = 0,989); N = 8: 1,210; 9: 1,035; 10: 1,0066; 11: 1,0013; 12: 1,00025. Kalman, P_f = I (stabil für jedes N): N = 1: 1,791; 2: 1,094; 3: 1,017. Wagen, R = 1, P_f = Q: N = 1 nicht endlich (ρ = 1), N = 2: 1,102; 3: 1,0026; 4: 1,00004. „Stabil“ heißt nicht „fast optimal“. |
| **B7** zeitvarianter Ablauf K_N, …, K_1 (Kalman, P_f = Q) | Das geometrische Mittel des Spektralradius pro Schritt liegt für N = 1 bis 12 bei mindestens 1 (1,495 bei N = 1; 1,379 bei N = 6; 1,070 bei N = 11; 1,029 bei N = 12) und von N = 13 bis 40 darunter; der größte Singulärwert des N-Schritt-Produkts steigt bis 8,28 (N = 6). |
| **B8** Wagen (Q = I, R = 1, 1/10, 1/100, 1/1000) | Bei P_f = Q ist ρ(N = 1) = 1 für alle vier R; K_1 = (0, 1/(R + 1)), Eigenwerte 1 und R/(R + 1) (exakt in Brüchen). Der Wagen bremst seine Geschwindigkeit, bringt aber die Position nicht zurück: Endlage x₁(0) + (R + 1)·x₂(0), bei R = 1 also x₁(0) + 2·x₂(0) (der Vektor (1, R + 1) ist Linkseigenvektor zum Eigenwert 1, von Hand). Ab N = 2 asymptotisch stabil: ρ = 0,535; 0,497; 0,500; 0,500 gegen ρ_∞ = 0,422; 0,362; 0,380; 0,382. Mit P_f = 0 sind N = 1 und N = 2 nicht asymptotisch stabil (4 von 4). |
| **B9** wann tritt die Falle auf? (Kalman-Familie, C = (−c, 1), ε = R = 10⁻³, P_f = Q) | Nullstelle 1/c innerhalb, auf dem Kreis oder keine (c = 3; 2; 1,2; 1; 0): kein instabiles N bis N = 400 (5 von 5). Nullstelle außerhalb (c = 0,9; 0,8; 2/3; 0,6; 0,5; 0,4; 0,3; 0,1): instabil, kleinstes stabiles N = 14, 10, 7, 6, 5, 4, 4, 2 (8 von 8). ρ(N = 1) liegt nahe an der Nullstelle (1,109 bei 1,111; 1,247 bei 1,250; 1,495 bei 1,500; 1,988 bei 2,000). **Abhängigkeit, nicht Ursache.** Gegenprobe über 200 Zufallswerte c in (0,05; 4) (Zufalls-Seed 31): bei Nullstelle außerhalb Falle in 42 von 43, sonst in 0 von 157; die eine Ausnahme ist c = 0,9988 (Nullstelle 1,0012), dort ist ρ(1) = 0,99988. Je billiger der Stelleingriff und je kleiner ε, desto länger der nötige Horizont (kleinstes stabiles N über R = 10⁻⁴ bis 100 bei ε = 10⁻³: 8; 7; 7; 4; 1; 1; 1; bei ε = 10⁻¹: 2; 2; 2; 2; 1; 1; 1; bei ε = 1 überall 1); in keiner der 28 Zellen folgt auf das kleinste stabile N noch ein instabiles (bis N = 400). |
| **B10** Vier-Tank (modellbasiert; Q = 3·C′C, R·I, P_f = Q bzw. 0; instabile N bis 150) | Die offene Strecke ist stabil (größter Betrag 0,937). R = 10⁻⁴: N = 1 bis 54 (P_f = Q) bzw. N = 2 bis 55 (P_f = 0); R = 10⁻³: 1 bis 35 bzw. 2 bis 36; R = 10⁻²: 1 bis 19 bzw. 2 bis 20; R = 0,1; 1; 10; 100: keines. ρ_∞ = 0,943 bei R = 10⁻⁴ und 10⁻³ (0,9426 bei 10⁻²). Mit P_f = P_∞ ist K_N = K_∞ für N = 1 bis 20 (relativ 10⁻⁸ geprüft). Knapp: bei R = 10⁻⁴ ist ρ(54) = 1,0008 und ρ(55) = 0,9973 (nur gemessen, Band; die Schwelle hängt an den drei Stellen der Matrixeinträge). |

Die Messungen B1, B2 (Gleitkomma), B4 bis B10 sind in diesem Stand der Demo neu gerechnet (Hinweise zu einzelnen Zahlen: siehe „Hinweise zu den Zahlen“). **Quelle zu B10 (gelesen):** Berberich/Köhler/Müller/Allgöwer nennen das System offen stabil und durch ein MPC ohne Terminalbedingung bei zu kurzem Horizont destabilisierbar; ihr Verfahren ist datenbasiert (L = 30, Q = 3·I, R = 10⁻⁴·I) und sie schreiben, dass bei hinreichend großem Eingangsgewicht auch ein Schema ohne Terminalbedingung das Gleichgewicht stabilisiert. B10 ist damit **verträglich** (L = 30 liegt für R = 10⁻⁴ im instabilen Bereich, R = 0,1 stabilisiert jedes N), aber **nicht gleich**: ob das Schema der Quelle (mit Schlupfvariable, Regularisierung und gemessenen Daten) dem hier gerechneten Modell-MPC entspricht, ist nicht gemessen.

### Zahl → Test (Band oder exakt)

| Zahl | Test |
|---|---|
| B1: ρ(N), kleinstes stabiles N, 0 von 28, Abstände | `test_claims.py::test_b1_*` (ρ als Band 5·10⁻⁴; Abstände als Faktor 1,05), exakt in Brüchen: `test_dynamics.py::test_kalman_horizons_one_to_twelve_in_fractions_*` |
| B2: Abweichungen 3,3·10⁻¹⁶ und 4,4·10⁻¹⁶ | `test_claims.py::test_b2_*` (als Schranke 10⁻¹², plattformabhängig im letzten Bit); exakte Brüche: `test_dynamics.py::test_rational_riccati_solution_*`, `test_oracle_horizont_falle.py::test_riccati_agrees_with_the_batch_solution_in_fractions_exactly` |
| B3: 119 624, 40 727, 1 238, 1 247, 9, Zufalls-Seed 12 | `test_claims.py::test_b3_*` (vorgerechnete Datei gegen die README; Neu-Rechnung der ersten 500 Ziehungen von Zufalls-Seed 1 als Band ± 2) |
| B4: Listen instabiler N, nicht monoton | `test_claims.py::test_b4_*` (exakte Zählungen, kein Wert liegt knapp an 1) |
| B5: ρ der harten Bedingung | `test_claims.py::test_b5_*` (Band 5·10⁻⁴; Deadbeat exakt in `test_dynamics.py`) |
| B6: Kostenverhältnisse | `test_claims.py::test_b6_*` (relativ 10⁻³; ein Orakel gegen Simulation und Winkelsuche: `test_oracle_horizont_falle.py`) |
| B7: geometrisches Mittel, Singulärwert | `test_claims.py::test_b7_*` (Band 10⁻³; N = 13 bis 40 als Zählung) |
| B8: Wagen | `test_claims.py::test_b8_*`, exakt: `test_dynamics.py::test_wagen_*` |
| B9: Nullstellen, Karten, Gegenprobe | `test_claims.py::test_b9_*` (Zählungen exakt; ρ(1) als Band; Gegenprobe über 200 Zufallswerte neu gerechnet, Zahlen aus der vorgerechneten Datei zusätzlich geprüft) |
| B10: Vier-Tank | `test_claims.py::test_b10_*` (Schwelle N = 55 ± 1, ρ(54) und ρ(55) als Band 10⁻³) |
| Verdikt je Preset | `test_claims.py::test_preset_verdicts_*` und die Hilfetexte der Presets (`test_preset_help_numbers_*`) |

Alles, was in dieser README steht und nicht in der Tabelle vorkommt (Einordnung, Quellenstand), ist keine Messung.

## Hinweise zu den Zahlen

- **Zufallssysteme:** Mit dem plattformfesten SplitMix64 gibt Zufalls-Seed 12 67 von 2 047 offen stabilen Strecken (3,3 %); über 20 Zufalls-Seeds liegt der Anteil bei 2,5 % bis 3,5 %, und in 9 Fällen (8 Zufalls-Seeds) ist N = 1 stabil, aber ein späteres N ≤ 10 instabil. Die größte Abweichung bei P_f = P_∞ ist 2,9·10⁻⁶ (viele Systeme, darunter schlecht konditionierte).
- **„Falle nur bei Nullstelle außerhalb“ hat eine Ausnahme:** Die Gegenprobe über 200 Zufallswerte c fand c = 0,9988 mit Nullstelle 1,0012 knapp außerhalb des Kreises und ρ(1) = 0,99988: dort kein instabiles N. Die Regel gilt in 199 von 200 Fällen dieser Gegenprobe; dicht an der Nullstelle 1 (c = 0,99, Nullstelle 1,0101: 5 instabile N, ρ(1) = 1,009) ist die Falle da, aber schmal. Ursache bleibt offen.
- **Karten live statt vorgerechnet:** Die Karte rechnet in der App rund eine Sekunde (91 + 28 Zellen bis N = 400, `st.cache_data`; Messung auf dem Entwicklungsrechner). Nur die Zufallssysteme (Normalzahlen über `decimal`, einige Sekunden je Zufalls-Seed) sind vorgerechnet.
- **Zählung bis N = 150:** Die App zählt instabile N bis 150 für alle Strecken.
- **Terminalgewicht „Q“ beim Vier-Tank-System** ist Q = 3·C′C (= C′QC mit dem Ausgangsgewicht 3·I).
- **Schwellen:** Die Vier-Tank-Schwelle (ρ(55) = 0,9973, 0,3 % unter 1) und die Kalman-Schwelle (ρ(7) = 0,989, 1,1 % unter 1) stehen in den Tests als Band, nicht bitgleich.

## Ehrliche Grenzen

- **Gilt im linearen Modell ohne Beschränkungen und ohne Störung.** „P_f = P_∞ gibt K_N = K_∞“ ist hier nachgerechnet (Induktion über die Rekursion, exakt in Brüchen am Wagen, 119 624 Zufallssysteme); mit Stell- oder Zustandsgrenzen ist das ein anderes Problem (QP, Terminalmenge, Zulässigkeit).
- **Die Abhängigkeit von der Nullstelle ist eine Zählung in einer Familie** (Kalmans Strecke mit C = (−c, 1), ein Q = C′C + ε·I, ein paar R). Dass die Nullstelle die **Ursache** ist, folgt daraus nicht; andere B, Mehrgrößenstrecken (außer dem Vier-Tank-System) und andere Q sind nicht gemessen.
- **Quellenvergleiche.** Grüne/Pannek/Seehafer/Worthmann (nur das Abstract gelesen, ein Aufsatz zu nichtlinearen Systemen): Suboptimalitätsabschätzung für MPC ohne Terminalbedingung unter einer Steuerbarkeitsannahme und Einfluss des Horizonts auf Stabilität und Güte; ob „stabil für genügend großes N“ gilt und ob die Zulässigkeit ausgeklammert ist, steht im Abstract nicht und ist nicht geprüft; die Mindest-N-Schranke dort ist nicht nachgerechnet und nicht mit B1 bis B10 verglichen. Rawlings, Kapitel 2, S. 144 bis 147 (Terminalgewicht γ ohne Terminalbedingung): hier nicht nachgerechnet; die Messung mit γ·P_∞ (γ = 0,5 bis 1 000) zeigt nur das Verhalten in Kalmans Beispiel. Yoon/Kim 2026 (nur Abstract und Einleitung gelesen) nennen für DARE-Terminalkosten „dieselbe erste Stellgröße wie der LQR für jeden Horizont“; B2 rechnet das selbst nach, ihr Beweis ist nicht geprüft.
- **Vier-Tank:** Matrizen aus einer Textextraktion; Schema der Quelle nicht nachgerechnet (siehe B10).
- **Plattform:** Eigenwerte, DARE- und Lyapunov-Lösungen können sich in der letzten Binärstelle unterscheiden; Schwellenwerte (N = 7 bei 1,1 %, N = 55 bei 0,3 %) und alle Zahlen aus Gleitkomma stehen in den Tests als Band. ρ = 1 (Wagen, N = 1) wird nie mit `==` geprüft, sondern als Band ρ ≥ 1 − 10⁻⁹. Zählungen ohne Gleichstand an einer Schwelle sind exakt.
- **Keine Neuheitsaussage.** Optimalität ohne Stabilität ist Lehrbuchstoff (Kalman 1960); neu ist hier nur die Zählung in diesem wachsenden Beispiel.
- **Bedienung:** AppTest löst alle Regler und Knöpfe aus. Im Browser ist die App beim Bau einmal gestartet worden (Seite laden, ein Preset anklicken, den Formel-Expander öffnen: keine Fehlermeldung, kein KaTeX-Fehler, 9 Diagramme); echte Mausklicks auf jeden Regler sind nicht systematisch geprüft.

## Verifikation

Die Suite hat 121 Tests (`pytest --collect-only -q tests`).

- **Handbeispiele:** Skalar (30/19, 570/1171, Pol 3/10 für N = 1 bis 7), Wagen (K_1 = (0, 1/2), K_2 = (2/7, 1), Determinanten 2/7 und 1/5), Wagen mit rationaler Riccati-Lösung, Kalman N = 1 (Spur 3006/2009, Determinante 3/2009), Deadbeat K = (4/3, −2/3): `tests/test_dynamics.py`, alle in Brüchen.
- **Orakel** (`tests/test_oracle_horizont_falle.py`, anderer Rechenweg, zusammen unter 10 Sekunden): Riccati gegen Batch-Lösung (Gleitkomma N ≤ 6 auf fünf Strecken und drei Terminalgewichten, 40 Zufallssysteme; Brüche exakt), gegen numerische Minimierung über Stellfolgen (BFGS mit analytischem Gradienten über die Adjungierte, N ≤ 4), ρ gegen Spur-und-Determinante-Formel und Jury-Kriterium (300 Matrizen), DARE gegen Fixpunkt, Kostenverhältnis gegen Simulation und Winkelsuche, harte Bedingung (KKT) gegen den Grenzwert P_f → ∞, Vier-Tank-Matrizen als Daten.
- **Eigenschaften** (`tests/test_eigenschaften_horizont_falle.py`): P_f = P_∞ gibt K_N unabhängig von N (50 Zufallssysteme, auch mit singulärem Q); Skalierung von Q, R, P_f ändert K nicht; K_N(P_f = 0) = K_(N−1)(P_f = Q); Koordinatenwechsel ändert ρ nicht; Q = 0 und P_f = 0 geben K = 0; harte Bedingung bei N = n ist Deadbeat; gleicher Zufalls-Seed gibt gleiche Ausgabe; benannter Gegenfall zu „ρ fällt mit N“.
- **Randfälle** (`tests/test_edge_cases.py`): N = 1, R = 10⁻⁸, ρ genau 1 als Band, Nullstelle auf dem Kreis, singuläres Q, nicht steuerbares System (Fehlermeldung), N bis 400, nicht symmetrisches Terminalgewicht, Randwerte der Regler.
- **Claims und Verdikte:** `tests/test_claims.py` (Zahlen der Befunde, je Preset die erwartete Verdikt-Zeile, Zahlen der Hilfetexte), `tests/test_verdicts.py` (Schwellen und jede Zahl im Text an handgesetzten Werten).
- **Presets und Layout:** vollständig, Werte innerhalb der Reglerbereiche; Schnellstart oben in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben; jede Änderung eines wirksamen Reglers hebt die Markierung auf; c und ε (nur bei Kalmans Gegenbeispiel) und γ (nur bei γ·I und γ·P_∞) sind sonst ausgegraut und zählen beim Vergleich nicht (`tests/test_layout.py`).
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an Grenzen und auf mehreren Optionen, jede Strecke mit jedem Terminalgewicht) und Footer-Test.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `data/hfa_frozen.json` | vorgerechnete Zählungen über Zufallssysteme (Experiment 4) |
| `hfa_constants.py` | Regler-Optionen, feste Annahmen, Farben, Presets |
| `hfa_dynamics.py` | Kern: Strecken, Riccati-Rekursion, ρ, DARE, Kosten, harte Bedingung, zeitvarianter Ablauf |
| `hfa_evaluation.py` | Analyse, Karten, Tabellen und alle Verdikte (reine Funktionen) |
| `hfa_exact.py` | Bruchrechnung für Handbeispiele und die Poltabelle am Wagen |
| `hfa_layout.py` | Fester Kopf (Stylesheet) |
| `hfa_presets.py` | Permalink (Begrenzen, Einrasten), Presets, aktives Preset |
| `hfa_random.py` | Zufallssysteme für Experiment 4 |
| `hfa_rng.py` | SplitMix64 mit plattformfesten Normalzahlen (Kopie der Vorlage aus dem Website-Repo, gekürzt) |
| `hfa_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tools/make_frozen.py` | erzeugt `data/hfa_frozen.json` |
| `tests/` | Handbeispiele, Orakel, Eigenschaften, Randfälle, Claims, Verdikte, Presets, Layout, AppTest-Smoke-Test, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Beschränkungen, Terminalmenge, Zulässigkeit:** [`mpc-stellgrenzen-demo`](https://github.com/sebastian-hanisch/mpc-stellgrenzen-demo) und [`mpc-zulaessigkeit-demo`](https://github.com/sebastian-hanisch/mpc-zulaessigkeit-demo).
- **Reserven und Robustheit des Gesetzes K_N:** [`lqr-reserven-demo`](https://github.com/sebastian-hanisch/lqr-reserven-demo); **Störung und Rauschen:** [`plan-stoerung-demo`](https://github.com/sebastian-hanisch/plan-stoerung-demo); **Rechenzeit als Verzug:** [`regelung-verzug-demo`](https://github.com/sebastian-hanisch/regelung-verzug-demo).
- **Steuerhorizont m > 1** (Grüne u. a.), **Terminalgewicht γ nach Rawlings (Kapitel 2, S. 144 bis 147)** als Satz, **Mindest-N-Schranke** aus der Literatur: nicht gemessen.
- **Ursachenprüfung der Nullstellen-Abhängigkeit** (andere B, Mehrgrößenstrecken, andere Q).
- **PDF-Export:** nicht gebaut; die Zahlen stehen in der App.
- **Nachbarn:** [`rolling-horizon-demo`](https://github.com/sebastian-hanisch/rolling-horizon-demo) (Quantilregeln auf einer AR(1)-Nachfrage, kein Zustandsraum, kein Stabilitätsbegriff; „Rolling“ meint dort etwas anderes als das Neu-Rechnen eines Optimierungsproblems hier), [`value-iteration-demo`](https://github.com/sebastian-hanisch/value-iteration-demo) (Bellman-Rückwärtsrechnung diskret) und [`energiefahrweise-demo`](https://github.com/sebastian-hanisch/energiefahrweise-demo) (offene Optimalsteuerung) sind nur Verweise.

## Quellenstand

Gelesen (nur Gleichungen und Befunde in eigenen Worten, nichts wörtlich aus dem Buch von Rawlings/Mayne/Diehl, dessen PDF die Weitergabe nur mit Erlaubnis des Verlags erlaubt): Rawlings/Mayne/Diehl, *Model Predictive Control: Theory, Computation, and Design*, Abschnitt 1.3.4 (S. 21 bis 22), 2.5.1 (S. 132) und Kapitel 2, S. 144 bis 147; Grüne/Pannek/Seehafer/Worthmann, arXiv:1006.2529 (Aufsatz, nicht das Buch); Berberich/Köhler/Müller/Allgöwer, arXiv:1906.04679, Abschnitt V; Yoon/Kim, arXiv:2607.04006 (nur Abstract und Einleitung). Alle übrigen Befunde sind **eigene Nachrechnung in diesem Modell**.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Die vorgerechneten Zählungen neu erzeugen (rund 3 Minuten): `python tools/make_frozen.py`.

Gebaut mit Streamlit, Plotly, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Regelung von LQR bis MPC: vom Plan zur Regel](https://sebastianhanisch.net/konzepte-regelung.html).
