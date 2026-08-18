# Projektübergabe: BiBaZu Geometry to Pose

Stand: 9. August 2026  
Repository: `https://github.com/match-BiBaZu/bibazu_geometry_to_pose`  
Referenzbasis: Branch `Tobias_tuning`, Commit `a6233b1`
Zusätzlicher lokaler Arbeitsstand: selbstständige JSON-Posenvorschaubilder für
die Pressure Control GUI.

## 1. Ziel des Projekts

Das Projekt sagt aus CAD-/Mesh-Geometrie vorher, welche Orientierungen ein starres
Bauteil beim Rutschen entlang einer doppelt geneigten, rechtwinkligen Rutsche
einnehmen kann. Zusätzlich wird eine gerichtete Posenroadmap erzeugt, mit der später
eine offene Folge von Luftimpulsen zu einer gewünschten stabilen Zielpose geplant
werden kann.

Der neue Code unter `src/chute_pose` ersetzt schrittweise die historischen Skripte im
Repository. Er umfasst:

- Validierung und Normierung der Bauteilgeometrie,
- theoretische Boden-Wand-Kontaktlagen,
- quasistatische Kraft-/Momentprüfung über einen Reibwertbereich,
- endliche Stör- und Kippbarrieren,
- Zusammenführung physisch identischer Posen über diskrete Rotationssymmetrie,
- robuste und metastabile Poseklassen,
- gerichtete Aktuator- und passive Kippübergänge,
- offene Routenplanung mit höchstens vier Luftimpulsen,
- JSON-, YAML-, GraphML-, SVG- und PNG-Export.

Die Software ist ein geometrisch-quasistatisches Entwicklungsmodell. Sie ersetzt
keine experimentelle Validierung und keine sicherheitsgerichtete Anlagensteuerung.

## 2. Physikalische und geometrische Konventionen

Das feste Rutschenkoordinatensystem ist rechtshändig:

- `+X`: bergab entlang der Rutsche,
- `+Y`: von der Seitenwand weg; zulässiger Innenraum `y >= 0`,
- `+Z`: vom Rutschenboden weg; zulässiger Innenraum `z >= 0`,
- Boden: `z = 0`,
- Wand: `y = 0`,
- Schnittlinie von Boden und Wand: X-Achse.

Aus der ebenen Neutrallage wird die Rutsche zuerst um die feste Y-Achse und danach
um die mitbewegte X-Achse gedreht. Der aktuelle Entwicklungsarbeitspunkt ist:

```text
alpha = 45 deg   # Drehung um die mitbewegte Rutschen-X-Achse
beta  = 20 deg   # Drehung um Y, Gefälle in +X
```

Die Rutschbeginnmessung wurde bei `beta = 15 deg` und zusätzlichem
`alpha = 45 deg` durchgeführt. Daraus wird ein vorläufiger Reibwertbereich
abgeleitet. Längen sind in Millimetern.

Eine katalogisierte Lage muss gleichzeitig Boden- und Wandkontakt besitzen.
Reiner Einpunktkontakt an einer der beiden Flächen ist keine stabile Pose, sondern
eine Übergangslage. Zulässig ist mindestens ein flächiger Kontakt an einer Seite
und flächiger oder kantenartiger Mehrpunktkontakt an der anderen Seite.

## 3. Reproduzierbare Umgebung

- Windows 10/11
- Python `>=3.11`
- Paketverwaltung mit `uv`
- NumPy, SciPy, trimesh, Matplotlib und NetworkX
- optional `cadquery-ocp` für STEP-Verifikation
- pytest als Entwicklungsabhängigkeit

Installation:

```powershell
git clone https://github.com/match-BiBaZu/bibazu_geometry_to_pose
cd bibazu_geometry_to_pose
uv sync --extra dev --extra step
```

CLI prüfen:

```powershell
uv run chute-pose --help
```

`pyproject.toml` und `uv.lock` sind versioniert. Neue Abhängigkeiten müssen in
beiden reproduzierbar abgebildet werden.

## 4. Architektur und wichtige Dateien

| Datei | Verantwortung |
| --- | --- |
| `src/chute_pose/frame.py` | festes Rutschenkoordinatensystem und Gravitation |
| `src/chute_pose/geometry.py` | Mesh laden, Solid-/Einheitenprüfung, Schwerpunkt |
| `src/chute_pose/contacts.py` | theoretische Boden-Wand-Kontaktposen und Topologie |
| `src/chute_pose/stability.py` | quasistatische Kraft-/Momentstabilität und Reibung |
| `src/chute_pose/disturbance.py` | Bremskraft- und Störmomentreserven |
| `src/chute_pose/rocking.py` | endliche Kippenergiebarrieren |
| `src/chute_pose/symmetry.py` | diskrete Rotationssymmetrie aus dem STL |
| `src/chute_pose/step_verification.py` | optionale Bestätigung von Symmetrien im STEP |
| `src/chute_pose/equivalence.py` | praktische physische Poseklassen |
| `src/chute_pose/roadmap.py` | Roadmap, Scores, Routen, Exporte und Darstellung |
| `src/chute_pose/visualization.py` | technische 3D-Posenbilder mit Boden/Wand/Kontakten |
| `src/chute_pose/cli.py` | Befehle `inspect` bis `route` |
| `tests/` | deterministische Unit- und Geometrieregressionen |
| `ROBUST_PIPELINE.md` | Entwicklungsgeschichte und fachliche Detailerklärung |
| `Werkstücke_STL_grob/` | STL/OBJ/STEP der Bauteile |

Die historischen Dateien `Main.py`, `PoseFinder.py`, `PoseEliminator.py` und
verwandte Skripte bleiben als Referenz erhalten, sind aber nicht die Grundlage der
neuen Roadmap. Neue Arbeiten sollten unter `src/chute_pose` erfolgen.

## 5. Berechnungspipeline

Die neue Pipeline arbeitet in dieser Reihenfolge:

1. Mesh laden, Einheit und geschlossenes Solid prüfen.
2. Vor der Flächenaufzählung kontinuierliche Rotationssymmetrie über
   Querträgheitsmomente und die azimutunabhängige konvexe Stützfunktion prüfen.
3. Konvexe Stützflächen und alle theoretischen Boden-Wand-Lagen bestimmen; bei
   `Cinf` werden komplette Kreisbahnen von STL-Facetten analytisch reduziert.
4. Über den abgeleiteten Reibwertbereich quasistatische Gleichgewichte prüfen.
5. Bremskraft- und reine Störmomentreserven berechnen.
6. Endliche Kippbarrieren durch wiedergesetzte Zwischenorientierungen bestimmen.
7. Verbleibende diskrete STL-Rotationssymmetrie erkennen und bei Bedarf mit STEP
   verifizieren.
8. Symmetrieäquivalente Darstellungen zu physischen Poseklassen zusammenführen.
9. Jede Klasse konservativ bewerten: nur wenn alle Darstellungen bestehen, ist
   die Klasse `robust`; sonst `metastable`.
10. Reine Aktuatorrotationen und passive Niedrigbarrierenübergänge erzeugen.
11. Roadmap exportieren oder eine offene Route mit maximal vier Impulsen planen.

Aktuell gilt als vorläufig kalibrierte Robustheitsregel:

```text
endliche Kippbarriere >= 0.20 mm
bei face-face zusätzlich Bremsreserve >= 0.10 g
```

Diese Werte sind keine Materialkonstanten. Sie wurden anhand der bisherigen
Beobachtungen für Df1a, Dl1a und Qk1a gewählt.

## 6. Symmetrie und Pose-IDs

Eine physische Roadmap-Pose kann mehrere Katalog-IDs enthalten. Beispiel Df1a:

```text
Roadmap-ID 9  -> Katalogdarstellungen 9/12/32
```

Die Roadmap-ID ist immer die Repräsentanten-ID der Klasse. Übergänge referenzieren
diese Roadmap-ID. Bei Austausch oder Neuvernetzung eines CAD-Modells dürfen sich
alle Pose-IDs ändern. Externe Systeme dürfen daher IDs nicht ohne zugehörige
Roadmap-Datei dauerhaft fest eincodieren.

STEP wird derzeit nur zur Symmetrieverifikation verwendet. Kontaktgeometrie und
Kippberechnungen verwenden das Mesh.

Kontinuierliche Rotationssymmetrie wird als `Cinf` gespeichert. Die Erkennung
vergleicht die konvexe Stützfunktion bei festen Neigungs- und wechselnden
Azimutwinkeln; dadurch wird die endliche Facettenzahl eines STL nicht als
physische Cn-Symmetrie missverstanden. Reine Drehungen um die Symmetrieachse sind
keine Posenänderung. Kippbarrieren werden deshalb im Quotientenraum der
Achsenrichtung und bei `Cinf` bis zum tatsächlichen Energieberg ausgewertet.
Der bisherige STEP-Boolean-Check unterstützt nur endliche Rotationen; ein
`Cinf`-Ergebnis bleibt bis zur visuellen/experimentellen Bestätigung
`provisional`.

## 7. Aktuatorregeln

Alle gesteuerten Drehungen sind Links-Multiplikationen im festen
Rutschenkoordinatensystem. Kombinierte gesteuerte Rotationen werden nicht erzeugt.

| Aktion | Bedingung | Winkel |
| --- | --- | --- |
| `floor_main_neg_x` | Hauptflächenfamilie am Boden | `[-180, 0)` |
| `floor_main_pos_x` | Hauptfläche am Boden und intrinsische Mindestbreite `> 25 mm` | `(0, 180]` |
| `wall_main_neg_x` | Hauptfläche an der Wand und intrinsische Mindestbreite `> 25 mm` | `[-180, 0)` |
| `wall_main_pos_x` | Hauptflächenfamilie an der Wand | `(0, 180]` |
| `free_y` | frei | beide Vorzeichen bis 180 Grad |
| `free_z` | frei | beide Vorzeichen bis 180 Grad |

Die 25-mm-Grenze ist über `--opposite-x-min-height-mm` einstellbar. Gemessen wird
die kleinere intrinsische planare Ausdehnung der größten Stützflächenfamilie. So
zählt bei Ql1i nicht fälschlich die 80-mm-Längsseite; die relevante Breite beträgt
20 mm. Df1a besitzt etwa 69.282 mm und besteht die Bedingung.

Passive Kippkanten dürfen sich um eine beliebige Achse bewegen und kosten im
Routenplaner keinen Aktuatorimpuls.

## 8. Einfangbereich und geometrischer Score

Für eine Aktuatorkante wird die eindimensionale gesetzte Energielandschaft um die
Zielpose mit 1 Grad abgetastet; die Einfanggrenze wird auf 0.1 Grad verfeinert.

```text
w = capture_width_deg
capture_fraction = w / verfügbarer Winkelbereich
barrier_score = min(1, Zielbarriere / 0.20 mm)
s = geometric_score = capture_fraction * barrier_score
```

`w` ist die zusammenhängende Einfangbreite in Grad. `s` liegt zwischen 0 und 1,
ist aber ausdrücklich **keine Erfolgswahrscheinlichkeit**. Reale Versuche sollen
später `empirical_success_rate` liefern.

## 9. Roadmap und YAML-Übergabe

Ein Roadmap-Export erzeugt:

```text
<Part>_roadmap.json
<Part>_roadmap.yaml
ROADMAP_YAML_README.md
<Part>_roadmap.graphml
<Part>_roadmap.svg
<Part>_roadmap.png
```

Die JSON bettet fuer jeden Knoten ein kompaktes PNG als
`thumbnail_png_base64` ein. Sie kann deshalb von der Pressure Control GUI ohne
Zugriff auf Mesh oder Matplotlib als visuelle Kalibrieruebergabe geladen werden.
Der Loader bleibt tolerant gegenueber aelteren JSON-Dateien ohne Vorschaubild.
Die experimentelle YAML enthaelt absichtlich keine Base64-Bilder und bleibt
damit gut diff- und editierbar.

Die YAML ist für die experimentelle Übergabe gedacht. Sie enthält robuste und
metastabile Posen, alle gerichteten direkten Übergänge und je Kante einen leeren
Block:

```yaml
experimental:
  status: "untested"
  trials: null
  successes: null
  empirical_success_rate: null
  difficulty_rating: null
  notes: ""
```

Nach Versuchen sollten diese Felder befüllt werden. `success_rate` ist als Wert
zwischen 0 und 1 zu führen. Die Bedeutung der YAML ist zusätzlich in der jeweils
exportierten `ROADMAP_YAML_README.md` beschrieben.

Der Ordner `Poses_Found_Robust` wird von `.gitignore` erfasst. Exporte sind daher
lokale Artefakte und müssen nach einem Clone neu erzeugt oder bewusst separat
übergeben werden.

## 10. CLI-Kurzreferenz

Geometrie prüfen:

```powershell
uv run chute-pose inspect "Werkstücke_STL_grob/Df1a.STL" --alpha 45 --beta 20
```

Katalog und Kontaktbilder:

```powershell
uv run chute-pose catalog "Werkstücke_STL_grob/Df1a.STL"
uv run chute-pose render "Werkstücke_STL_grob/Df1a.STL" `
  --output-dir "Poses_Found_Robust/Df1a_theoretical"
```

Alle vom Pose-Generator und von der Roadmap erzeugten 3D-Ansichten verwenden
dieselbe orthografische GUI-Kamera: Z zeigt nach oben, Y nach rechts unten und X
nach rechts oben zwischen Y und Z.

Die optionalen Stabilitaetsbilder nennen den ausgewaehlten quasistatischen
Kraft-/Momenten-Algorithmus. Unter jeder Pose steht als Zahlenwert die kleinste
Druckreserve ueber den abgetasteten Reibwertbereich.

Roadmap erzeugen:

```powershell
uv run chute-pose roadmap "Werkstücke_STL_grob/Df1a.STL" `
  --output-dir "Poses_Found_Robust/Df1a_roadmap_provisional" `
  --geometry-status provisional

uv run chute-pose roadmap "Werkstücke_STL_grob/Ql1i.STL" `
  --output-dir "Poses_Found_Robust/Ql1i_roadmap" `
  --geometry-status verified
```

Route mit höchstens vier Aktuatorimpulsen suchen:

```powershell
uv run chute-pose route `
  "Poses_Found_Robust/Df1a_roadmap_provisional/Df1a_roadmap.json" `
  --start-pose 9 --target-pose 60 --max-actions 4
```

Es gibt keine Zwischenzustandserkennung und keine Neuplanung zwischen Impulsen.
Das Ergebnis ist eine feste offene Aktionsfolge.

## 11. Aktuell validierte Referenzbauteile

### Df1a

**Wichtig:** Das aktuelle Df1a-CAD ist fachlich als fehlerhaft bestätigt. Alle
konkreten IDs und Übergänge sind `provisional`.

Aktueller Ersatzmodellstand:

- 108 theoretische Kontaktlagen,
- 27 quasistatisch zulässige Darstellungen,
- C3-Zusammenführung zu 11 Roadmap-Knoten,
- 4 robuste Klassen: `9`, `24`, `35`, `60`,
- 7 metastabile Klassen,
- 16 Aktuatorkanten und 7 passive Kippkanten,
- Hauptfläche 5, Mindestbreite etwa 69.282 mm,
- zusätzliche Gegenrichtungen:
  - `9 -> 35`: `wall_main_neg_x`, -90 Grad,
  - `24 -> 60`: `floor_main_pos_x`, +90 Grad.

Nach Austausch des CAD müssen Katalog, Symmetrie, Roadmap, Referenzbilder und
YAML vollständig neu erzeugt werden.

### Ql1i

- exakte C4-Symmetrie,
- 6 physische Roadmap-Knoten,
- 2 robuste Zielklassen: `2` und `3`,
- 4 metastabile Zwischenklassen: `0`, `6`, `10`, `13`,
- 32 Aktuatorkanten,
- keine aufgelöste passive Kippkante,
- passive Ziele für `0`, `6`, `10`, `13` derzeit ungelöst,
- vier gleich große Hauptflächen `0/1/4/5`,
- intrinsische Mindestbreite 20 mm; deshalb keine `floor_main_pos_x`- oder
  `wall_main_neg_x`-Kanten bei 25-mm-Grenze.

### Dl1a

- exakte C3-Symmetrie,
- 42 quasistatisch zulässige Darstellungen, 14 physische Klassen,
- beobachtete robuste Klassen `15`, `16`, `31`, `34`,
- robuste Barrieren etwa 0.248 mm,
- spiegelbildliche Längsklassen nur etwa 0.011 mm,
- freie Y-/Z-Einfangbereiche der Stirnflächenlagen bleiben schlechter als die
  der beobachteten Austragslagen.

### Qk1a

- keine nichttriviale Vollteilsymmetrie (`C1`),
- gekrümmte/facettierte Regionen erzeugen viele Rohkandidaten,
- nach endlicher Kippbarriere bleiben die beobachteten diagonalen
  Kanten-/Flächenlagen erhalten,
- weitere Konsolidierung gekrümmter Stützregionen ist vor einer allgemeinen
  Serienauswertung sinnvoll.

### Kk1a

- automatische kontinuierliche Rotationssymmetrie `Cinf`, Achse nahezu lokale
  Z-Achse,
- maximale azimutale Stützfunktionsabweichung etwa 0.052 mm bei 0.15 mm
  Prüftoleranz,
- 102 rohe konvexe Ebenen werden zu 5 axialen Flächenfamilien reduziert,
- 12 theoretische Boden-Wand-Lagen einschließlich der nur im
  Symmetriequotienten isolierten Mantel-Mantel-Lagen,
- 6 Roadmap-Knoten: die Mantel-Mantel-Lagen `5/10` sind robust; die vier
  Endflächenlagen `4/6/8/11` werden entsprechend der Anlagenbeobachtung als
  reibungsabhängig/metastabil geführt,
- die seltenen Endflächenlagen `4/6` auf beziehungsweise an der kleinen
  Endfläche besitzen eine vollständige Kippbarriere von jeweils etwa 1.474 mm,
- die Gegenlagen `8/11` mit der Hauptfläche an Wand beziehungsweise Boden
  besitzen etwa 2.465 mm Kippbarriere, scheitern im quasistatischen Modell
  jedoch ab einem gesampelten Reibwert von ungefähr 0.057 am
  Kraft-/Momentengleichgewicht; eine mögliche gyroskopische Stabilisierung ist
  dynamisch und nicht Teil des Robustheitsscores,
- die beobachteten, nahezu axialen Mantel-Mantel-Lagen `5/10` (dickes oder
  dünnes Ende voraus) besitzen mit etwa 4.88 mm eine deutlich höhere
  Kippbarriere; die Achse ist wegen der zwei Radien geometrisch um rund 4.6°
  gegen Y und Z geneigt,
- direkte symmetrieäquivalente Übergänge `5 <-> 10` sind sowohl um Y als auch
  um Z mit etwa 170.831 Grad zulässig,
- von `4/6/8/11` führen vorläufige 90-Grad-Aktuatorkanten über die instabilen,
  exakt axialen Kataloglagen `1/2` durch passives Einrasten nach `5/10`; die
  Zwischenpose steht als `settling_pose_ids` in JSON beziehungsweise
  `passive_settling_via_catalog_pose_ids` in YAML,
- Hauptfläche ist Face 4 mit 23.898 mm Mindestspanne und liegt unter der
  25-mm-Grenze; die beiden robusten Posen liegen auf beziehungsweise an der
  gegenüberliegenden kleineren Endfläche; die Hauptflächenannahme bleibt
  unverändert,
- Export: `Poses_Found_Robust/Kk1a_roadmap_provisional/`.

## 12. Tests

Vollständiger Testlauf:

```powershell
uv run --extra dev --extra step pytest -q -p no:cacheprovider
```

Aktueller Stand: **34 bestandene Tests**. Die Suite deckt unter anderem ab:

- Koordinatensystem und Gravitation,
- Geometrie- und Kontaktkatalog,
- Reibungs-/Gleichgewichtsfilter,
- Brems- und Störmomentreserven,
- endliche Kippbarrieren,
- STL- und STEP-Symmetrie,
- kontinuierliche `Cinf`-Erkennung und Kk1a-Facettenreduktion,
- praktische Poseklassen,
- Df1a-Knoten- und Kantenregression,
- Ql1i-25-mm-Regel,
- Dl1a-Einfangscore-Regression,
- Routenpriorisierung und Vier-Impuls-Grenze,
- Roadmapbilder und experimentelle YAML-Felder.
- selbststaendige JSON-Posenvorschaubilder fuer die Pressure Control GUI.

Die vollständige Suite benötigt auf dem aktuellen Rechner ungefähr 2.5 Minuten.
Für pytest-`--basetemp` möglichst ein Verzeichnis unter `%TEMP%` verwenden, damit
keine generierten Testbilder im Repository landen.

## 13. Bekannte Grenzen

- Df1a verwendet ein falsches CAD; Ergebnisse sind nur infrastrukturelle
  Referenzen.
- Das Modell ist quasistatisch. Falltests, Stöße, elastische Verformung,
  Luftströmung und zeitabhängige Reibung fehlen.
- Passive Kippkanten verwenden diskretisierte gesetzte Energiepfade und können
  ein reales dynamisches Ziel verfehlen.
- Ein STL kann durch Facettierung scheinbare Flächen, Kanten oder Symmetrien
  erzeugen. Verdächtige Symmetrien mit STEP prüfen.
- Die größte gleich große Stützflächenfamilie wird als Hauptfläche verwendet.
  Bei mehreren fachlich unterschiedlichen Flächen gleicher Fläche braucht es
  gegebenenfalls eine explizitere CAD-Klassifikation.
- Der geometrische Score ist nicht experimentell kalibriert.
- Metastabile Knoten sind mögliche Zwischenlagen, nicht automatisch sinnvolle
  Endziele.
- Der Routenplaner setzt eine offene Folge ohne Kamerakontrolle zwischen den
  Impulsen voraus.
- Nicht alle metastabilen Klassen besitzen bereits ein eindeutiges passives
  Ziel, beispielsweise Ql1i.
- Die Pipeline wurde noch nicht automatisiert für alle 39 Bauteile abgenommen.

## 14. Nächste sinnvolle Arbeiten

1. Korrektes Df1a-CAD einspielen und alle Df1a-Artefakte neu erzeugen.
2. YAML-Kanten experimentell vermessen und `trials`, `successes` sowie
   `empirical_success_rate` befüllen.
3. Einheitliche Versuchsbedingungen dokumentieren: Druck, Pulsdauer,
   Düsenarray, Startpose, Geschwindigkeit, Verschmutzungszustand und Fehlermodus.
4. Routenplanung auf empirische Wahrscheinlichkeiten umstellen; geometrischen
   Score nur als Fallback verwenden.
5. YAML/Resolver in `BiBaZu_Big_Boi/ReorientationControlGUI` anbinden und dabei
   gerichtete Mehrkantenpfade sowie metastabile Zwischenlagen unterstützen.
6. Die 25-mm-Grenze an mehreren Bauteilen experimentell validieren und bei Bedarf
   bauteil- oder aktuatortypabhängig machen.
7. Passive Entspannung in einer vollständigeren lokalen SO(3)-Energielandschaft
   verbessern und ungelöste Ql1i-Knoten untersuchen.
8. Batch-Auswertung für alle Bauteile mit Referenzstatistik und automatisch
   erzeugten Kontrollbildern ergänzen.
9. Gekrümmte/facettierte Stützregionen, insbesondere Qk1a, robuster
   konsolidieren.
10. Nach Schemaänderungen an JSON/YAML eine neue Schema-Version und
    rückwärtskompatible Ladelogik ergänzen.

## 15. Pflege dieses Dokuments

`PROJECT_HANDOFF.md` ist Teil der Definition of Done. Bei Änderungen an
Physikannahmen, Aktuatorregeln, Dateiformaten, Referenzbauteilen oder CLI muss
dieses Dokument im selben Arbeitsschritt aktualisiert werden.

Mindestens zu pflegen sind:

- Datum, Branch und Referenzcommit,
- aktueller Teststand,
- Roadmap-Zahlen der Referenzbauteile,
- neue oder geänderte CLI-Befehle,
- JSON-/YAML-Schema und experimentelle Felder,
- bekannte falsche CAD-Stände,
- neue experimentelle Kalibrierungen,
- offene Punkte und Hardwareannahmen.

Vor Übergabe an Kollegen:

1. `git status` prüfen und unbeabsichtigte Testartefakte entfernen.
2. Vollständige Tests ausführen.
3. Df1a/Ql1i-Roadmaps bei relevanten Änderungen neu exportieren.
4. YAML und Markdown-Anleitung stichprobenartig lesen.
5. Änderungen committen und pushen; lokale, ignorierte Roadmap-Artefakte bei
   Bedarf separat übergeben.

## 16. Kurzreferenz

```text
Paket:              src/chute_pose
Detaildokument:     ROBUST_PIPELINE.md
CLI:                uv run chute-pose --help
Roadmap:            uv run chute-pose roadmap <mesh> --output-dir <dir>
Route:              uv run chute-pose route <roadmap.json> --start-pose N --target-pose M
Test:               uv run --extra dev --extra step pytest -q -p no:cacheprovider
Teststand:          31 bestanden
Rutsche:            alpha=45 deg, beta=20 deg
Robustheit:         Kippbarriere >= 0.20 mm; face-face zusätzlich >= 0.10 g
Gegen-X-Schwelle:   intrinsische Hauptflächenbreite > 25 mm
Df1a:               provisional, 4 robust + 7 metastabil
Ql1i:               2 robust + 4 metastabil
YAML:               experimentell editierbare gerichtete Roadmap
```
