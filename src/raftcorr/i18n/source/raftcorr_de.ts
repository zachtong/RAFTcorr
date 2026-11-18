<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1" language="de" sourcelanguage="en_US">
<context>
    <name>AdvancedPanel</name>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="148"/>
        <source>Margin</source>
        <translation>Rand</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="167"/>
        <source>Fraction of the free GPU memory one window may use.
Large regions are cut into tiles to fit it.</source>
        <translation>Anteil des freien GPU-Speichers, den ein Fenster nutzen darf.
Große Bereiche werden passend dazu in Kacheln zerlegt.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="149"/>
        <source>Must cover the largest displacement of the region of interest.</source>
        <translation>Muss die größte Verschiebung in der Region of Interest abdecken.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="152"/>
        <source>Use the Model Reach</source>
        <translation>Modellreichweite verwenden</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="155"/>
        <source>Extra image context around the region of interest so that displaced points stay inside
the window given to the network. Automatic: the model&apos;s reach (32 s px).</source>
        <translation>Zusätzlicher Bildbereich um die Region of Interest, damit verschobene Punkte innerhalb
des Fensters bleiben, das das Netzwerk erhält. Automatisch: die Modellreichweite (32 s px).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="166"/>
        <source>GPU Memory</source>
        <translation>GPU-Speicher</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="174"/>
        <source>Tile Blending</source>
        <translation>Kachelüberblendung</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="175"/>
        <source>Width of the feathered overlap between neighbouring tiles.</source>
        <translation>Breite der weich überblendeten Überlappung zwischen benachbarten Kacheln.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="178"/>
        <source>Automatic</source>
        <translation>Automatisch</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="179"/>
        <source>On Demand</source>
        <translation>Bei Bedarf</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="180"/>
        <source>Stored Volume</source>
        <translation>Gespeichertes Volumen</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="182"/>
        <source>Correlation</source>
        <translation>Korrelation</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="183"/>
        <source>How the correlation lookup is computed; the results are identical.
Automatic: stored volume for s8 (fast), on demand for s1-s4
(memory linear in the image size).</source>
        <translation>Wie die Korrelationsabfrage berechnet wird; die Ergebnisse sind identisch.
Automatisch: gespeichertes Volumen für s8 (schnell), bei Bedarf für s1-s4
(Speicher linear in der Bildgröße).</translation>
    </message>
</context>
<context>
    <name>AnalysisChart</name>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="75"/>
        <source>too few valid points</source>
        <translation>zu wenige gültige Punkte</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="73"/>
        <source>partly untracked</source>
        <translation>teilweise nicht verfolgt</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="77"/>
        <source>no strain value (sparse strain gauge)</source>
        <translation>kein Dehnungswert (zu wenige Punkte im VSG)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="80"/>
        <source>gauge endpoint lost</source>
        <translation>Endpunkt der Messstrecke verloren</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="82"/>
        <source>not computed</source>
        <translation>nicht berechnet</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="83"/>
        <source>no data</source>
        <translation>keine Daten</translation>
    </message>
</context>
<context>
    <name>AnalysisTab</name>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="82"/>
        <source>Point</source>
        <comment>Placement tool: a single location</comment>
        <translation>Punkt</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="84"/>
        <source>Line</source>
        <comment>Placement tool: a two-point gauge</comment>
        <translation>Linie</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="86"/>
        <source>Rectangle</source>
        <comment>Placement tool</comment>
        <translation>Rechteck</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="88"/>
        <source>Circle</source>
        <comment>Placement tool</comment>
        <translation>Kreis</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="90"/>
        <source>Polygon</source>
        <comment>Placement tool</comment>
        <translation>Polygon</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="102"/>
        <source>Click once to place a point probe.</source>
        <translation>Einmal klicken, um eine Punktsonde zu setzen.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="108"/>
        <source>Click twice: opposite corners.</source>
        <translation>Zweimal klicken: gegenüberliegende Ecken.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="112"/>
        <source>Click each vertex, then double-click to close.</source>
        <translation>Jeden Eckpunkt anklicken, dann per Doppelklick schließen.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="170"/>
        <source>partly untracked from frame %1</source>
        <translation>ab Bild %1 teilweise nicht verfolgt</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="179"/>
        <source>gaps: no strain value</source>
        <translation>Lücken: kein Dehnungswert</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="201"/>
        <source>Nothing valid along %1: the strain gauge found too few points there (near an edge or a hole). Plot a displacement, or use a smaller gauge on the Strain Field tab.</source>
        <translation>Keine gültigen Daten entlang %1: Das VSG findet dort zu wenige Punkte (nahe einem Rand oder Loch). Stellen Sie eine Verschiebung dar oder verwenden Sie im Reiter „Dehnungsfeld“ ein kleineres VSG.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="207"/>
        <source>Nothing valid along %1: the material under it is untracked (cut out of the frame&apos;s ROI, or out of view).</source>
        <translation>Keine gültigen Daten entlang %1: Das Material darunter wird nicht verfolgt (aus der ROI des Bildes ausgeschnitten oder außerhalb des Sichtfelds).</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="89"/>
        <source>Show</source>
        <comment>Probe list column: visibility checkbox</comment>
        <translation>Anzeigen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="91"/>
        <source>Name</source>
        <comment>Probe list column: the probe&apos;s label</comment>
        <translation>Name</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="93"/>
        <source>Type</source>
        <comment>Probe list column: point, line or region</comment>
        <translation>Typ</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="95"/>
        <source>Color</source>
        <comment>Probe list column: color swatch</comment>
        <translation>Farbe</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="100"/>
        <source>Color…</source>
        <translation>Farbe…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="97"/>
        <source>Note</source>
        <comment>Probe list column: why a probe shows gaps</comment>
        <translation>Hinweis</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="101"/>
        <source>Delete</source>
        <comment>Button: delete the selected probe</comment>
        <translation>Löschen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="103"/>
        <source>Clear All</source>
        <translation>Alle löschen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="183"/>
        <source>Statistic:</source>
        <translation>Statistik:</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="56"/>
        <source>Mean</source>
        <comment>Statistic: arithmetic mean</comment>
        <translation>Mittelwert</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="58"/>
        <source>Median</source>
        <comment>Statistic</comment>
        <translation>Median</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="59"/>
        <source>Maximum</source>
        <comment>Statistic</comment>
        <translation>Maximum</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="60"/>
        <source>Minimum</source>
        <comment>Statistic</comment>
        <translation>Minimum</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="61"/>
        <source>Standard deviation</source>
        <translation>Standardabweichung</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="62"/>
        <source>Valid fraction</source>
        <translation>Gültiger Anteil</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="46"/>
        <source>Crack opening</source>
        <translation>Rissöffnung</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="489"/>
        <source>Strain has not been computed yet. Compute it on the Strain Field tab, or plot a displacement.</source>
        <translation>Die Dehnung wurde noch nicht berechnet. Berechnen Sie sie im Reiter „Dehnungsfeld“ oder stellen Sie eine Verschiebung dar.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="533"/>
        <source>Gauge quantities need a line probe.</source>
        <translation>Messstreckengrößen benötigen eine Liniensonde.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="530"/>
        <source>No visible probe can show this quantity.</source>
        <translation>Keine sichtbare Sonde kann diese Größe anzeigen.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="166"/>
        <source>no valid data: %1</source>
        <translation>keine gültigen Daten: %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="174"/>
        <source>endpoint lost from frame %1</source>
        <translation>Endpunkt ab Bild %1 verloren</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="177"/>
        <source>gaps: too few valid points</source>
        <translation>Lücken: zu wenige gültige Punkte</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="185"/>
        <source>not plotted: gauges need a line</source>
        <translation>nicht dargestellt: Messstrecken benötigen eine Linie</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="188"/>
        <source>not plotted: one point has no spread or coverage</source>
        <translation>nicht dargestellt: ein einzelner Punkt hat weder Streuung noch Abdeckung</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="190"/>
        <source>not plotted</source>
        <translation>nicht dargestellt</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="513"/>
        <source>Added probe &apos;%1&apos;.</source>
        <translation>Sonde &apos;%1&apos; hinzugefügt.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="599"/>
        <source>Clear All Probes</source>
        <translation>Alle Sonden löschen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="600"/>
        <source>Delete every probe? This cannot be undone.</source>
        <translation>Alle Sonden löschen? Das lässt sich nicht rückgängig machen.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="110"/>
        <source>Point</source>
        <comment>Probe type</comment>
        <translation>Punkt</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="111"/>
        <source>Line</source>
        <comment>Probe type</comment>
        <translation>Linie</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="112"/>
        <source>Region</source>
        <comment>Probe type: an enclosed area</comment>
        <translation>Bereich</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="480"/>
        <source>Run a DIC analysis to plot probes.</source>
        <translation>Führen Sie eine DIC-Analyse aus, um Sonden darzustellen.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="92"/>
        <source>Virtual extensometer</source>
        <comment>Placement tool</comment>
        <translation>Virtueller Dehnungsaufnehmer</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="94"/>
        <source>Crack gauge</source>
        <comment>Placement tool: a line across a crack</comment>
        <translation>Riss-Messstrecke</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="104"/>
        <source>Click twice: start and end. A line is also a virtual extensometer and a crack-opening gauge.</source>
        <translation>Zweimal klicken: Anfang und Ende. Eine Linie ist auch ein virtueller Dehnungsaufnehmer und eine Rissöffnungs-Messstrecke.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="110"/>
        <source>Click twice: center, then the edge.</source>
        <translation>Zweimal klicken: Mittelpunkt, dann Rand.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="114"/>
        <source>Click the two gauge points. The chart then shows the strain between them.</source>
        <translation>Klicken Sie auf die beiden Messpunkte. Das Diagramm zeigt dann die Dehnung zwischen ihnen.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="118"/>
        <source>Click one point on each side of the crack. The chart then shows how far it opens.</source>
        <translation>Klicken Sie auf je einen Punkt auf beiden Seiten des Risses. Das Diagramm zeigt dann, wie weit er sich öffnet.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="124"/>
        <source>Fit</source>
        <comment>Zoom button: fit the image to the view</comment>
        <translation>Anpassen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="131"/>
        <source>Fit image to viewport</source>
        <translation>Bild an den Ansichtsbereich anpassen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="132"/>
        <source>100%</source>
        <comment>Zoom button: one image pixel per screen pixel</comment>
        <translation>100%</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="135"/>
        <source>Zoom to 100% (1:1)</source>
        <translation>Auf 100% (1:1) zoomen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="136"/>
        <source>Zoom in</source>
        <translation>Vergrößern</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="137"/>
        <source>Zoom out</source>
        <translation>Verkleinern</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="138"/>
        <source>Show field</source>
        <comment>Analysis canvas: color the image by the field</comment>
        <translation>Feld anzeigen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="140"/>
        <source>Color the reference image with the plotted field at the current frame. For a gauge reading, the Strain Field tab&apos;s field is shown.</source>
        <translation>Färbt das Referenzbild mit dem dargestellten Feld im aktuellen Bild ein. Bei einer Messstreckengröße wird das Feld des Reiters „Dehnungsfeld“ angezeigt.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="181"/>
        <source>Plot:</source>
        <translation>Darstellen:</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="184"/>
        <source>X axis:</source>
        <translation>X-Achse:</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="185"/>
        <source>Strain as:</source>
        <translation>Dehnung als:</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="187"/>
        <source>Min. valid fraction:</source>
        <translation>Min. gültiger Anteil:</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="188"/>
        <source>A frame is left blank when fewer than this fraction of a line&apos;s or region&apos;s points are reliable. Guards against a curve that stays smooth while its sample shrinks away.</source>
        <translation>Ein Bild bleibt leer, wenn weniger als dieser Anteil der Punkte einer Linie oder eines Bereichs zuverlässig ist. Verhindert eine Kurve, die glatt bleibt, während ihre Stichprobe schrumpft.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="195"/>
        <source>Over time</source>
        <comment>Chart view: every frame of each probe</comment>
        <translation>Zeitverlauf</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="197"/>
        <source>Each probe&apos;s reading at every frame.</source>
        <translation>Der Messwert jeder Sonde in jedem Bild.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="200"/>
        <source>Along the line</source>
        <comment>Chart view: a profile</comment>
        <translation>Entlang der Linie</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="207"/>
        <source>Kymograph</source>
        <comment>Chart view: distance against frame</comment>
        <translation>Kymogramm</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="202"/>
        <source>The field along the selected line at the current frame, over the other frames in gray.</source>
        <translation>Das Feld entlang der ausgewählten Linie im aktuellen Bild, dahinter die übrigen Bilder in Grau.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="209"/>
        <source>The field along the selected line at every frame: distance against frame, value as color.</source>
        <translation>Das Feld entlang der ausgewählten Linie in jedem Bild: Abstand über Bild, Wert als Farbe.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="214"/>
        <source>Stress–strain</source>
        <comment>Chart view: stress against strain</comment>
        <translation>Spannung–Dehnung</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="216"/>
        <source>Stress (or load, without A0) against the plotted quantity, one curve per probe: stress-strain with an extensometer, load against opening with a crack gauge. Needs load data.</source>
        <translation>Spannung (ohne A0 die Kraft) über der dargestellten Größe, eine Kurve je Sonde: Spannungs-Dehnungs-Kurve mit einem Dehnungsaufnehmer, Kraft über Rissöffnung mit einer Riss-Messstrecke. Benötigt Kraftdaten.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="227"/>
        <source>Other frames</source>
        <translation>Übrige Bilder</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="228"/>
        <source>Draw the other frames&apos; profiles faintly behind the current one (at most twelve, evenly spaced).</source>
        <translation>Die Profile der übrigen Bilder blass hinter dem aktuellen zeichnen (höchstens zwölf, gleichmäßig verteilt).</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="233"/>
        <source>Line data (CSV)…</source>
        <translation>Liniendaten (CSV)…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="234"/>
        <source>Export</source>
        <translation>Exportieren</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="235"/>
        <source>Load data…</source>
        <translation>Kraftdaten…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="236"/>
        <source>Import a testing machine&apos;s load record (CSV) to plot against load or stress, and to draw stress-strain curves.</source>
        <translation>Kraftaufzeichnung einer Prüfmaschine (CSV) importieren, um über Kraft oder Spannung darzustellen und Spannungs-Dehnungs-Kurven zu zeichnen.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="241"/>
        <source>Probe data (CSV)…</source>
        <translation>Sondendaten (CSV)…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="243"/>
        <source>Chart image…</source>
        <translation>Diagrammbild…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="245"/>
        <source>Copy chart</source>
        <translation>Diagramm kopieren</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="247"/>
        <source>Copy plotted data</source>
        <translation>Dargestellte Daten kopieren</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="559"/>
        <source>Import the testing machine&apos;s load record with Load data… to draw stress-strain curves.</source>
        <translation>Importieren Sie die Kraftaufzeichnung der Prüfmaschine über „Kraftdaten…“, um Spannungs-Dehnungs-Kurven zu zeichnen.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="31"/>
        <source>Displacement U</source>
        <translation>Verschiebung U</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="32"/>
        <source>Displacement V</source>
        <translation>Verschiebung V</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="33"/>
        <source>Displacement magnitude</source>
        <translation>Verschiebungsbetrag</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="42"/>
        <source>Extensometer strain</source>
        <translation>Dehnungsaufnehmer-Dehnung</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="43"/>
        <source>Extensometer true strain</source>
        <translation>Wahre Dehnung (Dehnungsaufnehmer)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="45"/>
        <source>Elongation ΔL</source>
        <translation>Verlängerung ΔL</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="47"/>
        <source>Crack sliding</source>
        <translation>Rissgleiten</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="48"/>
        <source>Crack opening magnitude</source>
        <translation>Betrag der Rissöffnung</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="279"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="434"/>
        <source>Frame</source>
        <translation>Bild</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="281"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="429"/>
        <source>Time (s)</source>
        <translation>Zeit (s)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="283"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="431"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="574"/>
        <source>Load (N)</source>
        <translation>Kraft (N)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="286"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="433"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="571"/>
        <source>Stress (MPa)</source>
        <translation>Spannung (MPa)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="295"/>
        <source>ratio</source>
        <comment>Strain display unit: plain number</comment>
        <translation>Verhältnis</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="420"/>
        <source>Could not draw the field: %1</source>
        <translation>Das Feld konnte nicht gezeichnet werden: %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="130"/>
        <source>Esc cancels placement</source>
        <translation>Esc bricht das Setzen ab</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="138"/>
        <source>Drag to move the probe, or drag a handle to reshape it. Delete removes it; F2 renames it.</source>
        <translation>Ziehen verschiebt die Sonde, Ziehen an einem Griff ändert ihre Form. Entf löscht sie, F2 benennt sie um.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="485"/>
        <source>Place a probe on the reference image to begin.</source>
        <translation>Setzen Sie eine Sonde auf das Referenzbild, um zu beginnen.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="212"/>
        <source>Nothing valid along %1: it lies off the measured area.</source>
        <translation>Keine gültigen Daten entlang %1: Die Linie liegt außerhalb des Messbereichs.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="610"/>
        <source>A line view shows a field. Choose a field to plot.</source>
        <translation>Eine Linienansicht zeigt ein Feld. Wählen Sie ein Feld zum Darstellen.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="614"/>
        <source>Place a line probe, or select one, to see the field along it.</source>
        <translation>Setzen oder wählen Sie eine Liniensonde, um das Feld entlang der Linie zu sehen.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="632"/>
        <source>Distance along %1 (%2)</source>
        <translation>Abstand entlang %1 (%2)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="641"/>
        <source>%1, frame %2</source>
        <translation>%1, Bild %2</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="153"/>
        <source>not plotted: off the measured area</source>
        <translation>nicht dargestellt: außerhalb des Messbereichs</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="156"/>
        <source>not plotted: a gauge end is off the measured area</source>
        <translation>nicht dargestellt: ein Messstreckenende liegt außerhalb des Messbereichs</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="671"/>
        <source>Export Probe Data</source>
        <translation>Sondendaten exportieren</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="664"/>
        <source>CSV Files</source>
        <translation>CSV-Dateien</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="442"/>
        <source>The load data cannot be matched to the frames: %1</source>
        <translation>Die Kraftdaten lassen sich den Bildern nicht zuordnen: %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="664"/>
        <location filename="../../gui/panels/analysis/tab.py" line="728"/>
        <source>All Files</source>
        <translation>Alle Dateien</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="699"/>
        <source>Probe export failed: %1</source>
        <translation>Sondenexport fehlgeschlagen: %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="701"/>
        <source>Probe data written to %1</source>
        <translation>Sondendaten nach %1 geschrieben</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="709"/>
        <source>Export Line Data</source>
        <translation>Liniendaten exportieren</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="718"/>
        <source>Line export failed: %1</source>
        <translation>Linienexport fehlgeschlagen: %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="720"/>
        <source>Line data written to %1</source>
        <translation>Liniendaten nach %1 geschrieben</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="726"/>
        <source>SVG Images</source>
        <translation>SVG-Bilder</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="737"/>
        <source>Export Chart</source>
        <translation>Diagramm exportieren</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="756"/>
        <source>Chart copied to the clipboard.</source>
        <translation>Diagramm in die Zwischenablage kopiert.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="764"/>
        <source>Plotted data copied to the clipboard.</source>
        <translation>Dargestellte Daten in die Zwischenablage kopiert.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="725"/>
        <source>PNG Images</source>
        <translation>PNG-Bilder</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="727"/>
        <source>PDF Documents</source>
        <translation>PDF-Dokumente</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="748"/>
        <source>Chart export failed: %1</source>
        <translation>Diagrammexport fehlgeschlagen: %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="750"/>
        <source>Chart written to %1</source>
        <translation>Diagramm nach %1 geschrieben</translation>
    </message>
</context>
<context>
    <name>App</name>
    <message>
        <location filename="../../gui/app.py" line="746"/>
        <source>Run DIC first -- no displacement results to post-process.</source>
        <translation>DIC zuerst ausführen — keine Verschiebungs-Ergebnisse zur Nachbearbeitung.</translation>
    </message>
</context>
<context>
    <name>Application</name>
    <message>
        <location filename="../../gui/app.py" line="879"/>
        <source>RAFTcorr has hit an error</source>
        <translation>In RAFTcorr ist ein Fehler aufgetreten</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="880"/>
        <source>An unexpected error occurred. The application may not behave correctly from here on, so saving your project and restarting is recommended.</source>
        <translation>Ein unerwarteter Fehler ist aufgetreten. Die Anwendung verhält sich danach möglicherweise nicht mehr korrekt; es wird empfohlen, das Projekt zu speichern und neu zu starten.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="890"/>
        <source>Details were written to %1</source>
        <translation>Details wurden nach %1 geschrieben</translation>
    </message>
</context>
<context>
    <name>AutoFixedSelector</name>
    <message>
        <location filename="../../gui/widgets/range_mode.py" line="33"/>
        <source>Auto</source>
        <comment>Color range mode: rescale to the data range</comment>
        <translation>Auto</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/range_mode.py" line="36"/>
        <source>Rescale the color range to each frame&apos;s data range</source>
        <translation>Farbbereich an den Datenbereich jedes Bildes anpassen</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/range_mode.py" line="39"/>
        <source>Fixed</source>
        <comment>Color range mode: manual min/max bounds</comment>
        <translation>Fest</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/range_mode.py" line="42"/>
        <source>Keep the manual Min/Max bounds for every frame</source>
        <translation>Manuelle Min/Max-Grenzen für alle Bilder beibehalten</translation>
    </message>
</context>
<context>
    <name>BatchImportDialog</name>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="368"/>
        <source>Batch Import Region of Interest Masks</source>
        <translation>Region-of-Interest-Masken stapelweise importieren</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="419"/>
        <source>Mask Folder:</source>
        <translation>Maskenordner:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="420"/>
        <source>(none)</source>
        <translation>(keine)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="425"/>
        <source>Browse...</source>
        <translation>Durchsuchen…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="444"/>
        <source>Available Masks</source>
        <translation>Verfügbare Masken</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="455"/>
        <source>Auto-Match by Name</source>
        <translation>Automatisch nach Name zuordnen</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="457"/>
        <source>Match mask files to frames by number in filename</source>
        <translation>Maskendateien anhand der Zahl im Dateinamen den Bildern zuordnen</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="461"/>
        <source>Assign Sequential</source>
        <translation>Sequentiell zuweisen</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="463"/>
        <source>Assign masks to frames in order starting from frame 0</source>
        <translation>Masken den Bildern der Reihe nach ab Bild 0 zuweisen</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="471"/>
        <source>Frame Assignments</source>
        <translation>Bildzuweisungen</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="474"/>
        <source>Frame</source>
        <translation>Bild</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="474"/>
        <source>Image</source>
        <translation>Bilddatei</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="474"/>
        <source>Mask</source>
        <translation>Maske</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="485"/>
        <source>Assign Selected -&gt;</source>
        <translation>Auswahl zuweisen -&gt;</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="487"/>
        <source>Pair selected mask(s) with selected frame(s)</source>
        <translation>Ausgewählte Maske(n) mit ausgewähltem/ausgewählten Bild(ern) koppeln</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="491"/>
        <source>Clear All</source>
        <translation>Alle löschen</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="536"/>
        <source>Select Mask Folder</source>
        <translation>Maskenordner auswählen</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="590"/>
        <source>Failed to read mask file.</source>
        <translation>Maskendatei konnte nicht gelesen werden.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="594"/>
        <source>Mismatched shape: %1×%2 (expected %3×%4)</source>
        <translation>Größe stimmt nicht: %1×%2 (erwartet %3×%4)</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="666"/>
        <source>%n mask(s) have mismatched sizes and are disabled.</source>
        <translation>
            <numerusform>%n Maske hat eine abweichende Größe und ist deaktiviert.</numerusform>
            <numerusform>%n Masken haben abweichende Größen und sind deaktiviert.</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="719"/>
        <source>Invalid assignment</source>
        <translation>Ungültige Zuordnung</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="721"/>
        <source>A frame can only have one mask. Select exactly one mask, or select multiple frames to assign one mask to many.</source>
        <translation>Ein Bild kann nur eine Maske haben. Wählen Sie genau eine Maske aus oder wählen Sie mehrere Bilder, um eine Maske mehreren zuzuweisen.</translation>
    </message>
</context>
<context>
    <name>CanvasArea</name>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1223"/>
        <source>Fit</source>
        <translation>Anpassen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1224"/>
        <source>Fit image to viewport</source>
        <translation>Bild an den Ansichtsbereich anpassen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1228"/>
        <source>100%</source>
        <translation>100%</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1229"/>
        <source>Zoom to 100% (1:1)</source>
        <translation>Auf 100% (1:1) zoomen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1232"/>
        <source>Zoom in</source>
        <translation>Vergrößern</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1237"/>
        <source>–</source>
        <translation>–</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1238"/>
        <source>Zoom out</source>
        <translation>Verkleinern</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1329"/>
        <source>EDITING REGION OF INTEREST FOR FRAME %1</source>
        <translation>REGION OF INTEREST VON BILD %1 WIRD BEARBEITET</translation>
    </message>
</context>
<context>
    <name>CanvasConfigOverlay</name>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="40"/>
        <source>Mode</source>
        <translation>Modus</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="41"/>
        <source>Model</source>
        <translation>Modell</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="42"/>
        <source>Iterations</source>
        <translation>Iterationen</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="85"/>
        <source>Accumulative</source>
        <translation>Akkumulativ</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="87"/>
        <source>Incremental, every %1 frames</source>
        <translation>Inkrementell, alle %1 Bilder</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="89"/>
        <source>Custom reference frames</source>
        <translation>Benutzerdefinierte Referenzbilder</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="90"/>
        <source>Incremental</source>
        <translation>Inkrementell</translation>
    </message>
</context>
<context>
    <name>ColorRange</name>
    <message>
        <location filename="../../gui/widgets/color_range.py" line="27"/>
        <source>Range</source>
        <translation>Bereich</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/color_range.py" line="37"/>
        <source>Min</source>
        <translation>Min</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/color_range.py" line="47"/>
        <source>Max</source>
        <translation>Max</translation>
    </message>
</context>
<context>
    <name>ExportDialog</name>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="45"/>
        <source>Magnitude</source>
        <translation>Betrag</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="46"/>
        <source>Residual</source>
        <translation>Residuum</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="66"/>
        <source>Open this tab to render a preview.</source>
        <translation>Diesen Reiter öffnen, um eine Vorschau zu rendern.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="73"/>
        <location filename="../../gui/dialogs/export_widgets.py" line="370"/>
        <source>Field</source>
        <translation>Feld</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="80"/>
        <source>Frame</source>
        <translation>Bild</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="101"/>
        <source>FIELD APPEARANCE</source>
        <translation>FELDDARSTELLUNG</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="106"/>
        <location filename="../../gui/dialogs/export_widgets.py" line="370"/>
        <source>Colormap</source>
        <translation>Farbskala</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="110"/>
        <source>Range</source>
        <translation>Bereich</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="114"/>
        <location filename="../../gui/dialogs/export_widgets.py" line="370"/>
        <source>Min</source>
        <translation>Min</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="118"/>
        <location filename="../../gui/dialogs/export_widgets.py" line="370"/>
        <source>Max</source>
        <translation>Max</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="125"/>
        <location filename="../../gui/dialogs/export_widgets.py" line="176"/>
        <source>Opacity</source>
        <extracomment>Column widths of the field rows (export, field, colormap, range mode, min / max), shared by the header labels of the Images and Animation tabs. Older names that share a field&apos;s colour state.</extracomment>
        <translation>Deckkraft</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="127"/>
        <source>Apply to all fields</source>
        <translation>Auf alle Felder anwenden</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="128"/>
        <source>Apply this field&apos;s colormap, opacity and auto-range to every enabled field (each field keeps its own min/max).</source>
        <translation>Farbskala, Deckkraft und Auto-Bereich dieses Felds auf alle aktivierten Felder anwenden (jedes Feld behält sein eigenes Min/Max).</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="133"/>
        <source>Refresh preview</source>
        <translation>Vorschau aktualisieren</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="160"/>
        <source>Preview failed: </source>
        <translation>Vorschau fehlgeschlagen: </translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="194"/>
        <source>Enable a field on the Images tab to preview.</source>
        <translation>Aktivieren Sie ein Feld im Reiter „Bilder“ für die Vorschau.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="198"/>
        <source>No data for this field/frame.</source>
        <translation>Keine Daten für dieses Feld/Bild.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="177"/>
        <source>Field opacity (0 = transparent, 1 = fully opaque)</source>
        <translation>Feld-Deckkraft (0 = transparent, 1 = vollständig deckend)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="284"/>
        <source>All</source>
        <translation>Alle</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="286"/>
        <source>None</source>
        <translation>Keine</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="357"/>
        <source>Select:</source>
        <translation>Auswählen:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="370"/>
        <source>Export</source>
        <translation>Export</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="370"/>
        <source>Auto</source>
        <translation>Auto</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="387"/>
        <source>PHYSICAL UNITS</source>
        <translation>PHYSIKALISCHE EINHEITEN</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="391"/>
        <source>Enable physical units</source>
        <translation>Physikalische Einheiten aktivieren</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="393"/>
        <source>Scale displacement values by pixel size and show physical units on colorbar labels. Strain is dimensionless and unaffected.</source>
        <translation>Verschiebungswerte mit der Pixelgröße skalieren und physikalische Einheiten auf den Farbleistenbeschriftungen anzeigen. Dehnung ist dimensionslos und wird nicht beeinflusst.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="410"/>
        <source>/ pixel</source>
        <translation>/ Pixel</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="412"/>
        <source>Pixel size</source>
        <translation>Pixelgröße</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="421"/>
        <source>fps</source>
        <translation>fps</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="423"/>
        <source>Frame rate</source>
        <translation>Bildrate</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="449"/>
        <source>FRAME RANGE</source>
        <translation>BILDBEREICH</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="451"/>
        <source>All frames</source>
        <translation>Alle Bilder</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="456"/>
        <source>Numbered like the frame slider: frame 1 is the reference image.
File names use the same numbers: frame 1 is written as _0001.</source>
        <translation>Nummeriert wie der Bildschieberegler: Bild 1 ist das Referenzbild.
Dateinamen verwenden dieselben Nummern: Bild 1 wird als _0001 geschrieben.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="459"/>
        <source>From</source>
        <comment>Frame range: starting frame</comment>
        <translation>Von</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="464"/>
        <source>to</source>
        <comment>Frame range: ending frame</comment>
        <translation>bis</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="161"/>
        <source>Export Results</source>
        <extracomment>GUI field name -&gt; variable name in the data files (raftcorr.export.scientific).</extracomment>
        <translation>Ergebnisse exportieren</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="188"/>
        <source>OUTPUT FOLDER</source>
        <translation>AUSGABEORDNER</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="194"/>
        <source>Select output folder…</source>
        <translation>Ausgabeordner auswählen…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="200"/>
        <source>Browse…</source>
        <translation>Durchsuchen…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="205"/>
        <source>Open Folder</source>
        <translation>Ordner öffnen</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="219"/>
        <source>Data</source>
        <translation>Daten</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="220"/>
        <source>Images</source>
        <translation>Bilder</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="221"/>
        <source>Animation</source>
        <translation>Animation</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="222"/>
        <source>Report</source>
        <translation>Bericht</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="304"/>
        <source>FORMAT</source>
        <translation>FORMAT</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="306"/>
        <source>NumPy Archive (.npz)</source>
        <translation>NumPy-Archiv (.npz)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="313"/>
        <source>CSV (per frame)</source>
        <translation>CSV (pro Bild)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="323"/>
        <source>DISPLACEMENT</source>
        <translation>VERSCHIEBUNG</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="326"/>
        <source>STRAIN</source>
        <translation>DEHNUNG</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="271"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="329"/>
        <source>Run Compute Strain first.</source>
        <translation>Zuerst „Dehnung berechnen“ ausführen.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="223"/>
        <source>Preview</source>
        <translation>Vorschau</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="252"/>
        <source>Export: images not readable (%1); pictures go on white.</source>
        <translation>Export: Bilder nicht lesbar (%1); Abbildungen werden auf Weiß gezeichnet.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="256"/>
        <source>Export: the images do not match the results&apos; image size; pictures go on white.</source>
        <translation>Export: Die Bilder passen nicht zur Bildgröße der Ergebnisse; Abbildungen werden auf Weiß gezeichnet.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="270"/>
        <source>This run has no intensity residual.</source>
        <translation>Dieser Lauf hat kein Intensitätsresiduum.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="308"/>
        <source>MATLAB v7.3 (.mat)</source>
        <translation>MATLAB v7.3 (.mat)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="310"/>
        <source>An HDF5-based MAT file: load it in MATLAB with load().</source>
        <translation>Eine HDF5-basierte MAT-Datei: in MATLAB mit load() laden.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="311"/>
        <source>HDF5 (.h5)</source>
        <translation>HDF5 (.h5)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="316"/>
        <source>One table per computed frame, one row per point of the region of interest.
Dense fields make large files: a 1000 x 1000 px region is a million rows per frame.</source>
        <translation>Eine Tabelle pro berechnetem Bild, eine Zeile pro Punkt der Region of Interest.
Dichte Felder ergeben große Dateien: Ein Bereich von 1000 x 1000 px sind eine Million Zeilen pro Bild.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="333"/>
        <source>✓ Parameters file (JSON) always exported</source>
        <translation>✓ Parameterdatei (JSON) wird immer exportiert</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="334"/>
        <source>✓ Grid coordinates (X, Y), ROI and computed-frame flags always included</source>
        <translation>✓ Gitterkoordinaten (X, Y), ROI und Kennzeichnung berechneter Bilder immer enthalten</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="335"/>
        <source>Displacements are written in pixels (+u right, +v down); the pixel size and frame rate above are recorded with them.</source>
        <translation>Verschiebungen werden in Pixeln geschrieben (+u nach rechts, +v nach unten); Pixelgröße und Bildrate von oben werden mit abgelegt.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="341"/>
        <source>Export Data</source>
        <translation>Daten exportieren</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="377"/>
        <source>IMAGE SETTINGS</source>
        <translation>BILDEINSTELLUNGEN</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="387"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="464"/>
        <source>Format</source>
        <translation>Format</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="351"/>
        <source>Deformed frame</source>
        <translation>Verformtes Bild</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="352"/>
        <source>Reference frame</source>
        <translation>Referenzbild</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="358"/>
        <source>Show background image</source>
        <translation>Hintergrundbild anzeigen</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="398"/>
        <source>DPI</source>
        <translation>DPI</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="412"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="490"/>
        <source>Include colorbar</source>
        <translation>Farbleiste einfügen</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="421"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="496"/>
        <source>Render as</source>
        <translation>Darstellen als</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="431"/>
        <source>Export Images</source>
        <translation>Bilder exportieren</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="445"/>
        <source>ANIMATION SETTINGS</source>
        <translation>ANIMATIONSEINSTELLUNGEN</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="473"/>
        <source>FPS</source>
        <translation>FPS</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="483"/>
        <source>Export every Nth frame (1 = every frame). Higher is faster and smaller
but looks choppier. Playback duration is preserved (the FPS above is the pre-decimation rate).</source>
        <translation>Jedes N-te Bild exportieren (1 = jedes Bild). Höher = schneller und kleiner,
wirkt aber ruckeliger. Die Abspieldauer bleibt erhalten (die FPS oben sind die Rate vor der Dezimierung).</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="488"/>
        <source>Frame step</source>
        <translation>Bildschritt</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="511"/>
        <source>Export Animation</source>
        <translation>Animation exportieren</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="521"/>
        <source>CONTENT</source>
        <translation>INHALT</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="523"/>
        <source>Parameter summary table</source>
        <translation>Parameter-Übersichtstabelle</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="525"/>
        <source>Field statistics (min/max/mean/std per frame)</source>
        <translation>Feldstatistik (min/max/Mittelwert/Stdabw. pro Bild)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="527"/>
        <source>Sample field images</source>
        <translation>Beispiel-Feldbilder</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="535"/>
        <source>Sample every</source>
        <translation>Alle</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="541"/>
        <source>frames</source>
        <comment>Report: sample every N frames</comment>
        <translation>Bilder</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="547"/>
        <source>FIELDS</source>
        <translation>FELDER</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="549"/>
        <source>Displacement:</source>
        <translation>Verschiebung:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="550"/>
        <source>Strain:</source>
        <translation>Dehnung:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="559"/>
        <source>Format: HTML (self-contained, view in any browser)</source>
        <translation>Format: HTML (eigenständig, in jedem Browser anzeigbar)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="563"/>
        <source>Generate Report</source>
        <translation>Bericht erstellen</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="707"/>
        <source>Select Output Folder</source>
        <translation>Ausgabeordner auswählen</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="808"/>
        <source>Writing %1 (%2%)</source>
        <translation>Schreibe %1 (%2%)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="841"/>
        <source>Error: %1</source>
        <translation>Fehler: %1</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="783"/>
        <source>Starting…</source>
        <translation>Wird gestartet…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="355"/>
        <source>Deformed: the field is drawn where each point is in that frame, over the frame&apos;s own photo.
Reference: drawn at the points&apos; positions in frame 1, over frame 1.</source>
        <translation>Verformtes Bild: Das Feld wird dort gezeichnet, wo sich jeder Punkt in diesem Bild befindet, über dem eigenen Foto des Bildes.
Referenzbild: an den Positionen der Punkte in Bild 1 gezeichnet, über Bild 1.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="362"/>
        <source>Uncheck to export the field on its own, on white, with no speckle image behind it.</source>
        <translation>Deaktivieren, um nur das Feld auf Weiß ohne Speckle-Bild dahinter zu exportieren.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="363"/>
        <source>The images of this run are not available.</source>
        <translation>Die Bilder dieses Laufs sind nicht verfügbar.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="395"/>
        <source>Resolution of the written figure: about 8.6 x 7 inches at this many dots per inch.</source>
        <translation>Auflösung der geschriebenen Abbildung: etwa 8,6 x 7 Zoll bei dieser Anzahl Punkte pro Zoll.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="402"/>
        <source>Each frame</source>
        <translation>Pro Bild</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="403"/>
        <source>Whole sequence</source>
        <translation>Gesamte Sequenz</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="405"/>
        <source>How Auto picks the colour range: from each frame&apos;s own values (like the main window),
or one range from every exported frame, so a colour means the same value in every image.</source>
        <translation>Wie Auto den Farbbereich wählt: aus den Werten jedes einzelnen Bildes (wie im Hauptfenster)
oder ein Bereich aus allen exportierten Bildern, sodass eine Farbe in jedem Bild denselben Wert bedeutet.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="410"/>
        <source>Auto range</source>
        <translation>Auto-Bereich</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="414"/>
        <source>Add a vertical colorbar to the right of each image.</source>
        <translation>Fügt rechts neben jedem Bild eine vertikale Farbleiste hinzu.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="423"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="498"/>
        <source>Crop to ROI</source>
        <translation>Auf ROI zuschneiden</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="424"/>
        <source>Cut the picture to the region of interest plus a small margin.</source>
        <translation>Schneidet die Abbildung auf die Region of Interest plus einen kleinen Rand zu.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="457"/>
        <source>MP4 needs the imageio and imageio-ffmpeg packages (pip install imageio imageio-ffmpeg). Save as GIF instead.</source>
        <translation>MP4 benötigt die Pakete imageio und imageio-ffmpeg (pip install imageio imageio-ffmpeg). Speichern Sie stattdessen als GIF.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="492"/>
        <source>Add a vertical colorbar to the right of each frame.</source>
        <translation>Fügt rechts neben jedem Bild eine vertikale Farbleiste hinzu.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="499"/>
        <source>Cut the frames to the region of interest plus a small margin.</source>
        <translation>Schneidet die Bilder auf die Region of Interest plus einen kleinen Rand zu.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="503"/>
        <source>Auto range: one range for the whole animation, so a colour means the same value in every frame.</source>
        <translation>Auto-Bereich: ein Bereich für die ganze Animation, sodass eine Farbe in jedem Bild denselben Wert bedeutet.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="530"/>
        <source>Drawn like the Images tab: same colormap, range, opacity and configuration.</source>
        <translation>Wie im Reiter „Bilder“ gezeichnet: gleiche Farbskala, gleicher Bereich, gleiche Deckkraft und Konfiguration.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="640"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="688"/>
        <source>Min must be below Max for %1.</source>
        <translation>Min muss für %1 kleiner als Max sein.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="646"/>
        <source>Nothing to export: enable at least one field with computed frames in the frame range.</source>
        <translation>Nichts zu exportieren: Aktivieren Sie mindestens ein Feld mit berechneten Bildern im Bildbereich.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="654"/>
        <source>Select at least one format.</source>
        <translation>Wählen Sie mindestens ein Format.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="658"/>
        <source>Select at least one variable to export.</source>
        <translation>Wählen Sie mindestens eine zu exportierende Variable.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="691"/>
        <source>Nothing to export: select a field or the parameter table.</source>
        <translation>Nichts zu exportieren: Wählen Sie ein Feld oder die Parametertabelle.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="744"/>
        <source>… and %1 more</source>
        <translation>… und %1 weitere</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="746"/>
        <source>Replace Existing Files?</source>
        <translation>Vorhandene Dateien ersetzen?</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="747"/>
        <source>These already exist in the output folder and will be replaced:

%1</source>
        <translation>Diese Dateien existieren bereits im Ausgabeordner und werden ersetzt:

%1</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="766"/>
        <source>The results changed after this dialog was opened (a new run, strain or project). Close it and open Export again to export them.</source>
        <translation>Die Ergebnisse haben sich geändert, seit dieser Dialog geöffnet wurde (neuer Lauf, neue Dehnung oder neues Projekt). Schließen Sie ihn und öffnen Sie den Export erneut, um sie zu exportieren.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="770"/>
        <source>Nothing exported: the existing files were kept.</source>
        <translation>Nichts exportiert: Die vorhandenen Dateien wurden beibehalten.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="795"/>
        <source>Stopping…</source>
        <translation>Wird gestoppt…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="806"/>
        <source>Rendering %1 (%2/%3)</source>
        <translation>Rendere %1 (%2/%3)</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/dialogs/export_dialog.py" line="824"/>
        <source>Exported %n file(s) → %1</source>
        <translation>
            <numerusform>%n Datei exportiert → %1</numerusform>
            <numerusform>%n Dateien exportiert → %1</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/dialogs/export_dialog.py" line="826"/>
        <source>Exported %n image(s) → %1</source>
        <translation>
            <numerusform>%n Bild exportiert → %1</numerusform>
            <numerusform>%n Bilder exportiert → %1</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="842"/>
        <source>Export failed: %1</source>
        <translation>Export fehlgeschlagen: %1</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/dialogs/export_dialog.py" line="850"/>
        <source>Export cancelled: %n file(s) written before stopping.</source>
        <translation>
            <numerusform>Export abgebrochen: %n Datei vor dem Abbruch geschrieben.</numerusform>
            <numerusform>Export abgebrochen: %n Dateien vor dem Abbruch geschrieben.</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="830"/>
        <source>No animation was written. See the log for details.</source>
        <translation>Es wurde keine Animation geschrieben. Einzelheiten siehe Protokoll.</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/dialogs/export_dialog.py" line="832"/>
        <source>Exported %n animation(s) → %1</source>
        <translation>
            <numerusform>%n Animation exportiert → %1</numerusform>
            <numerusform>%n Animationen exportiert → %1</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="834"/>
        <source>Report saved → %1</source>
        <translation>Bericht gespeichert → %1</translation>
    </message>
</context>
<context>
    <name>FieldSelector</name>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="19"/>
        <source>Horizontal displacement (+ right)</source>
        <translation>Horizontale Verschiebung (+ nach rechts)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="20"/>
        <source>Vertical displacement (+ down)</source>
        <translation>Vertikale Verschiebung (+ nach unten)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="21"/>
        <source>Displacement magnitude</source>
        <translation>Verschiebungsbetrag</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="22"/>
        <source>Intensity residual: local RMS difference between the reference image and
the deformed image mapped back by the displacement. Large where the
tracking failed.</source>
        <translation>Intensitätsresiduum: lokale RMS-Differenz zwischen dem Referenzbild und
dem mit der Verschiebung zurückabgebildeten verformten Bild. Groß, wo das
Tracking fehlgeschlagen ist.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="26"/>
        <source>Disp U</source>
        <translation>Versch. U</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="26"/>
        <source>Disp V</source>
        <translation>Versch. V</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="27"/>
        <source>Magnitude</source>
        <translation>Betrag</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/viz_controller.py" line="150"/>
        <location filename="../../gui/widgets/field_selector.py" line="27"/>
        <source>Residual</source>
        <translation>Residuum</translation>
    </message>
</context>
<context>
    <name>FrameNavigator</name>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="59"/>
        <source>Previous frame</source>
        <translation>Vorheriges Bild</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="69"/>
        <location filename="../../gui/widgets/frame_navigator.py" line="170"/>
        <source>Play animation</source>
        <translation>Animation abspielen</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="73"/>
        <location filename="../../gui/widgets/frame_navigator.py" line="169"/>
        <source>▶</source>
        <translation>▶</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="80"/>
        <source>Next frame</source>
        <translation>Nächstes Bild</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="93"/>
        <source>Playback speed</source>
        <translation>Wiedergabegeschwindigkeit</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="98"/>
        <location filename="../../gui/widgets/frame_navigator.py" line="194"/>
        <source>FRAME 0/0</source>
        <translation>BILD 0/0</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="159"/>
        <source>⏸</source>
        <translation>⏸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="160"/>
        <source>Pause animation</source>
        <translation>Animation pausieren</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="191"/>
        <source>FRAME %1/%2</source>
        <translation>BILD %1/%2</translation>
    </message>
</context>
<context>
    <name>ImageCanvas</name>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1067"/>
        <source>Load images first before drawing a Region of Interest.</source>
        <translation>Laden Sie zuerst Bilder, bevor Sie eine Region of Interest zeichnen.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1086"/>
        <source>The three points are nearly collinear — pick points spread around the circle&apos;s edge.</source>
        <translation>Die drei Punkte sind fast kollinear — wählen Sie Punkte, die über den Kreisrand verteilt sind.</translation>
    </message>
</context>
<context>
    <name>ImageList</name>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="97"/>
        <source>#</source>
        <comment>Image list column: frame index</comment>
        <translation>#</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="98"/>
        <source>Filename</source>
        <translation>Dateiname</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="99"/>
        <source>Region</source>
        <comment>Image list column: ROI status</comment>
        <translation>Region</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="193"/>
        <location filename="../../gui/widgets/image_list.py" line="255"/>
        <source>Add</source>
        <translation>Hinzu.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="249"/>
        <source>Edit</source>
        <translation>Bearb.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="252"/>
        <source>Need</source>
        <translation>Offen</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/widgets/image_list.py" line="368"/>
        <source>Import Region of Interest for %n frame(s)</source>
        <translation>
            <numerusform>Region of Interest für %n Bild importieren</numerusform>
            <numerusform>Region of Interest für %n Bilder importieren</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="379"/>
        <source>Clear Region of Interest (%1 with region)</source>
        <translation>Region of Interest löschen (%1 mit Region)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="383"/>
        <source>Clear Region of Interest</source>
        <translation>Region of Interest löschen</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/widgets/image_list.py" line="394"/>
        <source>Delete %n image(s)</source>
        <translation>
            <numerusform>%n Bild löschen</numerusform>
            <numerusform>%n Bilder löschen</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/widgets/image_list.py" line="435"/>
        <source>Select %n Mask File(s)</source>
        <translation>
            <numerusform>%n Maskendatei auswählen</numerusform>
            <numerusform>%n Maskendateien auswählen</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="440"/>
        <source>Images</source>
        <translation>Bilder</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="441"/>
        <source>All Files</source>
        <translation>Alle Dateien</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="450"/>
        <source>Selected %1 files for %2 frames — count must match</source>
        <translation>%1 Dateien für %2 Bilder ausgewählt — Anzahl muss übereinstimmen</translation>
    </message>
</context>
<context>
    <name>ImageMessages</name>
    <message>
        <location filename="../../gui/controllers/image_controller.py" line="43"/>
        <source>Cannot open the images</source>
        <translation>Bilder können nicht geöffnet werden</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/image_controller.py" line="47"/>
        <source>%n image(s), %1 x %2 px, %3</source>
        <translation>
            <numerusform>%n Bild, %1 x %2 px, %3</numerusform>
            <numerusform>%n Bilder, %1 x %2 px, %3</numerusform>
        </translation>
    </message>
</context>
<context>
    <name>LeftSidebar</name>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="176"/>
        <source>IMAGES</source>
        <translation>BILDER</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="183"/>
        <source>Natural Sort (1, 2, …, 10)</source>
        <translation>Natürliche Sortierung (1, 2, …, 10)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="259"/>
        <source>WORKFLOW TYPE</source>
        <translation>WORKFLOW-TYP</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="186"/>
        <source>Sort by embedded numbers: image1, image2, ..., image10 (default).
Unchecked: plain alphabetical order, for zero-padded names.</source>
        <translation>Nach eingebetteten Zahlen sortieren: image1, image2, ..., image10 (Standard).
Nicht aktiviert: rein alphabetisch, für nullgefüllte Namen.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="266"/>
        <source>REGION OF INTEREST</source>
        <translation>REGION OF INTEREST</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="278"/>
        <source>PARAMETERS</source>
        <translation>PARAMETER</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="284"/>
        <source>ADVANCED</source>
        <translation>ERWEITERT</translation>
    </message>
</context>
<context>
    <name>LoadDataDialog</name>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="103"/>
        <source>Load Data</source>
        <translation>Kraftdaten</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="115"/>
        <source>No file chosen.</source>
        <translation>Keine Datei gewählt.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="117"/>
        <source>Choose file…</source>
        <translation>Datei wählen…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="137"/>
        <source>Load column:</source>
        <translation>Kraftspalte:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="139"/>
        <source>By time</source>
        <translation>Nach Zeit</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="140"/>
        <source>By frame number</source>
        <translation>Nach Bildnummer</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="144"/>
        <source>Match rows to frames:</source>
        <translation>Zeilen den Bildern zuordnen:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="147"/>
        <source>Time column:</source>
        <translation>Zeitspalte:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="153"/>
        <source>The machine&apos;s time at the reference image. If the camera started 2 s after the machine, enter 2.</source>
        <translation>Die Maschinenzeit beim Referenzbild. Startete die Kamera 2 s nach der Maschine, geben Sie 2 ein.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="155"/>
        <source>Offset:</source>
        <translation>Versatz:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="163"/>
        <source>Frame column:</source>
        <translation>Bildspalte:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="167"/>
        <source>The first image is numbered:</source>
        <translation>Das erste Bild hat die Nummer:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="174"/>
        <source>Initial cross-section, for engineering stress F / A0 in MPa. Leave at 0 for load only.</source>
        <translation>Anfangsquerschnitt für die technische Spannung F / A0 in MPa. Für nur Kraft bei 0 lassen.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="176"/>
        <source>Cross-section A0:</source>
        <translation>Querschnitt A0:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="192"/>
        <source>Remove Load Data</source>
        <translation>Kraftdaten entfernen</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="201"/>
        <source>Camera frame rate: %1 fps, from Physical Units.</source>
        <translation>Kamera-Bildrate: %1 fps, aus „Physikalische Einheiten“.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="204"/>
        <source>Set the camera frame rate under Physical Units to match by time.</source>
        <translation>Legen Sie unter „Physikalische Einheiten“ die Kamera-Bildrate fest, um nach Zeit zuzuordnen.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="227"/>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="245"/>
        <source>(unnamed)</source>
        <translation>(unbenannt)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="324"/>
        <source>No frame falls within the record: check the columns and the offset.</source>
        <translation>Kein Bild liegt im Aufzeichnungsbereich: Prüfen Sie die Spalten und den Versatz.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="328"/>
        <source>Frames with a load: %1 of %2.</source>
        <translation>Bilder mit Kraftwert: %1 von %2.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="334"/>
        <source>Open Load Data</source>
        <translation>Kraftdaten öffnen</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="335"/>
        <source>CSV Files</source>
        <translation>CSV-Dateien</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="335"/>
        <source>All Files</source>
        <translation>Alle Dateien</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="341"/>
        <source>Could not read %1: %2</source>
        <translation>%1 konnte nicht gelesen werden: %2</translation>
    </message>
</context>
<context>
    <name>MainWindow</name>
    <message>
        <location filename="../../gui/app.py" line="136"/>
        <source>&amp;File</source>
        <translation>Datei</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="161"/>
        <source>Quit</source>
        <translation>Beenden</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="165"/>
        <source>&amp;Settings</source>
        <translation>Einstellungen</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="168"/>
        <source>Theme</source>
        <translation>Farbschema</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="172"/>
        <source>Dark</source>
        <comment>Color theme</comment>
        <translation>Dunkel</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="172"/>
        <source>Light</source>
        <comment>Color theme</comment>
        <translation>Hell</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="182"/>
        <source>Language</source>
        <translation>Sprache</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="363"/>
        <source>Suggested: RAFTcorr-%1. Largest displacement of the region of interest about %2 px (99th percentile over frames %3); RAFTcorr-%1 reaches %4 px.</source>
        <translation>Vorschlag: RAFTcorr-%1. Größte Verschiebung der Region of Interest etwa %2 px (99. Perzentil über die Bilder %3); RAFTcorr-%1 reicht %4 px weit.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="456"/>
        <source>Language changed</source>
        <translation>Sprache geändert</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="627"/>
        <source>All Files</source>
        <translation>Alle Dateien</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="200"/>
        <source>Select Image Folder</source>
        <translation>Bildordner auswählen</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="322"/>
        <source>File Association Failed</source>
        <translation>Dateiverknüpfung fehlgeschlagen</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="325"/>
        <source>File Association</source>
        <translation>Dateiverknüpfung</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="146"/>
        <source>Open Image Folder…</source>
        <translation>Bildordner öffnen…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="148"/>
        <source>Open Project…</source>
        <translation>Projekt öffnen…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="149"/>
        <location filename="../../gui/app.py" line="230"/>
        <location filename="../../gui/app.py" line="237"/>
        <source>Save Project</source>
        <translation>Projekt speichern</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="150"/>
        <source>Save Project As…</source>
        <translation>Projekt speichern unter…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="152"/>
        <source>Export Results…</source>
        <translation>Ergebnisse exportieren…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="157"/>
        <source>Associate .raftcorr files with RAFTcorr…</source>
        <translation>.raftcorr-Dateien mit RAFTcorr verknüpfen…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="158"/>
        <source>Register .raftcorr so double-clicking a project file opens RAFTcorr (current user only, no admin rights needed).</source>
        <translation>Registriert .raftcorr, sodass ein Doppelklick auf eine Projektdatei RAFTcorr öffnet (nur aktueller Benutzer, keine Administratorrechte nötig).</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="191"/>
        <source>&amp;Help</source>
        <translation>Hilfe</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="192"/>
        <source>Installed Models…</source>
        <translation>Installierte Modelle…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="193"/>
        <location filename="../../gui/app.py" line="392"/>
        <location filename="../../gui/app.py" line="394"/>
        <source>Self-test</source>
        <translation>Selbsttest</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="194"/>
        <location filename="../../gui/app.py" line="399"/>
        <source>About RAFTcorr</source>
        <translation>Über RAFTcorr</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="209"/>
        <source>Cancel</source>
        <translation>Abbrechen</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="231"/>
        <source>Load the images first: a project keeps the images, the regions of interest, the settings and the results together.</source>
        <translation>Laden Sie zuerst die Bilder: Ein Projekt speichert Bilder, Regions of Interest, Einstellungen und Ergebnisse zusammen.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="252"/>
        <source>Project saved: %1</source>
        <translation>Projekt gespeichert: %1</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="254"/>
        <source>Saving the project</source>
        <translation>Projekt wird gespeichert</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="268"/>
        <source>Include Results?</source>
        <translation>Ergebnisse einschließen?</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="269"/>
        <source>Include the computed results in this project?</source>
        <translation>Die berechneten Ergebnisse in dieses Projekt aufnehmen?</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="271"/>
        <source>Including results (about %1 uncompressed) lets you reopen the project without recomputing. Choose No to save a small settings-only file for sharing.</source>
        <translation>Mit den Ergebnissen (unkomprimiert etwa %1) lässt sich das Projekt ohne Neuberechnung wieder öffnen. Wählen Sie „Nein“, um eine kleine Datei nur mit den Einstellungen zum Weitergeben zu speichern.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="283"/>
        <source>Open Project</source>
        <translation>Projekt öffnen</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="290"/>
        <source>Opening the project</source>
        <translation>Projekt wird geöffnet</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="298"/>
        <source>Images not found</source>
        <translation>Bilder nicht gefunden</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="299"/>
        <source>The image folder of this project was not found. Choose its new location.</source>
        <translation>Der Bildordner dieses Projekts wurde nicht gefunden. Wählen Sie seinen neuen Speicherort.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="300"/>
        <source>Image folder of the project</source>
        <translation>Bildordner des Projekts</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="307"/>
        <source>Cannot open the project</source>
        <translation>Projekt kann nicht geöffnet werden</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="313"/>
        <source>Project opened: %1</source>
        <translation>Projekt geöffnet: %1</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="323"/>
        <source>Could not register .raftcorr files: </source>
        <translation>.raftcorr-Dateien konnten nicht registriert werden: </translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="326"/>
        <source>Done. Double-clicking a .raftcorr file will now open RAFTcorr and restore that project.</source>
        <translation>Fertig. Ein Doppelklick auf eine .raftcorr-Datei öffnet nun RAFTcorr und stellt dieses Projekt wieder her.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="331"/>
        <source>Load at least two images first.</source>
        <translation>Laden Sie zuerst mindestens zwei Bilder.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="334"/>
        <source>Wait for the running task to finish.</source>
        <translation>Warten Sie, bis die laufende Aufgabe beendet ist.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="367"/>
        <source>This exceeds even the reach of RAFTcorr-s8: use incremental tracking.</source>
        <translation>Das übersteigt sogar die Reichweite von RAFTcorr-s8: Verwenden Sie inkrementelles Tracking.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="372"/>
        <source>Measuring the motion with RAFTcorr-s8</source>
        <translation>Bewegung wird mit RAFTcorr-s8 gemessen</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="381"/>
        <source>%1  (s = %2, reach %3 px): %4</source>
        <translation>%1  (s = %2, Reichweite %3 px): %4</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="382"/>
        <source>not installed</source>
        <translation>nicht installiert</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="384"/>
        <source>Searched:</source>
        <translation>Durchsucht:</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="385"/>
        <source>Installed Models</source>
        <translation>Installierte Modelle</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="400"/>
        <source>&lt;b&gt;RAFTcorr %1&lt;/b&gt;&lt;br&gt;Learned digital image correlation with the RAFTcorr-s1, -s2, -s4 and -s8 models.&lt;br&gt;&lt;br&gt;Displacements: reference to deformed, +u right, +v down, in pixels, frame-1 (Lagrangian) coordinates.&lt;br&gt;&lt;br&gt;Interface derived from pyALDIC.&lt;br&gt;Zixiang (Zach) Tong, The University of Texas at Austin.</source>
        <translation>&lt;b&gt;RAFTcorr %1&lt;/b&gt;&lt;br&gt;Gelernte digitale Bildkorrelation mit den Modellen RAFTcorr-s1, -s2, -s4 und -s8.&lt;br&gt;&lt;br&gt;Verschiebungen: von Referenz zu verformt, +u nach rechts, +v nach unten, in Pixeln, in Koordinaten von Bild 1 (Lagrange).&lt;br&gt;&lt;br&gt;Oberfläche abgeleitet von pyALDIC.&lt;br&gt;Zixiang (Zach) Tong, The University of Texas at Austin.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="459"/>
        <source>Language set to %1. Please restart RAFTcorr for all widgets to pick up the new language.</source>
        <translation>Sprache auf %1 eingestellt. Bitte starten Sie RAFTcorr neu, damit alle Elemente die neue Sprache übernehmen.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="594"/>
        <source>Frame %1 has no Region of Interest of its own — frame 1&apos;s is used for computation. Switch to frame 1 to edit it, or import a mask to give this frame its own.</source>
        <translation>Bild %1 hat keine eigene Region of Interest — für die Berechnung wird die von Bild 1 verwendet. Wechseln Sie zu Bild 1, um sie zu bearbeiten, oder importieren Sie eine Maske, um diesem Bild eine eigene zu geben.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="605"/>
        <source>No Region of Interest to save — load images first.</source>
        <translation>Keine Region of Interest zum Speichern — laden Sie zuerst Bilder.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="612"/>
        <source>Region of Interest mask is empty.</source>
        <translation>Die Region-of-Interest-Maske ist leer.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="624"/>
        <source>Save Region of Interest Mask</source>
        <translation>Region-of-Interest-Maske speichern</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="626"/>
        <source>PNG Images</source>
        <translation>PNG-Bilder</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="635"/>
        <source>Mask saved to %1</source>
        <translation>Maske gespeichert unter %1</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="644"/>
        <source>No Region of Interest to invert — load images first.</source>
        <translation>Keine Region of Interest zum Invertieren — laden Sie zuerst Bilder.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="667"/>
        <source>Load images first.</source>
        <translation>Bitte zuerst Bilder laden.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="703"/>
        <source>  Imported mask for frame %1</source>
        <translation>  Maske für Bild %1 importiert</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/app.py" line="709"/>
        <source>Batch import: %n mask(s) loaded</source>
        <translation>
            <numerusform>Stapelimport: %n Maske geladen</numerusform>
            <numerusform>Stapelimport: %n Masken geladen</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/app.py" line="736"/>
        <source>Imported Region of Interest for %n frame(s)</source>
        <translation>
            <numerusform>Region of Interest für %n Bild importiert</numerusform>
            <numerusform>Region of Interest für %n Bilder importiert</numerusform>
        </translation>
    </message>
</context>
<context>
    <name>ParamPanel</name>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="90"/>
        <source>Iterations</source>
        <extracomment>The label column is as wide as its longest label in the current language, never narrower than the English layout and, past about 28 characters of the label font, wrapped rather than squeezing the inputs (pyALDIC&apos;s ParamPanel).</extracomment>
        <translation>Iterationen</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="91"/>
        <source>Refinement iterations of the network (default: 20).</source>
        <translation>Verfeinerungsiterationen des Netzwerks (Standard: 20).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="98"/>
        <source>Off</source>
        <comment>Displacement smoothing</comment>
        <translation>Aus</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="101"/>
        <source>Smoothing</source>
        <translation>Glättung</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="102"/>
        <source>Gaussian smoothing (σ in pixels) of the tracked displacement, kept in the results.
Off by default: smoothing coarsens the spatial resolution.</source>
        <translation>Gaußsche Glättung (σ in Pixeln) der verfolgten Verschiebung, in den Ergebnissen gespeichert.
Standardmäßig aus: Glättung vergröbert die räumliche Auflösung.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="106"/>
        <source>Automatic</source>
        <translation>Automatisch</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="110"/>
        <source>Device</source>
        <translation>Gerät</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="111"/>
        <source>Automatic uses the GPU when PyTorch sees one.</source>
        <translation>„Automatisch“ verwendet die GPU, wenn PyTorch eine erkennt.</translation>
    </message>
</context>
<context>
    <name>PhysicalUnitsWidget</name>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="56"/>
        <source>Use physical units</source>
        <translation>Physikalische Einheiten verwenden</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="67"/>
        <source>Physical size of one image pixel</source>
        <translation>Physikalische Größe eines Bildpixels</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="85"/>
        <source>/ px</source>
        <translation>/ px</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="88"/>
        <source>Pixel size</source>
        <translation>Pixelgröße</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="98"/>
        <source>Acquisition frame rate (used for velocity field)</source>
        <translation>Aufnahme-Bildrate (für das Geschwindigkeitsfeld verwendet)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="100"/>
        <source>Frame rate</source>
        <translation>Bildrate</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="168"/>
        <source>Disp: %1  Velocity: %2/s</source>
        <translation>Verschiebung: %1  Geschwindigkeit: %2/s</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="173"/>
        <source>Disp: px  Velocity: px/fr</source>
        <translation>Verschiebung: px  Geschwindigkeit: px/Bild</translation>
    </message>
</context>
<context>
    <name>PipelineController</name>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="77"/>
        <source>Stopped.</source>
        <extracomment>Results larger than this are kept in memory-mapped files in a temporary folder.</extracomment>
        <translation>Gestoppt.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="84"/>
        <source>The GPU ran out of memory. Close other GPU programs, choose a model with a larger s, or lower the memory fraction (Advanced).</source>
        <translation>Der GPU-Speicher reicht nicht aus. Schließen Sie andere GPU-Programme, wählen Sie ein Modell mit größerem s oder verringern Sie den Speicheranteil (Erweitert).</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="86"/>
        <source>Analysis failed</source>
        <translation>Analyse fehlgeschlagen</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="164"/>
        <source>Reference frames without a Region of Interest of their own carry the one of frame 1 forward: %1</source>
        <translation>Referenzbilder ohne eigene Region of Interest übernehmen die von Bild 1: %1</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="185"/>
        <source>Starting...</source>
        <translation>Wird gestartet…</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="211"/>
        <source>The images changed: the running analysis is cancelled.</source>
        <translation>Die Bilder haben sich geändert: Die laufende Analyse wird abgebrochen.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="235"/>
        <source>The images changed during the run: its results are discarded.</source>
        <translation>Die Bilder haben sich während des Laufs geändert: Seine Ergebnisse werden verworfen.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="248"/>
        <source>No frame was computed; the previous results are kept.</source>
        <translation>Es wurde kein Bild berechnet; die vorherigen Ergebnisse bleiben erhalten.</translation>
    </message>
</context>
<context>
    <name>PipelineMessages</name>
    <message numerus="yes">
        <location filename="../../gui/controllers/pipeline_messages.py" line="73"/>
        <source>  %n frame(s) with custom ROI masks</source>
        <translation>
            <numerusform>  %n Bild mit benutzerdefinierter ROI-Maske</numerusform>
            <numerusform>  %n Bilder mit benutzerdefinierten ROI-Masken</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/pipeline_messages.py" line="77"/>
        <source>Results received: %n frame(s)</source>
        <translation>
            <numerusform>Ergebnisse empfangen: %n Bild</numerusform>
            <numerusform>Ergebnisse empfangen: %n Bilder</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/pipeline_messages.py" line="81"/>
        <source>  %n image(s), %1</source>
        <translation>
            <numerusform>  %n Bild, %1</numerusform>
            <numerusform>  %n Bilder, %1</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="84"/>
        <source>  ROI range: %1</source>
        <translation>  ROI-Bereich: %1</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="87"/>
        <source>  Refinement: %1</source>
        <translation>  Netzverfeinerung: %1</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="91"/>
        <source>Error: %1</source>
        <translation>Fehler: %1</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="95"/>
        <source>Run canceled. Give the listed reference frames a Region of Interest of their own, or run again and continue with frame 1&apos;s region.</source>
        <translation>Lauf abgebrochen. Geben Sie den aufgeführten Referenzbildern eine eigene Region of Interest, oder starten Sie erneut und fahren Sie mit der Region von Bild 1 fort.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="102"/>
        <source>Tracking Quality Warning</source>
        <translation>Warnung zur Tracking-Qualität</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="105"/>
        <source>Part of the sequence may not be tracked correctly.</source>
        <translation>Ein Teil der Sequenz wurde möglicherweise nicht korrekt verfolgt.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="110"/>
        <source>The tracking quality drops from frame %1 on: the intensity residual rises to %2 grey levels, against %3 before.</source>
        <translation>Die Tracking-Qualität sinkt ab Bild %1: Das Intensitätsresiduum steigt auf %2 Graustufen, gegenüber %3 davor.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="116"/>
        <source>The displacement from frame 1 may exceed the reach of the model. Try incremental tracking, or a model with a larger s.</source>
        <translation>Die Verschiebung gegenüber Bild 1 übersteigt möglicherweise die Reichweite des Modells. Versuchen Sie inkrementelles Tracking oder ein Modell mit größerem s.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="119"/>
        <source>The step between reference frames may exceed the reach of the model. Use more reference frames, or a model with a larger s.</source>
        <translation>Der Schritt zwischen Referenzbildern übersteigt möglicherweise die Reichweite des Modells. Verwenden Sie mehr Referenzbilder oder ein Modell mit größerem s.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="123"/>
        <source>Frames with a high residual: %1</source>
        <translation>Bilder mit hohem Residuum: %1</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="126"/>
        <source>Show the Residual field to see where the images do not match after the displacement is applied.</source>
        <translation>Zeigen Sie das Feld „Residuum“ an, um zu sehen, wo die Bilder nach Anwenden der Verschiebung nicht übereinstimmen.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="132"/>
        <source>Cannot start analysis</source>
        <translation>Analyse kann nicht gestartet werden</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="136"/>
        <source>Need at least 2 images. Load an image folder in the %1 section before running DIC.</source>
        <translation>Mindestens 2 Bilder erforderlich. Laden Sie vor dem Ausführen der DIC einen Bildordner im Abschnitt %1.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="141"/>
        <source>Define a Region of Interest first: draw or import one for frame 1 in the %1 section.</source>
        <translation>Definieren Sie zuerst eine Region of Interest: Zeichnen oder importieren Sie im Abschnitt %1 eine für Bild 1.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="146"/>
        <source>The Region of Interest of frame 1 is empty (no pixels selected). Draw it again in the %1 section.</source>
        <translation>Die Region of Interest von Bild 1 ist leer (keine Pixel ausgewählt). Zeichnen Sie sie im Abschnitt %1 neu.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="151"/>
        <source>Region of Interest too small</source>
        <translation>Region of Interest zu klein</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="157"/>
        <source>The Region of Interest is %1×%2 px, but the current %3 (%4) needs at least %5×%5 px. Enlarge the Region of Interest or reduce %3.</source>
        <translation>Die Region of Interest ist %1×%2 px groß, aber der aktuelle %3 (%4) erfordert mindestens %5×%5 px. Vergrößern Sie die Region of Interest oder verringern Sie den %3.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="163"/>
        <source>Starting Points required</source>
        <translation>Startpunkte erforderlich</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="167"/>
        <source>Place at least one Starting Point on the canvas before running: click “%1” or “%2” in the %3 section.</source>
        <translation>Platzieren Sie vor dem Ausführen mindestens einen Startpunkt auf der Zeichenfläche: Klicken Sie im Abschnitt %3 auf „%1“ oder „%2“.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="173"/>
        <source>Invalid Starting Points</source>
        <translation>Ungültige Startpunkte</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="177"/>
        <source>One or more Starting Points do not lie in any part of the Region of Interest. Click “%1” in the %2 section, then place the Starting Points again.</source>
        <translation>Ein oder mehrere Startpunkte liegen in keinem Teil der Region of Interest. Klicken Sie im Abschnitt %2 auf „%1“ und platzieren Sie die Startpunkte erneut.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="185"/>
        <source>DIC analysis failed</source>
        <translation>DIC-Analyse fehlgeschlagen</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="188"/>
        <source>Failed to start analysis</source>
        <translation>Analyse konnte nicht gestartet werden</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="191"/>
        <source>Failed to process results</source>
        <translation>Ergebnisse konnten nicht verarbeitet werden</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="196"/>
        <source>The analysis stopped with a runtime error:

%1

See the console log for the full traceback.</source>
        <translation>Die Analyse wurde mit einem Laufzeitfehler beendet:

%1

Den vollständigen Traceback finden Sie im Konsolenprotokoll.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="202"/>
        <source>%1

See the console log for the full traceback.</source>
        <translation>%1

Den vollständigen Traceback finden Sie im Konsolenprotokoll.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="207"/>
        <source>Incomplete Region of Interest Coverage</source>
        <translation>Unvollständige Abdeckung durch Regions of Interest</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/pipeline_messages.py" line="212"/>
        <source>This incremental run has %n reference frame(s) with no Region of Interest of their own: frame(s) %1.</source>
        <translation>
            <numerusform>Dieser inkrementelle Lauf hat %n Referenzbild ohne eigene Region of Interest: Bild %1.</numerusform>
            <numerusform>Dieser inkrementelle Lauf hat %n Referenzbilder ohne eigene Region of Interest: Bilder %1.</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/pipeline_messages.py" line="218"/>
        <source>%1 and %n more</source>
        <comment>List of frame numbers cut short: %1 = the frame numbers shown, %n = how many more frames there are</comment>
        <translation>
            <numerusform>%1 und %n weiteres</numerusform>
            <numerusform>%1 und %n weitere</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="224"/>
        <source>A reference frame without a Region of Interest of its own reuses frame 1&apos;s region unchanged: nothing moves the region with the specimen. Where the specimen has moved out of that region, the part outside it is not measured from such a frame, and the nodes there stop being tracked or have their displacement copied from the nearest measured node. The run reports these nodes when it finishes.</source>
        <translation>Ein Referenzbild ohne eigene Region of Interest übernimmt die Region von Bild 1 unverändert: Nichts bewegt die Region mit der Probe mit. Wo sich die Probe aus dieser Region hinausbewegt hat, wird der Teil außerhalb von einem solchen Bild aus nicht gemessen, und die Knoten dort fallen aus der Verfolgung oder übernehmen ihre Verschiebung vom nächstgelegenen gemessenen Knoten. Der Lauf meldet diese Knoten, wenn er abgeschlossen ist.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="234"/>
        <source>To measure the whole specimen, give each reference frame its own Region of Interest that covers the specimen in that frame (Batch Import can assign Regions of Interest to many frames at once). A reference frame with a region of its own is measured on a grid that covers that region.</source>
        <translation>Um die ganze Probe zu messen, geben Sie jedem Referenzbild eine eigene Region of Interest, die die Probe in diesem Bild abdeckt (mit dem Stapelimport lassen sich vielen Bildern auf einmal Regions of Interest zuweisen). Ein Referenzbild mit eigener Region wird auf einem Gitter gemessen, das diese Region abdeckt.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="241"/>
        <source>Continue with frame 1&apos;s region?</source>
        <translation>Mit der Region von Bild 1 fortfahren?</translation>
    </message>
</context>
<context>
    <name>ROIHint</name>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="67"/>
        <source>Load images first, then draw a Region of Interest on frame 1.</source>
        <extracomment>The same card for other hints of the left sidebar (the model advice).</extracomment>
        <translation>Laden Sie zuerst Bilder und zeichnen Sie dann eine Region of Interest auf Bild 1.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="74"/>
        <source>&lt;b&gt;Accumulative mode&lt;/b&gt; — only frame 1 needs a Region of Interest. All later frames are compared against it directly.</source>
        <translation>&lt;b&gt;Akkumulativer Modus&lt;/b&gt; — nur Bild 1 benötigt eine Region of Interest. Alle späteren Bilder werden direkt damit verglichen.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="87"/>
        <source>&lt;b&gt;Incremental&lt;/b&gt; — only frame 1 needs a Region of Interest. RAFTcorr carries it to the reference frames (&lt;b&gt;%1&lt;/b&gt;) with the measured displacement. Draw on another frame only to replace that region or to leave out cracks and holes.</source>
        <translation>&lt;b&gt;Inkrementell&lt;/b&gt; — nur Bild 1 benötigt eine Region of Interest. RAFTcorr überträgt sie mit der gemessenen Verschiebung auf die Referenzbilder (&lt;b&gt;%1&lt;/b&gt;). Zeichnen Sie auf einem anderen Bild nur, um diesen Bereich zu ersetzen oder Risse und Löcher auszusparen.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="96"/>
        <source>Draw a Region of Interest on frame 1.</source>
        <translation>Zeichnen Sie eine Region of Interest auf Bild 1.</translation>
    </message>
</context>
<context>
    <name>ROIToolbar</name>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="72"/>
        <source>+ Add</source>
        <translation>+ Hinzufügen</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="74"/>
        <source>Add region to the Region of Interest (Polygon / Rectangle / Circle)</source>
        <translation>Region zur Region of Interest hinzufügen (Polygon / Rechteck / Kreis)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="79"/>
        <source>Cut</source>
        <translation>Ausschneiden</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="81"/>
        <source>Cut region from the Region of Interest (Polygon / Rectangle / Circle)</source>
        <translation>Region aus der Region of Interest ausschneiden (Polygon / Rechteck / Kreis)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="86"/>
        <source>+ Refine</source>
        <translation>+ Verfeinern</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="90"/>
        <source>Paint extra mesh-refinement zones with a brush
(only on frame 1 — material points auto-warped to later frames)</source>
        <translation>Zusätzliche Netzverfeinerungszonen mit einem Pinsel malen
(nur auf Bild 1 — Materialpunkte werden automatisch auf spätere Bilder übertragen)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="94"/>
        <source>Refine brush is only available on frame 1. Switch to frame 1 to paint refinement zones; they are automatically warped to later frames.</source>
        <translation>Der Verfeinerungspinsel ist nur auf Bild 1 verfügbar. Wechseln Sie zu Bild 1, um Verfeinerungszonen zu malen; sie werden automatisch auf spätere Bilder übertragen.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="118"/>
        <source>Import</source>
        <translation>Importieren</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="119"/>
        <source>Import mask from image file</source>
        <translation>Maske aus Bilddatei importieren</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="124"/>
        <source>Batch Import</source>
        <translation>Stapelimport</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="126"/>
        <source>Batch import mask files for multiple frames</source>
        <translation>Maskendateien für mehrere Bilder stapelweise importieren</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="139"/>
        <source>Save</source>
        <translation>Speichern</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="140"/>
        <source>Save current mask to PNG file</source>
        <translation>Aktuelle Maske als PNG-Datei speichern</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="145"/>
        <source>Invert</source>
        <translation>Invertieren</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="147"/>
        <source>Invert the Region of Interest mask</source>
        <translation>Region-of-Interest-Maske invertieren</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="152"/>
        <source>Clear</source>
        <translation>Löschen</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="154"/>
        <source>Clear all Region of Interest masks</source>
        <translation>Alle Region-of-Interest-Masken löschen</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="186"/>
        <source>Radius</source>
        <translation>Radius</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="204"/>
        <source>Paint</source>
        <translation>Malen</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="209"/>
        <source>Erase</source>
        <translation>Radieren</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="219"/>
        <source>Clear Brush</source>
        <translation>Pinsel löschen</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="247"/>
        <source>Polygon</source>
        <translation>Polygon</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="251"/>
        <source>Rectangle</source>
        <translation>Rechteck</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="255"/>
        <source>Circle</source>
        <translation>Kreis</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="259"/>
        <source>Circle (3-point)</source>
        <translation>Kreis (3 Punkte)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="352"/>
        <source>Import Mask Image</source>
        <translation>Maskenbild importieren</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="354"/>
        <source>Images</source>
        <translation>Bilder</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="355"/>
        <source>All Files</source>
        <translation>Alle Dateien</translation>
    </message>
</context>
<context>
    <name>RightSidebar</name>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="76"/>
        <source>Run DIC Analysis</source>
        <translation>DIC-Analyse ausführen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="89"/>
        <source>Cancel</source>
        <translation>Abbrechen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="94"/>
        <source>Cancel the current analysis. Frames already computed are kept so you can review or export the partial run.</source>
        <translation>Aktuelle Analyse abbrechen. Bereits berechnete Bilder bleiben erhalten, sodass Sie den Teillauf ansehen oder exportieren können.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="102"/>
        <source>Export Results</source>
        <translation>Ergebnisse exportieren</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="110"/>
        <source>Open Strain Window</source>
        <translation>Dehnungsfenster öffnen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="113"/>
        <source>Compute and visualize strain in a separate post-processing window. Requires displacement results from a completed Run.</source>
        <translation>Dehnung in einem separaten Nachbearbeitungsfenster berechnen und visualisieren. Benötigt Verschiebungsergebnisse eines abgeschlossenen Laufs.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="122"/>
        <source>PROGRESS</source>
        <translation>FORTSCHRITT</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="131"/>
        <location filename="../../gui/panels/right_sidebar.py" line="435"/>
        <source>Ready</source>
        <translation>Bereit</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="139"/>
        <location filename="../../gui/panels/right_sidebar.py" line="437"/>
        <location filename="../../gui/panels/right_sidebar.py" line="537"/>
        <source>ELAPSED  %1</source>
        <translation>VERSTRICHEN  %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="144"/>
        <location filename="../../gui/panels/right_sidebar.py" line="439"/>
        <location filename="../../gui/panels/right_sidebar.py" line="545"/>
        <location filename="../../gui/panels/right_sidebar.py" line="549"/>
        <source>REMAINING  %1</source>
        <translation>VERBLEIBEND  %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="152"/>
        <source>FIELD</source>
        <translation>FELD</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="161"/>
        <source>Show on</source>
        <translation>Anzeigen auf</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="166"/>
        <source>Deformed frame</source>
        <translation>Verformtes Bild</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="167"/>
        <source>Reference frame</source>
        <translation>Referenzbild</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="169"/>
        <source>Plot the field where the material points are in the current frame, or at their positions in the reference frame.</source>
        <translation>Das Feld dort zeichnen, wo sich die Materialpunkte im aktuellen Bild befinden, oder an ihren Positionen im Referenzbild.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="215"/>
        <source>Leave points without a result transparent, for example material that left the field of view, instead of filling them with the nearest value inside the region of interest. Also used for exported images and animations; data files always keep NaN there.</source>
        <translation>Punkte ohne Ergebnis (etwa Material, das das Sichtfeld verlassen hat) transparent lassen, statt sie mit dem nächstgelegenen Wert innerhalb der Region of Interest zu füllen. Gilt auch für exportierte Bilder und Animationen; Datendateien behalten dort immer NaN.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="177"/>
        <source>Show background image</source>
        <translation>Hintergrundbild anzeigen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="180"/>
        <source>Uncheck to show the field on its own, with no speckle image behind it.</source>
        <translation>Deaktivieren, um nur das Feld ohne Speckle-Bild dahinter anzuzeigen.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="196"/>
        <source>Hidden background</source>
        <translation>Ausgeblendeter Hintergrund</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="198"/>
        <source>White</source>
        <translation>Weiß</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="199"/>
        <source>Black</source>
        <translation>Schwarz</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="200"/>
        <source>Transparent</source>
        <translation>Transparent</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="203"/>
        <source>What replaces the image when it is hidden. Transparency is kept for PNG and TIFF on export; other formats get white.</source>
        <translation>Was das Bild ersetzt, wenn es ausgeblendet ist. Beim Export bleibt die Transparenz bei PNG und TIFF erhalten; andere Formate erhalten Weiß.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="212"/>
        <source>Hide untracked regions</source>
        <translation>Nicht verfolgte Bereiche ausblenden</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="225"/>
        <source>VISUALIZATION</source>
        <translation>VISUALISIERUNG</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="230"/>
        <source>Colormap</source>
        <translation>Farbskala</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="249"/>
        <source>Opacity</source>
        <translation>Deckkraft</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="256"/>
        <source>Overlay opacity (0 = transparent, 100 = opaque)</source>
        <translation>Deckkraft der Überlagerung (0 = transparent, 100 = deckend)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="262"/>
        <source>PHYSICAL UNITS</source>
        <translation>PHYSIKALISCHE EINHEITEN</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="268"/>
        <source>LOG</source>
        <translation>PROTOKOLL</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="274"/>
        <source>Clear</source>
        <translation>Löschen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="404"/>
        <source>Place at least one Starting Point in each red region before running (red = needs a Starting Point).</source>
        <translation>Platzieren Sie vor dem Ausführen mindestens einen Startpunkt in jeder roten Region (rot = Startpunkt benötigt).</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="520"/>
        <source>%1  —  Frame %2</source>
        <translation>%1  —  Bild %2</translation>
    </message>
</context>
<context>
    <name>StrainController</name>
    <message>
        <location filename="../../gui/controllers/strain_controller.py" line="96"/>
        <source>Strain frame %1/%2</source>
        <translation>Dehnung: Bild %1/%2</translation>
    </message>
</context>
<context>
    <name>StrainFieldSelector</name>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="78"/>
        <source>Magnitude</source>
        <extracomment>GUI field name -&gt; FieldProvider key.</extracomment>
        <translation>Betrag</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="79"/>
        <source>Velocity</source>
        <translation>Geschwindigkeit</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="200"/>
        <source>DISPLACEMENT</source>
        <translation>VERSCHIEBUNG</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="211"/>
        <source>STRAIN</source>
        <translation>DEHNUNG</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="265"/>
        <source>Horizontal displacement u (+ right)</source>
        <translation>Horizontale Verschiebung u (+ nach rechts)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="266"/>
        <source>Vertical displacement v (+ down)</source>
        <translation>Vertikale Verschiebung v (+ nach unten)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="267"/>
        <source>Displacement magnitude |u|</source>
        <translation>Verschiebungsbetrag |u|</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="269"/>
        <source>Speed |du/dt|: central difference of the displacement over the neighbouring computed frames, times the frame rate</source>
        <translation>Geschwindigkeitsbetrag |du/dt|: zentrale Differenz der Verschiebung über die benachbarten berechneten Bilder, mal Bildrate</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="271"/>
        <source>Normal strain εxx</source>
        <translation>Normaldehnung εxx</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="272"/>
        <source>Normal strain εyy</source>
        <translation>Normaldehnung εyy</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="274"/>
        <source>Shear strain εxy (tensor component: half the engineering shear γxy)</source>
        <translation>Scherdehnung εxy (Tensorkomponente: die Hälfte der technischen Scherung γxy)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="276"/>
        <source>Major principal strain ε₁</source>
        <translation>Größte Hauptdehnung ε₁</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="277"/>
        <source>Minor principal strain ε₂</source>
        <translation>Kleinste Hauptdehnung ε₂</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="279"/>
        <source>Maximum in-plane shear strain (ε₁ − ε₂) / 2 (tensor component: half the engineering shear)</source>
        <translation>Maximale Scherdehnung in der Ebene (ε₁ − ε₂) / 2 (Tensorkomponente: die Hälfte der technischen Scherung)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="282"/>
        <source>Von Mises equivalent strain (incompressible material)</source>
        <translation>Von-Mises-Vergleichsdehnung (inkompressibles Material)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="284"/>
        <source>Rigid rotation in degrees (polar decomposition), clockwise on screen positive</source>
        <translation>Starrkörperdrehung in Grad (polare Zerlegung), im Uhrzeigersinn auf dem Bildschirm positiv</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="287"/>
        <source>Direction of ε₁ in degrees from +x, clockwise on screen positive</source>
        <translation>Richtung von ε₁ in Grad ab +x, im Uhrzeigersinn auf dem Bildschirm positiv</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="290"/>
        <source>Relative area change ΔA/A₀ = det F − 1</source>
        <translation>Relative Flächenänderung ΔA/A₀ = det F − 1</translation>
    </message>
</context>
<context>
    <name>StrainNavigator</name>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="73"/>
        <source>Previous frame</source>
        <translation>Vorheriges Bild</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="84"/>
        <location filename="../../gui/widgets/strain_navigator.py" line="216"/>
        <source>Play animation</source>
        <translation>Animation abspielen</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="88"/>
        <location filename="../../gui/widgets/strain_navigator.py" line="215"/>
        <source>▶</source>
        <translation>▶</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="95"/>
        <source>Next frame</source>
        <translation>Nächstes Bild</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="109"/>
        <source>Playback speed</source>
        <translation>Wiedergabegeschwindigkeit</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="114"/>
        <location filename="../../gui/widgets/strain_navigator.py" line="228"/>
        <source>FRAME 0/0</source>
        <translation>BILD 0/0</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="205"/>
        <source>⏸</source>
        <translation>⏸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="206"/>
        <source>Pause animation</source>
        <translation>Animation pausieren</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="223"/>
        <source>FRAME %1/%2</source>
        <translation>BILD %1/%2</translation>
    </message>
</context>
<context>
    <name>StrainParamPanel</name>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="88"/>
        <source>Plane fitting</source>
        <extracomment>Strain measures in combo order (pyALDIC&apos;s order: infinitesimal, Eulerian, Green-Lagrange). The configuration in the controls changed (edited by the user).</extracomment>
        <translation>Ebenenanpassung</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="91"/>
        <source>Polynomial fitted to each displacement component over the VSG window. Its slopes at the centre are the displacement gradients.
• Plane fitting (1st order): the standard VSG (default).
• Quadratic fitting (2nd order): follows strain gradients inside the window better, at the cost of more noise.</source>
        <translation>Polynom, das über das VSG-Fenster an jede Verschiebungskomponente angepasst wird. Seine Steigungen im Zentrum sind die Verschiebungsgradienten.
• Ebenenanpassung (1. Ordnung): das Standard-VSG (Standard).
• Quadratische Anpassung (2. Ordnung): folgt Dehnungsgradienten im Fenster besser, auf Kosten von mehr Rauschen.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="97"/>
        <source>Method</source>
        <translation>Methode</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="112"/>
        <source>VSG size</source>
        <translation>VSG-Größe</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="123"/>
        <source>Hides low-confidence strain along the edges of the region of interest, holes and cracks, where the edge cuts the VSG window and the local fit becomes one-sided and unreliable.

• Coefficient × VSG half-width (Lg − 1) / 2 = width of the trimmed band.
• 0.00 = keep every point (no trimming).
• 0.70 = default.
• 1.00 = strictest (trim every point whose window touches an edge).

Trimmed points are empty (NaN) in exported data files.</source>
        <translation>Blendet Dehnung mit geringer Konfidenz an den Rändern der Region of Interest, an Löchern und Rissen aus, wo der Rand das VSG-Fenster abschneidet und die lokale Anpassung einseitig und unzuverlässig wird.

• Koeffizient × VSG-Halbbreite (Lg − 1) / 2 = Breite des beschnittenen Bands.
• 0.00 = jeden Punkt behalten (kein Beschneiden).
• 0.70 = Standard.
• 1.00 = strengste Einstellung (beschneidet jeden Punkt, dessen Fenster einen Rand berührt).

Beschnittene Punkte sind in exportierten Datendateien leer (NaN).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="154"/>
        <source>Weight of the points inside the VSG window: all equal (uniform, default), or decreasing with the distance from the centre (Gaussian).</source>
        <translation>Gewicht der Punkte im VSG-Fenster: alle gleich (gleichmäßig, Standard) oder mit dem Abstand vom Zentrum abnehmend (Gauß).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="189"/>
        <source>Off</source>
        <comment>Strain smoothing preset</comment>
        <translation>Aus</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="89"/>
        <source>Quadratic fitting</source>
        <translation>Quadratische Anpassung</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="105"/>
        <source>VSG (Virtual Strain Gauge) size Lg is the side, in pixels, of the square window around each point over which the displacement is fitted. Strain is then taken from the fit&apos;s slopes.

• Larger VSG → smoother strain, lower spatial resolution.
• Smaller VSG → sharper strain, more noise.
• Odd, so that the window has a centre pixel (default: 41 px).</source>
        <translation>Die VSG-Größe Lg (Virtual Strain Gauge, virtueller Dehnungsmessstreifen) ist die Seitenlänge in Pixeln des quadratischen Fensters um jeden Punkt, über das die Verschiebung angepasst wird. Die Dehnung ergibt sich dann aus den Steigungen der Anpassung.

• Größeres VSG → glattere Dehnung, geringere räumliche Auflösung.
• Kleineres VSG → schärfere Dehnung, mehr Rauschen.
• Ungerade, damit das Fenster ein Mittelpixel hat (Standard: 41 px).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="133"/>
        <source>Trim low-confidence edges</source>
        <translation>Ränder mit geringer Konfidenz beschneiden</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="151"/>
        <source>Uniform</source>
        <translation>Gleichmäßig</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="152"/>
        <source>Gaussian</source>
        <translation>Gauß</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="158"/>
        <source>Weighting</source>
        <translation>Gewichtung</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="166"/>
        <source>Auto (Lg / 4)</source>
        <translation>Auto (Lg / 4)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="168"/>
        <source>Standard deviation of the Gaussian weight. Auto uses a quarter of the VSG size.</source>
        <translation>Standardabweichung der Gauß-Gewichtung. Auto verwendet ein Viertel der VSG-Größe.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="171"/>
        <source>Gaussian σ</source>
        <translation>Gauß-σ</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="178"/>
        <source>Spacing of the strain points: 1 computes strain at every pixel; larger values are faster and lighter for long sequences.</source>
        <translation>Abstand der Dehnungspunkte: 1 berechnet die Dehnung an jedem Pixel; größere Werte sind für lange Sequenzen schneller und sparsamer.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="181"/>
        <source>Output spacing</source>
        <translation>Ausgabeabstand</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="191"/>
        <source>Gaussian smoothing (σ in pixels) of the displacement used by the strain fit only; the displacement results are not changed. Off (0) differentiates the measured displacement as it is (default).</source>
        <translation>Gaußsche Glättung (σ in Pixeln) nur der Verschiebung, die die Dehnungsanpassung verwendet; die Verschiebungsergebnisse bleiben unverändert. Aus (0) differenziert die gemessene Verschiebung unverändert (Standard).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="195"/>
        <source>Pre-fit smoothing</source>
        <translation>Glättung vor der Anpassung</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="198"/>
        <source>Never fit across cracks or holes</source>
        <translation>Nie über Risse oder Löcher hinweg anpassen</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="200"/>
        <source>Fit each connected part of the tracked region separately, so no VSG window ever spans a crack, a hole or the gap between two regions.</source>
        <translation>Jeden zusammenhängenden Teil des verfolgten Bereichs getrennt anpassen, sodass kein VSG-Fenster je einen Riss, ein Loch oder die Lücke zwischen zwei Bereichen überspannt.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="209"/>
        <source>Infinitesimal</source>
        <translation>Infinitesimal</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="210"/>
        <source>Eulerian</source>
        <translation>Euler</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="211"/>
        <source>Green-Lagrange</source>
        <translation>Green-Lagrange</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="213"/>
        <source>Strain measure from the deformation gradient F = I + ∇u:
• Infinitesimal: (∇u + ∇uᵀ) / 2 (small strains only).
• Eulerian (Euler-Almansi): (I − (F Fᵀ)⁻¹) / 2.
• Green-Lagrange: (Fᵀ F − I) / 2 (default).</source>
        <translation>Dehnungsmaß aus dem Deformationsgradienten F = I + ∇u:
• Infinitesimal: (∇u + ∇uᵀ) / 2 (nur kleine Dehnungen).
• Euler (Euler-Almansi): (I − (F Fᵀ)⁻¹) / 2.
• Green-Lagrange: (Fᵀ F − I) / 2 (Standard).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="218"/>
        <source>Strain type</source>
        <translation>Dehnungstyp</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="272"/>
        <source>Trimmed: %1 points (%2%)</source>
        <translation>Beschnitten: %1 Punkte (%2%)</translation>
    </message>
</context>
<context>
    <name>StrainVizPanel</name>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="67"/>
        <source>Deformed frame</source>
        <extracomment>Alias so existing references read unchanged; the list itself lives in raftcorr.gui.support.colormaps, so every chooser offers the same options.</extracomment>
        <translation>Verformtes Bild</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="68"/>
        <source>Reference frame</source>
        <translation>Referenzbild</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="70"/>
        <source>Plot the field where the material points are in the current frame, or at their positions in the reference frame.</source>
        <translation>Das Feld dort zeichnen, wo sich die Materialpunkte im aktuellen Bild befinden, oder an ihren Positionen im Referenzbild.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="73"/>
        <source>Show on</source>
        <translation>Anzeigen auf</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="75"/>
        <source>Show background image</source>
        <translation>Hintergrundbild anzeigen</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="78"/>
        <source>Uncheck to show the field on its own, with no speckle image behind it.</source>
        <translation>Deaktivieren, um nur das Feld ohne Speckle-Bild dahinter anzuzeigen.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="84"/>
        <source>White</source>
        <translation>Weiß</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="85"/>
        <source>Black</source>
        <translation>Schwarz</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="86"/>
        <source>Transparent</source>
        <translation>Transparent</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="89"/>
        <source>What replaces the image when it is hidden. Transparency is kept for PNG and TIFF on export; other formats get white.</source>
        <translation>Was das Bild ersetzt, wenn es ausgeblendet ist. Beim Export bleibt die Transparenz bei PNG und TIFF erhalten; andere Formate erhalten Weiß.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="141"/>
        <source>Fill trimmed edges (display only)</source>
        <translation>Beschnittene Ränder füllen (nur Anzeige)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="144"/>
        <source>Fill the edge-trimmed strain band from the nearest reliable point of the same region. Affects the view and exported images and animations; exported data files always keep the trimmed band as NaN.</source>
        <translation>Füllt das randbeschnittene Dehnungsband mit dem nächstgelegenen zuverlässigen Punkt desselben Bereichs. Betrifft die Ansicht sowie exportierte Bilder und Animationen; exportierte Datendateien behalten das beschnittene Band immer als NaN.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="151"/>
        <source>Hide untracked regions</source>
        <translation>Nicht verfolgte Bereiche ausblenden</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="154"/>
        <source>Leave points without a result transparent, for example material that left the field of view, instead of filling them with the nearest value inside the region of interest. Also used for exported images and animations; data files always keep NaN there.</source>
        <translation>Punkte ohne Ergebnis (etwa Material, das das Sichtfeld verlassen hat) transparent lassen, statt sie mit dem nächstgelegenen Wert innerhalb der Region of Interest zu füllen. Gilt auch für exportierte Bilder und Animationen; Datendateien behalten dort immer NaN.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="81"/>
        <source>Background</source>
        <translation>Hintergrund</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="93"/>
        <source>Hidden background</source>
        <translation>Ausgeblendeter Hintergrund</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="100"/>
        <source>Colormap</source>
        <translation>Farbskala</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="104"/>
        <source>Range</source>
        <translation>Bereich</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="124"/>
        <source>Min</source>
        <translation>Min</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="126"/>
        <source>Max</source>
        <translation>Max</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="136"/>
        <source>Opacity</source>
        <translation>Deckkraft</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="148"/>
        <source>Edges</source>
        <translation>Ränder</translation>
    </message>
</context>
<context>
    <name>StrainWindow</name>
    <message>
        <location filename="../../gui/strain_window.py" line="125"/>
        <source>Strain Post-Processing</source>
        <translation>Dehnungs-Nachbearbeitung</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="210"/>
        <source>Fit</source>
        <translation>Anpassen</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="211"/>
        <source>Fit image to viewport</source>
        <translation>Bild an den Ansichtsbereich anpassen</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="217"/>
        <source>100%</source>
        <translation>100%</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="218"/>
        <source>Zoom to 100% (1:1)</source>
        <translation>Auf 100% (1:1) zoomen</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="221"/>
        <source>Zoom in</source>
        <translation>Vergrößern</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="226"/>
        <source>–</source>
        <translation>–</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="284"/>
        <source>STRAIN PARAMETERS</source>
        <translation>DEHNUNGSPARAMETER</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="293"/>
        <source>ADVANCED</source>
        <translation>ERWEITERT</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="308"/>
        <source>Cancel</source>
        <translation>Abbrechen</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="312"/>
        <source>Cancel the running strain computation. The previous strain result is kept.</source>
        <translation>Laufende Dehnungsberechnung abbrechen. Das vorherige Dehnungsergebnis bleibt erhalten.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="324"/>
        <source>Export displacement and strain results to NPZ / MAT / CSV / PNG</source>
        <translation>Verschiebungs- und Dehnungsergebnisse als NPZ / MAT / CSV / PNG exportieren</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="354"/>
        <source>FIELD</source>
        <translation>FELD</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="361"/>
        <source>VISUALIZATION</source>
        <translation>VISUALISIERUNG</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="373"/>
        <source>PHYSICAL UNITS</source>
        <translation>PHYSIKALISCHE EINHEITEN</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="379"/>
        <source>LOG</source>
        <translation>PROTOKOLL</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="861"/>
        <location filename="../../gui/strain_window.py" line="899"/>
        <source>No displacement results.</source>
        <translation>Keine Verschiebungsergebnisse.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="901"/>
        <source>Frame %1 was not computed.</source>
        <translation>Bild %1 wurde nicht berechnet.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="909"/>
        <source>Click Compute Strain to show this field.</source>
        <translation>Klicken Sie auf „Dehnung berechnen“, um dieses Feld anzuzeigen.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="910"/>
        <source>No strain for frame %1: click Compute Strain.</source>
        <translation>Keine Dehnung für Bild %1: Klicken Sie auf „Dehnung berechnen“.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="912"/>
        <source>Velocity needs at least two computed frames.</source>
        <translation>Die Geschwindigkeit benötigt mindestens zwei berechnete Bilder.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="913"/>
        <source>No valid values in frame %1.</source>
        <translation>Keine gültigen Werte in Bild %1.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="170"/>
        <source>Strain Field</source>
        <translation>Dehnungsfeld</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="419"/>
        <source>Analysis is not available</source>
        <translation>Analyse ist nicht verfügbar</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="422"/>
        <location filename="../../gui/strain_window.py" line="425"/>
        <source>Analysis</source>
        <translation>Analyse</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="423"/>
        <source>Analysis is not available (see the application log).</source>
        <translation>Analyse ist nicht verfügbar (siehe Anwendungsprotokoll).</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="534"/>
        <source>Strain compute failed: %1: %2</source>
        <translation>Dehnungsberechnung fehlgeschlagen: %1: %2</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="540"/>
        <location filename="../../gui/strain_window.py" line="603"/>
        <source>Strain computation complete.</source>
        <translation>Dehnungsberechnung abgeschlossen.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="551"/>
        <source>Strain window: no displacement results to post-process.</source>
        <translation>Dehnungsfenster: Keine Verschiebungs-Ergebnisse zur Nachbearbeitung.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="596"/>
        <source>The displacement results changed during the strain computation; its result was discarded.</source>
        <translation>Die Verschiebungsergebnisse haben sich während der Dehnungsberechnung geändert; ihr Ergebnis wurde verworfen.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="620"/>
        <source>Strain computation canceled.</source>
        <translation>Dehnungsberechnung abgebrochen.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="627"/>
        <source>Strain compute failed: %1</source>
        <translation>Dehnungsberechnung fehlgeschlagen: %1</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="631"/>
        <source>Strain Computation Failed</source>
        <translation>Dehnungsberechnung fehlgeschlagen</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="671"/>
        <location filename="../../gui/strain_window.py" line="673"/>
        <source>Export is not available: %1</source>
        <translation>Export ist nicht verfügbar: %1</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="685"/>
        <source>⚠ Params changed -- click Compute Strain</source>
        <translation>⚠ Parameter geändert — „Dehnung berechnen“ klicken</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="227"/>
        <source>Zoom out</source>
        <translation>Verkleinern</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="299"/>
        <source>Compute Strain</source>
        <translation>Dehnung berechnen</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="321"/>
        <location filename="../../gui/strain_window.py" line="672"/>
        <source>Export Results</source>
        <translation>Ergebnisse exportieren</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="563"/>
        <source>Starting…</source>
        <translation>Wird gestartet…</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="581"/>
        <source>Canceling…</source>
        <translation>Wird abgebrochen…</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="600"/>
        <source>Complete</source>
        <translation>Abgeschlossen</translation>
    </message>
</context>
<context>
    <name>VelocitySettingsWidget</name>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="47"/>
        <source>Use physical units</source>
        <translation>Physikalische Einheiten verwenden</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="69"/>
        <source>/ px</source>
        <translation>/ px</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="84"/>
        <source>Unit: px/frame</source>
        <translation>Einheit: px/Bild</translation>
    </message>
</context>
<context>
    <name>WorkflowTypePanel</name>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="59"/>
        <source>Incremental</source>
        <translation>Inkrementell</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="58"/>
        <source>Accumulative</source>
        <translation>Akkumulativ</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="73"/>
        <source>Tracking Mode</source>
        <translation>Tracking-Modus</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="64"/>
        <source>Accumulative: every frame is compared directly with frame 1.
Preferred whenever the motion stays within the model&apos;s reach:
no error accumulates from frame to frame.

Incremental: each frame is compared with a later reference frame
and the steps are composed. Needed when the motion from frame 1
exceeds the model&apos;s reach (large stretch or rotation); every step
adds its error.</source>
        <translation>Akkumulativ: Jedes Bild wird direkt mit Bild 1 verglichen.
Zu bevorzugen, solange die Bewegung in der Reichweite des Modells bleibt:
Von Bild zu Bild sammelt sich kein Fehler an.

Inkrementell: Jedes Bild wird mit einem späteren Referenzbild verglichen,
und die Schritte werden verknüpft. Nötig, wenn die Bewegung ab Bild 1
die Reichweite des Modells übersteigt (große Streckung oder Drehung); jeder Schritt
fügt seinen Fehler hinzu.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="85"/>
        <source>finest spatial resolution; small displacements</source>
        <translation>feinste räumliche Auflösung; kleine Verschiebungen</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="86"/>
        <source>fine resolution, several times faster than s1; small to moderate displacements</source>
        <translation>feine Auflösung, mehrfach schneller als s1; kleine bis mittlere Verschiebungen</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="87"/>
        <source>balanced resolution and capture range</source>
        <translation>ausgewogene Auflösung und Reichweite</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="88"/>
        <source>largest capture range, coarsest resolution; large motion and rotation</source>
        <translation>größte Reichweite, gröbste Auflösung; große Bewegung und Drehung</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="92"/>
        <source>%1: %2.
Reach (nominal lookup radius 32 s): %3 px.</source>
        <translation>%1: %2.
Reichweite (nomineller Suchradius 32 s): %3 px.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="95"/>
        <source>Weights not installed.</source>
        <translation>Gewichte nicht installiert.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="100"/>
        <source>RAFTcorr model. A smaller s resolves finer detail; a larger s
reaches further (the coarsest correlation level looks 32 s px away).

s1, s2: small displacements, strain concentrations.
s4: balanced.
s8: large motion or rotation, or no prior knowledge.</source>
        <translation>RAFTcorr-Modell. Ein kleineres s löst feinere Details auf; ein größeres s
reicht weiter (die gröbste Korrelationsebene schaut 32 s px weit).

s1, s2: kleine Verschiebungen, Dehnungskonzentrationen.
s4: ausgewogen.
s8: große Bewegung oder Drehung, oder kein Vorwissen.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="107"/>
        <source>Model</source>
        <translation>Modell</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="114"/>
        <source>Suggest a Model</source>
        <translation>Modell vorschlagen</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="173"/>
        <source>e.g. 5, 10, 20</source>
        <translation>z. B. 5, 10, 20</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="175"/>
        <source>Frame numbers (as in the image list) to use as reference frames, separated by commas. Frame 1 is always the first reference.</source>
        <translation>Bildnummern (wie in der Bildliste) der Referenzbilder, durch Kommas getrennt. Bild 1 ist immer die erste Referenz.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="272"/>
        <source>Invalid reference frames: use frame numbers from 1 on, separated by commas.</source>
        <translation>Ungültige Referenzbilder: Bildnummern ab 1 verwenden, durch Kommas getrennt.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="118"/>
        <source>Measure the motion of the sequence quickly with s8 and pick the
finest model whose reach covers it.</source>
        <translation>Misst die Bewegung der Sequenz schnell mit s8 und wählt das
feinste Modell, dessen Reichweite sie abdeckt.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="140"/>
        <source>Every Frame</source>
        <translation>Jedes Bild</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="141"/>
        <source>Every N Frames</source>
        <translation>Alle N Bilder</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="142"/>
        <source>Custom Frames</source>
        <translation>Benutzerdefiniert</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="145"/>
        <source>When the reference frame refreshes during incremental tracking.
Every Frame: reset reference every frame (smallest per-step displacement,
most robust for large deformation).
Every N Frames: reset every N frames (balance speed vs robustness).
Custom Frames: user-defined list of reference frame indices.</source>
        <translation>Zeitpunkt der Referenzaktualisierung beim inkrementellen Tracking.
Jedes Bild: Referenz bei jedem Bild zurücksetzen (kleinste Schrittverschiebung,
am robustesten für große Verformungen).
Alle N Bilder: alle N Bilder zurücksetzen (Balance zwischen Geschwindigkeit und Robustheit).
Benutzerdefiniert: vom Benutzer festgelegte Liste der Referenzbild-Indizes.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="154"/>
        <source>Reference Update</source>
        <translation>Referenzaktualisierung</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="164"/>
        <source>Update reference every N frames</source>
        <translation>Referenz alle N Bilder aktualisieren</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="166"/>
        <source>Interval</source>
        <translation>Intervall</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="179"/>
        <source>Reference Frames</source>
        <translation>Referenzbilder</translation>
    </message>
</context>
<context>
    <name>_DropZone</name>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="118"/>
        <source>Drop image folder
or Browse</source>
        <translation>Bildordner hier ablegen
oder Durchsuchen</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="128"/>
        <source>Select Image Folder</source>
        <translation>Bildordner auswählen</translation>
    </message>
</context>
<context>
    <name>_MaskPreviewPanel</name>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="133"/>
        <source>Preview</source>
        <translation>Vorschau</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="137"/>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="263"/>
        <source>(no image)</source>
        <translation>(kein Bild)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="157"/>
        <source>Image only</source>
        <translation>Nur Bild</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="158"/>
        <source>Image + Mask</source>
        <translation>Bild + Maske</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="159"/>
        <source>Mask only</source>
        <translation>Nur Maske</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="162"/>
        <source>View:</source>
        <translation>Ansicht:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="168"/>
        <source>Alpha:</source>
        <translation>Alpha:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="176"/>
        <source>Blue</source>
        <comment>Mask overlay color</comment>
        <translation>Blau</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="180"/>
        <source>Red</source>
        <comment>Mask overlay color</comment>
        <translation>Rot</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="184"/>
        <source>Green</source>
        <comment>Mask overlay color</comment>
        <translation>Grün</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="188"/>
        <source>Yellow</source>
        <comment>Mask overlay color</comment>
        <translation>Gelb</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="192"/>
        <source>Mask color:</source>
        <translation>Maskenfarbe:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="269"/>
        <source>No mask assigned</source>
        <translation>Keine Maske zugewiesen</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="275"/>
        <source>Frame %1 — %2</source>
        <translation>Bild %1 — %2</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="290"/>
        <source>Failed to load image</source>
        <translation>Bild konnte nicht geladen werden</translation>
    </message>
</context>
<context>
    <name>dialog</name>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="309"/>
        <source>Cancel Export</source>
        <translation>Export abbrechen</translation>
    </message>
</context>
</TS>
