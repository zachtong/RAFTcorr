<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1" language="fr" sourcelanguage="en_US">
<context>
    <name>AdvancedPanel</name>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="117"/>
        <source>Margin = model reach</source>
        <translation>Marge = portée du modèle</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="120"/>
        <source>Extra image context around the ROI so that displaced points stay inside
the window given to the network. Automatic: the model&apos;s reach (32 s px).</source>
        <translation>Contexte d&apos;image supplémentaire autour de la ROI, pour que les points déplacés restent
dans la fenêtre fournie au réseau. Automatique : la portée du modèle (32 s px).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="128"/>
        <source>Margin</source>
        <translation>Marge</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="129"/>
        <source>Must cover the largest displacement of the ROI.</source>
        <translation>Doit couvrir le plus grand déplacement de la ROI.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="136"/>
        <source>GPU memory</source>
        <translation>Mémoire GPU</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="137"/>
        <source>Fraction of the free GPU memory one window may use.
Large regions are cut into tiles to fit it.</source>
        <translation>Fraction de la mémoire GPU libre qu&apos;une fenêtre peut utiliser.
Les grandes régions sont découpées en tuiles pour s&apos;y conformer.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="144"/>
        <source>Tile blending</source>
        <translation>Fondu des tuiles</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="145"/>
        <source>Width of the feathered overlap between neighbouring tiles.</source>
        <translation>Largeur du chevauchement en fondu entre tuiles voisines.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="148"/>
        <source>Automatic</source>
        <translation>Automatique</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="149"/>
        <source>On demand</source>
        <translation>À la demande</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="150"/>
        <source>Stored volume</source>
        <translation>Volume stocké</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="152"/>
        <source>Correlation</source>
        <translation>Corrélation</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="153"/>
        <source>How the correlation lookup is computed; the results are identical.
Automatic: stored volume for s8 (fast), on demand for s1-s4
(memory linear in the image size).</source>
        <translation>Mode de calcul de la recherche de corrélation ; les résultats sont identiques.
Automatique : volume stocké pour s8 (rapide), à la demande pour s1-s4
(mémoire proportionnelle à la taille de l&apos;image).</translation>
    </message>
</context>
<context>
    <name>AnalysisChart</name>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="75"/>
        <source>too few valid points</source>
        <translation>trop peu de points valides</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="73"/>
        <source>partly untracked</source>
        <translation>en partie hors suivi</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="77"/>
        <source>no strain value (sparse strain gauge)</source>
        <translation>aucune valeur de déformation (trop peu de points dans le VSG)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="80"/>
        <source>gauge endpoint lost</source>
        <translation>extrémité de jauge perdue</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="82"/>
        <source>not computed</source>
        <translation>non calculé</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="83"/>
        <source>no data</source>
        <translation>aucune donnée</translation>
    </message>
</context>
<context>
    <name>AnalysisTab</name>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="82"/>
        <source>Point</source>
        <comment>Placement tool: a single location</comment>
        <translation>Point</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="84"/>
        <source>Line</source>
        <comment>Placement tool: a two-point gauge</comment>
        <translation>Ligne</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="86"/>
        <source>Rectangle</source>
        <comment>Placement tool</comment>
        <translation>Rectangle</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="88"/>
        <source>Circle</source>
        <comment>Placement tool</comment>
        <translation>Cercle</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="90"/>
        <source>Polygon</source>
        <comment>Placement tool</comment>
        <translation>Polygone</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="102"/>
        <source>Click once to place a point probe.</source>
        <translation>Cliquez une fois pour placer une sonde ponctuelle.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="108"/>
        <source>Click twice: opposite corners.</source>
        <translation>Cliquez deux fois : coins opposés.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="112"/>
        <source>Click each vertex, then double-click to close.</source>
        <translation>Cliquez sur chaque sommet, puis double-cliquez pour fermer.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="170"/>
        <source>partly untracked from frame %1</source>
        <translation>en partie hors suivi dès l&apos;image %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="179"/>
        <source>gaps: no strain value</source>
        <translation>lacunes : aucune valeur de déformation</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="201"/>
        <source>Nothing valid along %1: the strain gauge found too few points there (near an edge or a hole). Plot a displacement, or use a smaller gauge on the Strain Field tab.</source>
        <translation>Aucune donnée valide le long de %1 : le VSG y trouve trop peu de points (près d&apos;un bord ou d&apos;un trou). Tracez un déplacement, ou utilisez un VSG plus petit dans l&apos;onglet « Champ de déformation ».</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="207"/>
        <source>Nothing valid along %1: the material under it is untracked (cut out of the frame&apos;s ROI, or out of view).</source>
        <translation>Aucune donnée valide le long de %1 : le matériau en dessous n&apos;est pas suivi (exclu de la ROI de l&apos;image, ou hors du champ de vision).</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="89"/>
        <source>Show</source>
        <comment>Probe list column: visibility checkbox</comment>
        <translation>Afficher</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="91"/>
        <source>Name</source>
        <comment>Probe list column: the probe&apos;s label</comment>
        <translation>Nom</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="93"/>
        <source>Type</source>
        <comment>Probe list column: point, line or region</comment>
        <translation>Type</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="95"/>
        <source>Color</source>
        <comment>Probe list column: color swatch</comment>
        <translation>Couleur</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="100"/>
        <source>Color…</source>
        <translation>Couleur…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="97"/>
        <source>Note</source>
        <comment>Probe list column: why a probe shows gaps</comment>
        <translation>Remarque</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="101"/>
        <source>Delete</source>
        <comment>Button: delete the selected probe</comment>
        <translation>Supprimer</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="103"/>
        <source>Clear All</source>
        <translation>Tout effacer</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="183"/>
        <source>Statistic:</source>
        <translation>Statistique :</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="56"/>
        <source>Mean</source>
        <comment>Statistic: arithmetic mean</comment>
        <translation>Moyenne</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="58"/>
        <source>Median</source>
        <comment>Statistic</comment>
        <translation>Médiane</translation>
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
        <translation>Écart-type</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="62"/>
        <source>Valid fraction</source>
        <translation>Fraction valide</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="46"/>
        <source>Crack opening</source>
        <translation>Ouverture de fissure</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="489"/>
        <source>Strain has not been computed yet. Compute it on the Strain Field tab, or plot a displacement.</source>
        <translation>La déformation n&apos;a pas encore été calculée. Calculez-la dans l&apos;onglet « Champ de déformation » ou affichez un déplacement.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="533"/>
        <source>Gauge quantities need a line probe.</source>
        <translation>Les grandeurs de jauge nécessitent une sonde linéaire.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="530"/>
        <source>No visible probe can show this quantity.</source>
        <translation>Aucune sonde visible ne peut afficher cette grandeur.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="166"/>
        <source>no valid data: %1</source>
        <translation>aucune donnée valide : %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="174"/>
        <source>endpoint lost from frame %1</source>
        <translation>extrémité perdue dès l&apos;image %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="177"/>
        <source>gaps: too few valid points</source>
        <translation>lacunes : trop peu de points valides</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="185"/>
        <source>not plotted: gauges need a line</source>
        <translation>non tracée : les jauges nécessitent une ligne</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="188"/>
        <source>not plotted: one point has no spread or coverage</source>
        <translation>non tracée : un seul point n&apos;a ni dispersion ni couverture</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="190"/>
        <source>not plotted</source>
        <translation>non tracée</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="513"/>
        <source>Added probe &apos;%1&apos;.</source>
        <translation>Sonde « %1 » ajoutée.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="599"/>
        <source>Clear All Probes</source>
        <translation>Effacer toutes les sondes</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="600"/>
        <source>Delete every probe? This cannot be undone.</source>
        <translation>Supprimer toutes les sondes ? Cette action est irréversible.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="110"/>
        <source>Point</source>
        <comment>Probe type</comment>
        <translation>Point</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="111"/>
        <source>Line</source>
        <comment>Probe type</comment>
        <translation>Ligne</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="112"/>
        <source>Region</source>
        <comment>Probe type: an enclosed area</comment>
        <translation>Région</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="480"/>
        <source>Run a DIC analysis to plot probes.</source>
        <translation>Exécutez une analyse DIC pour tracer les sondes.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="92"/>
        <source>Virtual extensometer</source>
        <comment>Placement tool</comment>
        <translation>Extensomètre virtuel</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="94"/>
        <source>Crack gauge</source>
        <comment>Placement tool: a line across a crack</comment>
        <translation>Jauge de fissure</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="104"/>
        <source>Click twice: start and end. A line is also a virtual extensometer and a crack-opening gauge.</source>
        <translation>Cliquez deux fois : début et fin. Une ligne est aussi un extensomètre virtuel et une jauge d&apos;ouverture de fissure.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="110"/>
        <source>Click twice: center, then the edge.</source>
        <translation>Cliquez deux fois : centre, puis bord.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="114"/>
        <source>Click the two gauge points. The chart then shows the strain between them.</source>
        <translation>Cliquez sur les deux points de mesure. Le graphique affiche alors la déformation entre eux.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="118"/>
        <source>Click one point on each side of the crack. The chart then shows how far it opens.</source>
        <translation>Cliquez sur un point de chaque côté de la fissure. Le graphique affiche alors son ouverture.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="124"/>
        <source>Fit</source>
        <comment>Zoom button: fit the image to the view</comment>
        <translation>Ajuster</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="131"/>
        <source>Fit image to viewport</source>
        <translation>Ajuster l&apos;image à la vue</translation>
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
        <translation>Zoomer à 100% (1:1)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="136"/>
        <source>Zoom in</source>
        <translation>Zoom avant</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="137"/>
        <source>Zoom out</source>
        <translation>Zoom arrière</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="138"/>
        <source>Show field</source>
        <comment>Analysis canvas: color the image by the field</comment>
        <translation>Afficher le champ</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="140"/>
        <source>Color the reference image with the plotted field at the current frame. For a gauge reading, the Strain Field tab&apos;s field is shown.</source>
        <translation>Colore l&apos;image de référence avec le champ tracé à l&apos;image courante. Pour une lecture de jauge, le champ de l&apos;onglet « Champ de déformation » est affiché.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="181"/>
        <source>Plot:</source>
        <translation>Tracer :</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="184"/>
        <source>X axis:</source>
        <translation>Axe X :</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="185"/>
        <source>Strain as:</source>
        <translation>Déformation en :</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="187"/>
        <source>Min. valid fraction:</source>
        <translation>Fraction valide min. :</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="188"/>
        <source>A frame is left blank when fewer than this fraction of a line&apos;s or region&apos;s points are reliable. Guards against a curve that stays smooth while its sample shrinks away.</source>
        <translation>Une image reste vide lorsque moins de cette fraction des points d&apos;une ligne ou d&apos;une région est fiable. Évite une courbe qui reste lisse alors que son échantillon se réduit.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="195"/>
        <source>Over time</source>
        <comment>Chart view: every frame of each probe</comment>
        <translation>Évolution temporelle</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="197"/>
        <source>Each probe&apos;s reading at every frame.</source>
        <translation>La lecture de chaque sonde à chaque image.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="200"/>
        <source>Along the line</source>
        <comment>Chart view: a profile</comment>
        <translation>Le long de la ligne</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="207"/>
        <source>Kymograph</source>
        <comment>Chart view: distance against frame</comment>
        <translation>Kymographe</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="202"/>
        <source>The field along the selected line at the current frame, over the other frames in gray.</source>
        <translation>Le champ le long de la ligne sélectionnée à l&apos;image courante, sur les autres images en gris.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="209"/>
        <source>The field along the selected line at every frame: distance against frame, value as color.</source>
        <translation>Le champ le long de la ligne sélectionnée à chaque image : distance en fonction de l&apos;image, valeur en couleur.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="214"/>
        <source>Stress–strain</source>
        <comment>Chart view: stress against strain</comment>
        <translation>Contrainte–déformation</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="216"/>
        <source>Stress (or load, without A0) against the plotted quantity, one curve per probe: stress-strain with an extensometer, load against opening with a crack gauge. Needs load data.</source>
        <translation>Contrainte (ou charge, sans A0) en fonction de la grandeur tracée, une courbe par sonde : contrainte-déformation avec un extensomètre, charge-ouverture avec une jauge de fissure. Nécessite des données de charge.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="227"/>
        <source>Other frames</source>
        <translation>Autres images</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="228"/>
        <source>Draw the other frames&apos; profiles faintly behind the current one (at most twelve, evenly spaced).</source>
        <translation>Tracer en pâle les profils des autres images derrière l&apos;image courante (au plus douze, régulièrement espacées).</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="233"/>
        <source>Line data (CSV)…</source>
        <translation>Données de ligne (CSV)…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="234"/>
        <source>Export</source>
        <translation>Exporter</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="235"/>
        <source>Load data…</source>
        <translation>Données de charge…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="236"/>
        <source>Import a testing machine&apos;s load record (CSV) to plot against load or stress, and to draw stress-strain curves.</source>
        <translation>Importer l&apos;enregistrement de charge d&apos;une machine d&apos;essai (CSV) pour tracer en fonction de la charge ou de la contrainte, et tracer des courbes contrainte-déformation.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="241"/>
        <source>Probe data (CSV)…</source>
        <translation>Données de sonde (CSV)…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="243"/>
        <source>Chart image…</source>
        <translation>Image du graphique…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="245"/>
        <source>Copy chart</source>
        <translation>Copier le graphique</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="247"/>
        <source>Copy plotted data</source>
        <translation>Copier les données tracées</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="559"/>
        <source>Import the testing machine&apos;s load record with Load data… to draw stress-strain curves.</source>
        <translation>Importez l&apos;enregistrement de charge de la machine d&apos;essai via « Données de charge… » pour tracer des courbes contrainte-déformation.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="31"/>
        <source>Displacement U</source>
        <translation>Déplacement U</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="32"/>
        <source>Displacement V</source>
        <translation>Déplacement V</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="33"/>
        <source>Displacement magnitude</source>
        <translation>Norme du déplacement</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="42"/>
        <source>Extensometer strain</source>
        <translation>Déformation d&apos;extensomètre</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="43"/>
        <source>Extensometer true strain</source>
        <translation>Déformation vraie d&apos;extensomètre</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="45"/>
        <source>Elongation ΔL</source>
        <translation>Allongement ΔL</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="47"/>
        <source>Crack sliding</source>
        <translation>Glissement de fissure</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="48"/>
        <source>Crack opening magnitude</source>
        <translation>Norme de l&apos;ouverture de fissure</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="279"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="434"/>
        <source>Frame</source>
        <translation>Image</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="281"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="429"/>
        <source>Time (s)</source>
        <translation>Temps (s)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="283"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="431"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="574"/>
        <source>Load (N)</source>
        <translation>Charge (N)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="286"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="433"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="571"/>
        <source>Stress (MPa)</source>
        <translation>Contrainte (MPa)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="295"/>
        <source>ratio</source>
        <comment>Strain display unit: plain number</comment>
        <translation>rapport</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="420"/>
        <source>Could not draw the field: %1</source>
        <translation>Impossible de dessiner le champ : %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="130"/>
        <source>Esc cancels placement</source>
        <translation>Échap annule le placement</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="138"/>
        <source>Drag to move the probe, or drag a handle to reshape it. Delete removes it; F2 renames it.</source>
        <translation>Faites glisser pour déplacer la sonde, ou une poignée pour la remodeler. Suppr la supprime ; F2 la renomme.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="485"/>
        <source>Place a probe on the reference image to begin.</source>
        <translation>Placez une sonde sur l&apos;image de référence pour commencer.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="212"/>
        <source>Nothing valid along %1: it lies off the measured area.</source>
        <translation>Aucune donnée valide le long de %1 : elle est hors de la zone mesurée.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="610"/>
        <source>A line view shows a field. Choose a field to plot.</source>
        <translation>Une vue de ligne affiche un champ. Choisissez un champ à tracer.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="614"/>
        <source>Place a line probe, or select one, to see the field along it.</source>
        <translation>Placez ou sélectionnez une sonde linéaire pour voir le champ le long de celle-ci.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="632"/>
        <source>Distance along %1 (%2)</source>
        <translation>Distance le long de %1 (%2)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="641"/>
        <source>%1, frame %2</source>
        <translation>%1, image %2</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="153"/>
        <source>not plotted: off the measured area</source>
        <translation>non tracée : hors de la zone mesurée</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="156"/>
        <source>not plotted: a gauge end is off the measured area</source>
        <translation>non tracée : une extrémité de la jauge est hors de la zone mesurée</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="671"/>
        <source>Export Probe Data</source>
        <translation>Exporter les données de sonde</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="664"/>
        <source>CSV Files</source>
        <translation>Fichiers CSV</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="442"/>
        <source>The load data cannot be matched to the frames: %1</source>
        <translation>Les données de charge ne peuvent pas être associées aux images : %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="664"/>
        <location filename="../../gui/panels/analysis/tab.py" line="728"/>
        <source>All Files</source>
        <translation>Tous les fichiers</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="699"/>
        <source>Probe export failed: %1</source>
        <translation>Échec de l&apos;export de la sonde : %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="701"/>
        <source>Probe data written to %1</source>
        <translation>Données de sonde écrites dans %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="709"/>
        <source>Export Line Data</source>
        <translation>Exporter les données de ligne</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="718"/>
        <source>Line export failed: %1</source>
        <translation>Échec de l&apos;export de la ligne : %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="720"/>
        <source>Line data written to %1</source>
        <translation>Données de ligne écrites dans %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="726"/>
        <source>SVG Images</source>
        <translation>Images SVG</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="737"/>
        <source>Export Chart</source>
        <translation>Exporter le graphique</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="756"/>
        <source>Chart copied to the clipboard.</source>
        <translation>Graphique copié dans le presse-papiers.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="764"/>
        <source>Plotted data copied to the clipboard.</source>
        <translation>Données tracées copiées dans le presse-papiers.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="725"/>
        <source>PNG Images</source>
        <translation>Images PNG</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="727"/>
        <source>PDF Documents</source>
        <translation>Documents PDF</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="748"/>
        <source>Chart export failed: %1</source>
        <translation>Échec de l&apos;export du graphique : %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="750"/>
        <source>Chart written to %1</source>
        <translation>Graphique écrit dans %1</translation>
    </message>
</context>
<context>
    <name>App</name>
    <message>
        <location filename="../../gui/app.py" line="713"/>
        <source>Run DIC first -- no displacement results to post-process.</source>
        <translation>Exécutez d&apos;abord le DIC — aucun résultat de déplacement à post-traiter.</translation>
    </message>
</context>
<context>
    <name>Application</name>
    <message>
        <location filename="../../gui/app.py" line="846"/>
        <source>RAFTcorr has hit an error</source>
        <translation>RAFTcorr a rencontré une erreur</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="847"/>
        <source>An unexpected error occurred. The application may not behave correctly from here on, so saving your session and restarting is recommended.</source>
        <translation>Une erreur inattendue s&apos;est produite. L&apos;application risque de ne plus fonctionner correctement ; il est recommandé d&apos;enregistrer la session et de redémarrer.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="857"/>
        <source>Details were written to %1</source>
        <translation>Les détails ont été écrits dans %1</translation>
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
        <translation>Ajuster la plage de couleurs à la plage de données de chaque image</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/range_mode.py" line="39"/>
        <source>Fixed</source>
        <comment>Color range mode: manual min/max bounds</comment>
        <translation>Fixe</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/range_mode.py" line="42"/>
        <source>Keep the manual Min/Max bounds for every frame</source>
        <translation>Conserver les bornes Min/Max manuelles pour toutes les images</translation>
    </message>
</context>
<context>
    <name>BatchImportDialog</name>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="368"/>
        <source>Batch Import Region of Interest Masks</source>
        <translation>Import par lot des masques de région d&apos;intérêt</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="419"/>
        <source>Mask Folder:</source>
        <translation>Dossier de masques :</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="420"/>
        <source>(none)</source>
        <translation>(aucun)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="425"/>
        <source>Browse...</source>
        <translation>Parcourir…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="444"/>
        <source>Available Masks</source>
        <translation>Masques disponibles</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="455"/>
        <source>Auto-Match by Name</source>
        <translation>Correspondance auto par nom</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="457"/>
        <source>Match mask files to frames by number in filename</source>
        <translation>Associer les fichiers de masque aux images d&apos;après le numéro du nom de fichier</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="461"/>
        <source>Assign Sequential</source>
        <translation>Attribuer séquentiellement</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="463"/>
        <source>Assign masks to frames in order starting from frame 0</source>
        <translation>Attribuer les masques aux images dans l&apos;ordre à partir de l&apos;image 0</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="471"/>
        <source>Frame Assignments</source>
        <translation>Attributions d&apos;images</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="474"/>
        <source>Frame</source>
        <translation>Image</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="474"/>
        <source>Image</source>
        <translation>Fichier image</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="474"/>
        <source>Mask</source>
        <translation>Masque</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="485"/>
        <source>Assign Selected -&gt;</source>
        <translation>Attribuer la sélection -&gt;</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="487"/>
        <source>Pair selected mask(s) with selected frame(s)</source>
        <translation>Associer les masques sélectionnés aux images sélectionnées</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="491"/>
        <source>Clear All</source>
        <translation>Tout effacer</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="536"/>
        <source>Select Mask Folder</source>
        <translation>Sélectionner le dossier de masques</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="590"/>
        <source>Failed to read mask file.</source>
        <translation>Échec de la lecture du fichier de masque.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="594"/>
        <source>Mismatched shape: %1×%2 (expected %3×%4)</source>
        <translation>Forme incompatible : %1×%2 (attendu %3×%4)</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="666"/>
        <source>%n mask(s) have mismatched sizes and are disabled.</source>
        <translation>
            <numerusform>%n masque a une taille incompatible et est désactivé.</numerusform>
            <numerusform>%n masques ont des tailles incompatibles et sont désactivés.</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="719"/>
        <source>Invalid assignment</source>
        <translation>Affectation non valide</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="721"/>
        <source>A frame can only have one mask. Select exactly one mask, or select multiple frames to assign one mask to many.</source>
        <translation>Une image ne peut avoir qu&apos;un seul masque. Sélectionnez exactement un masque, ou sélectionnez plusieurs images pour attribuer un masque à plusieurs.</translation>
    </message>
</context>
<context>
    <name>CanvasArea</name>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1222"/>
        <source>Fit</source>
        <translation>Ajuster</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1223"/>
        <source>Fit image to viewport</source>
        <translation>Ajuster l&apos;image à la vue</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1227"/>
        <source>100%</source>
        <translation>100%</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1228"/>
        <source>Zoom to 100% (1:1)</source>
        <translation>Zoomer à 100% (1:1)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1231"/>
        <source>Zoom in</source>
        <translation>Zoom avant</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1236"/>
        <source>–</source>
        <translation>–</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1237"/>
        <source>Zoom out</source>
        <translation>Zoom arrière</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1321"/>
        <source>Editing the ROI of frame 1 (the reference)</source>
        <translation>Modification de la ROI de l&apos;image 1 (la référence)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1322"/>
        <source>Editing the mask of frame %1 (regions left out are excluded)</source>
        <translation>Modification du masque de l&apos;image %1 (les zones laissées de côté sont exclues)</translation>
    </message>
</context>
<context>
    <name>ColorRange</name>
    <message>
        <location filename="../../gui/widgets/color_range.py" line="27"/>
        <source>Range</source>
        <translation>Plage</translation>
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
        <translation>Norme</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="46"/>
        <source>Residual</source>
        <translation>Résidu</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="66"/>
        <source>Open this tab to render a preview.</source>
        <translation>Ouvrez cet onglet pour générer un aperçu.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="73"/>
        <location filename="../../gui/dialogs/export_widgets.py" line="368"/>
        <source>Field</source>
        <translation>Champ</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="80"/>
        <source>Frame</source>
        <translation>Image</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="101"/>
        <source>FIELD APPEARANCE</source>
        <translation>APPARENCE DU CHAMP</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="106"/>
        <location filename="../../gui/dialogs/export_widgets.py" line="368"/>
        <source>Colormap</source>
        <translation>Palette</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="110"/>
        <source>Range</source>
        <translation>Plage</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="114"/>
        <location filename="../../gui/dialogs/export_widgets.py" line="368"/>
        <source>Min</source>
        <translation>Min</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="118"/>
        <location filename="../../gui/dialogs/export_widgets.py" line="368"/>
        <source>Max</source>
        <translation>Max</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="125"/>
        <location filename="../../gui/dialogs/export_widgets.py" line="174"/>
        <source>Opacity</source>
        <extracomment>Column widths of the field rows (export, field, colormap, range mode, min / max), shared by the header labels of the Images and Animation tabs. Older names that share a field&apos;s colour state.</extracomment>
        <translation>Opacité</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="127"/>
        <source>Apply to all fields</source>
        <translation>Appliquer à tous les champs</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="128"/>
        <source>Apply this field&apos;s colormap, opacity and auto-range to every enabled field (each field keeps its own min/max).</source>
        <translation>Applique la palette, l&apos;opacité et la plage auto de ce champ à tous les champs activés (chaque champ garde ses propres min/max).</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="133"/>
        <source>Refresh preview</source>
        <translation>Actualiser l&apos;aperçu</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="160"/>
        <source>Preview failed: </source>
        <translation>Échec de l&apos;aperçu : </translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="194"/>
        <source>Enable a field on the Images tab to preview.</source>
        <translation>Activez un champ dans l&apos;onglet « Images » pour l&apos;aperçu.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="198"/>
        <source>No data for this field/frame.</source>
        <translation>Aucune donnée pour ce champ/cette image.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="175"/>
        <source>Field opacity (0 = transparent, 1 = fully opaque)</source>
        <translation>Opacité du champ (0 = transparent, 1 = opaque)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="282"/>
        <source>All</source>
        <translation>Tous</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="284"/>
        <source>None</source>
        <translation>Aucun</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="355"/>
        <source>Select:</source>
        <translation>Sélectionner :</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="368"/>
        <source>Export</source>
        <translation>Exporter</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="368"/>
        <source>Auto</source>
        <translation>Auto</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="385"/>
        <source>PHYSICAL UNITS</source>
        <translation>UNITÉS PHYSIQUES</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="389"/>
        <source>Enable physical units</source>
        <translation>Activer les unités physiques</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="391"/>
        <source>Scale displacement values by pixel size and show physical units on colorbar labels. Strain is dimensionless and unaffected.</source>
        <translation>Met à l&apos;échelle les valeurs de déplacement par la taille de pixel et affiche les unités physiques sur les étiquettes de la barre de couleur. La déformation est sans dimension et n&apos;est pas affectée.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="408"/>
        <source>/ pixel</source>
        <translation>/ pixel</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="410"/>
        <source>Pixel size</source>
        <translation>Taille de pixel</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="419"/>
        <source>fps</source>
        <translation>fps</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="421"/>
        <source>Frame rate</source>
        <translation>Fréquence d&apos;images</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="447"/>
        <source>FRAME RANGE</source>
        <translation>PLAGE D&apos;IMAGES</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="449"/>
        <source>All frames</source>
        <translation>Toutes les images</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="454"/>
        <source>Numbered like the frame slider: frame 1 is the reference image.
File names use the same numbers: frame 1 is written as _0001.</source>
        <translation>Numérotées comme le curseur d&apos;images : l&apos;image 1 est l&apos;image de référence.
Les noms de fichiers utilisent les mêmes numéros : l&apos;image 1 est écrite _0001.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="457"/>
        <source>From</source>
        <comment>Frame range: starting frame</comment>
        <translation>De</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="462"/>
        <source>to</source>
        <comment>Frame range: ending frame</comment>
        <translation>à</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="161"/>
        <source>Export Results</source>
        <extracomment>GUI field name -&gt; variable name in the data files (raftcorr.export.scientific).</extracomment>
        <translation>Exporter les résultats</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="188"/>
        <source>OUTPUT FOLDER</source>
        <translation>DOSSIER DE SORTIE</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="194"/>
        <source>Select output folder…</source>
        <translation>Sélectionner le dossier de sortie…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="200"/>
        <source>Browse…</source>
        <translation>Parcourir…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="205"/>
        <source>Open Folder</source>
        <translation>Ouvrir le dossier</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="219"/>
        <source>Data</source>
        <translation>Données</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="220"/>
        <source>Images</source>
        <translation>Images</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="221"/>
        <source>Animation</source>
        <translation>Animation</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="222"/>
        <source>Report</source>
        <translation>Rapport</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="304"/>
        <source>FORMAT</source>
        <translation>FORMAT</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="306"/>
        <source>NumPy Archive (.npz)</source>
        <translation>Archive NumPy (.npz)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="313"/>
        <source>CSV (per frame)</source>
        <translation>CSV (par image)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="323"/>
        <source>DISPLACEMENT</source>
        <translation>DÉPLACEMENT</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="326"/>
        <source>STRAIN</source>
        <translation>DÉFORMATION</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="271"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="329"/>
        <source>Run Compute Strain first.</source>
        <translation>Exécutez d&apos;abord « Calculer la déformation ».</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="223"/>
        <source>Preview</source>
        <translation>Aperçu</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="252"/>
        <source>Export: images not readable (%1); pictures go on white.</source>
        <translation>Export : images illisibles (%1) ; les figures sont tracées sur fond blanc.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="256"/>
        <source>Export: the images do not match the results&apos; image size; pictures go on white.</source>
        <translation>Export : les images ne correspondent pas à la taille d&apos;image des résultats ; les figures sont tracées sur fond blanc.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="270"/>
        <source>This run has no intensity residual.</source>
        <translation>Cette exécution n&apos;a pas de résidu d&apos;intensité.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="308"/>
        <source>MATLAB v7.3 (.mat)</source>
        <translation>MATLAB v7.3 (.mat)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="310"/>
        <source>An HDF5-based MAT file: load it in MATLAB with load().</source>
        <translation>Un fichier MAT basé sur HDF5 : chargez-le dans MATLAB avec load().</translation>
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
        <translation>Une table par image calculée, une ligne par point de la région d&apos;intérêt.
Les champs denses produisent de gros fichiers : une région de 1000 x 1000 px fait un million de lignes par image.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="333"/>
        <source>✓ Parameters file (JSON) always exported</source>
        <translation>✓ Le fichier de paramètres (JSON) est toujours exporté</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="334"/>
        <source>✓ Grid coordinates (X, Y), ROI and computed-frame flags always included</source>
        <translation>✓ Coordonnées de grille (X, Y), ROI et indicateurs d&apos;images calculées toujours inclus</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="335"/>
        <source>Displacements are written in pixels (+u right, +v down); the pixel size and frame rate above are recorded with them.</source>
        <translation>Les déplacements sont écrits en pixels (+u vers la droite, +v vers le bas) ; la taille de pixel et la fréquence d&apos;images ci-dessus sont enregistrées avec eux.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="341"/>
        <source>Export Data</source>
        <translation>Exporter les données</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="377"/>
        <source>IMAGE SETTINGS</source>
        <translation>PARAMÈTRES D&apos;IMAGE</translation>
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
        <translation>Image déformée</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="352"/>
        <source>Reference frame</source>
        <translation>Image de référence</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="358"/>
        <source>Show background image</source>
        <translation>Afficher l&apos;image de fond</translation>
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
        <translation>Inclure la barre de couleur</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="421"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="496"/>
        <source>Render as</source>
        <translation>Rendu</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="431"/>
        <source>Export Images</source>
        <translation>Exporter les images</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="445"/>
        <source>ANIMATION SETTINGS</source>
        <translation>PARAMÈTRES D&apos;ANIMATION</translation>
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
        <translation>Exporte une image sur N (1 = toutes les images). Plus élevé = plus rapide et plus léger,
mais plus saccadé. La durée de lecture est conservée (les FPS ci-dessus sont la fréquence d&apos;images avant décimation).</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="488"/>
        <source>Frame step</source>
        <translation>Pas d&apos;image</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="511"/>
        <source>Export Animation</source>
        <translation>Exporter l&apos;animation</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="521"/>
        <source>CONTENT</source>
        <translation>CONTENU</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="523"/>
        <source>Parameter summary table</source>
        <translation>Tableau récapitulatif des paramètres</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="525"/>
        <source>Field statistics (min/max/mean/std per frame)</source>
        <translation>Statistiques de champ (min/max/moyenne/écart-type par image)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="527"/>
        <source>Sample field images</source>
        <translation>Exemples d&apos;images de champ</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="535"/>
        <source>Sample every</source>
        <translation>Échantillonner toutes les</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="541"/>
        <source>frames</source>
        <comment>Report: sample every N frames</comment>
        <translation>images</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="547"/>
        <source>FIELDS</source>
        <translation>CHAMPS</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="549"/>
        <source>Displacement:</source>
        <translation>Déplacement :</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="550"/>
        <source>Strain:</source>
        <translation>Déformation :</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="559"/>
        <source>Format: HTML (self-contained, view in any browser)</source>
        <translation>Format : HTML (autonome, consultable dans n&apos;importe quel navigateur)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="563"/>
        <source>Generate Report</source>
        <translation>Générer le rapport</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="706"/>
        <source>Select Output Folder</source>
        <translation>Sélectionner le dossier de sortie</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="807"/>
        <source>Writing %1 (%2%)</source>
        <translation>Écriture de %1 (%2%)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="840"/>
        <source>Error: %1</source>
        <translation>Erreur : %1</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="782"/>
        <source>Starting…</source>
        <translation>Démarrage…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="355"/>
        <source>Deformed: the field is drawn where each point is in that frame, over the frame&apos;s own photo.
Reference: drawn at the points&apos; positions in frame 1, over frame 1.</source>
        <translation>Image déformée : le champ est tracé là où se trouve chaque point dans cette image, par-dessus la photo propre à l&apos;image.
Image de référence : tracé aux positions des points dans l&apos;image 1, par-dessus l&apos;image 1.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="362"/>
        <source>Uncheck to export the field on its own, on white, with no speckle image behind it.</source>
        <translation>Décochez pour n&apos;exporter que le champ, sur fond blanc, sans image de mouchetis derrière.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="363"/>
        <source>The images of this run are not available.</source>
        <translation>Les images de cette exécution ne sont pas disponibles.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="395"/>
        <source>Resolution of the written figure: about 8.6 x 7 inches at this many dots per inch.</source>
        <translation>Résolution de la figure écrite : environ 8,6 x 7 pouces à ce nombre de points par pouce.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="402"/>
        <source>Each frame</source>
        <translation>Par image</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="403"/>
        <source>Whole sequence</source>
        <translation>Séquence entière</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="405"/>
        <source>How Auto picks the colour range: from each frame&apos;s own values (like the main window),
or one range from every exported frame, so a colour means the same value in every image.</source>
        <translation>Comment Auto choisit la plage de couleurs : à partir des valeurs propres à chaque image (comme la fenêtre principale),
ou une seule plage pour toutes les images exportées, pour qu&apos;une couleur ait la même valeur dans chaque image.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="410"/>
        <source>Auto range</source>
        <translation>Plage auto</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="414"/>
        <source>Add a vertical colorbar to the right of each image.</source>
        <translation>Ajoute une barre de couleur verticale à droite de chaque image.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="423"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="498"/>
        <source>Crop to ROI</source>
        <translation>Rogner à la ROI</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="424"/>
        <source>Cut the picture to the region of interest plus a small margin.</source>
        <translation>Rogne la figure à la région d&apos;intérêt plus une petite marge.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="457"/>
        <source>MP4 needs the imageio and imageio-ffmpeg packages (pip install imageio imageio-ffmpeg). Save as GIF instead.</source>
        <translation>Le MP4 nécessite les paquets imageio et imageio-ffmpeg (pip install imageio imageio-ffmpeg). Enregistrez plutôt en GIF.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="492"/>
        <source>Add a vertical colorbar to the right of each frame.</source>
        <translation>Ajoute une barre de couleur verticale à droite de chaque image.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="499"/>
        <source>Cut the frames to the region of interest plus a small margin.</source>
        <translation>Rogne les images à la région d&apos;intérêt plus une petite marge.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="503"/>
        <source>Auto range: one range for the whole animation, so a colour means the same value in every frame.</source>
        <translation>Plage auto : une seule plage pour toute l&apos;animation, pour qu&apos;une couleur ait la même valeur dans chaque image.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="530"/>
        <source>Drawn like the Images tab: same colormap, range, opacity and configuration.</source>
        <translation>Tracé comme dans l&apos;onglet « Images » : même palette, plage, opacité et configuration.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="639"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="687"/>
        <source>Min must be below Max for %1.</source>
        <translation>Min doit être inférieur à Max pour %1.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="645"/>
        <source>Nothing to export: enable at least one field with computed frames in the frame range.</source>
        <translation>Rien à exporter : activez au moins un champ ayant des images calculées dans la plage d&apos;images.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="653"/>
        <source>Select at least one format.</source>
        <translation>Sélectionnez au moins un format.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="657"/>
        <source>Select at least one variable to export.</source>
        <translation>Sélectionnez au moins une variable à exporter.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="690"/>
        <source>Nothing to export: select a field or the parameter table.</source>
        <translation>Rien à exporter : sélectionnez un champ ou le tableau des paramètres.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="743"/>
        <source>… and %1 more</source>
        <translation>… et %1 de plus</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="745"/>
        <source>Replace Existing Files?</source>
        <translation>Remplacer les fichiers existants ?</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="746"/>
        <source>These already exist in the output folder and will be replaced:

%1</source>
        <translation>Ces fichiers existent déjà dans le dossier de sortie et seront remplacés :

%1</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="765"/>
        <source>The results changed after this dialog was opened (a new run, strain or project). Close it and open Export again to export them.</source>
        <translation>Les résultats ont changé depuis l&apos;ouverture de cette boîte de dialogue (nouvelle exécution, déformation ou projet). Fermez-la et rouvrez l&apos;export pour les exporter.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="769"/>
        <source>Nothing exported: the existing files were kept.</source>
        <translation>Rien n&apos;a été exporté : les fichiers existants ont été conservés.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="794"/>
        <source>Stopping…</source>
        <translation>Arrêt en cours…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="805"/>
        <source>Rendering %1 (%2/%3)</source>
        <translation>Rendu de %1 (%2/%3)</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/dialogs/export_dialog.py" line="823"/>
        <source>Exported %n file(s) → %1</source>
        <translation>
            <numerusform>%n fichier exporté → %1</numerusform>
            <numerusform>%n fichiers exportés → %1</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/dialogs/export_dialog.py" line="825"/>
        <source>Exported %n image(s) → %1</source>
        <translation>
            <numerusform>%n image exportée → %1</numerusform>
            <numerusform>%n images exportées → %1</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="841"/>
        <source>Export failed: %1</source>
        <translation>Échec de l&apos;export : %1</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/dialogs/export_dialog.py" line="849"/>
        <source>Export cancelled: %n file(s) written before stopping.</source>
        <translation>
            <numerusform>Export annulé : %n fichier écrit avant l&apos;arrêt.</numerusform>
            <numerusform>Export annulé : %n fichiers écrits avant l&apos;arrêt.</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="829"/>
        <source>No animation was written. See the log for details.</source>
        <translation>Aucune animation n&apos;a été écrite. Consultez le journal pour plus de détails.</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/dialogs/export_dialog.py" line="831"/>
        <source>Exported %n animation(s) → %1</source>
        <translation>
            <numerusform>%n animation exportée → %1</numerusform>
            <numerusform>%n animations exportées → %1</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="833"/>
        <source>Report saved → %1</source>
        <translation>Rapport enregistré → %1</translation>
    </message>
</context>
<context>
    <name>FieldSelector</name>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="19"/>
        <source>Horizontal displacement (+ right)</source>
        <translation>Déplacement horizontal (+ vers la droite)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="20"/>
        <source>Vertical displacement (+ down)</source>
        <translation>Déplacement vertical (+ vers le bas)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="21"/>
        <source>Displacement magnitude</source>
        <translation>Norme du déplacement</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="22"/>
        <source>Intensity residual: local RMS difference between the reference image and
the deformed image mapped back by the displacement. Large where the
tracking failed.</source>
        <translation>Résidu d&apos;intensité : différence RMS locale entre l&apos;image de référence et
l&apos;image déformée ramenée par le déplacement. Élevé là où le
suivi a échoué.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="26"/>
        <source>Disp U</source>
        <translation>Dépl. U</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="26"/>
        <source>Disp V</source>
        <translation>Dépl. V</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="27"/>
        <source>|U|</source>
        <translation>|U|</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="27"/>
        <source>Residual</source>
        <translation>Résidu</translation>
    </message>
</context>
<context>
    <name>FrameNavigator</name>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="59"/>
        <source>Previous frame</source>
        <translation>Image précédente</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="69"/>
        <location filename="../../gui/widgets/frame_navigator.py" line="170"/>
        <source>Play animation</source>
        <translation>Lire l&apos;animation</translation>
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
        <translation>Image suivante</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="93"/>
        <source>Playback speed</source>
        <translation>Vitesse de lecture</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="98"/>
        <location filename="../../gui/widgets/frame_navigator.py" line="194"/>
        <source>FRAME 0/0</source>
        <translation>IMAGE 0/0</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="159"/>
        <source>⏸</source>
        <translation>⏸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="160"/>
        <source>Pause animation</source>
        <translation>Pause</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="191"/>
        <source>FRAME %1/%2</source>
        <translation>IMAGE %1/%2</translation>
    </message>
</context>
<context>
    <name>ImageCanvas</name>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1066"/>
        <source>Load images first before drawing a Region of Interest.</source>
        <translation>Chargez d&apos;abord des images avant de dessiner une région d&apos;intérêt.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1085"/>
        <source>The three points are nearly collinear — pick points spread around the circle&apos;s edge.</source>
        <translation>Les trois points sont presque colinéaires — choisissez des points répartis sur le bord du cercle.</translation>
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
        <translation>Nom de fichier</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="99"/>
        <source>Region</source>
        <comment>Image list column: ROI status</comment>
        <translation>Région</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="193"/>
        <location filename="../../gui/widgets/image_list.py" line="255"/>
        <source>Add</source>
        <translation>Ajouter</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="249"/>
        <source>Edit</source>
        <translation>Modifier</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="252"/>
        <source>Need</source>
        <translation>Requis</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/widgets/image_list.py" line="368"/>
        <source>Import Region of Interest for %n frame(s)</source>
        <translation>
            <numerusform>Importer la région d&apos;intérêt pour %n image</numerusform>
            <numerusform>Importer la région d&apos;intérêt pour %n images</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="379"/>
        <source>Clear Region of Interest (%1 with region)</source>
        <translation>Effacer la région d&apos;intérêt (%1 avec région)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="383"/>
        <source>Clear Region of Interest</source>
        <translation>Effacer la région d&apos;intérêt</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/widgets/image_list.py" line="394"/>
        <source>Delete %n image(s)</source>
        <translation>
            <numerusform>Supprimer %n image</numerusform>
            <numerusform>Supprimer %n images</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/widgets/image_list.py" line="435"/>
        <source>Select %n Mask File(s)</source>
        <translation>
            <numerusform>Sélectionner %n fichier de masque</numerusform>
            <numerusform>Sélectionner %n fichiers de masque</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="440"/>
        <source>Images</source>
        <translation>Images</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="441"/>
        <source>All Files</source>
        <translation>Tous les fichiers</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="450"/>
        <source>Selected %1 files for %2 frames — count must match</source>
        <translation>%1 fichiers sélectionnés pour %2 images — le nombre doit correspondre</translation>
    </message>
</context>
<context>
    <name>ImageMessages</name>
    <message>
        <location filename="../../gui/controllers/image_controller.py" line="43"/>
        <source>Cannot open the images</source>
        <translation>Impossible d&apos;ouvrir les images</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/image_controller.py" line="47"/>
        <source>%n image(s), %1 x %2 px, %3</source>
        <translation>
            <numerusform>%n image, %1 x %2 px, %3</numerusform>
            <numerusform>%n images, %1 x %2 px, %3</numerusform>
        </translation>
    </message>
</context>
<context>
    <name>LeftSidebar</name>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="176"/>
        <source>IMAGES</source>
        <translation>IMAGES</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="183"/>
        <source>Natural Sort (1, 2, …, 10)</source>
        <translation>Tri naturel (1, 2, …, 10)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="259"/>
        <source>WORKFLOW TYPE</source>
        <translation>TYPE DE FLUX</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="186"/>
        <source>Sort by embedded numbers: image1, image2, ..., image10 (default).
Unchecked: plain alphabetical order, for zero-padded names.</source>
        <translation>Tri par numéros intégrés : image1, image2, ..., image10 (par défaut).
Non coché : ordre alphabétique simple, pour les noms avec zéros de remplissage.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="266"/>
        <source>REGION OF INTEREST</source>
        <translation>RÉGION D&apos;INTÉRÊT</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="278"/>
        <source>PARAMETERS</source>
        <translation>PARAMÈTRES</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="284"/>
        <source>ADVANCED</source>
        <translation>AVANCÉ</translation>
    </message>
</context>
<context>
    <name>LoadDataDialog</name>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="103"/>
        <source>Load Data</source>
        <translation>Données de charge</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="115"/>
        <source>No file chosen.</source>
        <translation>Aucun fichier choisi.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="117"/>
        <source>Choose file…</source>
        <translation>Choisir un fichier…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="137"/>
        <source>Load column:</source>
        <translation>Colonne de charge :</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="139"/>
        <source>By time</source>
        <translation>Par le temps</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="140"/>
        <source>By frame number</source>
        <translation>Par numéro d&apos;image</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="144"/>
        <source>Match rows to frames:</source>
        <translation>Associer les lignes aux images :</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="147"/>
        <source>Time column:</source>
        <translation>Colonne de temps :</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="153"/>
        <source>The machine&apos;s time at the reference image. If the camera started 2 s after the machine, enter 2.</source>
        <translation>Le temps de la machine à l&apos;image de référence. Si la caméra a démarré 2 s après la machine, saisissez 2.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="155"/>
        <source>Offset:</source>
        <translation>Décalage :</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="163"/>
        <source>Frame column:</source>
        <translation>Colonne d&apos;image :</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="167"/>
        <source>The first image is numbered:</source>
        <translation>Numéro de la première image :</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="174"/>
        <source>Initial cross-section, for engineering stress F / A0 in MPa. Leave at 0 for load only.</source>
        <translation>Section initiale, pour la contrainte nominale F / A0 en MPa. Laisser à 0 pour la charge seule.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="176"/>
        <source>Cross-section A0:</source>
        <translation>Section A0 :</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="192"/>
        <source>Remove Load Data</source>
        <translation>Supprimer les données de charge</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="201"/>
        <source>Camera frame rate: %1 fps, from Physical Units.</source>
        <translation>Fréquence d&apos;images de la caméra : %1 fps, issue des « Unités physiques ».</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="204"/>
        <source>Set the camera frame rate under Physical Units to match by time.</source>
        <translation>Définissez la fréquence d&apos;images de la caméra dans « Unités physiques » pour associer par le temps.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="227"/>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="245"/>
        <source>(unnamed)</source>
        <translation>(sans nom)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="324"/>
        <source>No frame falls within the record: check the columns and the offset.</source>
        <translation>Aucune image ne tombe dans l&apos;enregistrement : vérifiez les colonnes et le décalage.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="328"/>
        <source>Frames with a load: %1 of %2.</source>
        <translation>Images avec une charge : %1 sur %2.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="334"/>
        <source>Open Load Data</source>
        <translation>Ouvrir les données de charge</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="335"/>
        <source>CSV Files</source>
        <translation>Fichiers CSV</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="335"/>
        <source>All Files</source>
        <translation>Tous les fichiers</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="341"/>
        <source>Could not read %1: %2</source>
        <translation>Impossible de lire %1 : %2</translation>
    </message>
</context>
<context>
    <name>MainWindow</name>
    <message>
        <location filename="../../gui/app.py" line="136"/>
        <source>&amp;File</source>
        <translation>Fichier</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="161"/>
        <source>Quit</source>
        <translation>Quitter</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="165"/>
        <source>&amp;Settings</source>
        <translation>Paramètres</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="168"/>
        <source>Theme</source>
        <translation>Thème</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="172"/>
        <source>Dark</source>
        <comment>Color theme</comment>
        <translation>Sombre</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="172"/>
        <source>Light</source>
        <comment>Color theme</comment>
        <translation>Clair</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="182"/>
        <source>Language</source>
        <translation>Langue</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="423"/>
        <source>Language changed</source>
        <translation>Langue modifiée</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="594"/>
        <source>All Files</source>
        <translation>Tous les fichiers</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="200"/>
        <source>Select Image Folder</source>
        <translation>Sélectionner le dossier d&apos;images</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="289"/>
        <source>File Association Failed</source>
        <translation>Échec de l&apos;association de fichiers</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="292"/>
        <source>File Association</source>
        <translation>Association de fichiers</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="146"/>
        <source>Open Image Folder…</source>
        <translation>Ouvrir un dossier d&apos;images…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="148"/>
        <source>Open Project…</source>
        <translation>Ouvrir un projet…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="149"/>
        <location filename="../../gui/app.py" line="233"/>
        <source>Save Project</source>
        <translation>Enregistrer le projet</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="150"/>
        <source>Save Project As…</source>
        <translation>Enregistrer le projet sous…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="152"/>
        <source>Export Results…</source>
        <translation>Exporter les résultats…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="157"/>
        <source>Associate .raftcorr files with RAFTcorr…</source>
        <translation>Associer les fichiers .raftcorr à RAFTcorr…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="158"/>
        <source>Register .raftcorr so double-clicking a project file opens RAFTcorr (current user only, no admin rights needed).</source>
        <translation>Enregistre .raftcorr pour qu&apos;un double-clic sur un fichier de projet ouvre RAFTcorr (utilisateur actuel uniquement, sans droits administrateur).</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="191"/>
        <source>&amp;Help</source>
        <translation>Aide</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="192"/>
        <source>Installed Models…</source>
        <translation>Modèles installés…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="193"/>
        <location filename="../../gui/app.py" line="359"/>
        <location filename="../../gui/app.py" line="361"/>
        <source>Self-test</source>
        <translation>Autotest</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="194"/>
        <location filename="../../gui/app.py" line="366"/>
        <source>About RAFTcorr</source>
        <translation>À propos de RAFTcorr</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="209"/>
        <source>Cancel</source>
        <translation>Annuler</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="245"/>
        <source>Project saved: %1</source>
        <translation>Projet enregistré : %1</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="247"/>
        <source>Saving the project</source>
        <translation>Enregistrement du projet</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="250"/>
        <source>Open Project</source>
        <translation>Ouvrir un projet</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="257"/>
        <source>Opening the project</source>
        <translation>Ouverture du projet</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="265"/>
        <source>Images not found</source>
        <translation>Images introuvables</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="266"/>
        <source>The image folder of this project was not found. Choose its new location.</source>
        <translation>Le dossier d&apos;images de ce projet est introuvable. Choisissez son nouvel emplacement.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="267"/>
        <source>Image folder of the project</source>
        <translation>Dossier d&apos;images du projet</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="274"/>
        <source>Cannot open the project</source>
        <translation>Impossible d&apos;ouvrir le projet</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="280"/>
        <source>Project opened: %1</source>
        <translation>Projet ouvert : %1</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="290"/>
        <source>Could not register .raftcorr files: </source>
        <translation>Impossible d&apos;enregistrer les fichiers .raftcorr : </translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="293"/>
        <source>Done. Double-clicking a .raftcorr file will now open RAFTcorr and restore that project.</source>
        <translation>Terminé. Un double-clic sur un fichier .raftcorr ouvrira désormais RAFTcorr et restaurera ce projet.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="298"/>
        <source>Load at least two images first.</source>
        <translation>Chargez d&apos;abord au moins deux images.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="301"/>
        <source>Wait for the running task to finish.</source>
        <translation>Attendez la fin de la tâche en cours.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="330"/>
        <source>Suggested: RAFTcorr-%1. Largest displacement of the ROI about %2 px (99th percentile over frames %3); RAFTcorr-%1 reaches %4 px.</source>
        <translation>Suggestion : RAFTcorr-%1. Plus grand déplacement de la ROI d&apos;environ %2 px (99e centile sur les images %3) ; la portée de RAFTcorr-%1 est de %4 px.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="334"/>
        <source>This exceeds even the reach of RAFTcorr-s8: use incremental tracking.</source>
        <translation>Cela dépasse même la portée de RAFTcorr-s8 : utilisez le suivi incrémental.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="339"/>
        <source>Measuring the motion with RAFTcorr-s8</source>
        <translation>Mesure du mouvement avec RAFTcorr-s8</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="348"/>
        <source>%1  (s = %2, reach %3 px): %4</source>
        <translation>%1  (s = %2, portée %3 px) : %4</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="349"/>
        <source>not installed</source>
        <translation>non installé</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="351"/>
        <source>Searched:</source>
        <translation>Emplacements recherchés :</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="352"/>
        <source>Installed Models</source>
        <translation>Modèles installés</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="367"/>
        <source>&lt;b&gt;RAFTcorr %1&lt;/b&gt;&lt;br&gt;Learned digital image correlation with the RAFTcorr-s1, -s2, -s4 and -s8 models.&lt;br&gt;&lt;br&gt;Displacements: reference to deformed, +u right, +v down, in pixels, frame-1 (Lagrangian) coordinates.&lt;br&gt;&lt;br&gt;Interface derived from pyALDIC.&lt;br&gt;Zixiang (Zach) Tong, The University of Texas at Austin.</source>
        <translation>&lt;b&gt;RAFTcorr %1&lt;/b&gt;&lt;br&gt;Corrélation d&apos;images numériques apprise avec les modèles RAFTcorr-s1, -s2, -s4 et -s8.&lt;br&gt;&lt;br&gt;Déplacements : de la référence vers l&apos;image déformée, +u vers la droite, +v vers le bas, en pixels, dans les coordonnées de l&apos;image 1 (lagrangiennes).&lt;br&gt;&lt;br&gt;Interface dérivée de pyALDIC.&lt;br&gt;Zixiang (Zach) Tong, The University of Texas at Austin.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="426"/>
        <source>Language set to %1. Please restart RAFTcorr for all widgets to pick up the new language.</source>
        <translation>Langue définie sur %1. Veuillez redémarrer RAFTcorr pour que tous les éléments prennent en compte la nouvelle langue.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="561"/>
        <source>Frame %1 has no Region of Interest of its own — frame 1&apos;s is used for computation. Switch to frame 1 to edit it, or import a mask to give this frame its own.</source>
        <translation>L&apos;image %1 n&apos;a pas de région d&apos;intérêt propre — celle de l&apos;image 1 est utilisée pour le calcul. Passez à l&apos;image 1 pour la modifier, ou importez un masque pour donner à cette image la sienne.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="572"/>
        <source>No Region of Interest to save — load images first.</source>
        <translation>Aucune région d&apos;intérêt à enregistrer — chargez d&apos;abord des images.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="579"/>
        <source>Region of Interest mask is empty.</source>
        <translation>Le masque de la région d&apos;intérêt est vide.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="591"/>
        <source>Save Region of Interest Mask</source>
        <translation>Enregistrer le masque de région d&apos;intérêt</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="593"/>
        <source>PNG Images</source>
        <translation>Images PNG</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="602"/>
        <source>Mask saved to %1</source>
        <translation>Masque enregistré dans %1</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="611"/>
        <source>No Region of Interest to invert — load images first.</source>
        <translation>Aucune région d&apos;intérêt à inverser — chargez d&apos;abord des images.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="634"/>
        <source>Load images first.</source>
        <translation>Veuillez d&apos;abord charger des images.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="670"/>
        <source>  Imported mask for frame %1</source>
        <translation>  Masque importé pour l&apos;image %1</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/app.py" line="676"/>
        <source>Batch import: %n mask(s) loaded</source>
        <translation>
            <numerusform>Import par lot : %n masque chargé</numerusform>
            <numerusform>Import par lot : %n masques chargés</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/app.py" line="703"/>
        <source>Imported Region of Interest for %n frame(s)</source>
        <translation>
            <numerusform>Région d&apos;intérêt importée pour %n image</numerusform>
            <numerusform>Région d&apos;intérêt importée pour %n images</numerusform>
        </translation>
    </message>
</context>
<context>
    <name>ParamPanel</name>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="65"/>
        <source>Iterations</source>
        <translation>Itérations</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="66"/>
        <source>Refinement iterations of the network.
The paper evaluates every model with 20.</source>
        <translation>Itérations de raffinement du réseau.
L&apos;article évalue chaque modèle avec 20.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="73"/>
        <source>off (raw)</source>
        <translation>désactivé (brut)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="75"/>
        <source>Smoothing σ</source>
        <translation>Lissage σ</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="76"/>
        <source>Gaussian smoothing of the displacement (standard deviation).
Off by default, as in the paper: smoothing lengthens the
spatial resolution.</source>
        <translation>Lissage gaussien du déplacement (écart-type).
Désactivé par défaut, comme dans l&apos;article : le lissage dégrade la
résolution spatiale.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="81"/>
        <source>Automatic</source>
        <translation>Automatique</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="85"/>
        <source>Device</source>
        <translation>Périphérique</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="86"/>
        <source>Automatic uses the GPU when PyTorch sees one.</source>
        <translation>« Automatique » utilise le GPU lorsque PyTorch en détecte un.</translation>
    </message>
</context>
<context>
    <name>PhysicalUnitsWidget</name>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="57"/>
        <source>Use physical units</source>
        <translation>Utiliser les unités physiques</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="68"/>
        <source>Physical size of one image pixel</source>
        <translation>Taille physique d&apos;un pixel de l&apos;image</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="83"/>
        <source>/ px</source>
        <translation>/ px</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="86"/>
        <source>Pixel size</source>
        <translation>Taille de pixel</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="96"/>
        <source>Acquisition frame rate (used for velocity field)</source>
        <translation>Fréquence d&apos;images d&apos;acquisition (utilisée pour le champ de vitesse)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="97"/>
        <source>Frame rate</source>
        <translation>Fréquence d&apos;images</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="165"/>
        <source>Disp: %1  Velocity: %2/s</source>
        <translation>Dépl. : %1  Vitesse : %2/s</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="170"/>
        <source>Disp: px  Velocity: px/fr</source>
        <translation>Dépl. : px  Vitesse : px/im</translation>
    </message>
</context>
<context>
    <name>PipelineController</name>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="77"/>
        <source>Stopped.</source>
        <extracomment>Results larger than this are kept in memory-mapped files in a temporary folder.</extracomment>
        <translation>Arrêté.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="84"/>
        <source>The GPU ran out of memory. Close other GPU programs, choose a model with a larger s, or lower the memory fraction (Advanced).</source>
        <translation>La mémoire du GPU est épuisée. Fermez les autres programmes qui utilisent le GPU, choisissez un modèle avec un s plus grand, ou réduisez la fraction de mémoire (Avancé).</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="86"/>
        <source>Analysis failed</source>
        <translation>Échec de l&apos;analyse</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="164"/>
        <source>Reference frames without their own ROI carry the ROI of frame 1 forward: %1</source>
        <translation>Les images de référence sans ROI propre reprennent la ROI de l&apos;image 1 : %1</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="183"/>
        <source>Starting...</source>
        <translation>Démarrage…</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="209"/>
        <source>The images changed: the running analysis is cancelled.</source>
        <translation>Les images ont changé : l&apos;analyse en cours est annulée.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="233"/>
        <source>The images changed during the run: its results are discarded.</source>
        <translation>Les images ont changé pendant l&apos;exécution : ses résultats sont abandonnés.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="246"/>
        <source>No frame was computed; the previous results are kept.</source>
        <translation>Aucune image n&apos;a été calculée ; les résultats précédents sont conservés.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="260"/>
        <source>Tracking quality</source>
        <translation>Qualité du suivi</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="261"/>
        <source>Part of the sequence may not be tracked correctly.</source>
        <translation>Une partie de la séquence n&apos;a peut-être pas été suivie correctement.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="263"/>
        <source>Show the &apos;Residual&apos; field to see where the images do not match after the displacement is applied.</source>
        <translation>Affichez le champ « Résidu » pour voir où les images ne correspondent pas une fois le déplacement appliqué.</translation>
    </message>
</context>
<context>
    <name>PipelineMessages</name>
    <message numerus="yes">
        <location filename="../../gui/controllers/pipeline_messages.py" line="73"/>
        <source>  %n frame(s) with custom ROI masks</source>
        <translation>
            <numerusform>  %n image avec un masque ROI personnalisé</numerusform>
            <numerusform>  %n images avec des masques ROI personnalisés</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/pipeline_messages.py" line="77"/>
        <source>Results received: %n frame(s)</source>
        <translation>
            <numerusform>Résultats reçus : %n image</numerusform>
            <numerusform>Résultats reçus : %n images</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/pipeline_messages.py" line="81"/>
        <source>  %n image(s), %1</source>
        <translation>
            <numerusform>  %n image, %1</numerusform>
            <numerusform>  %n images, %1</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="84"/>
        <source>  ROI range: %1</source>
        <translation>  Étendue de la ROI : %1</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="87"/>
        <source>  Refinement: %1</source>
        <translation>  Raffinement de maillage : %1</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="91"/>
        <source>Error: %1</source>
        <translation>Erreur : %1</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="95"/>
        <source>Run canceled. Give the listed reference frames a Region of Interest of their own, or run again and continue with frame 1&apos;s region.</source>
        <translation>Exécution annulée. Donnez aux images de référence listées leur propre région d&apos;intérêt, ou relancez et continuez avec la région de l&apos;image 1.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="102"/>
        <source>Cannot start analysis</source>
        <translation>Impossible de lancer l&apos;analyse</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="106"/>
        <source>Need at least 2 images. Load an image folder in the %1 section before running DIC.</source>
        <translation>Au moins 2 images sont nécessaires. Chargez un dossier d&apos;images dans la section %1 avant de lancer la DIC.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="111"/>
        <source>Define a Region of Interest first: draw or import one for frame 1 in the %1 section.</source>
        <translation>Définissez d&apos;abord une région d&apos;intérêt : dessinez-en ou importez-en une pour l&apos;image 1 dans la section %1.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="116"/>
        <source>The Region of Interest of frame 1 is empty (no pixels selected). Draw it again in the %1 section.</source>
        <translation>La région d&apos;intérêt de l&apos;image 1 est vide (aucun pixel sélectionné). Redessinez-la dans la section %1.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="121"/>
        <source>Region of Interest too small</source>
        <translation>Région d&apos;intérêt trop petite</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="127"/>
        <source>The Region of Interest is %1×%2 px, but the current %3 (%4) needs at least %5×%5 px. Enlarge the Region of Interest or reduce %3.</source>
        <translation>La région d&apos;intérêt mesure %1×%2 px, mais le %3 actuel (%4) exige au moins %5×%5 px. Agrandissez la région d&apos;intérêt ou réduisez le %3.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="133"/>
        <source>Starting Points required</source>
        <translation>Points de départ requis</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="137"/>
        <source>Place at least one Starting Point on the canvas before running: click “%1” or “%2” in the %3 section.</source>
        <translation>Placez au moins un point de départ sur le canevas avant de lancer l&apos;exécution : cliquez sur « %1 » ou « %2 » dans la section %3.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="143"/>
        <source>Invalid Starting Points</source>
        <translation>Points de départ non valides</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="147"/>
        <source>One or more Starting Points do not lie in any part of the Region of Interest. Click “%1” in the %2 section, then place the Starting Points again.</source>
        <translation>Un ou plusieurs points de départ ne se trouvent dans aucune partie de la région d&apos;intérêt. Cliquez sur « %1 » dans la section %2, puis placez à nouveau les points de départ.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="155"/>
        <source>DIC analysis failed</source>
        <translation>Échec de l&apos;analyse DIC</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="158"/>
        <source>Failed to start analysis</source>
        <translation>Échec du lancement de l&apos;analyse</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="161"/>
        <source>Failed to process results</source>
        <translation>Échec du traitement des résultats</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="166"/>
        <source>The analysis stopped with a runtime error:

%1

See the console log for the full traceback.</source>
        <translation>L&apos;analyse s&apos;est arrêtée sur une erreur d&apos;exécution :

%1

Consultez le journal de la console pour la trace complète.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="172"/>
        <source>%1

See the console log for the full traceback.</source>
        <translation>%1

Consultez le journal de la console pour la trace complète.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="177"/>
        <source>Incomplete Region of Interest Coverage</source>
        <translation>Couverture incomplète par les régions d&apos;intérêt</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/pipeline_messages.py" line="182"/>
        <source>This incremental run has %n reference frame(s) with no Region of Interest of their own: frame(s) %1.</source>
        <translation>
            <numerusform>Cette exécution incrémentale compte %n image de référence sans région d&apos;intérêt propre : image %1.</numerusform>
            <numerusform>Cette exécution incrémentale compte %n images de référence sans région d&apos;intérêt propre : images %1.</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/pipeline_messages.py" line="188"/>
        <source>%1 and %n more</source>
        <comment>List of frame numbers cut short: %1 = the frame numbers shown, %n = how many more frames there are</comment>
        <translation>
            <numerusform>%1 et %n autre</numerusform>
            <numerusform>%1 et %n autres</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="194"/>
        <source>A reference frame without a Region of Interest of its own reuses frame 1&apos;s region unchanged: nothing moves the region with the specimen. Where the specimen has moved out of that region, the part outside it is not measured from such a frame, and the nodes there stop being tracked or have their displacement copied from the nearest measured node. The run reports these nodes when it finishes.</source>
        <translation>Une image de référence sans région d&apos;intérêt propre reprend telle quelle la région de l&apos;image 1 : rien ne déplace la région avec l&apos;éprouvette. Là où l&apos;éprouvette est sortie de cette région, la partie extérieure n&apos;est pas mesurée à partir d&apos;une telle image, et les nœuds qui s&apos;y trouvent cessent d&apos;être suivis ou voient leur déplacement copié depuis le nœud mesuré le plus proche. L&apos;exécution signale ces nœuds à la fin.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="204"/>
        <source>To measure the whole specimen, give each reference frame its own Region of Interest that covers the specimen in that frame (Batch Import can assign Regions of Interest to many frames at once). A reference frame with a region of its own is measured on a grid that covers that region.</source>
        <translation>Pour mesurer toute l&apos;éprouvette, donnez à chaque image de référence sa propre région d&apos;intérêt couvrant l&apos;éprouvette dans cette image (l&apos;import par lot permet d&apos;attribuer des régions d&apos;intérêt à de nombreuses images à la fois). Une image de référence dotée de sa propre région est mesurée sur une grille qui couvre cette région.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="211"/>
        <source>Continue with frame 1&apos;s region?</source>
        <translation>Continuer avec la région de l&apos;image 1 ?</translation>
    </message>
</context>
<context>
    <name>ROIHint</name>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="63"/>
        <source>Load images first, then draw a Region of Interest on frame 1.</source>
        <translation>Chargez d&apos;abord des images, puis dessinez une région d&apos;intérêt sur l&apos;image 1.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="70"/>
        <source>&lt;b&gt;Accumulative mode&lt;/b&gt; — only frame 1 needs a Region of Interest. All later frames are compared against it directly.</source>
        <translation>&lt;b&gt;Mode cumulatif&lt;/b&gt; — seule l&apos;image 1 a besoin d&apos;une région d&apos;intérêt. Toutes les images suivantes sont comparées directement à celle-ci.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="83"/>
        <source>&lt;b&gt;Incremental&lt;/b&gt; — only frame 1 needs a Region of Interest. RAFTcorr carries it to the reference frames (&lt;b&gt;%1&lt;/b&gt;) with the measured displacement. Draw on another frame only to replace that region or to leave out cracks and holes.</source>
        <translation>&lt;b&gt;Incrémental&lt;/b&gt; — seule l&apos;image 1 a besoin d&apos;une région d&apos;intérêt. RAFTcorr la reporte sur les images de référence (&lt;b&gt;%1&lt;/b&gt;) avec le déplacement mesuré. Ne dessinez sur une autre image que pour y remplacer cette région ou pour exclure des fissures et des trous.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="92"/>
        <source>Draw a Region of Interest on frame 1.</source>
        <translation>Dessinez une région d&apos;intérêt sur l&apos;image 1.</translation>
    </message>
</context>
<context>
    <name>ROIToolbar</name>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="72"/>
        <source>+ Add</source>
        <translation>+ Ajouter</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="74"/>
        <source>Add region to the Region of Interest (Polygon / Rectangle / Circle)</source>
        <translation>Ajouter une région à la région d&apos;intérêt (Polygone / Rectangle / Cercle)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="79"/>
        <source>Cut</source>
        <translation>Découper</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="81"/>
        <source>Cut region from the Region of Interest (Polygon / Rectangle / Circle)</source>
        <translation>Découper une région de la région d&apos;intérêt (Polygone / Rectangle / Cercle)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="86"/>
        <source>+ Refine</source>
        <translation>+ Raffiner</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="90"/>
        <source>Paint extra mesh-refinement zones with a brush
(only on frame 1 — material points auto-warped to later frames)</source>
        <translation>Peindre des zones de raffinement de maillage supplémentaires au pinceau
(uniquement sur l&apos;image 1 — les points matériels sont automatiquement reportés sur les images suivantes)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="94"/>
        <source>Refine brush is only available on frame 1. Switch to frame 1 to paint refinement zones; they are automatically warped to later frames.</source>
        <translation>Le pinceau de raffinement n&apos;est disponible que sur l&apos;image 1. Passez à l&apos;image 1 pour peindre les zones de raffinement ; elles sont automatiquement reportées sur les images suivantes.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="118"/>
        <source>Import</source>
        <translation>Importer</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="119"/>
        <source>Import mask from image file</source>
        <translation>Importer le masque depuis un fichier image</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="124"/>
        <source>Batch Import</source>
        <translation>Import par lot</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="126"/>
        <source>Batch import mask files for multiple frames</source>
        <translation>Importer par lot des fichiers de masques pour plusieurs images</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="139"/>
        <source>Save</source>
        <translation>Enregistrer</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="140"/>
        <source>Save current mask to PNG file</source>
        <translation>Enregistrer le masque actuel en PNG</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="145"/>
        <source>Invert</source>
        <translation>Inverser</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="147"/>
        <source>Invert the Region of Interest mask</source>
        <translation>Inverser le masque de la région d&apos;intérêt</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="152"/>
        <source>Clear</source>
        <translation>Effacer</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="154"/>
        <source>Clear all Region of Interest masks</source>
        <translation>Effacer tous les masques de région d&apos;intérêt</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="186"/>
        <source>Radius</source>
        <translation>Rayon</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="204"/>
        <source>Paint</source>
        <translation>Peindre</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="209"/>
        <source>Erase</source>
        <translation>Gommer</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="219"/>
        <source>Clear Brush</source>
        <translation>Effacer le pinceau</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="247"/>
        <source>Polygon</source>
        <translation>Polygone</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="251"/>
        <source>Rectangle</source>
        <translation>Rectangle</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="255"/>
        <source>Circle</source>
        <translation>Cercle</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="259"/>
        <source>Circle (3-point)</source>
        <translation>Cercle (3 points)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="352"/>
        <source>Import Mask Image</source>
        <translation>Importer une image de masque</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="354"/>
        <source>Images</source>
        <translation>Images</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="355"/>
        <source>All Files</source>
        <translation>Tous les fichiers</translation>
    </message>
</context>
<context>
    <name>RightSidebar</name>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="75"/>
        <source>Run DIC Analysis</source>
        <translation>Exécuter l&apos;analyse DIC</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="88"/>
        <source>Cancel</source>
        <translation>Annuler</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="93"/>
        <source>Cancel the current analysis. Frames already computed are kept so you can review or export the partial run.</source>
        <translation>Annuler l&apos;analyse en cours. Les images déjà calculées sont conservées, ce qui permet de consulter ou d&apos;exporter le résultat partiel.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="101"/>
        <source>Export Results</source>
        <translation>Exporter les résultats</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="109"/>
        <source>Open Strain Window</source>
        <translation>Ouvrir la fenêtre de déformation</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="112"/>
        <source>Compute and visualize strain in a separate post-processing window. Requires displacement results from a completed Run.</source>
        <translation>Calculer et visualiser la déformation dans une fenêtre de post-traitement séparée. Nécessite des résultats de déplacement d&apos;une exécution terminée.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="121"/>
        <source>PROGRESS</source>
        <translation>PROGRESSION</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="130"/>
        <location filename="../../gui/panels/right_sidebar.py" line="430"/>
        <source>Ready</source>
        <translation>Prêt</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="138"/>
        <location filename="../../gui/panels/right_sidebar.py" line="432"/>
        <location filename="../../gui/panels/right_sidebar.py" line="532"/>
        <source>ELAPSED  %1</source>
        <translation>ÉCOULÉ  %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="143"/>
        <location filename="../../gui/panels/right_sidebar.py" line="434"/>
        <location filename="../../gui/panels/right_sidebar.py" line="540"/>
        <location filename="../../gui/panels/right_sidebar.py" line="544"/>
        <source>REMAINING  %1</source>
        <translation>RESTANT  %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="151"/>
        <source>FIELD</source>
        <translation>CHAMP</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="160"/>
        <source>Show on</source>
        <translation>Afficher sur</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="165"/>
        <source>Deformed frame</source>
        <translation>Image déformée</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="166"/>
        <source>Reference frame</source>
        <translation>Image de référence</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="168"/>
        <source>Plot the field at the deformed node positions, or at their positions in the reference frame.</source>
        <translation>Tracer le champ aux positions déformées des nœuds, ou à leurs positions dans l&apos;image de référence.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="176"/>
        <source>Show background image</source>
        <translation>Afficher l&apos;image de fond</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="179"/>
        <source>Uncheck to show the field on its own, with no speckle image behind it.</source>
        <translation>Décochez pour n&apos;afficher que le champ, sans image de mouchetis derrière.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="190"/>
        <source>Hidden background</source>
        <translation>Arrière-plan masqué</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="194"/>
        <source>White</source>
        <translation>Blanc</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="195"/>
        <source>Black</source>
        <translation>Noir</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="196"/>
        <source>Transparent</source>
        <translation>Transparent</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="199"/>
        <source>What replaces the image when it is hidden. Transparency is kept for PNG and TIFF on export; other formats get white.</source>
        <translation>Ce qui remplace l&apos;image lorsqu&apos;elle est masquée. À l&apos;export, la transparence est conservée pour PNG et TIFF ; les autres formats reçoivent du blanc.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="208"/>
        <source>Hide untracked regions</source>
        <translation>Masquer les zones non suivies</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="211"/>
        <source>Leave areas without a tracked node transparent, for example material that left the field of view, instead of filling them with values copied from distant nodes. Also used for exported images and animations; data files always keep NaN there.</source>
        <translation>Laisser transparentes les zones sans nœud suivi, par exemple le matériau sorti du champ de vision, au lieu de les remplir avec des valeurs copiées depuis des nœuds éloignés. S&apos;applique aussi aux images et animations exportées ; les fichiers de données y gardent toujours NaN.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="221"/>
        <source>VISUALIZATION</source>
        <translation>VISUALISATION</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="226"/>
        <source>Colormap</source>
        <translation>Palette</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="245"/>
        <source>Opacity</source>
        <translation>Opacité</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="252"/>
        <source>Overlay opacity (0 = transparent, 100 = opaque)</source>
        <translation>Opacité de la superposition (0 = transparent, 100 = opaque)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="258"/>
        <source>PHYSICAL UNITS</source>
        <translation>UNITÉS PHYSIQUES</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="264"/>
        <source>LOG</source>
        <translation>JOURNAL</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="270"/>
        <source>Clear</source>
        <translation>Effacer</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="399"/>
        <source>Place at least one Starting Point in each red region before running (red = needs a Starting Point).</source>
        <translation>Placez au moins un point de départ dans chaque région rouge avant de lancer l&apos;exécution (rouge = point de départ requis).</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="515"/>
        <source>%1  —  Frame %2</source>
        <translation>%1  —  Image %2</translation>
    </message>
</context>
<context>
    <name>StrainController</name>
    <message>
        <location filename="../../gui/controllers/strain_controller.py" line="96"/>
        <source>Strain frame %1/%2</source>
        <translation>Déformation : image %1/%2</translation>
    </message>
</context>
<context>
    <name>StrainFieldSelector</name>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="78"/>
        <source>Magnitude</source>
        <extracomment>GUI field name -&gt; FieldProvider key.</extracomment>
        <translation>Norme</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="79"/>
        <source>Velocity</source>
        <translation>Vitesse</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="200"/>
        <source>DISPLACEMENT</source>
        <translation>DÉPLACEMENT</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="211"/>
        <source>STRAIN</source>
        <translation>DÉFORMATION</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="265"/>
        <source>Horizontal displacement u (+ right)</source>
        <translation>Déplacement horizontal u (+ vers la droite)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="266"/>
        <source>Vertical displacement v (+ down)</source>
        <translation>Déplacement vertical v (+ vers le bas)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="267"/>
        <source>Displacement magnitude |u|</source>
        <translation>Norme du déplacement |u|</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="269"/>
        <source>Speed |du/dt|: central difference of the displacement over the neighbouring computed frames, times the frame rate</source>
        <translation>Vitesse |du/dt| : différence centrée du déplacement sur les images calculées voisines, multipliée par la fréquence d&apos;images</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="271"/>
        <source>Normal strain εxx</source>
        <translation>Déformation normale εxx</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="272"/>
        <source>Normal strain εyy</source>
        <translation>Déformation normale εyy</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="274"/>
        <source>Shear strain εxy (tensor component: half the engineering shear γxy)</source>
        <translation>Déformation de cisaillement εxy (composante du tenseur : moitié du cisaillement de l&apos;ingénieur γxy)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="276"/>
        <source>Major principal strain ε₁</source>
        <translation>Déformation principale majeure ε₁</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="277"/>
        <source>Minor principal strain ε₂</source>
        <translation>Déformation principale mineure ε₂</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="279"/>
        <source>Maximum in-plane shear strain (ε₁ − ε₂) / 2 (tensor component: half the engineering shear)</source>
        <translation>Déformation de cisaillement maximale dans le plan (ε₁ − ε₂) / 2 (composante du tenseur : moitié du cisaillement de l&apos;ingénieur)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="282"/>
        <source>Von Mises equivalent strain (incompressible material)</source>
        <translation>Déformation équivalente de von Mises (matériau incompressible)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="284"/>
        <source>Rigid rotation in degrees (polar decomposition), clockwise on screen positive</source>
        <translation>Rotation de corps rigide en degrés (décomposition polaire), positive dans le sens horaire à l&apos;écran</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="287"/>
        <source>Direction of ε₁ in degrees from +x, clockwise on screen positive</source>
        <translation>Direction de ε₁ en degrés depuis +x, positive dans le sens horaire à l&apos;écran</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="290"/>
        <source>Relative area change ΔA/A₀ = det F − 1</source>
        <translation>Variation relative d&apos;aire ΔA/A₀ = det F − 1</translation>
    </message>
</context>
<context>
    <name>StrainNavigator</name>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="73"/>
        <source>Previous frame</source>
        <translation>Image précédente</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="84"/>
        <location filename="../../gui/widgets/strain_navigator.py" line="216"/>
        <source>Play animation</source>
        <translation>Lire l&apos;animation</translation>
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
        <translation>Image suivante</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="109"/>
        <source>Playback speed</source>
        <translation>Vitesse de lecture</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="114"/>
        <location filename="../../gui/widgets/strain_navigator.py" line="228"/>
        <source>FRAME 0/0</source>
        <translation>IMAGE 0/0</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="205"/>
        <source>⏸</source>
        <translation>⏸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="206"/>
        <source>Pause animation</source>
        <translation>Pause</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="223"/>
        <source>FRAME %1/%2</source>
        <translation>IMAGE %1/%2</translation>
    </message>
</context>
<context>
    <name>StrainParamPanel</name>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="86"/>
        <source>Plane fitting</source>
        <extracomment>Strain measures in combo order (pyALDIC&apos;s order: infinitesimal, Eulerian, Green-Lagrangian). The configuration in the controls changed (by the user or Paper settings).</extracomment>
        <translation>Ajustement de plan</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="95"/>
        <source>Method</source>
        <translation>Méthode</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="110"/>
        <source>VSG size</source>
        <translation>Taille VSG</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="152"/>
        <source>Off</source>
        <comment>Strain smoothing preset</comment>
        <translation>Désactivé</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="87"/>
        <source>Quadratic fitting</source>
        <translation>Ajustement quadratique</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="89"/>
        <source>Polynomial fitted to each displacement component over the VSG window. Its slopes at the centre are the displacement gradients.
• Plane fitting (1st order): the standard VSG (paper setting).
• Quadratic fitting (2nd order): follows strain gradients inside the window better, at the cost of more noise.</source>
        <translation>Polynôme ajusté à chaque composante du déplacement sur la fenêtre VSG. Ses pentes au centre sont les gradients de déplacement.
• Ajustement de plan (ordre 1) : le VSG standard (réglage de l&apos;article).
• Ajustement quadratique (ordre 2) : suit mieux les gradients de déformation dans la fenêtre, au prix de plus de bruit.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="103"/>
        <source>VSG (Virtual Strain Gauge) size Lg is the side, in pixels, of the square window around each point over which the displacement is fitted. Strain is then taken from the fit&apos;s slopes.

• Larger VSG → smoother strain, lower spatial resolution.
• Smaller VSG → sharper strain, more noise.
• Odd, so that the window has a centre pixel (default: 41 px).</source>
        <translation>La taille VSG (Virtual Strain Gauge, jauge de déformation virtuelle) Lg est le côté, en pixels, de la fenêtre carrée autour de chaque point sur laquelle le déplacement est ajusté. La déformation est ensuite tirée des pentes de l&apos;ajustement.

• VSG plus grand → déformation plus lisse, résolution spatiale plus faible.
• VSG plus petit → déformation plus nette, plus de bruit.
• Impair, pour que la fenêtre ait un pixel central (par défaut : 41 px).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="114"/>
        <source>Uniform</source>
        <translation>Uniforme</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="115"/>
        <source>Gaussian</source>
        <translation>Gaussienne</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="117"/>
        <source>Weight of the points inside the VSG window: all equal (uniform, paper setting), or decreasing with the distance from the centre (Gaussian).</source>
        <translation>Poids des points dans la fenêtre VSG : tous égaux (uniforme, réglage de l&apos;article), ou décroissant avec la distance au centre (gaussienne).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="121"/>
        <source>Weighting</source>
        <translation>Pondération</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="129"/>
        <source>Auto (Lg / 4)</source>
        <translation>Auto (Lg / 4)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="131"/>
        <source>Standard deviation of the Gaussian weight. Auto uses a quarter of the VSG size.</source>
        <translation>Écart-type de la pondération gaussienne. Auto utilise un quart de la taille VSG.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="134"/>
        <source>Gaussian σ</source>
        <translation>σ gaussien</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="141"/>
        <source>Spacing of the strain points: 1 computes strain at every pixel; larger values are faster and lighter for long sequences.</source>
        <translation>Espacement des points de déformation : 1 calcule la déformation à chaque pixel ; des valeurs plus grandes sont plus rapides et plus légères pour les longues séquences.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="144"/>
        <source>Output spacing</source>
        <translation>Espacement de sortie</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="154"/>
        <source>Gaussian smoothing (σ in pixels) of the displacement before the fit. Off (0) differentiates the measured displacement as it is (paper setting).</source>
        <translation>Lissage gaussien (σ en pixels) du déplacement avant l&apos;ajustement. Désactivé (0) dérive le déplacement mesuré tel quel (réglage de l&apos;article).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="158"/>
        <source>Displacement smoothing</source>
        <translation>Lissage du déplacement</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="161"/>
        <source>Never fit across cracks or holes</source>
        <translation>Ne jamais ajuster à travers fissures ou trous</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="163"/>
        <source>Fit each connected part of the tracked region separately, so no VSG window ever spans a crack, a hole or the gap between two regions.</source>
        <translation>Ajuste séparément chaque partie connexe de la région suivie, pour qu&apos;aucune fenêtre VSG ne recouvre jamais une fissure, un trou ou l&apos;espace entre deux régions.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="172"/>
        <source>Infinitesimal</source>
        <translation>Infinitésimal</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="173"/>
        <source>Eulerian</source>
        <translation>Eulérien</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="174"/>
        <source>Green-Lagrangian</source>
        <translation>Green-Lagrange</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="176"/>
        <source>Strain measure from the deformation gradient F = I + ∇u:
• Infinitesimal: (∇u + ∇uᵀ) / 2 (small strains only).
• Eulerian (Euler-Almansi): (I − (F Fᵀ)⁻¹) / 2.
• Green-Lagrangian: (Fᵀ F − I) / 2 (paper setting).</source>
        <translation>Mesure de déformation à partir du gradient de transformation F = I + ∇u :
• Infinitésimal : (∇u + ∇uᵀ) / 2 (petites déformations uniquement).
• Eulérien (Euler-Almansi) : (I − (F Fᵀ)⁻¹) / 2.
• Green-Lagrange : (Fᵀ F − I) / 2 (réglage de l&apos;article).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="181"/>
        <source>Strain type</source>
        <translation>Type de déformation</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="184"/>
        <source>Paper settings</source>
        <translation>Réglages de l&apos;article</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="186"/>
        <source>Green-Lagrangian strain, unweighted plane fit, VSG 41 px, every pixel, no smoothing, never across cracks (the settings of the RAFTcorr paper).</source>
        <translation>Déformation de Green-Lagrange, ajustement de plan non pondéré, VSG 41 px, chaque pixel, sans lissage, jamais à travers les fissures (les réglages de l&apos;article RAFTcorr).</translation>
    </message>
</context>
<context>
    <name>StrainVizPanel</name>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="67"/>
        <source>Deformed frame</source>
        <extracomment>Alias so existing references read unchanged; the list itself lives in raftcorr.gui.support.colormaps, so every chooser offers the same options.</extracomment>
        <translation>Image déformée</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="68"/>
        <source>Reference frame</source>
        <translation>Image de référence</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="70"/>
        <source>Plot the field where the material points are in the current frame, or at their positions in the reference frame.</source>
        <translation>Tracer le champ là où se trouvent les points matériels dans l&apos;image courante, ou à leurs positions dans l&apos;image de référence.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="73"/>
        <source>Show on</source>
        <translation>Afficher sur</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="75"/>
        <source>Show background image</source>
        <translation>Afficher l&apos;image de fond</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="78"/>
        <source>Uncheck to show the field on its own, with no speckle image behind it.</source>
        <translation>Décochez pour n&apos;afficher que le champ, sans image de mouchetis derrière.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="84"/>
        <source>White</source>
        <translation>Blanc</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="85"/>
        <source>Black</source>
        <translation>Noir</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="86"/>
        <source>Transparent</source>
        <translation>Transparent</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="89"/>
        <source>What replaces the image when it is hidden. Transparency is kept for PNG and TIFF on export; other formats get white.</source>
        <translation>Ce qui remplace l&apos;image lorsqu&apos;elle est masquée. À l&apos;export, la transparence est conservée pour PNG et TIFF ; les autres formats reçoivent du blanc.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="139"/>
        <source>Hide untracked regions</source>
        <translation>Masquer les zones non suivies</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="142"/>
        <source>Leave points without a result transparent, for example material that left the field of view, instead of filling them with the nearest value inside the region of interest (reference frame view). Data files always keep NaN there.</source>
        <translation>Laisser transparents les points sans résultat, par exemple le matériau sorti du champ de vision, au lieu de les remplir avec la valeur la plus proche dans la région d&apos;intérêt (vue sur l&apos;image de référence). Les fichiers de données y gardent toujours NaN.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="81"/>
        <source>Background</source>
        <translation>Arrière-plan</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="93"/>
        <source>Hidden background</source>
        <translation>Arrière-plan masqué</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="100"/>
        <source>Colormap</source>
        <translation>Palette</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="104"/>
        <source>Range</source>
        <translation>Plage</translation>
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
        <translation>Opacité</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="147"/>
        <source>Edges</source>
        <translation>Bords</translation>
    </message>
</context>
<context>
    <name>StrainWindow</name>
    <message>
        <location filename="../../gui/strain_window.py" line="125"/>
        <source>Strain Post-Processing</source>
        <translation>Post-traitement de la déformation</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="210"/>
        <source>Fit</source>
        <translation>Ajuster</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="211"/>
        <source>Fit image to viewport</source>
        <translation>Ajuster l&apos;image à la vue</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="217"/>
        <source>100%</source>
        <translation>100%</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="218"/>
        <source>Zoom to 100% (1:1)</source>
        <translation>Zoomer à 100% (1:1)</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="221"/>
        <source>Zoom in</source>
        <translation>Zoom avant</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="226"/>
        <source>–</source>
        <translation>–</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="284"/>
        <source>STRAIN PARAMETERS</source>
        <translation>PARAMÈTRES DE DÉFORMATION</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="303"/>
        <source>Cancel</source>
        <translation>Annuler</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="307"/>
        <source>Cancel the running strain computation. The previous strain result is kept.</source>
        <translation>Annuler le calcul de déformation en cours. Le résultat de déformation précédent est conservé.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="319"/>
        <source>Export displacement and strain results to NPZ / MAT / CSV / PNG</source>
        <translation>Exporter les résultats de déplacement et de déformation en NPZ / MAT / CSV / PNG</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="349"/>
        <source>FIELD</source>
        <translation>CHAMP</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="356"/>
        <source>VISUALIZATION</source>
        <translation>VISUALISATION</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="368"/>
        <source>PHYSICAL UNITS</source>
        <translation>UNITÉS PHYSIQUES</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="374"/>
        <source>LOG</source>
        <translation>JOURNAL</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="855"/>
        <location filename="../../gui/strain_window.py" line="891"/>
        <source>No displacement results.</source>
        <translation>Aucun résultat de déplacement.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="893"/>
        <source>Frame %1 was not computed.</source>
        <translation>L&apos;image %1 n&apos;a pas été calculée.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="901"/>
        <source>Click Compute Strain to show this field.</source>
        <translation>Cliquez sur « Calculer la déformation » pour afficher ce champ.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="902"/>
        <source>No strain for frame %1: click Compute Strain.</source>
        <translation>Aucune déformation pour l&apos;image %1 : cliquez sur « Calculer la déformation ».</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="904"/>
        <source>Velocity needs at least two computed frames.</source>
        <translation>La vitesse nécessite au moins deux images calculées.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="905"/>
        <source>No valid values in frame %1.</source>
        <translation>Aucune valeur valide dans l&apos;image %1.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="170"/>
        <source>Strain Field</source>
        <translation>Champ de déformation</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="414"/>
        <source>Analysis is not available</source>
        <translation>L&apos;analyse n&apos;est pas disponible</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="417"/>
        <location filename="../../gui/strain_window.py" line="420"/>
        <source>Analysis</source>
        <translation>Analyse</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="418"/>
        <source>Analysis is not available (see the application log).</source>
        <translation>L&apos;analyse n&apos;est pas disponible (voir le journal de l&apos;application).</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="529"/>
        <source>Strain compute failed: %1: %2</source>
        <translation>Échec du calcul de déformation : %1 : %2</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="535"/>
        <location filename="../../gui/strain_window.py" line="598"/>
        <source>Strain computation complete.</source>
        <translation>Calcul de déformation terminé.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="546"/>
        <source>Strain window: no displacement results to post-process.</source>
        <translation>Fenêtre de déformation : aucun résultat de déplacement à post-traiter.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="591"/>
        <source>The displacement results changed during the strain computation; its result was discarded.</source>
        <translation>Les résultats de déplacement ont changé pendant le calcul de déformation ; son résultat a été abandonné.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="615"/>
        <source>Strain computation canceled.</source>
        <translation>Calcul de déformation annulé.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="622"/>
        <source>Strain compute failed: %1</source>
        <translation>Échec du calcul de déformation : %1</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="626"/>
        <source>Strain Computation Failed</source>
        <translation>Échec du calcul de déformation</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="665"/>
        <location filename="../../gui/strain_window.py" line="667"/>
        <source>Export is not available: %1</source>
        <translation>L&apos;export n&apos;est pas disponible : %1</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="679"/>
        <source>⚠ Params changed -- click Compute Strain</source>
        <translation>⚠ Paramètres modifiés — cliquez sur « Calculer la déformation »</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="227"/>
        <source>Zoom out</source>
        <translation>Zoom arrière</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="294"/>
        <source>Compute Strain</source>
        <translation>Calculer la déformation</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="316"/>
        <location filename="../../gui/strain_window.py" line="666"/>
        <source>Export Results</source>
        <translation>Exporter les résultats</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="558"/>
        <source>Starting…</source>
        <translation>Démarrage…</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="576"/>
        <source>Canceling…</source>
        <translation>Annulation…</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="595"/>
        <source>Complete</source>
        <translation>Terminé</translation>
    </message>
</context>
<context>
    <name>VelocitySettingsWidget</name>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="47"/>
        <source>Use physical units</source>
        <translation>Utiliser les unités physiques</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="69"/>
        <source>/ px</source>
        <translation>/ px</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="84"/>
        <source>Unit: px/frame</source>
        <translation>Unité : px/image</translation>
    </message>
</context>
<context>
    <name>WorkflowTypePanel</name>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="56"/>
        <source>Incremental</source>
        <translation>Incrémental</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="55"/>
        <source>Accumulative</source>
        <translation>Cumulatif</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="70"/>
        <source>Tracking Mode</source>
        <translation>Mode de suivi</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="61"/>
        <source>Accumulative: every frame is compared directly with frame 1.
Preferred whenever the motion stays within the model&apos;s reach:
no error accumulates from frame to frame.

Incremental: each frame is compared with a later reference frame
and the steps are composed. Needed when the motion from frame 1
exceeds the model&apos;s reach (large stretch or rotation); every step
adds its error.</source>
        <translation>Cumulatif : chaque image est comparée directement à l&apos;image 1.
À privilégier tant que le mouvement reste dans la portée du modèle :
aucune erreur ne s&apos;accumule d&apos;une image à l&apos;autre.

Incrémental : chaque image est comparée à une image de référence ultérieure
et les étapes sont composées. Nécessaire quand le mouvement depuis l&apos;image 1
dépasse la portée du modèle (grand étirement ou rotation) ; chaque étape
ajoute son erreur.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="82"/>
        <source>finest spatial resolution; small displacements</source>
        <translation>résolution spatiale la plus fine ; petits déplacements</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="83"/>
        <source>fine resolution, several times faster than s1; small to moderate displacements</source>
        <translation>résolution fine, plusieurs fois plus rapide que s1 ; déplacements petits à modérés</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="84"/>
        <source>balanced resolution and capture range</source>
        <translation>résolution et portée équilibrées</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="85"/>
        <source>largest capture range, coarsest resolution; large motion and rotation</source>
        <translation>portée la plus grande, résolution la plus grossière ; grands mouvements et rotations</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="89"/>
        <source>%1: %2.
Reach (nominal lookup radius 32 s): %3 px.</source>
        <translation>%1 : %2.
Portée (rayon de recherche nominal 32 s) : %3 px.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="92"/>
        <source>Weights not installed.</source>
        <translation>Poids non installés.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="97"/>
        <source>RAFTcorr model. A smaller s resolves finer detail; a larger s
reaches further (the coarsest correlation level looks 32 s px away).

s1, s2: small displacements, strain concentrations.
s4: balanced.
s8: large motion or rotation, or no prior knowledge.</source>
        <translation>Modèle RAFTcorr. Un s plus petit résout des détails plus fins ; un s plus grand
porte plus loin (le niveau de corrélation le plus grossier regarde à 32 s px).

s1, s2 : petits déplacements, concentrations de déformation.
s4 : équilibré.
s8 : grand mouvement ou rotation, ou aucune connaissance préalable.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="104"/>
        <source>Model</source>
        <translation>Modèle</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="110"/>
        <source>Suggest a model</source>
        <translation>Suggérer un modèle</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="112"/>
        <source>Measure the motion of the sequence quickly with s8 and pick the
finest model whose reach covers it.</source>
        <translation>Mesure rapidement le mouvement de la séquence avec s8 et choisit le
modèle le plus fin dont la portée le couvre.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="134"/>
        <source>Every Frame</source>
        <translation>Chaque image</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="135"/>
        <source>Every N Frames</source>
        <translation>Toutes les N images</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="136"/>
        <source>Custom Frames</source>
        <translation>Images personnalisées</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="139"/>
        <source>When the reference frame refreshes during incremental tracking.
Every Frame: reset reference every frame (smallest per-step displacement,
most robust for large deformation).
Every N Frames: reset every N frames (balance speed vs robustness).
Custom Frames: user-defined list of reference frame indices.</source>
        <translation>Quand l&apos;image de référence est rafraîchie lors du suivi incrémental.
Chaque image : réinitialiser la référence à chaque image (plus petit déplacement par étape,
plus robuste pour les grandes déformations).
Toutes les N images : réinitialiser toutes les N images (compromis vitesse/robustesse).
Images personnalisées : liste d&apos;indices définis par l&apos;utilisateur.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="148"/>
        <source>Reference Update</source>
        <translation>Mise à jour de la référence</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="158"/>
        <source>Update reference every N frames</source>
        <translation>Mettre à jour la référence toutes les N images</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="160"/>
        <source>Interval</source>
        <translation>Intervalle</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="169"/>
        <source>Comma-separated frame indices to use as reference frames (0-based)</source>
        <translation>Indices d&apos;images séparés par des virgules pour les images de référence (base 0)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="173"/>
        <source>Reference Frames</source>
        <translation>Images de référence</translation>
    </message>
</context>
<context>
    <name>_DropZone</name>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="118"/>
        <source>Drop image folder
or Browse</source>
        <translation>Déposez un dossier d&apos;images
ou Parcourir</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="128"/>
        <source>Select Image Folder</source>
        <translation>Sélectionner le dossier d&apos;images</translation>
    </message>
</context>
<context>
    <name>_MaskPreviewPanel</name>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="133"/>
        <source>Preview</source>
        <translation>Aperçu</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="137"/>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="263"/>
        <source>(no image)</source>
        <translation>(aucune image)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="157"/>
        <source>Image only</source>
        <translation>Image seule</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="158"/>
        <source>Image + Mask</source>
        <translation>Image + masque</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="159"/>
        <source>Mask only</source>
        <translation>Masque seul</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="162"/>
        <source>View:</source>
        <translation>Affichage :</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="168"/>
        <source>Alpha:</source>
        <translation>Alpha :</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="176"/>
        <source>Blue</source>
        <comment>Mask overlay color</comment>
        <translation>Bleu</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="180"/>
        <source>Red</source>
        <comment>Mask overlay color</comment>
        <translation>Rouge</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="184"/>
        <source>Green</source>
        <comment>Mask overlay color</comment>
        <translation>Vert</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="188"/>
        <source>Yellow</source>
        <comment>Mask overlay color</comment>
        <translation>Jaune</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="192"/>
        <source>Mask color:</source>
        <translation>Couleur du masque :</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="269"/>
        <source>No mask assigned</source>
        <translation>Aucun masque attribué</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="275"/>
        <source>Frame %1 — %2</source>
        <translation>Image %1 — %2</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="290"/>
        <source>Failed to load image</source>
        <translation>Échec du chargement de l&apos;image</translation>
    </message>
</context>
<context>
    <name>dialog</name>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="307"/>
        <source>Cancel Export</source>
        <translation>Annuler l&apos;export</translation>
    </message>
</context>
</TS>
