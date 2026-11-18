<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1" language="ja" sourcelanguage="en_US">
<context>
    <name>AdvancedPanel</name>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="148"/>
        <source>Margin</source>
        <translation>マージン</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="167"/>
        <source>Fraction of the free GPU memory one window may use.
Large regions are cut into tiles to fit it.</source>
        <translation>1 つのウィンドウが使用できる空き GPU メモリの割合です。
大きな領域は、これに収まるようタイルに分割されます。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="149"/>
        <source>Must cover the largest displacement of the region of interest.</source>
        <translation>関心領域内の最大変位をカバーする必要があります。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="152"/>
        <source>Use the Model Reach</source>
        <translation>モデルの到達範囲を使う</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="155"/>
        <source>Extra image context around the region of interest so that displaced points stay inside
the window given to the network. Automatic: the model&apos;s reach (32 s px).</source>
        <translation>変位した点がネットワークに渡すウィンドウ内に収まるよう、関心領域の周囲に
追加で含める画像範囲です。自動：モデルの到達範囲（32 s px）。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="166"/>
        <source>GPU Memory</source>
        <translation>GPU メモリ</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="174"/>
        <source>Tile Blending</source>
        <translation>タイルのブレンド</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="175"/>
        <source>Width of the feathered overlap between neighbouring tiles.</source>
        <translation>隣接タイル間でフェザー処理する重なり領域の幅です。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="178"/>
        <source>Automatic</source>
        <translation>自動</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="179"/>
        <source>On Demand</source>
        <translation>オンデマンド</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="180"/>
        <source>Stored Volume</source>
        <translation>ボリューム保持</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="182"/>
        <source>Correlation</source>
        <translation>相関計算</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="183"/>
        <source>How the correlation lookup is computed; the results are identical.
Automatic: stored volume for s8 (fast), on demand for s1-s4
(memory linear in the image size).</source>
        <translation>相関ルックアップの計算方法です。結果はどちらも同一です。
自動：s8 ではボリューム保持（高速）、s1-s4 ではオンデマンド
（メモリは画像サイズに比例）。</translation>
    </message>
</context>
<context>
    <name>AnalysisChart</name>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="75"/>
        <source>too few valid points</source>
        <translation>有効な点が少なすぎます</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="73"/>
        <source>partly untracked</source>
        <translation>一部未追跡</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="77"/>
        <source>no strain value (sparse strain gauge)</source>
        <translation>ひずみ値なし（VSG 内の点が不足）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="80"/>
        <source>gauge endpoint lost</source>
        <translation>ゲージ端点が無効</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="82"/>
        <source>not computed</source>
        <translation>未計算</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="83"/>
        <source>no data</source>
        <translation>データなし</translation>
    </message>
</context>
<context>
    <name>AnalysisTab</name>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="82"/>
        <source>Point</source>
        <comment>Placement tool: a single location</comment>
        <translation>点</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="84"/>
        <source>Line</source>
        <comment>Placement tool: a two-point gauge</comment>
        <translation>線分</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="86"/>
        <source>Rectangle</source>
        <comment>Placement tool</comment>
        <translation>矩形</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="88"/>
        <source>Circle</source>
        <comment>Placement tool</comment>
        <translation>円</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="90"/>
        <source>Polygon</source>
        <comment>Placement tool</comment>
        <translation>多角形</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="102"/>
        <source>Click once to place a point probe.</source>
        <translation>1 回クリックして点プローブを配置します。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="108"/>
        <source>Click twice: opposite corners.</source>
        <translation>2 回クリック：対角の 2 点。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="112"/>
        <source>Click each vertex, then double-click to close.</source>
        <translation>各頂点をクリックし、ダブルクリックで閉じます。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="170"/>
        <source>partly untracked from frame %1</source>
        <translation>フレーム %1 から一部未追跡</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="179"/>
        <source>gaps: no strain value</source>
        <translation>欠損あり：ひずみ値なし</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="201"/>
        <source>Nothing valid along %1: the strain gauge found too few points there (near an edge or a hole). Plot a displacement, or use a smaller gauge on the Strain Field tab.</source>
        <translation>%1 に沿って有効なデータがありません：縁や穴の近くで、VSG 内の点が不足しています。変位を表示するか、「ひずみ場」タブで VSG サイズを小さくしてください。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="207"/>
        <source>Nothing valid along %1: the material under it is untracked (cut out of the frame&apos;s ROI, or out of view).</source>
        <translation>%1 に沿って有効なデータがありません：その下の材料が追跡されていません（そのフレームの ROI から除外されたか、視野の外に出ています）。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="89"/>
        <source>Show</source>
        <comment>Probe list column: visibility checkbox</comment>
        <translation>表示</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="91"/>
        <source>Name</source>
        <comment>Probe list column: the probe&apos;s label</comment>
        <translation>名前</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="93"/>
        <source>Type</source>
        <comment>Probe list column: point, line or region</comment>
        <translation>種類</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="95"/>
        <source>Color</source>
        <comment>Probe list column: color swatch</comment>
        <translation>色</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="100"/>
        <source>Color…</source>
        <translation>色…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="97"/>
        <source>Note</source>
        <comment>Probe list column: why a probe shows gaps</comment>
        <translation>備考</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="101"/>
        <source>Delete</source>
        <comment>Button: delete the selected probe</comment>
        <translation>削除</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="103"/>
        <source>Clear All</source>
        <translation>すべてクリア</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="183"/>
        <source>Statistic:</source>
        <translation>統計量：</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="56"/>
        <source>Mean</source>
        <comment>Statistic: arithmetic mean</comment>
        <translation>平均</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="58"/>
        <source>Median</source>
        <comment>Statistic</comment>
        <translation>中央値</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="59"/>
        <source>Maximum</source>
        <comment>Statistic</comment>
        <translation>最大値</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="60"/>
        <source>Minimum</source>
        <comment>Statistic</comment>
        <translation>最小値</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="61"/>
        <source>Standard deviation</source>
        <translation>標準偏差</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="62"/>
        <source>Valid fraction</source>
        <translation>有効割合</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="46"/>
        <source>Crack opening</source>
        <translation>き裂開口変位</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="489"/>
        <source>Strain has not been computed yet. Compute it on the Strain Field tab, or plot a displacement.</source>
        <translation>ひずみはまだ計算されていません。「ひずみ場」タブで計算するか、変位を表示してください。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="533"/>
        <source>Gauge quantities need a line probe.</source>
        <translation>ゲージ量には線プローブが必要です。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="530"/>
        <source>No visible probe can show this quantity.</source>
        <translation>この量を表示できる可視プローブがありません。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="166"/>
        <source>no valid data: %1</source>
        <translation>有効なデータなし：%1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="174"/>
        <source>endpoint lost from frame %1</source>
        <translation>フレーム %1 から端点が無効</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="177"/>
        <source>gaps: too few valid points</source>
        <translation>欠損あり：有効点が少なすぎる</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="185"/>
        <source>not plotted: gauges need a line</source>
        <translation>未表示：ゲージには線が必要</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="188"/>
        <source>not plotted: one point has no spread or coverage</source>
        <translation>未表示：1 点にはばらつきも被覆率もない</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="190"/>
        <source>not plotted</source>
        <translation>未表示</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="513"/>
        <source>Added probe &apos;%1&apos;.</source>
        <translation>プローブ「%1」を追加しました。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="599"/>
        <source>Clear All Probes</source>
        <translation>すべてのプローブをクリア</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="600"/>
        <source>Delete every probe? This cannot be undone.</source>
        <translation>すべてのプローブを削除しますか？この操作は取り消せません。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="110"/>
        <source>Point</source>
        <comment>Probe type</comment>
        <translation>点</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="111"/>
        <source>Line</source>
        <comment>Probe type</comment>
        <translation>線分</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="112"/>
        <source>Region</source>
        <comment>Probe type: an enclosed area</comment>
        <translation>領域</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="480"/>
        <source>Run a DIC analysis to plot probes.</source>
        <translation>プローブを描画するには、先に DIC 解析を実行してください。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="92"/>
        <source>Virtual extensometer</source>
        <comment>Placement tool</comment>
        <translation>仮想伸び計</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="94"/>
        <source>Crack gauge</source>
        <comment>Placement tool: a line across a crack</comment>
        <translation>き裂ゲージ</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="104"/>
        <source>Click twice: start and end. A line is also a virtual extensometer and a crack-opening gauge.</source>
        <translation>2 回クリック：始点と終点。線は仮想伸び計およびき裂開口ゲージとしても使えます。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="110"/>
        <source>Click twice: center, then the edge.</source>
        <translation>2 回クリック：中心、次に円周。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="114"/>
        <source>Click the two gauge points. The chart then shows the strain between them.</source>
        <translation>2 つの標点をクリックします。グラフにはその間のひずみが表示されます。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="118"/>
        <source>Click one point on each side of the crack. The chart then shows how far it opens.</source>
        <translation>き裂の両側に 1 点ずつクリックします。グラフにはき裂の開口量が表示されます。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="124"/>
        <source>Fit</source>
        <comment>Zoom button: fit the image to the view</comment>
        <translation>フィット</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="131"/>
        <source>Fit image to viewport</source>
        <translation>画像をビューポートに合わせる</translation>
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
        <translation>100% (1:1) ズーム</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="136"/>
        <source>Zoom in</source>
        <translation>拡大</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="137"/>
        <source>Zoom out</source>
        <translation>縮小</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="138"/>
        <source>Show field</source>
        <comment>Analysis canvas: color the image by the field</comment>
        <translation>場を表示</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="140"/>
        <source>Color the reference image with the plotted field at the current frame. For a gauge reading, the Strain Field tab&apos;s field is shown.</source>
        <translation>現在のフレームで表示中の量の場で参照画像を色付けします。ゲージの読み取り値では「ひずみ場」タブの場を表示します。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="181"/>
        <source>Plot:</source>
        <translation>表示：</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="184"/>
        <source>X axis:</source>
        <translation>X 軸：</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="185"/>
        <source>Strain as:</source>
        <translation>ひずみの表示：</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="187"/>
        <source>Min. valid fraction:</source>
        <translation>最小有効割合：</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="188"/>
        <source>A frame is left blank when fewer than this fraction of a line&apos;s or region&apos;s points are reliable. Guards against a curve that stays smooth while its sample shrinks away.</source>
        <translation>線または領域の信頼できる点の割合がこの値を下回るフレームは空白になります。サンプルが減っていても曲線が滑らかに見えてしまうことを防ぎます。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="195"/>
        <source>Over time</source>
        <comment>Chart view: every frame of each probe</comment>
        <translation>時間変化</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="197"/>
        <source>Each probe&apos;s reading at every frame.</source>
        <translation>各プローブの全フレームでの読み取り値。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="200"/>
        <source>Along the line</source>
        <comment>Chart view: a profile</comment>
        <translation>線に沿った分布</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="207"/>
        <source>Kymograph</source>
        <comment>Chart view: distance against frame</comment>
        <translation>キモグラフ</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="202"/>
        <source>The field along the selected line at the current frame, over the other frames in gray.</source>
        <translation>現在のフレームにおける選択した線に沿った場。他のフレームは背後に灰色で表示されます。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="209"/>
        <source>The field along the selected line at every frame: distance against frame, value as color.</source>
        <translation>選択した線に沿った全フレームの場：距離とフレームの関係を、値を色で表示します。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="214"/>
        <source>Stress–strain</source>
        <comment>Chart view: stress against strain</comment>
        <translation>応力–ひずみ</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="216"/>
        <source>Stress (or load, without A0) against the plotted quantity, one curve per probe: stress-strain with an extensometer, load against opening with a crack gauge. Needs load data.</source>
        <translation>表示中の量に対する応力（A0 がない場合は荷重）の曲線を、プローブごとに描きます：伸び計なら応力–ひずみ曲線、き裂ゲージなら荷重–開口量曲線です。荷重データが必要です。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="227"/>
        <source>Other frames</source>
        <translation>他のフレーム</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="228"/>
        <source>Draw the other frames&apos; profiles faintly behind the current one (at most twelve, evenly spaced).</source>
        <translation>現在のフレームの背後に他のフレームの分布を薄く描画します（最大 12 フレーム、等間隔）。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="233"/>
        <source>Line data (CSV)…</source>
        <translation>線データ（CSV）…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="234"/>
        <source>Export</source>
        <translation>エクスポート</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="235"/>
        <source>Load data…</source>
        <translation>荷重データ…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="236"/>
        <source>Import a testing machine&apos;s load record (CSV) to plot against load or stress, and to draw stress-strain curves.</source>
        <translation>試験機の荷重記録（CSV）をインポートし、荷重または応力に対してプロットし、応力–ひずみ曲線を描きます。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="241"/>
        <source>Probe data (CSV)…</source>
        <translation>プローブデータ（CSV）…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="243"/>
        <source>Chart image…</source>
        <translation>グラフ画像…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="245"/>
        <source>Copy chart</source>
        <translation>グラフをコピー</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="247"/>
        <source>Copy plotted data</source>
        <translation>表示中のデータをコピー</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="559"/>
        <source>Import the testing machine&apos;s load record with Load data… to draw stress-strain curves.</source>
        <translation>応力–ひずみ曲線を描くには、「荷重データ…」で試験機の荷重記録をインポートしてください。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="31"/>
        <source>Displacement U</source>
        <translation>変位 U</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="32"/>
        <source>Displacement V</source>
        <translation>変位 V</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="33"/>
        <source>Displacement magnitude</source>
        <translation>変位の大きさ</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="42"/>
        <source>Extensometer strain</source>
        <translation>伸び計ひずみ</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="43"/>
        <source>Extensometer true strain</source>
        <translation>伸び計の真ひずみ</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="45"/>
        <source>Elongation ΔL</source>
        <translation>伸び ΔL</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="47"/>
        <source>Crack sliding</source>
        <translation>き裂すべり</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="48"/>
        <source>Crack opening magnitude</source>
        <translation>き裂開口変位の大きさ</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="279"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="434"/>
        <source>Frame</source>
        <translation>フレーム</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="281"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="429"/>
        <source>Time (s)</source>
        <translation>時間 (s)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="283"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="431"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="574"/>
        <source>Load (N)</source>
        <translation>荷重 (N)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="286"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="433"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="571"/>
        <source>Stress (MPa)</source>
        <translation>応力 (MPa)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="295"/>
        <source>ratio</source>
        <comment>Strain display unit: plain number</comment>
        <translation>比率</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="420"/>
        <source>Could not draw the field: %1</source>
        <translation>場を描画できませんでした：%1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="130"/>
        <source>Esc cancels placement</source>
        <translation>Esc で配置をキャンセル</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="138"/>
        <source>Drag to move the probe, or drag a handle to reshape it. Delete removes it; F2 renames it.</source>
        <translation>ドラッグでプローブを移動、ハンドルのドラッグで形状を変更します。Delete で削除、F2 で名前を変更します。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="485"/>
        <source>Place a probe on the reference image to begin.</source>
        <translation>参照画像にプローブを配置すると始まります。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="212"/>
        <source>Nothing valid along %1: it lies off the measured area.</source>
        <translation>%1 に沿って有効なデータがありません：測定領域の外にあります。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="610"/>
        <source>A line view shows a field. Choose a field to plot.</source>
        <translation>線のビューは場を表示します。表示する場を選んでください。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="614"/>
        <source>Place a line probe, or select one, to see the field along it.</source>
        <translation>線プローブを配置または選択すると、線に沿った場が表示されます。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="632"/>
        <source>Distance along %1 (%2)</source>
        <translation>%1 に沿った距離（%2）</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="641"/>
        <source>%1, frame %2</source>
        <translation>%1、フレーム %2</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="153"/>
        <source>not plotted: off the measured area</source>
        <translation>未表示：測定領域の外</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="156"/>
        <source>not plotted: a gauge end is off the measured area</source>
        <translation>未表示：ゲージの端点が測定領域の外</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="671"/>
        <source>Export Probe Data</source>
        <translation>プローブデータをエクスポート</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="664"/>
        <source>CSV Files</source>
        <translation>CSV ファイル</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="442"/>
        <source>The load data cannot be matched to the frames: %1</source>
        <translation>荷重データをフレームに対応付けられません：%1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="664"/>
        <location filename="../../gui/panels/analysis/tab.py" line="728"/>
        <source>All Files</source>
        <translation>すべてのファイル</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="699"/>
        <source>Probe export failed: %1</source>
        <translation>プローブのエクスポートに失敗しました：%1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="701"/>
        <source>Probe data written to %1</source>
        <translation>プローブデータを %1 に書き込みました</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="709"/>
        <source>Export Line Data</source>
        <translation>線データをエクスポート</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="718"/>
        <source>Line export failed: %1</source>
        <translation>線データのエクスポートに失敗しました：%1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="720"/>
        <source>Line data written to %1</source>
        <translation>線データを %1 に書き込みました</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="726"/>
        <source>SVG Images</source>
        <translation>SVG 画像</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="737"/>
        <source>Export Chart</source>
        <translation>グラフをエクスポート</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="756"/>
        <source>Chart copied to the clipboard.</source>
        <translation>グラフをクリップボードにコピーしました。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="764"/>
        <source>Plotted data copied to the clipboard.</source>
        <translation>表示中のデータをクリップボードにコピーしました。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="725"/>
        <source>PNG Images</source>
        <translation>PNG 画像</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="727"/>
        <source>PDF Documents</source>
        <translation>PDF ドキュメント</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="748"/>
        <source>Chart export failed: %1</source>
        <translation>グラフのエクスポートに失敗しました：%1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="750"/>
        <source>Chart written to %1</source>
        <translation>グラフを %1 に書き込みました</translation>
    </message>
</context>
<context>
    <name>App</name>
    <message>
        <location filename="../../gui/app.py" line="746"/>
        <source>Run DIC first -- no displacement results to post-process.</source>
        <translation>先に DIC を実行してください —— 後処理する変位結果がありません。</translation>
    </message>
</context>
<context>
    <name>Application</name>
    <message>
        <location filename="../../gui/app.py" line="879"/>
        <source>RAFTcorr has hit an error</source>
        <translation>RAFTcorr でエラーが発生しました</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="880"/>
        <source>An unexpected error occurred. The application may not behave correctly from here on, so saving your project and restarting is recommended.</source>
        <translation>予期しないエラーが発生しました。これ以降アプリケーションが正しく動作しない可能性があるため、プロジェクトを保存して再起動することをお勧めします。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="890"/>
        <source>Details were written to %1</source>
        <translation>詳細を %1 に書き込みました</translation>
    </message>
</context>
<context>
    <name>AutoFixedSelector</name>
    <message>
        <location filename="../../gui/widgets/range_mode.py" line="33"/>
        <source>Auto</source>
        <comment>Color range mode: rescale to the data range</comment>
        <translation>自動</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/range_mode.py" line="36"/>
        <source>Rescale the color range to each frame&apos;s data range</source>
        <translation>各フレームのデータ範囲に合わせてカラー範囲を再スケールします</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/range_mode.py" line="39"/>
        <source>Fixed</source>
        <comment>Color range mode: manual min/max bounds</comment>
        <translation>固定</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/range_mode.py" line="42"/>
        <source>Keep the manual Min/Max bounds for every frame</source>
        <translation>すべてのフレームで手動の最小/最大値を使用します</translation>
    </message>
</context>
<context>
    <name>BatchImportDialog</name>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="368"/>
        <source>Batch Import Region of Interest Masks</source>
        <translation>関心領域マスクを一括インポート</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="419"/>
        <source>Mask Folder:</source>
        <translation>マスクフォルダ:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="420"/>
        <source>(none)</source>
        <translation>(なし)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="425"/>
        <source>Browse...</source>
        <translation>参照…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="444"/>
        <source>Available Masks</source>
        <translation>利用可能なマスク</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="455"/>
        <source>Auto-Match by Name</source>
        <translation>ファイル名で自動マッチ</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="457"/>
        <source>Match mask files to frames by number in filename</source>
        <translation>ファイル名中の番号でマスクをフレームに対応付けます</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="461"/>
        <source>Assign Sequential</source>
        <translation>順次割り当て</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="463"/>
        <source>Assign masks to frames in order starting from frame 0</source>
        <translation>フレーム 0 から順番にマスクを割り当てます</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="471"/>
        <source>Frame Assignments</source>
        <translation>フレーム割り当て</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="474"/>
        <source>Frame</source>
        <translation>フレーム</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="474"/>
        <source>Image</source>
        <translation>画像</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="474"/>
        <source>Mask</source>
        <translation>マスク</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="485"/>
        <source>Assign Selected -&gt;</source>
        <translation>選択項目を割り当て -&gt;</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="487"/>
        <source>Pair selected mask(s) with selected frame(s)</source>
        <translation>選択したマスクと選択したフレームを対応付けます</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="491"/>
        <source>Clear All</source>
        <translation>すべてクリア</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="536"/>
        <source>Select Mask Folder</source>
        <translation>マスクフォルダを選択</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="590"/>
        <source>Failed to read mask file.</source>
        <translation>マスクファイルの読み込みに失敗しました。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="594"/>
        <source>Mismatched shape: %1×%2 (expected %3×%4)</source>
        <translation>形状が一致しません: %1×%2 (期待値 %3×%4)</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="666"/>
        <source>%n mask(s) have mismatched sizes and are disabled.</source>
        <translation>
            <numerusform>%n 個のマスクはサイズが一致しないため無効化されました。</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="719"/>
        <source>Invalid assignment</source>
        <translation>無効な割り当て</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="721"/>
        <source>A frame can only have one mask. Select exactly one mask, or select multiple frames to assign one mask to many.</source>
        <translation>1 つのフレームに割り当てられるマスクは 1 つだけです。マスクを 1 つだけ選択するか、複数のフレームを選択して 1 つのマスクを複数に割り当ててください。</translation>
    </message>
</context>
<context>
    <name>CanvasArea</name>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1223"/>
        <source>Fit</source>
        <translation>フィット</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1224"/>
        <source>Fit image to viewport</source>
        <translation>画像をビューポートに合わせる</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1228"/>
        <source>100%</source>
        <translation>100%</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1229"/>
        <source>Zoom to 100% (1:1)</source>
        <translation>100% (1:1) ズーム</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1232"/>
        <source>Zoom in</source>
        <translation>拡大</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1237"/>
        <source>–</source>
        <translation>–</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1238"/>
        <source>Zoom out</source>
        <translation>縮小</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1329"/>
        <source>EDITING REGION OF INTEREST FOR FRAME %1</source>
        <translation>フレーム %1 の関心領域を編集中</translation>
    </message>
</context>
<context>
    <name>CanvasConfigOverlay</name>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="40"/>
        <source>Mode</source>
        <translation>モード</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="41"/>
        <source>Model</source>
        <translation>モデル</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="42"/>
        <source>Iterations</source>
        <translation>反復回数</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="85"/>
        <source>Accumulative</source>
        <translation>累積式</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="87"/>
        <source>Incremental, every %1 frames</source>
        <translation>逐次式、%1 フレームごと</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="89"/>
        <source>Custom reference frames</source>
        <translation>カスタム参照フレーム</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="90"/>
        <source>Incremental</source>
        <translation>逐次式</translation>
    </message>
</context>
<context>
    <name>ColorRange</name>
    <message>
        <location filename="../../gui/widgets/color_range.py" line="27"/>
        <source>Range</source>
        <translation>範囲</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/color_range.py" line="37"/>
        <source>Min</source>
        <translation>最小</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/color_range.py" line="47"/>
        <source>Max</source>
        <translation>最大</translation>
    </message>
</context>
<context>
    <name>ExportDialog</name>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="45"/>
        <source>Magnitude</source>
        <translation>変位の大きさ</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="46"/>
        <source>Residual</source>
        <translation>残差</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="194"/>
        <source>Open this tab to render a preview.</source>
        <translation>このタブを開くとプレビューが描画されます。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="201"/>
        <location filename="../../gui/dialogs/export_widgets.py" line="383"/>
        <source>Field</source>
        <translation>フィールド</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="208"/>
        <source>Frame</source>
        <translation>フレーム</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="239"/>
        <source>FIELD APPEARANCE</source>
        <translation>フィールドの外観</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="244"/>
        <location filename="../../gui/dialogs/export_widgets.py" line="383"/>
        <source>Colormap</source>
        <translation>カラーマップ</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="248"/>
        <source>Range</source>
        <translation>範囲</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="252"/>
        <location filename="../../gui/dialogs/export_widgets.py" line="383"/>
        <source>Min</source>
        <translation>最小</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="256"/>
        <location filename="../../gui/dialogs/export_widgets.py" line="383"/>
        <source>Max</source>
        <translation>最大</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="263"/>
        <location filename="../../gui/dialogs/export_widgets.py" line="179"/>
        <source>Opacity</source>
        <extracomment>Column widths of the field rows (export, field, colormap, range mode, min / max), shared by the header labels of the Images and Animation tabs. Long-edge presets (pixels) of the &quot;Resolution (long edge)&quot; combos, in pyALDIC&apos;s order; the last entry, &quot;Full resolution&quot;, is 0 (no cap). Older names that share a field&apos;s colour state.</extracomment>
        <translation>不透明度</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="265"/>
        <source>Apply to all fields</source>
        <translation>すべてのフィールドに適用</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="266"/>
        <source>Apply this field&apos;s colormap, opacity and auto-range to every enabled field (each field keeps its own min/max).</source>
        <translation>このフィールドのカラーマップ・不透明度・自動範囲を、有効なすべてのフィールドに適用します（各フィールドの最小/最大は保持）。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="222"/>
        <source>Refresh preview</source>
        <translation>プレビューを更新</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="66"/>
        <source>COLORBAR STYLE</source>
        <translation>カラーバーのスタイル</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="71"/>
        <source>Right</source>
        <translation>右</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="72"/>
        <source>Left</source>
        <translation>左</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="73"/>
        <source>Top</source>
        <translation>上</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="74"/>
        <source>Bottom</source>
        <translation>下</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="77"/>
        <source>Position</source>
        <translation>位置</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="82"/>
        <source>Font size</source>
        <translation>フォントサイズ</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="88"/>
        <source>Font family</source>
        <translation>フォント</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="91"/>
        <source>Bar thickness as a fraction of the picture&apos;s width (bar on the left or right) or height (bar on top or at the bottom).</source>
        <translation>カラーバーの太さ。画像の幅（左右に配置したとき）または高さ（上下に配置したとき）に対する比率です。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="94"/>
        <source>Bar thickness</source>
        <translation>バーの太さ</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="97"/>
        <location filename="../../gui/dialogs/export_preview.py" line="114"/>
        <location filename="../../gui/dialogs/export_preview.py" line="120"/>
        <source>Black</source>
        <translation>黒</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="98"/>
        <location filename="../../gui/dialogs/export_preview.py" line="113"/>
        <location filename="../../gui/dialogs/export_preview.py" line="119"/>
        <source>White</source>
        <translation>白</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="101"/>
        <source>Colour behind the colorbar and around the picture; text and ticks are drawn in the opposite colour.</source>
        <translation>カラーバーの背後と画像の周囲の色です。文字と目盛りは反対の色で描きます。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="104"/>
        <source>Background</source>
        <translation>背景</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="107"/>
        <source>Add a blank border around the exported content, as a fraction of the long edge (0 = none).</source>
        <translation>エクスポートする内容の周囲に空白の枠を追加します。幅は長辺に対する割合です（0 = なし）。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="110"/>
        <source>Margin</source>
        <translation>余白</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="116"/>
        <source>Margin color</source>
        <translation>余白の色</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="121"/>
        <source>Transparent</source>
        <translation>透明</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="124"/>
        <source>Fill used where the background image would have been, when &apos;Show background image&apos; is off.
Transparency is kept for PNG and TIFF; the other formats get white instead.</source>
        <translation>「背景画像を表示」がオフのとき、背景画像があった場所を埋める色です。
PNG と TIFF では透明を保持し、その他の形式では白になります。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="128"/>
        <source>Hidden background</source>
        <translation>非表示時の背景</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="290"/>
        <source>Preview failed: </source>
        <translation>プレビューに失敗しました：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="324"/>
        <source>Enable a field on the Images tab to preview.</source>
        <translation>プレビューするには「画像」タブでフィールドを有効にしてください。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="328"/>
        <source>No data for this field/frame.</source>
        <translation>このフィールド/フレームにはデータがありません。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="180"/>
        <source>Field opacity (0 = transparent, 1 = fully opaque)</source>
        <translation>場の不透明度（0 = 透明、1 = 完全に不透明）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="290"/>
        <source>Full resolution</source>
        <translation>フル解像度</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="297"/>
        <source>All</source>
        <translation>すべて</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="299"/>
        <source>None</source>
        <translation>なし</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="370"/>
        <source>Select:</source>
        <translation>選択：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="383"/>
        <source>Export</source>
        <translation>出力</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="383"/>
        <source>Auto</source>
        <translation>自動</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="400"/>
        <source>PHYSICAL UNITS</source>
        <translation>物理単位</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="404"/>
        <source>Enable physical units</source>
        <translation>物理単位を有効化</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="406"/>
        <source>Scale displacement values by pixel size and show physical units on colorbar labels. Strain is dimensionless and unaffected.</source>
        <translation>変位値をピクセルサイズでスケールし、カラーバーのラベルに物理単位を表示します。ひずみは無次元量のため影響を受けません。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="423"/>
        <source>/ pixel</source>
        <translation>/ ピクセル</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="425"/>
        <source>Pixel size</source>
        <translation>ピクセルサイズ</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="434"/>
        <source>fps</source>
        <translation>fps</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="436"/>
        <source>Frame rate</source>
        <translation>フレームレート</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="462"/>
        <source>FRAME RANGE</source>
        <translation>フレーム範囲</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="464"/>
        <source>All frames</source>
        <translation>すべてのフレーム</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="469"/>
        <source>Numbered like the frame slider: frame 1 is the reference image.
File names use the same numbers: frame 1 is written as _0001.</source>
        <translation>フレームスライダーと同じ番号です。フレーム 1 は参照画像です。
ファイル名も同じ番号を使い、フレーム 1 は _0001 として書き出されます。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="472"/>
        <source>From</source>
        <comment>Frame range: starting frame</comment>
        <translation>開始</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="477"/>
        <source>to</source>
        <comment>Frame range: ending frame</comment>
        <translation>まで</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="174"/>
        <source>Export Results</source>
        <extracomment>GUI field name -&gt; variable name in the data files (raftcorr.export.scientific). Long-edge caps the tabs start from (pixels, 0 = full resolution): pictures keep the size their DPI gives, animations start at pyALDIC&apos;s 1024 px.</extracomment>
        <translation>結果をエクスポート</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="201"/>
        <source>OUTPUT FOLDER</source>
        <translation>出力フォルダ</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="207"/>
        <source>Select output folder…</source>
        <translation>出力フォルダを選択…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="213"/>
        <source>Browse…</source>
        <translation>参照…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="218"/>
        <source>Open Folder</source>
        <translation>フォルダを開く</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="232"/>
        <source>Data</source>
        <translation>データ</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="233"/>
        <source>Images</source>
        <translation>画像</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="234"/>
        <source>Animation</source>
        <translation>アニメーション</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="235"/>
        <source>Report</source>
        <translation>レポート</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="320"/>
        <source>FORMAT</source>
        <translation>形式</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="322"/>
        <source>NumPy Archive (.npz)</source>
        <translation>NumPy アーカイブ (.npz)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="329"/>
        <source>CSV (per frame)</source>
        <translation>CSV(フレーム単位)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="345"/>
        <source>DISPLACEMENT</source>
        <translation>変位</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="348"/>
        <source>STRAIN</source>
        <translation>ひずみ</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="287"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="351"/>
        <source>Run Compute Strain first.</source>
        <translation>先に「ひずみを計算」を実行してください。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="268"/>
        <source>Export: images not readable (%1); pictures go on white.</source>
        <translation>エクスポート：画像を読み込めません（%1）。図は白背景で出力されます。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="272"/>
        <source>Export: the images do not match the results&apos; image size; pictures go on white.</source>
        <translation>エクスポート：画像のサイズが結果の画像サイズと一致しません。図は白背景で出力されます。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="286"/>
        <source>This run has no intensity residual.</source>
        <translation>この実行には輝度残差がありません。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="324"/>
        <source>MATLAB v7.3 (.mat)</source>
        <translation>MATLAB v7.3 (.mat)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="326"/>
        <source>An HDF5-based MAT file: load it in MATLAB with load().</source>
        <translation>HDF5 ベースの MAT ファイルです。MATLAB で load() により読み込めます。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="327"/>
        <source>HDF5 (.h5)</source>
        <translation>HDF5 (.h5)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="332"/>
        <source>One table per computed frame, one row per point of the region of interest.
Dense fields make large files: a 1000 x 1000 px region is a million rows per frame.</source>
        <translation>計算済みフレームごとに 1 つの表、関心領域の点ごとに 1 行です。
密な場はファイルが大きくなります：1000 x 1000 px の領域はフレームあたり 100 万行です。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="355"/>
        <source>✓ Parameters file (JSON) always exported</source>
        <translation>✓ パラメータファイル (JSON) は常にエクスポートされます</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="356"/>
        <source>✓ Grid coordinates (X, Y), ROI and computed-frame flags always included</source>
        <translation>✓ グリッド座標 (X, Y)、ROI、計算済みフレームのフラグは常に含まれます</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="357"/>
        <source>Displacements are written in pixels (+u right, +v down); the pixel size and frame rate above are recorded with them.</source>
        <translation>変位はピクセル単位で書き出されます（+u 右、+v 下）。上のピクセルサイズとフレームレートも併せて記録されます。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="363"/>
        <source>Export Data</source>
        <translation>データをエクスポート</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="412"/>
        <source>IMAGE SETTINGS</source>
        <translation>画像設定</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="423"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="530"/>
        <source>Format</source>
        <translation>形式</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="376"/>
        <source>Deformed frame</source>
        <translation>変形フレーム</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="239"/>
        <source>Preview &amp; Colorbar</source>
        <translation>プレビューとカラーバー</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="334"/>
        <source>NPZ: one file per frame (default: single merged file)</source>
        <translation>NPZ: フレーム単位で 1 ファイル(デフォルト: 統合 1 ファイル)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="337"/>
        <source>One .npz per computed frame, in a folder of its own: that frame&apos;s variables, the grid coordinates, the ROI and the metadata.</source>
        <translation>計算済みのフレームごとに 1 つの .npz を専用フォルダーに保存します：そのフレームの変数、格子座標、ROI、メタデータ。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="377"/>
        <source>Reference frame</source>
        <translation>参照フレーム</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="383"/>
        <source>Show background image</source>
        <translation>背景画像を表示</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="394"/>
        <source>Hide untracked regions</source>
        <translation>追跡されていない領域を隠す</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="397"/>
        <source>Leave points without a result transparent in the pictures, for example material that left the field of view, instead of filling them with the nearest value inside the region of interest. Data files always keep NaN there.</source>
        <translation>結果のない点（視野の外に出た材料など）を、画像では関心領域内で最も近い値で埋めずに透明のままにします。データファイルではこれらの点は常に NaN です。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="430"/>
        <source>Cap the long edge of each written image (the larger of width/height, in pixels; the aspect ratio and layout are kept).
Lower = faster and much smaller on disk; for PDF and SVG it caps the embedded picture. &apos;Full resolution&apos; writes the size set by DPI.</source>
        <translation>書き出す各画像の長辺（幅と高さの大きい方、ピクセル単位。縦横比とレイアウトは保持）を制限します。
小さいほど速く、ファイルもずっと小さくなります。PDF と SVG では埋め込む画像を制限します。「フル解像度」は DPI で決まるサイズで書き出します。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="435"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="543"/>
        <source>Resolution (long edge)</source>
        <translation>解像度（長辺）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="443"/>
        <source>JPEG quality (higher = larger file). Ignored for the other formats.</source>
        <translation>JPEG の品質（高いほどファイルが大きくなります）。その他の形式では無視されます。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="446"/>
        <source>JPEG quality</source>
        <translation>JPEG 品質</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="454"/>
        <source>Resolution of the written figure: about 8.6 x 7 inches at this many dots per inch.
A resolution cap (long edge) lowers it.</source>
        <translation>書き出す図の解像度：約 8.6 x 7 インチをこの DPI で出力します。
解像度の上限（長辺）を設定すると下がります。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="469"/>
        <source>DPI</source>
        <translation>DPI</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="471"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="569"/>
        <source>Include colorbar</source>
        <translation>カラーバーを含める</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="474"/>
        <source>Add a colorbar to each image; its position and style are set on the Preview &amp; Colorbar tab.</source>
        <translation>各画像にカラーバーを付けます。位置とスタイルは「プレビューとカラーバー」タブで設定します。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="479"/>
        <source>Uncheck to export the field on its own, with no speckle image behind it. Pick the fill on the Preview &amp; Colorbar tab.</source>
        <translation>チェックを外すと、背後にスペックル画像を含めず場のみをエクスポートします。塗りつぶしは「プレビューとカラーバー」タブで選択します。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="487"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="582"/>
        <source>Render as</source>
        <translation>描画方法</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="497"/>
        <source>Export Images</source>
        <translation>画像をエクスポート</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="511"/>
        <source>ANIMATION SETTINGS</source>
        <translation>アニメーション設定</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="537"/>
        <source>Cap the animation&apos;s long edge (the larger of width/height).
Lower = faster and much smaller. Strongly recommended for GIF, whose size explodes at native resolution. &apos;Full resolution&apos; draws the images at their own pixel size.</source>
        <translation>アニメーションの長辺（幅と高さの大きい方）を制限します。
小さいほど速く、ずっと小さくなります。GIF では強く推奨します（元の解像度ではサイズが急増します）。「フル解像度」は画像を元のピクセルサイズで描きます。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="552"/>
        <source>FPS</source>
        <translation>FPS</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="562"/>
        <source>Export every Nth frame (1 = every frame). Higher is faster and smaller
but looks choppier. Playback duration is preserved (the FPS above is the pre-decimation rate).</source>
        <translation>N フレームごとに 1 枚エクスポートします（1 = 全フレーム）。大きいほど高速・小容量ですが、
カクついて見えます。再生時間は維持されます（上の FPS は間引き前のレート）。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="567"/>
        <source>Frame step</source>
        <translation>フレーム間引き</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="572"/>
        <source>Add a colorbar to each frame; its position and style are set on the Preview &amp; Colorbar tab.</source>
        <translation>各フレームにカラーバーを付けます。位置とスタイルは「プレビューとカラーバー」タブで設定します。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="576"/>
        <source>Uncheck to export the field on its own, with no speckle image behind it. GIF and MP4 cannot store transparency, so a transparent fill is written as white.</source>
        <translation>チェックを外すと、背後にスペックル画像を含めず場のみをエクスポートします。GIF と MP4 は透明度を保存できないため、透明の塗りつぶしは白として書き出されます。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="595"/>
        <source>Export Animation</source>
        <translation>アニメーションをエクスポート</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="605"/>
        <source>CONTENT</source>
        <translation>内容</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="607"/>
        <source>Parameter summary table</source>
        <translation>パラメータ要約表</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="609"/>
        <source>Field statistics (min/max/mean/std per frame)</source>
        <translation>フィールド統計（フレームごとの最小/最大/平均/標準偏差）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="611"/>
        <source>Sample field images</source>
        <translation>フィールド画像のサンプル</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="619"/>
        <source>Sample every</source>
        <translation>抽出間隔</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="625"/>
        <source>frames</source>
        <comment>Report: sample every N frames</comment>
        <translation>フレーム</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="631"/>
        <source>FIELDS</source>
        <translation>フィールド</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="633"/>
        <source>Displacement:</source>
        <translation>変位：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="634"/>
        <source>Strain:</source>
        <translation>ひずみ：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="643"/>
        <source>Format: HTML (self-contained, view in any browser)</source>
        <translation>形式：HTML（自己完結型、任意のブラウザで表示可能）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="647"/>
        <source>Generate Report</source>
        <translation>レポートを生成</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="808"/>
        <source>Select Output Folder</source>
        <translation>出力フォルダを選択</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="909"/>
        <source>Writing %1 (%2%)</source>
        <translation>%1 を書き込み中 (%2%)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="942"/>
        <source>Error: %1</source>
        <translation>エラー：%1</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="884"/>
        <source>Starting…</source>
        <translation>開始中…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="380"/>
        <source>Deformed: the field is drawn where each point is in that frame, over the frame&apos;s own photo.
Reference: drawn at the points&apos; positions in frame 1, over frame 1.</source>
        <translation>変形フレーム：各点がそのフレームで位置する場所に場を描画し、そのフレーム自身の写真に重ねます。
参照フレーム：フレーム 1 での点の位置に描画し、フレーム 1 に重ねます。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="387"/>
        <source>The images of this run are not available.</source>
        <translation>この実行の画像は利用できません。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="461"/>
        <source>Each frame</source>
        <translation>フレームごと</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="462"/>
        <source>Whole sequence</source>
        <translation>シーケンス全体</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="464"/>
        <source>How Auto picks the colour range: from each frame&apos;s own values (like the main window),
or one range from every exported frame, so a colour means the same value in every image.</source>
        <translation>自動で色範囲を決める方法：各フレーム自身の値から決める（メインウィンドウと同じ）か、
エクスポートする全フレームから 1 つの範囲を決め、どの画像でも同じ色が同じ値を表すようにします。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="459"/>
        <source>Auto range</source>
        <translation>自動範囲</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="483"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="579"/>
        <source>Crop to ROI</source>
        <translation>ROI で切り抜く</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="484"/>
        <source>Cut the picture to the region of interest plus a small margin.</source>
        <translation>図を関心領域に少し余白を加えた範囲で切り抜きます。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="523"/>
        <source>MP4 needs the imageio and imageio-ffmpeg packages (pip install imageio imageio-ffmpeg). Save as GIF instead.</source>
        <translation>MP4 には imageio と imageio-ffmpeg パッケージが必要です（pip install imageio imageio-ffmpeg）。代わりに GIF で保存してください。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="580"/>
        <source>Cut the frames to the region of interest plus a small margin.</source>
        <translation>各フレームを関心領域に少し余白を加えた範囲で切り抜きます。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="587"/>
        <source>Auto range: one range for the whole animation, so a colour means the same value in every frame.</source>
        <translation>自動範囲：アニメーション全体で 1 つの範囲を使うため、どのフレームでも同じ色が同じ値を表します。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="614"/>
        <source>Drawn like the Images tab: same colormap, range, opacity and configuration.</source>
        <translation>「画像」タブと同じように描画します：カラーマップ、範囲、不透明度、構成が同じです。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="737"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="789"/>
        <source>Min must be below Max for %1.</source>
        <translation>%1 の最小値は最大値より小さくする必要があります。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="743"/>
        <source>Nothing to export: enable at least one field with computed frames in the frame range.</source>
        <translation>エクスポートするものがありません：フレーム範囲内に計算済みフレームがある場を少なくとも 1 つ有効にしてください。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="751"/>
        <source>Select at least one format.</source>
        <translation>形式を少なくとも 1 つ選択してください。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="755"/>
        <source>Select at least one variable to export.</source>
        <translation>エクスポートする変数を少なくとも 1 つ選択してください。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="792"/>
        <source>Nothing to export: select a field or the parameter table.</source>
        <translation>エクスポートするものがありません：場またはパラメータ表を選択してください。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="845"/>
        <source>… and %1 more</source>
        <translation>… ほか %1 件</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="847"/>
        <source>Replace Existing Files?</source>
        <translation>既存のファイルを置き換えますか？</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="848"/>
        <source>These already exist in the output folder and will be replaced:

%1</source>
        <translation>次のファイルは出力フォルダに既に存在し、置き換えられます：

%1</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="867"/>
        <source>The results changed after this dialog was opened (a new run, strain or project). Close it and open Export again to export them.</source>
        <translation>このダイアログを開いた後に結果が変わりました（新しい実行、ひずみ、またはプロジェクト）。ダイアログを閉じ、もう一度「エクスポート」を開いてからエクスポートしてください。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="871"/>
        <source>Nothing exported: the existing files were kept.</source>
        <translation>何もエクスポートされませんでした：既存のファイルは保持されています。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="896"/>
        <source>Stopping…</source>
        <translation>停止中…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="907"/>
        <source>Rendering %1 (%2/%3)</source>
        <translation>%1 を描画中 (%2/%3)</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/dialogs/export_dialog.py" line="925"/>
        <source>Exported %n file(s) → %1</source>
        <translation>
            <numerusform>%n 個のファイルをエクスポートしました → %1</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/dialogs/export_dialog.py" line="927"/>
        <source>Exported %n image(s) → %1</source>
        <translation>
            <numerusform>%n 枚の画像をエクスポートしました → %1</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="943"/>
        <source>Export failed: %1</source>
        <translation>エクスポートに失敗しました：%1</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/dialogs/export_dialog.py" line="951"/>
        <source>Export cancelled: %n file(s) written before stopping.</source>
        <translation>
            <numerusform>エクスポートをキャンセルしました：停止前に %n 個のファイルを書き込みました。</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="931"/>
        <source>No animation was written. See the log for details.</source>
        <translation>アニメーションは書き込まれませんでした。詳細はログを参照してください。</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/dialogs/export_dialog.py" line="933"/>
        <source>Exported %n animation(s) → %1</source>
        <translation>
            <numerusform>%n 個のアニメーションをエクスポートしました → %1</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="935"/>
        <source>Report saved → %1</source>
        <translation>レポートを保存しました → %1</translation>
    </message>
</context>
<context>
    <name>FieldSelector</name>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="19"/>
        <source>Horizontal displacement (+ right)</source>
        <translation>水平変位（右向きが正）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="20"/>
        <source>Vertical displacement (+ down)</source>
        <translation>垂直変位（下向きが正）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="21"/>
        <source>Displacement magnitude</source>
        <translation>変位の大きさ</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="22"/>
        <source>Intensity residual: local RMS difference between the reference image and
the deformed image mapped back by the displacement. Large where the
tracking failed.</source>
        <translation>輝度残差：参照画像と、変位で逆写像した変形画像との
局所 RMS 差です。追跡に失敗した箇所で大きくなります。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="26"/>
        <source>Disp U</source>
        <translation>変位 U</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="26"/>
        <source>Disp V</source>
        <translation>変位 V</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="27"/>
        <source>Magnitude</source>
        <translation>変位の大きさ</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/viz_controller.py" line="150"/>
        <location filename="../../gui/widgets/field_selector.py" line="27"/>
        <source>Residual</source>
        <translation>残差</translation>
    </message>
</context>
<context>
    <name>FrameNavigator</name>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="59"/>
        <source>Previous frame</source>
        <translation>前フレーム</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="69"/>
        <location filename="../../gui/widgets/frame_navigator.py" line="170"/>
        <source>Play animation</source>
        <translation>アニメーションを再生</translation>
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
        <translation>次のフレーム</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="93"/>
        <source>Playback speed</source>
        <translation>再生速度</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="98"/>
        <location filename="../../gui/widgets/frame_navigator.py" line="194"/>
        <source>FRAME 0/0</source>
        <translation>フレーム 0/0</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="159"/>
        <source>⏸</source>
        <translation>⏸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="160"/>
        <source>Pause animation</source>
        <translation>アニメーションを一時停止</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="191"/>
        <source>FRAME %1/%2</source>
        <translation>フレーム %1/%2</translation>
    </message>
</context>
<context>
    <name>ImageCanvas</name>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1067"/>
        <source>Load images first before drawing a Region of Interest.</source>
        <translation>関心領域を描画する前に、まず画像を読み込んでください。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1086"/>
        <source>The three points are nearly collinear — pick points spread around the circle&apos;s edge.</source>
        <translation>3 点がほぼ一直線です — 円周上に分散させて 3 点を選んでください。</translation>
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
        <translation>ファイル名</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="99"/>
        <source>Region</source>
        <comment>Image list column: ROI status</comment>
        <translation>領域</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="193"/>
        <location filename="../../gui/widgets/image_list.py" line="255"/>
        <source>Add</source>
        <translation>追加</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="249"/>
        <source>Edit</source>
        <translation>編集</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="252"/>
        <source>Need</source>
        <translation>未設定</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/widgets/image_list.py" line="368"/>
        <source>Import Region of Interest for %n frame(s)</source>
        <translation>
            <numerusform>%n フレームに関心領域をインポート</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="379"/>
        <source>Clear Region of Interest (%1 with region)</source>
        <translation>関心領域をクリア（%1 フレームに領域あり）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="383"/>
        <source>Clear Region of Interest</source>
        <translation>関心領域をクリア</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/widgets/image_list.py" line="394"/>
        <source>Delete %n image(s)</source>
        <translation>
            <numerusform>%n 個の画像を削除</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/widgets/image_list.py" line="435"/>
        <source>Select %n Mask File(s)</source>
        <translation>
            <numerusform>%n 個のマスクファイルを選択</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="440"/>
        <source>Images</source>
        <translation>画像</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="441"/>
        <source>All Files</source>
        <translation>すべてのファイル</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="450"/>
        <source>Selected %1 files for %2 frames — count must match</source>
        <translation>%2 フレームに対し %1 個のファイルが選択されました — 数量が一致する必要があります</translation>
    </message>
</context>
<context>
    <name>ImageMessages</name>
    <message>
        <location filename="../../gui/controllers/image_controller.py" line="43"/>
        <source>Cannot open the images</source>
        <translation>画像を開けません</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/image_controller.py" line="47"/>
        <source>%n image(s), %1 x %2 px, %3</source>
        <translation>
            <numerusform>画像 %n 枚、%1 x %2 px、%3</numerusform>
        </translation>
    </message>
</context>
<context>
    <name>LeftSidebar</name>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="176"/>
        <source>IMAGES</source>
        <translation>画像</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="183"/>
        <source>Natural Sort (1, 2, …, 10)</source>
        <translation>自然順ソート (1, 2, …, 10)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="259"/>
        <source>WORKFLOW TYPE</source>
        <translation>ワークフロー種別</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="186"/>
        <source>Sort by embedded numbers: image1, image2, ..., image10 (default).
Unchecked: plain alphabetical order, for zero-padded names.</source>
        <translation>ファイル名中の数字順でソート：image1, image2, ..., image10（デフォルト）。
オフ：単純なアルファベット順。ゼロ埋めされた名前向けです。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="266"/>
        <source>REGION OF INTEREST</source>
        <translation>関心領域</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="278"/>
        <source>PARAMETERS</source>
        <translation>パラメータ</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="284"/>
        <source>ADVANCED</source>
        <translation>詳細設定</translation>
    </message>
</context>
<context>
    <name>LoadDataDialog</name>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="103"/>
        <source>Load Data</source>
        <translation>荷重データ</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="115"/>
        <source>No file chosen.</source>
        <translation>ファイルが選択されていません。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="117"/>
        <source>Choose file…</source>
        <translation>ファイルを選択…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="137"/>
        <source>Load column:</source>
        <translation>荷重の列：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="139"/>
        <source>By time</source>
        <translation>時間で</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="140"/>
        <source>By frame number</source>
        <translation>フレーム番号で</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="144"/>
        <source>Match rows to frames:</source>
        <translation>行とフレームの対応：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="147"/>
        <source>Time column:</source>
        <translation>時間の列：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="153"/>
        <source>The machine&apos;s time at the reference image. If the camera started 2 s after the machine, enter 2.</source>
        <translation>参照画像の時点における試験機の時間。カメラが試験機より 2 s 遅れて開始した場合は 2 と入力します。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="155"/>
        <source>Offset:</source>
        <translation>オフセット：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="163"/>
        <source>Frame column:</source>
        <translation>フレームの列：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="167"/>
        <source>The first image is numbered:</source>
        <translation>最初の画像の番号：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="174"/>
        <source>Initial cross-section, for engineering stress F / A0 in MPa. Leave at 0 for load only.</source>
        <translation>初期断面積。公称応力 F / A0（MPa）の計算に使います。荷重のみの場合は 0 のままにします。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="176"/>
        <source>Cross-section A0:</source>
        <translation>断面積 A0：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="192"/>
        <source>Remove Load Data</source>
        <translation>荷重データを削除</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="201"/>
        <source>Camera frame rate: %1 fps, from Physical Units.</source>
        <translation>カメラのフレームレート：%1 fps（「物理単位」の設定）。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="204"/>
        <source>Set the camera frame rate under Physical Units to match by time.</source>
        <translation>時間で対応付けるには、「物理単位」でカメラのフレームレートを設定してください。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="227"/>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="245"/>
        <source>(unnamed)</source>
        <translation>（名前なし）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="324"/>
        <source>No frame falls within the record: check the columns and the offset.</source>
        <translation>記録の範囲に入るフレームがありません：列とオフセットを確認してください。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="328"/>
        <source>Frames with a load: %1 of %2.</source>
        <translation>荷重のあるフレーム：%1 / %2。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="334"/>
        <source>Open Load Data</source>
        <translation>荷重データを開く</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="335"/>
        <source>CSV Files</source>
        <translation>CSV ファイル</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="335"/>
        <source>All Files</source>
        <translation>すべてのファイル</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="341"/>
        <source>Could not read %1: %2</source>
        <translation>%1 を読み込めませんでした：%2</translation>
    </message>
</context>
<context>
    <name>MainWindow</name>
    <message>
        <location filename="../../gui/app.py" line="136"/>
        <source>&amp;File</source>
        <translation>ファイル</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="161"/>
        <source>Quit</source>
        <translation>終了</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="165"/>
        <source>&amp;Settings</source>
        <translation>設定</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="168"/>
        <source>Theme</source>
        <translation>テーマ</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="172"/>
        <source>Dark</source>
        <comment>Color theme</comment>
        <translation>ダーク</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="172"/>
        <source>Light</source>
        <comment>Color theme</comment>
        <translation>ライト</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="182"/>
        <source>Language</source>
        <translation>言語</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="363"/>
        <source>Suggested: RAFTcorr-%1. Largest displacement of the region of interest about %2 px (99th percentile over frames %3); RAFTcorr-%1 reaches %4 px.</source>
        <translation>推奨：RAFTcorr-%1。関心領域の最大変位は約 %2 px（フレーム %3 での 99 パーセンタイル）、RAFTcorr-%1 の到達範囲は %4 px です。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="456"/>
        <source>Language changed</source>
        <translation>言語を変更しました</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="627"/>
        <source>All Files</source>
        <translation>すべてのファイル</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="200"/>
        <source>Select Image Folder</source>
        <translation>画像フォルダを選択</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="322"/>
        <source>File Association Failed</source>
        <translation>ファイル関連付けに失敗</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="325"/>
        <source>File Association</source>
        <translation>ファイル関連付け</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="146"/>
        <source>Open Image Folder…</source>
        <translation>画像フォルダを開く…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="148"/>
        <source>Open Project…</source>
        <translation>プロジェクトを開く…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="149"/>
        <location filename="../../gui/app.py" line="230"/>
        <location filename="../../gui/app.py" line="237"/>
        <source>Save Project</source>
        <translation>プロジェクトを保存</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="150"/>
        <source>Save Project As…</source>
        <translation>名前を付けてプロジェクトを保存…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="152"/>
        <source>Export Results…</source>
        <translation>結果をエクスポート…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="157"/>
        <source>Associate .raftcorr files with RAFTcorr…</source>
        <translation>.raftcorr ファイルを RAFTcorr に関連付け…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="158"/>
        <source>Register .raftcorr so double-clicking a project file opens RAFTcorr (current user only, no admin rights needed).</source>
        <translation>.raftcorr を登録し、プロジェクトファイルをダブルクリックすると RAFTcorr が開くようにします（現在のユーザーのみ、管理者権限は不要）。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="191"/>
        <source>&amp;Help</source>
        <translation>ヘルプ</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="192"/>
        <source>Installed Models…</source>
        <translation>インストール済みモデル…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="193"/>
        <location filename="../../gui/app.py" line="392"/>
        <location filename="../../gui/app.py" line="394"/>
        <source>Self-test</source>
        <translation>セルフテスト</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="194"/>
        <location filename="../../gui/app.py" line="399"/>
        <source>About RAFTcorr</source>
        <translation>RAFTcorr について</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="209"/>
        <source>Cancel</source>
        <translation>キャンセル</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="231"/>
        <source>Load the images first: a project keeps the images, the regions of interest, the settings and the results together.</source>
        <translation>先に画像を読み込んでください：プロジェクトは画像、関心領域、設定、結果をまとめて保存します。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="252"/>
        <source>Project saved: %1</source>
        <translation>プロジェクトを保存しました：%1</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="254"/>
        <source>Saving the project</source>
        <translation>プロジェクトを保存中</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="268"/>
        <source>Include Results?</source>
        <translation>結果を含めますか？</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="269"/>
        <source>Include the computed results in this project?</source>
        <translation>このプロジェクトに計算結果を含めますか？</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="271"/>
        <source>Including results (about %1 uncompressed) lets you reopen the project without recomputing. Choose No to save a small settings-only file for sharing.</source>
        <translation>結果（非圧縮で約 %1）を含めると、再計算せずにプロジェクトを開き直せます。「いいえ」を選ぶと、共有用に設定だけの小さなファイルを保存します。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="283"/>
        <source>Open Project</source>
        <translation>プロジェクトを開く</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="290"/>
        <source>Opening the project</source>
        <translation>プロジェクトを開いています</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="298"/>
        <source>Images not found</source>
        <translation>画像が見つかりません</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="299"/>
        <source>The image folder of this project was not found. Choose its new location.</source>
        <translation>このプロジェクトの画像フォルダが見つかりませんでした。新しい場所を選択してください。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="300"/>
        <source>Image folder of the project</source>
        <translation>プロジェクトの画像フォルダ</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="307"/>
        <source>Cannot open the project</source>
        <translation>プロジェクトを開けません</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="313"/>
        <source>Project opened: %1</source>
        <translation>プロジェクトを開きました：%1</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="323"/>
        <source>Could not register .raftcorr files: </source>
        <translation>.raftcorr ファイルを登録できませんでした：</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="326"/>
        <source>Done. Double-clicking a .raftcorr file will now open RAFTcorr and restore that project.</source>
        <translation>完了しました。これで .raftcorr ファイルをダブルクリックすると RAFTcorr が開き、そのプロジェクトが復元されます。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="331"/>
        <source>Load at least two images first.</source>
        <translation>先に画像を 2 枚以上読み込んでください。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="334"/>
        <source>Wait for the running task to finish.</source>
        <translation>実行中のタスクが終わるまでお待ちください。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="367"/>
        <source>This exceeds even the reach of RAFTcorr-s8: use incremental tracking.</source>
        <translation>これは RAFTcorr-s8 の到達範囲さえ超えています。逐次追跡を使用してください。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="372"/>
        <source>Measuring the motion with RAFTcorr-s8</source>
        <translation>RAFTcorr-s8 で動きを計測中</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="381"/>
        <source>%1  (s = %2, reach %3 px): %4</source>
        <translation>%1  （s = %2、到達範囲 %3 px）：%4</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="382"/>
        <source>not installed</source>
        <translation>未インストール</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="384"/>
        <source>Searched:</source>
        <translation>検索した場所：</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="385"/>
        <source>Installed Models</source>
        <translation>インストール済みモデル</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="400"/>
        <source>&lt;b&gt;RAFTcorr %1&lt;/b&gt;&lt;br&gt;Learned digital image correlation with the RAFTcorr-s1, -s2, -s4 and -s8 models.&lt;br&gt;&lt;br&gt;Displacements: reference to deformed, +u right, +v down, in pixels, frame-1 (Lagrangian) coordinates.&lt;br&gt;&lt;br&gt;Interface derived from pyALDIC.&lt;br&gt;Zixiang (Zach) Tong, The University of Texas at Austin.</source>
        <translation>&lt;b&gt;RAFTcorr %1&lt;/b&gt;&lt;br&gt;RAFTcorr-s1、-s2、-s4、-s8 モデルによる学習型デジタル画像相関。&lt;br&gt;&lt;br&gt;変位：参照から変形へ、+u 右、+v 下、ピクセル単位、フレーム 1（ラグランジュ）座標。&lt;br&gt;&lt;br&gt;インターフェースは pyALDIC に由来します。&lt;br&gt;Zixiang (Zach) Tong, The University of Texas at Austin.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="459"/>
        <source>Language set to %1. Please restart RAFTcorr for all widgets to pick up the new language.</source>
        <translation>言語を %1 に変更しました。すべての画面に反映するには RAFTcorr を再起動してください。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="594"/>
        <source>Frame %1 has no Region of Interest of its own — frame 1&apos;s is used for computation. Switch to frame 1 to edit it, or import a mask to give this frame its own.</source>
        <translation>フレーム %1 には独自の関心領域がありません — 計算にはフレーム 1 の関心領域を使用します。フレーム 1 に切り替えて編集するか、マスクをインポートしてこのフレーム専用の領域を設定してください。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="605"/>
        <source>No Region of Interest to save — load images first.</source>
        <translation>保存できる関心領域がありません — 先に画像を読み込んでください。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="612"/>
        <source>Region of Interest mask is empty.</source>
        <translation>関心領域のマスクが空です。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="624"/>
        <source>Save Region of Interest Mask</source>
        <translation>関心領域マスクを保存</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="626"/>
        <source>PNG Images</source>
        <translation>PNG 画像</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="635"/>
        <source>Mask saved to %1</source>
        <translation>マスクを %1 に保存しました</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="644"/>
        <source>No Region of Interest to invert — load images first.</source>
        <translation>反転できる関心領域がありません — 先に画像を読み込んでください。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="667"/>
        <source>Load images first.</source>
        <translation>先に画像を読み込んでください。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="703"/>
        <source>  Imported mask for frame %1</source>
        <translation>  フレーム %1 のマスクをインポートしました</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/app.py" line="709"/>
        <source>Batch import: %n mask(s) loaded</source>
        <translation>
            <numerusform>一括インポート: %n 個のマスクを読み込みました</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/app.py" line="736"/>
        <source>Imported Region of Interest for %n frame(s)</source>
        <translation>
            <numerusform>%n フレームに関心領域をインポートしました</numerusform>
        </translation>
    </message>
</context>
<context>
    <name>ParamPanel</name>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="90"/>
        <source>Iterations</source>
        <extracomment>The label column is as wide as its longest label in the current language, never narrower than the English layout and, past about 28 characters of the label font, wrapped rather than squeezing the inputs (pyALDIC&apos;s ParamPanel).</extracomment>
        <translation>反復回数</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="91"/>
        <source>Refinement iterations of the network (default: 20).</source>
        <translation>ネットワークの反復改良回数です（既定値：20）。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="98"/>
        <source>Off</source>
        <comment>Displacement smoothing</comment>
        <translation>オフ</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="101"/>
        <source>Smoothing</source>
        <translation>平滑化</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="102"/>
        <source>Gaussian smoothing (σ in pixels) of the tracked displacement, kept in the results.
Off by default: smoothing coarsens the spatial resolution.</source>
        <translation>追跡した変位のガウス平滑化（σ はピクセル単位）で、結果に保存されます。
既定ではオフ：平滑化すると空間分解能が粗くなります。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="106"/>
        <source>Automatic</source>
        <translation>自動</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="110"/>
        <source>Device</source>
        <translation>デバイス</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="111"/>
        <source>Automatic uses the GPU when PyTorch sees one.</source>
        <translation>「自動」では、PyTorch が GPU を検出すると GPU を使用します。</translation>
    </message>
</context>
<context>
    <name>PhysicalUnitsWidget</name>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="56"/>
        <source>Use physical units</source>
        <translation>物理単位を使用</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="67"/>
        <source>Physical size of one image pixel</source>
        <translation>1 画像ピクセルの物理サイズ</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="85"/>
        <source>/ px</source>
        <translation>/ px</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="88"/>
        <source>Pixel size</source>
        <translation>ピクセルサイズ</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="98"/>
        <source>Acquisition frame rate (used for velocity field)</source>
        <translation>取得フレームレート(速度場に使用)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="100"/>
        <source>Frame rate</source>
        <translation>フレームレート</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="168"/>
        <source>Disp: %1  Velocity: %2/s</source>
        <translation>変位：%1  速度：%2/s</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="173"/>
        <source>Disp: px  Velocity: px/fr</source>
        <translation>変位: px  速度: px/fr</translation>
    </message>
</context>
<context>
    <name>PipelineController</name>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="77"/>
        <source>Stopped.</source>
        <extracomment>Results larger than this are kept in memory-mapped files in a temporary folder.</extracomment>
        <translation>停止しました。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="84"/>
        <source>The GPU ran out of memory. Close other GPU programs, choose a model with a larger s, or lower the memory fraction (Advanced).</source>
        <translation>GPU のメモリが不足しました。GPU を使う他のプログラムを閉じるか、s の大きいモデルを選ぶか、「詳細設定」のメモリの割合を下げてください。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="86"/>
        <source>Analysis failed</source>
        <translation>解析に失敗しました</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="164"/>
        <source>Reference frames without a Region of Interest of their own carry the one of frame 1 forward: %1</source>
        <translation>独自の関心領域がない参照フレームはフレーム 1 の関心領域を引き継ぎます：%1</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="185"/>
        <source>Starting...</source>
        <translation>開始中…</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="211"/>
        <source>The images changed: the running analysis is cancelled.</source>
        <translation>画像が変更されたため、実行中の解析をキャンセルしました。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="235"/>
        <source>The images changed during the run: its results are discarded.</source>
        <translation>実行中に画像が変更されたため、その結果は破棄されました。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="248"/>
        <source>No frame was computed; the previous results are kept.</source>
        <translation>計算されたフレームはありません。以前の結果を保持します。</translation>
    </message>
</context>
<context>
    <name>PipelineMessages</name>
    <message numerus="yes">
        <location filename="../../gui/controllers/pipeline_messages.py" line="73"/>
        <source>  %n frame(s) with custom ROI masks</source>
        <translation>
            <numerusform>  %n 個のフレームでカスタム ROI マスクを使用</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/pipeline_messages.py" line="77"/>
        <source>Results received: %n frame(s)</source>
        <translation>
            <numerusform>結果を受信：%n フレーム</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/pipeline_messages.py" line="81"/>
        <source>  %n image(s), %1</source>
        <translation>
            <numerusform>  画像 %n 枚、%1</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="84"/>
        <source>  ROI range: %1</source>
        <translation>  ROI 範囲：%1</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="87"/>
        <source>  Refinement: %1</source>
        <translation>  メッシュ細分化：%1</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="91"/>
        <source>Error: %1</source>
        <translation>エラー：%1</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="95"/>
        <source>Run canceled. Give the listed reference frames a Region of Interest of their own, or run again and continue with frame 1&apos;s region.</source>
        <translation>実行をキャンセルしました。一覧の参照フレームにそれぞれ専用の関心領域を設定するか、もう一度実行してフレーム 1 の領域で続行してください。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="102"/>
        <source>Tracking Quality Warning</source>
        <translation>追跡品質の警告</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="105"/>
        <source>Part of the sequence may not be tracked correctly.</source>
        <translation>シーケンスの一部が正しく追跡されていない可能性があります。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="110"/>
        <source>The tracking quality drops from frame %1 on: the intensity residual rises to %2 grey levels, against %3 before.</source>
        <translation>フレーム %1 以降で追跡品質が低下しています：輝度残差が %2 階調に上がりました（それ以前は %3）。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="116"/>
        <source>The displacement from frame 1 may exceed the reach of the model. Try incremental tracking, or a model with a larger s.</source>
        <translation>フレーム 1 からの変位がモデルの到達範囲を超えている可能性があります。逐次式の追跡か、s の大きいモデルを試してください。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="119"/>
        <source>The step between reference frames may exceed the reach of the model. Use more reference frames, or a model with a larger s.</source>
        <translation>参照フレーム間の変位がモデルの到達範囲を超えている可能性があります。参照フレームを増やすか、s の大きいモデルを使ってください。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="123"/>
        <source>Frames with a high residual: %1</source>
        <translation>残差の大きいフレーム：%1</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="126"/>
        <source>Show the Residual field to see where the images do not match after the displacement is applied.</source>
        <translation>「残差」の場を表示すると、変位を適用した後に画像が一致しない場所を確認できます。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="132"/>
        <source>Cannot start analysis</source>
        <translation>解析を開始できません</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="136"/>
        <source>Need at least 2 images. Load an image folder in the %1 section before running DIC.</source>
        <translation>少なくとも 2 枚の画像が必要です。DIC を実行する前に、「%1」セクションで画像フォルダーを読み込んでください。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="141"/>
        <source>Define a Region of Interest first: draw or import one for frame 1 in the %1 section.</source>
        <translation>先に関心領域を定義してください：「%1」セクションでフレーム 1 の関心領域を描画またはインポートします。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="146"/>
        <source>The Region of Interest of frame 1 is empty (no pixels selected). Draw it again in the %1 section.</source>
        <translation>フレーム 1 の関心領域が空です（ピクセルが選択されていません）。「%1」セクションで描き直してください。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="151"/>
        <source>Region of Interest too small</source>
        <translation>関心領域が小さすぎます</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="157"/>
        <source>The Region of Interest is %1×%2 px, but the current %3 (%4) needs at least %5×%5 px. Enlarge the Region of Interest or reduce %3.</source>
        <translation>関心領域は %1×%2 px ですが、現在の%3（%4）では少なくとも %5×%5 px が必要です。関心領域を広げるか、%3を小さくしてください。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="163"/>
        <source>Starting Points required</source>
        <translation>シード点が必要です</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="167"/>
        <source>Place at least one Starting Point on the canvas before running: click “%1” or “%2” in the %3 section.</source>
        <translation>実行前にキャンバス上に少なくとも 1 つのシード点を配置してください：「%3」セクションの「%1」または「%2」をクリックします。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="173"/>
        <source>Invalid Starting Points</source>
        <translation>無効なシード点</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="177"/>
        <source>One or more Starting Points do not lie in any part of the Region of Interest. Click “%1” in the %2 section, then place the Starting Points again.</source>
        <translation>1 つ以上のシード点が関心領域のどの部分にもありません。「%2」セクションの「%1」をクリックしてから、シード点を配置し直してください。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="185"/>
        <source>DIC analysis failed</source>
        <translation>DIC 解析に失敗しました</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="188"/>
        <source>Failed to start analysis</source>
        <translation>解析の開始に失敗しました</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="191"/>
        <source>Failed to process results</source>
        <translation>結果の処理に失敗しました</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="196"/>
        <source>The analysis stopped with a runtime error:

%1

See the console log for the full traceback.</source>
        <translation>実行時エラーにより解析が停止しました：

%1

完全なトレースバックはコンソールログを参照してください。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="202"/>
        <source>%1

See the console log for the full traceback.</source>
        <translation>%1

完全なトレースバックはコンソールログを参照してください。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="207"/>
        <source>Incomplete Region of Interest Coverage</source>
        <translation>関心領域のカバー範囲が不完全です</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/pipeline_messages.py" line="212"/>
        <source>This incremental run has %n reference frame(s) with no Region of Interest of their own: frame(s) %1.</source>
        <translation>
            <numerusform>この逐次追跡の実行では、%n 個の参照フレームに専用の関心領域がありません：フレーム %1。</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/pipeline_messages.py" line="218"/>
        <source>%1 and %n more</source>
        <comment>List of frame numbers cut short: %1 = the frame numbers shown, %n = how many more frames there are</comment>
        <translation>
            <numerusform>%1 ほか %n フレーム</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="224"/>
        <source>A reference frame without a Region of Interest of its own reuses frame 1&apos;s region unchanged: nothing moves the region with the specimen. Where the specimen has moved out of that region, the part outside it is not measured from such a frame, and the nodes there stop being tracked or have their displacement copied from the nearest measured node. The run reports these nodes when it finishes.</source>
        <translation>専用の関心領域がない参照フレームは、フレーム 1 の領域をそのまま再利用します。この領域は試験片に合わせて移動しません。試験片がこの領域の外へ移動すると、その外側の部分はそのような参照フレームからは測定されず、そこにあるノードは追跡されなくなるか、最も近い測定済みノードから変位がコピーされます。これらのノードは実行終了時に報告されます。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="234"/>
        <source>To measure the whole specimen, give each reference frame its own Region of Interest that covers the specimen in that frame (Batch Import can assign Regions of Interest to many frames at once). A reference frame with a region of its own is measured on a grid that covers that region.</source>
        <translation>試験片全体を測定するには、各参照フレームに、そのフレームの試験片を覆う専用の関心領域を設定してください（一括インポートで多数のフレームに関心領域をまとめて割り当てられます）。専用の領域を持つ参照フレームは、その領域を覆う測定グリッド上で測定されます。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="241"/>
        <source>Continue with frame 1&apos;s region?</source>
        <translation>フレーム 1 の領域で続行しますか？</translation>
    </message>
</context>
<context>
    <name>ROIHint</name>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="67"/>
        <source>Load images first, then draw a Region of Interest on frame 1.</source>
        <extracomment>The same card for other hints of the left sidebar (the model advice).</extracomment>
        <translation>まず画像を読み込み、その後フレーム 1 に関心領域を描画してください。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="74"/>
        <source>&lt;b&gt;Accumulative mode&lt;/b&gt; — only frame 1 needs a Region of Interest. All later frames are compared against it directly.</source>
        <translation>&lt;b&gt;累積モード&lt;/b&gt; — 関心領域はフレーム 1 にのみ必要です。後続フレームはすべて直接比較されます。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="87"/>
        <source>&lt;b&gt;Incremental&lt;/b&gt; — only frame 1 needs a Region of Interest. RAFTcorr carries it to the reference frames (&lt;b&gt;%1&lt;/b&gt;) with the measured displacement. Draw on another frame only to replace that region or to leave out cracks and holes.</source>
        <translation>&lt;b&gt;逐次&lt;/b&gt; — 関心領域はフレーム 1 にのみ必要です。RAFTcorr が計測した変位でそれを参照フレーム（&lt;b&gt;%1&lt;/b&gt;）へ移します。他のフレームに描画するのは、その領域を置き換える場合や、き裂や穴を除外する場合だけです。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="96"/>
        <source>Draw a Region of Interest on frame 1.</source>
        <translation>フレーム 1 に関心領域を描画してください。</translation>
    </message>
</context>
<context>
    <name>ROIToolbar</name>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="72"/>
        <source>+ Add</source>
        <translation>+ 追加</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="74"/>
        <source>Add region to the Region of Interest (Polygon / Rectangle / Circle)</source>
        <translation>関心領域に形状を追加します(多角形 / 矩形 / 円)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="79"/>
        <source>Cut</source>
        <translation>切り取り</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="81"/>
        <source>Cut region from the Region of Interest (Polygon / Rectangle / Circle)</source>
        <translation>関心領域から形状を切り取ります(多角形 / 矩形 / 円)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="86"/>
        <source>+ Refine</source>
        <translation>+ 細分化</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="90"/>
        <source>Paint extra mesh-refinement zones with a brush
(only on frame 1 — material points auto-warped to later frames)</source>
        <translation>ブラシで追加のメッシュ細分化領域を塗ります
(フレーム 1 のみ — 後続フレームへ自動ワープされます)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="94"/>
        <source>Refine brush is only available on frame 1. Switch to frame 1 to paint refinement zones; they are automatically warped to later frames.</source>
        <translation>細分化ブラシはフレーム 1 でのみ使用可能です。フレーム 1 に切り替えて領域を塗ってください。後続フレームには自動でワープされます。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="118"/>
        <source>Import</source>
        <translation>インポート</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="119"/>
        <source>Import mask from image file</source>
        <translation>画像ファイルからマスクをインポート</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="124"/>
        <source>Batch Import</source>
        <translation>一括インポート</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="126"/>
        <source>Batch import mask files for multiple frames</source>
        <translation>複数フレームのマスクファイルを一括インポート</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="139"/>
        <source>Save</source>
        <translation>保存</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="140"/>
        <source>Save current mask to PNG file</source>
        <translation>現在のマスクを PNG ファイルに保存</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="145"/>
        <source>Invert</source>
        <translation>反転</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="147"/>
        <source>Invert the Region of Interest mask</source>
        <translation>関心領域マスクを反転</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="152"/>
        <source>Clear</source>
        <translation>クリア</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="154"/>
        <source>Clear all Region of Interest masks</source>
        <translation>すべての関心領域マスクをクリア</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="186"/>
        <source>Radius</source>
        <translation>半径</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="204"/>
        <source>Paint</source>
        <translation>塗り</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="209"/>
        <source>Erase</source>
        <translation>消去</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="219"/>
        <source>Clear Brush</source>
        <translation>ブラシをクリア</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="247"/>
        <source>Polygon</source>
        <translation>多角形</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="251"/>
        <source>Rectangle</source>
        <translation>矩形</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="255"/>
        <source>Circle</source>
        <translation>円</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="259"/>
        <source>Circle (3-point)</source>
        <translation>円（3 点）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="352"/>
        <source>Import Mask Image</source>
        <translation>マスク画像をインポート</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="354"/>
        <source>Images</source>
        <translation>画像</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="355"/>
        <source>All Files</source>
        <translation>すべてのファイル</translation>
    </message>
</context>
<context>
    <name>RightSidebar</name>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="76"/>
        <source>Run DIC Analysis</source>
        <translation>DIC 解析を実行</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="89"/>
        <source>Cancel</source>
        <translation>キャンセル</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="94"/>
        <source>Cancel the current analysis. Frames already computed are kept so you can review or export the partial run.</source>
        <translation>現在の解析をキャンセルします。計算済みのフレームは保持され、途中までの結果を確認またはエクスポートできます。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="102"/>
        <source>Export Results</source>
        <translation>結果をエクスポート</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="110"/>
        <source>Open Strain Window</source>
        <translation>ひずみウィンドウを開く</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="113"/>
        <source>Compute and visualize strain in a separate post-processing window. Requires displacement results from a completed Run.</source>
        <translation>別ウィンドウでひずみを計算・可視化します。完了した実行結果の変位データが必要です。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="122"/>
        <source>PROGRESS</source>
        <translation>進捗</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="131"/>
        <location filename="../../gui/panels/right_sidebar.py" line="435"/>
        <source>Ready</source>
        <translation>準備完了</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="139"/>
        <location filename="../../gui/panels/right_sidebar.py" line="437"/>
        <location filename="../../gui/panels/right_sidebar.py" line="537"/>
        <source>ELAPSED  %1</source>
        <translation>経過時間  %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="144"/>
        <location filename="../../gui/panels/right_sidebar.py" line="439"/>
        <location filename="../../gui/panels/right_sidebar.py" line="545"/>
        <location filename="../../gui/panels/right_sidebar.py" line="549"/>
        <source>REMAINING  %1</source>
        <translation>残り時間  %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="152"/>
        <source>FIELD</source>
        <translation>フィールド</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="161"/>
        <source>Show on</source>
        <translation>表示先</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="166"/>
        <source>Deformed frame</source>
        <translation>変形フレーム</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="167"/>
        <source>Reference frame</source>
        <translation>参照フレーム</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="169"/>
        <source>Plot the field where the material points are in the current frame, or at their positions in the reference frame.</source>
        <translation>場を材料点の現在フレームでの位置に描画するか、参照フレームでの位置に描画します。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="215"/>
        <source>Leave points without a result transparent, for example material that left the field of view, instead of filling them with the nearest value inside the region of interest. Also used for exported images and animations; data files always keep NaN there.</source>
        <translation>結果のない点（視野の外に出た材料など）を、関心領域内で最も近い値で埋めずに透明のままにします。エクスポートした画像・アニメーションにも適用されます。データファイルではこれらの点は常に NaN です。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="177"/>
        <source>Show background image</source>
        <translation>背景画像を表示</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="180"/>
        <source>Uncheck to show the field on its own, with no speckle image behind it.</source>
        <translation>チェックを外すと、背後にスペックル画像を表示せず場のみを表示します。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="196"/>
        <source>Hidden background</source>
        <translation>非表示時の背景</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="198"/>
        <source>White</source>
        <translation>白</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="199"/>
        <source>Black</source>
        <translation>黒</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="200"/>
        <source>Transparent</source>
        <translation>透明</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="203"/>
        <source>What replaces the image when it is hidden. Transparency is kept for PNG and TIFF on export; other formats get white.</source>
        <translation>背景を非表示にしたとき、画像の代わりに何を表示するかです。エクスポートでは PNG と TIFF が透明度を保持し、他の形式は白になります。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="212"/>
        <source>Hide untracked regions</source>
        <translation>追跡されていない領域を隠す</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="225"/>
        <source>VISUALIZATION</source>
        <translation>可視化</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="230"/>
        <source>Colormap</source>
        <translation>カラーマップ</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="249"/>
        <source>Opacity</source>
        <translation>不透明度</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="256"/>
        <source>Overlay opacity (0 = transparent, 100 = opaque)</source>
        <translation>オーバーレイの不透明度(0 = 透明、100 = 不透明)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="262"/>
        <source>PHYSICAL UNITS</source>
        <translation>物理単位</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="268"/>
        <source>LOG</source>
        <translation>ログ</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="274"/>
        <source>Clear</source>
        <translation>クリア</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="404"/>
        <source>Place at least one Starting Point in each red region before running (red = needs a Starting Point).</source>
        <translation>実行前に、各赤色領域に少なくとも 1 つのシード点を配置してください(赤色 = シード点が必要)。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="520"/>
        <source>%1  —  Frame %2</source>
        <translation>%1  —  フレーム %2</translation>
    </message>
</context>
<context>
    <name>StrainController</name>
    <message>
        <location filename="../../gui/controllers/strain_controller.py" line="96"/>
        <source>Strain frame %1/%2</source>
        <translation>ひずみ：フレーム %1/%2</translation>
    </message>
</context>
<context>
    <name>StrainFieldSelector</name>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="78"/>
        <source>Magnitude</source>
        <extracomment>GUI field name -&gt; FieldProvider key.</extracomment>
        <translation>変位の大きさ</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="79"/>
        <source>Velocity</source>
        <translation>速度</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="200"/>
        <source>DISPLACEMENT</source>
        <translation>変位</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="211"/>
        <source>STRAIN</source>
        <translation>ひずみ</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="265"/>
        <source>Horizontal displacement u (+ right)</source>
        <translation>水平変位 u（右向きが正）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="266"/>
        <source>Vertical displacement v (+ down)</source>
        <translation>垂直変位 v（下向きが正）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="267"/>
        <source>Displacement magnitude |u|</source>
        <translation>変位の大きさ |u|</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="269"/>
        <source>Speed |du/dt|: central difference of the displacement over the neighbouring computed frames, times the frame rate</source>
        <translation>速さ |du/dt|：隣接する計算済みフレーム間の変位の中心差分にフレームレートを掛けたもの</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="271"/>
        <source>Normal strain εxx</source>
        <translation>垂直ひずみ εxx</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="272"/>
        <source>Normal strain εyy</source>
        <translation>垂直ひずみ εyy</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="274"/>
        <source>Shear strain εxy (tensor component: half the engineering shear γxy)</source>
        <translation>せん断ひずみ εxy（テンソル成分：工学せん断ひずみ γxy の半分）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="276"/>
        <source>Major principal strain ε₁</source>
        <translation>最大主ひずみ ε₁</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="277"/>
        <source>Minor principal strain ε₂</source>
        <translation>最小主ひずみ ε₂</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="279"/>
        <source>Maximum in-plane shear strain (ε₁ − ε₂) / 2 (tensor component: half the engineering shear)</source>
        <translation>最大面内せん断ひずみ (ε₁ − ε₂) / 2（テンソル成分：工学せん断ひずみの半分）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="282"/>
        <source>Von Mises equivalent strain (incompressible material)</source>
        <translation>von Mises 相当ひずみ（非圧縮性材料）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="284"/>
        <source>Rigid rotation in degrees (polar decomposition), clockwise on screen positive</source>
        <translation>剛体回転角（度、極分解）、画面上で時計回りが正</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="287"/>
        <source>Direction of ε₁ in degrees from +x, clockwise on screen positive</source>
        <translation>ε₁ の方向（+x からの角度、度）、画面上で時計回りが正</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="290"/>
        <source>Relative area change ΔA/A₀ = det F − 1</source>
        <translation>相対面積変化 ΔA/A₀ = det F − 1</translation>
    </message>
</context>
<context>
    <name>StrainNavigator</name>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="73"/>
        <source>Previous frame</source>
        <translation>前フレーム</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="84"/>
        <location filename="../../gui/widgets/strain_navigator.py" line="216"/>
        <source>Play animation</source>
        <translation>アニメーションを再生</translation>
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
        <translation>次のフレーム</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="109"/>
        <source>Playback speed</source>
        <translation>再生速度</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="114"/>
        <location filename="../../gui/widgets/strain_navigator.py" line="228"/>
        <source>FRAME 0/0</source>
        <translation>フレーム 0/0</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="205"/>
        <source>⏸</source>
        <translation>⏸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="206"/>
        <source>Pause animation</source>
        <translation>アニメーションを一時停止</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="223"/>
        <source>FRAME %1/%2</source>
        <translation>フレーム %1/%2</translation>
    </message>
</context>
<context>
    <name>StrainParamPanel</name>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="88"/>
        <source>Plane fitting</source>
        <extracomment>Strain measures in combo order (pyALDIC&apos;s order: infinitesimal, Eulerian, Green-Lagrange). The configuration in the controls changed (edited by the user).</extracomment>
        <translation>平面フィット</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="91"/>
        <source>Polynomial fitted to each displacement component over the VSG window. Its slopes at the centre are the displacement gradients.
• Plane fitting (1st order): the standard VSG (default).
• Quadratic fitting (2nd order): follows strain gradients inside the window better, at the cost of more noise.</source>
        <translation>VSG ウィンドウ内で各変位成分に多項式をフィットします。中心での傾きが変位勾配です。
• 平面フィット（1 次）：標準的な VSG（既定値）。
• 二次フィット（2 次）：ウィンドウ内のひずみ勾配によく追従しますが、ノイズが増えます。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="97"/>
        <source>Method</source>
        <translation>手法</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="112"/>
        <source>VSG size</source>
        <translation>VSG サイズ</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="123"/>
        <source>Hides low-confidence strain along the edges of the region of interest, holes and cracks, where the edge cuts the VSG window and the local fit becomes one-sided and unreliable.

• Coefficient × VSG half-width (Lg − 1) / 2 = width of the trimmed band.
• 0.00 = keep every point (no trimming).
• 0.70 = default.
• 1.00 = strictest (trim every point whose window touches an edge).

Trimmed points are empty (NaN) in exported data files.</source>
        <translation>関心領域、穴、き裂の縁で、縁が VSG ウィンドウを切り取り、局所フィットが片側的で信頼できなくなる箇所の低信頼度のひずみを非表示にします。

• 係数 × VSG の半幅 (Lg − 1) / 2 = トリミングされる帯の幅。
• 0.00 = すべての点を保持（トリミングなし）。
• 0.70 = 既定値。
• 1.00 = 最も厳格（ウィンドウが縁に触れる点をすべてトリミング）。

トリミングされた点は、エクスポートしたデータファイルでは空 (NaN) です。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="154"/>
        <source>Weight of the points inside the VSG window: all equal (uniform, default), or decreasing with the distance from the centre (Gaussian).</source>
        <translation>VSG ウィンドウ内の点の重みです：すべて等しい（一様、既定値）か、中心からの距離とともに減少する（ガウス）。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="189"/>
        <source>Off</source>
        <comment>Strain smoothing preset</comment>
        <translation>オフ</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="89"/>
        <source>Quadratic fitting</source>
        <translation>二次フィット</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="105"/>
        <source>VSG (Virtual Strain Gauge) size Lg is the side, in pixels, of the square window around each point over which the displacement is fitted. Strain is then taken from the fit&apos;s slopes.

• Larger VSG → smoother strain, lower spatial resolution.
• Smaller VSG → sharper strain, more noise.
• Odd, so that the window has a centre pixel (default: 41 px).</source>
        <translation>VSG（仮想ひずみゲージ）サイズ Lg は、各点の周囲で変位をフィットする正方形ウィンドウの一辺の長さ（ピクセル）です。ひずみはフィットの傾きから求めます。

• VSG が大きい → ひずみは滑らか、空間分解能は低い。
• VSG が小さい → ひずみはシャープ、ノイズが多い。
• ウィンドウに中心ピクセルがあるよう奇数にします（既定値：41 px）。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="133"/>
        <source>Trim low-confidence edges</source>
        <translation>低信頼度の縁をトリミング</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="151"/>
        <source>Uniform</source>
        <translation>一様</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="152"/>
        <source>Gaussian</source>
        <translation>ガウス</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="158"/>
        <source>Weighting</source>
        <translation>重み付け</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="166"/>
        <source>Auto (Lg / 4)</source>
        <translation>自動（Lg / 4）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="168"/>
        <source>Standard deviation of the Gaussian weight. Auto uses a quarter of the VSG size.</source>
        <translation>ガウス重みの標準偏差です。「自動」では VSG サイズの 4 分の 1 を使います。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="171"/>
        <source>Gaussian σ</source>
        <translation>ガウス σ</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="178"/>
        <source>Spacing of the strain points: 1 computes strain at every pixel; larger values are faster and lighter for long sequences.</source>
        <translation>ひずみ点の間隔です：1 ですべてのピクセルでひずみを計算します。大きい値ほど高速・軽量で、長いシーケンスに向きます。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="181"/>
        <source>Output spacing</source>
        <translation>出力間隔</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="191"/>
        <source>Gaussian smoothing (σ in pixels) of the displacement used by the strain fit only; the displacement results are not changed. Off (0) differentiates the measured displacement as it is (default).</source>
        <translation>ひずみのフィットに使う変位だけをガウス平滑化します（σ はピクセル単位）。変位の結果は変わりません。オフ（0）では計測した変位をそのまま微分します（既定値）。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="195"/>
        <source>Pre-fit smoothing</source>
        <translation>フィット前の平滑化</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="198"/>
        <source>Never fit across cracks or holes</source>
        <translation>き裂や穴をまたいでフィットしない</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="200"/>
        <source>Fit each connected part of the tracked region separately, so no VSG window ever spans a crack, a hole or the gap between two regions.</source>
        <translation>追跡領域の連結した部分ごとに別々にフィットし、どの VSG ウィンドウもき裂、穴、2 つの領域の間の隙間をまたがないようにします。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="209"/>
        <source>Infinitesimal</source>
        <translation>微小ひずみ</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="210"/>
        <source>Eulerian</source>
        <translation>オイラーひずみ</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="211"/>
        <source>Green-Lagrange</source>
        <translation>グリーン-ラグランジュひずみ</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="213"/>
        <source>Strain measure from the deformation gradient F = I + ∇u:
• Infinitesimal: (∇u + ∇uᵀ) / 2 (small strains only).
• Eulerian (Euler-Almansi): (I − (F Fᵀ)⁻¹) / 2.
• Green-Lagrange: (Fᵀ F − I) / 2 (default).</source>
        <translation>変形勾配 F = I + ∇u から求めるひずみ尺度：
• 微小ひずみ：(∇u + ∇uᵀ) / 2（小ひずみのみ）。
• オイラーひずみ（Euler-Almansi）：(I − (F Fᵀ)⁻¹) / 2。
• グリーン-ラグランジュひずみ：(Fᵀ F − I) / 2（既定値）。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="218"/>
        <source>Strain type</source>
        <translation>ひずみ種別</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="272"/>
        <source>Trimmed: %1 points (%2%)</source>
        <translation>トリミング: %1 点 (%2%)</translation>
    </message>
</context>
<context>
    <name>StrainVizPanel</name>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="67"/>
        <source>Deformed frame</source>
        <extracomment>Alias so existing references read unchanged; the list itself lives in raftcorr.gui.support.colormaps, so every chooser offers the same options.</extracomment>
        <translation>変形フレーム</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="68"/>
        <source>Reference frame</source>
        <translation>参照フレーム</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="70"/>
        <source>Plot the field where the material points are in the current frame, or at their positions in the reference frame.</source>
        <translation>場を材料点の現在フレームでの位置に描画するか、参照フレームでの位置に描画します。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="73"/>
        <source>Show on</source>
        <translation>表示先</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="75"/>
        <source>Show background image</source>
        <translation>背景画像を表示</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="78"/>
        <source>Uncheck to show the field on its own, with no speckle image behind it.</source>
        <translation>チェックを外すと、背後にスペックル画像を表示せず場のみを表示します。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="84"/>
        <source>White</source>
        <translation>白</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="85"/>
        <source>Black</source>
        <translation>黒</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="86"/>
        <source>Transparent</source>
        <translation>透明</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="89"/>
        <source>What replaces the image when it is hidden. Transparency is kept for PNG and TIFF on export; other formats get white.</source>
        <translation>背景を非表示にしたとき、画像の代わりに何を表示するかです。エクスポートでは PNG と TIFF が透明度を保持し、他の形式は白になります。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="141"/>
        <source>Fill trimmed edges (display only)</source>
        <translation>トリミングされた縁を埋める（表示のみ）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="144"/>
        <source>Fill the edge-trimmed strain band from the nearest reliable point of the same region. Affects the view and exported images and animations; exported data files always keep the trimmed band as NaN.</source>
        <translation>縁がトリミングされたひずみ帯を、同じ領域内で最も近い信頼できる点の値で埋めます。表示とエクスポートした画像・アニメーションに影響します。エクスポートしたデータファイルでは、トリミングされた帯は常に NaN のままです。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="151"/>
        <source>Hide untracked regions</source>
        <translation>追跡されていない領域を隠す</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="154"/>
        <source>Leave points without a result transparent, for example material that left the field of view, instead of filling them with the nearest value inside the region of interest. Also used for exported images and animations; data files always keep NaN there.</source>
        <translation>結果のない点（視野の外に出た材料など）を、関心領域内で最も近い値で埋めずに透明のままにします。エクスポートした画像・アニメーションにも適用されます。データファイルではこれらの点は常に NaN です。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="81"/>
        <source>Background</source>
        <translation>背景</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="93"/>
        <source>Hidden background</source>
        <translation>非表示時の背景</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="100"/>
        <source>Colormap</source>
        <translation>カラーマップ</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="104"/>
        <source>Range</source>
        <translation>範囲</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="124"/>
        <source>Min</source>
        <translation>最小</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="126"/>
        <source>Max</source>
        <translation>最大</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="136"/>
        <source>Opacity</source>
        <translation>不透明度</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="148"/>
        <source>Edges</source>
        <translation>縁</translation>
    </message>
</context>
<context>
    <name>StrainWindow</name>
    <message>
        <location filename="../../gui/strain_window.py" line="125"/>
        <source>Strain Post-Processing</source>
        <translation>ひずみ後処理</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="210"/>
        <source>Fit</source>
        <translation>フィット</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="211"/>
        <source>Fit image to viewport</source>
        <translation>画像をビューポートに合わせる</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="217"/>
        <source>100%</source>
        <translation>100%</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="218"/>
        <source>Zoom to 100% (1:1)</source>
        <translation>100% (1:1) ズーム</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="221"/>
        <source>Zoom in</source>
        <translation>拡大</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="226"/>
        <source>–</source>
        <translation>–</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="284"/>
        <source>STRAIN PARAMETERS</source>
        <translation>ひずみパラメータ</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="293"/>
        <source>ADVANCED</source>
        <translation>詳細設定</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="308"/>
        <source>Cancel</source>
        <translation>キャンセル</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="312"/>
        <source>Cancel the running strain computation. The previous strain result is kept.</source>
        <translation>実行中のひずみ計算をキャンセルします。以前のひずみ結果は保持されます。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="324"/>
        <source>Export displacement and strain results to NPZ / MAT / CSV / PNG</source>
        <translation>変位とひずみ結果を NPZ / MAT / CSV / PNG にエクスポート</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="354"/>
        <source>FIELD</source>
        <translation>フィールド</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="361"/>
        <source>VISUALIZATION</source>
        <translation>可視化</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="373"/>
        <source>PHYSICAL UNITS</source>
        <translation>物理単位</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="379"/>
        <source>LOG</source>
        <translation>ログ</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="861"/>
        <location filename="../../gui/strain_window.py" line="899"/>
        <source>No displacement results.</source>
        <translation>変位結果がありません。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="901"/>
        <source>Frame %1 was not computed.</source>
        <translation>フレーム %1 は計算されていません。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="909"/>
        <source>Click Compute Strain to show this field.</source>
        <translation>「ひずみを計算」をクリックすると、この場を表示します。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="910"/>
        <source>No strain for frame %1: click Compute Strain.</source>
        <translation>フレーム %1 のひずみがありません：「ひずみを計算」をクリックしてください。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="912"/>
        <source>Velocity needs at least two computed frames.</source>
        <translation>速度には計算済みフレームが 2 つ以上必要です。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="913"/>
        <source>No valid values in frame %1.</source>
        <translation>フレーム %1 に有効な値がありません。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="170"/>
        <source>Strain Field</source>
        <translation>ひずみ場</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="419"/>
        <source>Analysis is not available</source>
        <translation>解析は利用できません</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="422"/>
        <location filename="../../gui/strain_window.py" line="425"/>
        <source>Analysis</source>
        <translation>解析</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="423"/>
        <source>Analysis is not available (see the application log).</source>
        <translation>解析は利用できません（アプリケーションのログを参照してください）。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="534"/>
        <source>Strain compute failed: %1: %2</source>
        <translation>ひずみ計算に失敗しました：%1：%2</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="540"/>
        <location filename="../../gui/strain_window.py" line="603"/>
        <source>Strain computation complete.</source>
        <translation>ひずみ計算が完了しました。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="551"/>
        <source>Strain window: no displacement results to post-process.</source>
        <translation>ひずみウィンドウ：後処理する変位結果がありません。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="596"/>
        <source>The displacement results changed during the strain computation; its result was discarded.</source>
        <translation>ひずみ計算中に変位結果が変更されたため、その結果は破棄されました。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="620"/>
        <source>Strain computation canceled.</source>
        <translation>ひずみ計算をキャンセルしました。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="627"/>
        <source>Strain compute failed: %1</source>
        <translation>ひずみ計算に失敗しました：%1</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="631"/>
        <source>Strain Computation Failed</source>
        <translation>ひずみ計算に失敗しました</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="671"/>
        <location filename="../../gui/strain_window.py" line="673"/>
        <source>Export is not available: %1</source>
        <translation>エクスポートは利用できません：%1</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="685"/>
        <source>⚠ Params changed -- click Compute Strain</source>
        <translation>⚠ パラメータが変更されました — 「ひずみを計算」をクリックしてください</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="227"/>
        <source>Zoom out</source>
        <translation>縮小</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="299"/>
        <source>Compute Strain</source>
        <translation>ひずみを計算</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="321"/>
        <location filename="../../gui/strain_window.py" line="672"/>
        <source>Export Results</source>
        <translation>結果をエクスポート</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="563"/>
        <source>Starting…</source>
        <translation>開始中…</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="581"/>
        <source>Canceling…</source>
        <translation>キャンセル中…</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="600"/>
        <source>Complete</source>
        <translation>完了</translation>
    </message>
</context>
<context>
    <name>VelocitySettingsWidget</name>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="47"/>
        <source>Use physical units</source>
        <translation>物理単位を使用</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="69"/>
        <source>/ px</source>
        <translation>/ px</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="84"/>
        <source>Unit: px/frame</source>
        <translation>単位: px/フレーム</translation>
    </message>
</context>
<context>
    <name>WorkflowTypePanel</name>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="59"/>
        <source>Incremental</source>
        <translation>逐次式</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="58"/>
        <source>Accumulative</source>
        <translation>累積式</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="73"/>
        <source>Tracking Mode</source>
        <translation>追跡モード</translation>
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
        <translation>累積：すべてのフレームをフレーム 1 と直接比較します。
動きがモデルの到達範囲内に収まる限り、こちらを推奨します：
フレーム間で誤差が累積しません。

逐次：各フレームをフレーム 1 より後の参照フレームと比較し、
各ステップを合成します。フレーム 1 からの動きが
モデルの到達範囲を超える場合（大きな伸びや回転）に必要です。各ステップで
誤差が加わります。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="85"/>
        <source>finest spatial resolution; small displacements</source>
        <translation>空間分解能が最も高い、小さな変位向け</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="86"/>
        <source>fine resolution, several times faster than s1; small to moderate displacements</source>
        <translation>分解能が高く s1 の数倍高速、小～中程度の変位向け</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="87"/>
        <source>balanced resolution and capture range</source>
        <translation>分解能と到達範囲のバランス型</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="88"/>
        <source>largest capture range, coarsest resolution; large motion and rotation</source>
        <translation>到達範囲が最大で分解能は最も粗い、大きな動きや回転向け</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="92"/>
        <source>%1: %2.
Reach (nominal lookup radius 32 s): %3 px.</source>
        <translation>%1：%2。
到達範囲（公称探索半径 32 s）：%3 px。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="95"/>
        <source>Weights not installed.</source>
        <translation>重みファイルがインストールされていません。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="100"/>
        <source>RAFTcorr model. A smaller s resolves finer detail; a larger s
reaches further (the coarsest correlation level looks 32 s px away).

s1, s2: small displacements, strain concentrations.
s4: balanced.
s8: large motion or rotation, or no prior knowledge.</source>
        <translation>RAFTcorr モデル。s が小さいほど細部を分解でき、s が大きいほど
遠くまで届きます（最も粗い相関レベルは 32 s px 先まで探索）。

s1、s2：小さな変位、ひずみ集中。
s4：バランス型。
s8：大きな動きや回転、または事前情報がない場合。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="107"/>
        <source>Model</source>
        <translation>モデル</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="114"/>
        <source>Suggest a Model</source>
        <translation>モデルを提案</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="173"/>
        <source>e.g. 5, 10, 20</source>
        <translation>例：5, 10, 20</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="175"/>
        <source>Frame numbers (as in the image list) to use as reference frames, separated by commas. Frame 1 is always the first reference.</source>
        <translation>参照フレームにするフレーム番号（画像リストと同じ）をコンマ区切りで指定します。フレーム 1 は常に最初の参照です。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="272"/>
        <source>Invalid reference frames: use frame numbers from 1 on, separated by commas.</source>
        <translation>参照フレームが無効です：1 以上のフレーム番号をコンマ区切りで指定してください。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="118"/>
        <source>Measure the motion of the sequence quickly with s8 and pick the
finest model whose reach covers it.</source>
        <translation>s8 でシーケンスの動きをすばやく計測し、その動きを到達範囲で
カバーできる最も細かいモデルを選びます。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="140"/>
        <source>Every Frame</source>
        <translation>毎フレーム</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="141"/>
        <source>Every N Frames</source>
        <translation>N フレームごと</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="142"/>
        <source>Custom Frames</source>
        <translation>カスタムフレーム</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="145"/>
        <source>When the reference frame refreshes during incremental tracking.
Every Frame: reset reference every frame (smallest per-step displacement,
most robust for large deformation).
Every N Frames: reset every N frames (balance speed vs robustness).
Custom Frames: user-defined list of reference frame indices.</source>
        <translation>逐次追跡中の参照フレームの更新タイミング。
毎フレーム: 毎フレーム参照をリセット(ステップ変位最小、
大変形に最もロバスト)。
N フレームごと: N フレームごとにリセット(速度と頑健性のバランス)。
カスタムフレーム: ユーザー指定の参照フレームインデックス。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="154"/>
        <source>Reference Update</source>
        <translation>参照フレーム更新</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="164"/>
        <source>Update reference every N frames</source>
        <translation>N フレームごとに参照を更新</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="166"/>
        <source>Interval</source>
        <translation>間隔</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="179"/>
        <source>Reference Frames</source>
        <translation>参照フレーム</translation>
    </message>
</context>
<context>
    <name>_DropZone</name>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="118"/>
        <source>Drop image folder
or Browse</source>
        <translation>画像フォルダをドロップ
または参照</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="128"/>
        <source>Select Image Folder</source>
        <translation>画像フォルダを選択</translation>
    </message>
</context>
<context>
    <name>_MaskPreviewPanel</name>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="133"/>
        <source>Preview</source>
        <translation>プレビュー</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="137"/>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="263"/>
        <source>(no image)</source>
        <translation>（画像なし）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="157"/>
        <source>Image only</source>
        <translation>画像のみ</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="158"/>
        <source>Image + Mask</source>
        <translation>画像 + マスク</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="159"/>
        <source>Mask only</source>
        <translation>マスクのみ</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="162"/>
        <source>View:</source>
        <translation>表示:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="168"/>
        <source>Alpha:</source>
        <translation>アルファ:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="176"/>
        <source>Blue</source>
        <comment>Mask overlay color</comment>
        <translation>青</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="180"/>
        <source>Red</source>
        <comment>Mask overlay color</comment>
        <translation>赤</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="184"/>
        <source>Green</source>
        <comment>Mask overlay color</comment>
        <translation>緑</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="188"/>
        <source>Yellow</source>
        <comment>Mask overlay color</comment>
        <translation>黄</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="192"/>
        <source>Mask color:</source>
        <translation>マスクの色:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="269"/>
        <source>No mask assigned</source>
        <translation>マスク未割り当て</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="275"/>
        <source>Frame %1 — %2</source>
        <translation>フレーム %1 — %2</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="290"/>
        <source>Failed to load image</source>
        <translation>画像の読み込みに失敗しました</translation>
    </message>
</context>
<context>
    <name>dialog</name>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="322"/>
        <source>Cancel Export</source>
        <translation>エクスポートをキャンセル</translation>
    </message>
</context>
</TS>
