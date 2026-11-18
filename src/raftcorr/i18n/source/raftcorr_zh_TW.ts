<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1" language="zh_TW" sourcelanguage="en_US">
<context>
    <name>AdvancedPanel</name>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="117"/>
        <source>Margin = model reach</source>
        <translation>邊距 = 模型量程</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="120"/>
        <source>Extra image context around the ROI so that displaced points stay inside
the window given to the network. Automatic: the model&apos;s reach (32 s px).</source>
        <translation>在 ROI 周圍額外保留的影像範圍，使發生位移的點仍位於
送入網路的視窗內。自動：取模型量程（32 s px）。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="128"/>
        <source>Margin</source>
        <translation>邊距</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="129"/>
        <source>Must cover the largest displacement of the ROI.</source>
        <translation>必須涵蓋 ROI 內的最大位移。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="136"/>
        <source>GPU memory</source>
        <translation>GPU 記憶體</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="137"/>
        <source>Fraction of the free GPU memory one window may use.
Large regions are cut into tiles to fit it.</source>
        <translation>單一視窗可使用的閒置 GPU 記憶體比例。
較大的區域會被切分成多個分塊以符合此限制。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="144"/>
        <source>Tile blending</source>
        <translation>分塊融合</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="145"/>
        <source>Width of the feathered overlap between neighbouring tiles.</source>
        <translation>相鄰分塊之間羽化重疊區的寬度。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="148"/>
        <source>Automatic</source>
        <translation>自動</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="149"/>
        <source>On demand</source>
        <translation>隨需計算</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="150"/>
        <source>Stored volume</source>
        <translation>預存相關體</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="152"/>
        <source>Correlation</source>
        <translation>相關計算</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="153"/>
        <source>How the correlation lookup is computed; the results are identical.
Automatic: stored volume for s8 (fast), on demand for s1-s4
(memory linear in the image size).</source>
        <translation>相關查找的計算方式；兩種方式的結果完全相同。
自動：s8 使用預存相關體（較快），s1-s4 使用隨需計算
（記憶體隨影像尺寸線性增長）。</translation>
    </message>
</context>
<context>
    <name>AnalysisChart</name>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="75"/>
        <source>too few valid points</source>
        <translation>有效點過少</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="73"/>
        <source>partly untracked</source>
        <translation>部分未追蹤</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="77"/>
        <source>no strain value (sparse strain gauge)</source>
        <translation>無應變值（VSG 內點過少）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="80"/>
        <source>gauge endpoint lost</source>
        <translation>量規端點失效</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="82"/>
        <source>not computed</source>
        <translation>未計算</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="83"/>
        <source>no data</source>
        <translation>無資料</translation>
    </message>
</context>
<context>
    <name>AnalysisTab</name>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="82"/>
        <source>Point</source>
        <comment>Placement tool: a single location</comment>
        <translation>點</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="84"/>
        <source>Line</source>
        <comment>Placement tool: a two-point gauge</comment>
        <translation>線段</translation>
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
        <translation>圓形</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="90"/>
        <source>Polygon</source>
        <comment>Placement tool</comment>
        <translation>多邊形</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="102"/>
        <source>Click once to place a point probe.</source>
        <translation>點擊一次放置點探針。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="108"/>
        <source>Click twice: opposite corners.</source>
        <translation>點擊兩次：對角兩點。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="112"/>
        <source>Click each vertex, then double-click to close.</source>
        <translation>逐一點擊頂點，雙擊閉合。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="170"/>
        <source>partly untracked from frame %1</source>
        <translation>自第 %1 影格起部分未追蹤</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="179"/>
        <source>gaps: no strain value</source>
        <translation>有缺口：無應變值</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="201"/>
        <source>Nothing valid along %1: the strain gauge found too few points there (near an edge or a hole). Plot a displacement, or use a smaller gauge on the Strain Field tab.</source>
        <translation>%1 沿線沒有有效資料：該處靠近邊緣或孔，VSG 內的點過少。請改為繪製位移，或在「應變場」頁減小 VSG 尺寸。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="207"/>
        <source>Nothing valid along %1: the material under it is untracked (cut out of the frame&apos;s ROI, or out of view).</source>
        <translation>%1 沿線沒有有效資料：其下方的材料未被追蹤（已被該影格的 ROI 排除，或已移出視野）。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="89"/>
        <source>Show</source>
        <comment>Probe list column: visibility checkbox</comment>
        <translation>顯示</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="91"/>
        <source>Name</source>
        <comment>Probe list column: the probe&apos;s label</comment>
        <translation>名稱</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="93"/>
        <source>Type</source>
        <comment>Probe list column: point, line or region</comment>
        <translation>類型</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="95"/>
        <source>Color</source>
        <comment>Probe list column: color swatch</comment>
        <translation>顏色</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="100"/>
        <source>Color…</source>
        <translation>顏色…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="97"/>
        <source>Note</source>
        <comment>Probe list column: why a probe shows gaps</comment>
        <translation>說明</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="101"/>
        <source>Delete</source>
        <comment>Button: delete the selected probe</comment>
        <translation>刪除</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="103"/>
        <source>Clear All</source>
        <translation>全部清除</translation>
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
        <translation>平均值</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="58"/>
        <source>Median</source>
        <comment>Statistic</comment>
        <translation>中位數</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="59"/>
        <source>Maximum</source>
        <comment>Statistic</comment>
        <translation>最大值</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="60"/>
        <source>Minimum</source>
        <comment>Statistic</comment>
        <translation>最小值</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="61"/>
        <source>Standard deviation</source>
        <translation>標準差</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="62"/>
        <source>Valid fraction</source>
        <translation>有效比例</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="46"/>
        <source>Crack opening</source>
        <translation>裂紋張開位移</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="489"/>
        <source>Strain has not been computed yet. Compute it on the Strain Field tab, or plot a displacement.</source>
        <translation>尚未計算應變。請在「應變場」頁中計算，或改為繪製位移。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="533"/>
        <source>Gauge quantities need a line probe.</source>
        <translation>量規類物理量需要線探針。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="530"/>
        <source>No visible probe can show this quantity.</source>
        <translation>沒有可見的探針能顯示此物理量。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="166"/>
        <source>no valid data: %1</source>
        <translation>無有效資料：%1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="174"/>
        <source>endpoint lost from frame %1</source>
        <translation>自第 %1 影格起端點失效</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="177"/>
        <source>gaps: too few valid points</source>
        <translation>有缺口：有效點過少</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="185"/>
        <source>not plotted: gauges need a line</source>
        <translation>未繪製：量規需要線探針</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="188"/>
        <source>not plotted: one point has no spread or coverage</source>
        <translation>未繪製：單一點沒有離散度或覆蓋率</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="190"/>
        <source>not plotted</source>
        <translation>未繪製</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="513"/>
        <source>Added probe &apos;%1&apos;.</source>
        <translation>已新增探針「%1」。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="599"/>
        <source>Clear All Probes</source>
        <translation>清除所有探針</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="600"/>
        <source>Delete every probe? This cannot be undone.</source>
        <translation>刪除所有探針？此操作無法復原。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="110"/>
        <source>Point</source>
        <comment>Probe type</comment>
        <translation>點</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="111"/>
        <source>Line</source>
        <comment>Probe type</comment>
        <translation>線段</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="112"/>
        <source>Region</source>
        <comment>Probe type: an enclosed area</comment>
        <translation>區域</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="480"/>
        <source>Run a DIC analysis to plot probes.</source>
        <translation>請先執行 DIC 分析，才能繪製探針曲線。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="92"/>
        <source>Virtual extensometer</source>
        <comment>Placement tool</comment>
        <translation>虛擬引伸計</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="94"/>
        <source>Crack gauge</source>
        <comment>Placement tool: a line across a crack</comment>
        <translation>裂紋量規</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="104"/>
        <source>Click twice: start and end. A line is also a virtual extensometer and a crack-opening gauge.</source>
        <translation>點擊兩次：起點和終點。線探針同時也是虛擬引伸計和裂紋張開量規。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="110"/>
        <source>Click twice: center, then the edge.</source>
        <translation>點擊兩次：先圓心，後邊緣。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="114"/>
        <source>Click the two gauge points. The chart then shows the strain between them.</source>
        <translation>點擊兩個標距點。圖表隨後顯示兩點之間的應變。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="118"/>
        <source>Click one point on each side of the crack. The chart then shows how far it opens.</source>
        <translation>在裂紋兩側各點擊一個點。圖表隨後顯示裂紋的張開量。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="124"/>
        <source>Fit</source>
        <comment>Zoom button: fit the image to the view</comment>
        <translation>適配</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="131"/>
        <source>Fit image to viewport</source>
        <translation>將影像適配到檢視區</translation>
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
        <translation>縮放到 100%（1:1）</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="136"/>
        <source>Zoom in</source>
        <translation>放大</translation>
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
        <translation>顯示場</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="140"/>
        <source>Color the reference image with the plotted field at the current frame. For a gauge reading, the Strain Field tab&apos;s field is shown.</source>
        <translation>以目前影格所繪物理量的場為參考影像著色。對於量規讀數，顯示「應變場」頁中的場。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="181"/>
        <source>Plot:</source>
        <translation>繪製：</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="184"/>
        <source>X axis:</source>
        <translation>X 軸：</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="185"/>
        <source>Strain as:</source>
        <translation>應變顯示為：</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="187"/>
        <source>Min. valid fraction:</source>
        <translation>最小有效比例：</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="188"/>
        <source>A frame is left blank when fewer than this fraction of a line&apos;s or region&apos;s points are reliable. Guards against a curve that stays smooth while its sample shrinks away.</source>
        <translation>當線或區域中可靠點的比例低於此值時，該影格留空。避免樣本逐漸縮小而曲線依然平滑的假象。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="195"/>
        <source>Over time</source>
        <comment>Chart view: every frame of each probe</comment>
        <translation>隨時間變化</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="197"/>
        <source>Each probe&apos;s reading at every frame.</source>
        <translation>每個探針在各影格的讀數。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="200"/>
        <source>Along the line</source>
        <comment>Chart view: a profile</comment>
        <translation>沿線分佈</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="207"/>
        <source>Kymograph</source>
        <comment>Chart view: distance against frame</comment>
        <translation>時空圖</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="202"/>
        <source>The field along the selected line at the current frame, over the other frames in gray.</source>
        <translation>目前影格沿所選線的場分佈，其他影格以灰色顯示在下方。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="209"/>
        <source>The field along the selected line at every frame: distance against frame, value as color.</source>
        <translation>所選線在每一影格的場分佈：距離對影格，數值以顏色表示。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="214"/>
        <source>Stress–strain</source>
        <comment>Chart view: stress against strain</comment>
        <translation>應力–應變</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="216"/>
        <source>Stress (or load, without A0) against the plotted quantity, one curve per probe: stress-strain with an extensometer, load against opening with a crack gauge. Needs load data.</source>
        <translation>以所繪物理量為橫軸的應力（無 A0 時為載荷）曲線，每個探針一條：搭配引伸計為應力–應變曲線，搭配裂紋量規為載荷–張開量曲線。需要載荷資料。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="227"/>
        <source>Other frames</source>
        <translation>其他影格</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="228"/>
        <source>Draw the other frames&apos; profiles faintly behind the current one (at most twelve, evenly spaced).</source>
        <translation>在目前影格後方以淡色繪製其他影格的分佈（最多十二個，均勻間隔）。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="233"/>
        <source>Line data (CSV)…</source>
        <translation>線資料（CSV）…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="234"/>
        <source>Export</source>
        <translation>匯出</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="235"/>
        <source>Load data…</source>
        <translation>載荷資料…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="236"/>
        <source>Import a testing machine&apos;s load record (CSV) to plot against load or stress, and to draw stress-strain curves.</source>
        <translation>匯入試驗機的載荷記錄（CSV），以載荷或應力為橫軸繪圖，並繪製應力–應變曲線。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="241"/>
        <source>Probe data (CSV)…</source>
        <translation>探針資料（CSV）…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="243"/>
        <source>Chart image…</source>
        <translation>圖表影像…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="245"/>
        <source>Copy chart</source>
        <translation>複製圖表</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="247"/>
        <source>Copy plotted data</source>
        <translation>複製所繪資料</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="559"/>
        <source>Import the testing machine&apos;s load record with Load data… to draw stress-strain curves.</source>
        <translation>請透過「載荷資料…」匯入試驗機的載荷記錄，以繪製應力–應變曲線。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="31"/>
        <source>Displacement U</source>
        <translation>位移 U</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="32"/>
        <source>Displacement V</source>
        <translation>位移 V</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="33"/>
        <source>Displacement magnitude</source>
        <translation>位移大小</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="42"/>
        <source>Extensometer strain</source>
        <translation>引伸計應變</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="43"/>
        <source>Extensometer true strain</source>
        <translation>引伸計真應變</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="45"/>
        <source>Elongation ΔL</source>
        <translation>伸長量 ΔL</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="47"/>
        <source>Crack sliding</source>
        <translation>裂紋滑移</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="48"/>
        <source>Crack opening magnitude</source>
        <translation>裂紋張開位移大小</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="279"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="434"/>
        <source>Frame</source>
        <translation>影格</translation>
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
        <translation>載荷 (N)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="286"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="433"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="571"/>
        <source>Stress (MPa)</source>
        <translation>應力 (MPa)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="295"/>
        <source>ratio</source>
        <comment>Strain display unit: plain number</comment>
        <translation>比值</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="420"/>
        <source>Could not draw the field: %1</source>
        <translation>無法繪製場：%1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="130"/>
        <source>Esc cancels placement</source>
        <translation>Esc 取消放置</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="138"/>
        <source>Drag to move the probe, or drag a handle to reshape it. Delete removes it; F2 renames it.</source>
        <translation>拖曳可移動探針，拖曳控制點可改變其形狀。按 Delete 刪除，按 F2 重新命名。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="485"/>
        <source>Place a probe on the reference image to begin.</source>
        <translation>在參考影像上放置一個探針即可開始。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="212"/>
        <source>Nothing valid along %1: it lies off the measured area.</source>
        <translation>%1 沿線沒有有效資料：它不在量測區域內。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="610"/>
        <source>A line view shows a field. Choose a field to plot.</source>
        <translation>沿線檢視顯示的是場。請選擇一個場來繪製。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="614"/>
        <source>Place a line probe, or select one, to see the field along it.</source>
        <translation>放置或選擇一個線探針，即可查看沿線的場。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="632"/>
        <source>Distance along %1 (%2)</source>
        <translation>沿 %1 的距離（%2）</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="641"/>
        <source>%1, frame %2</source>
        <translation>%1，第 %2 影格</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="153"/>
        <source>not plotted: off the measured area</source>
        <translation>未繪製：不在量測區域內</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="156"/>
        <source>not plotted: a gauge end is off the measured area</source>
        <translation>未繪製：量規端點不在量測區域內</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="671"/>
        <source>Export Probe Data</source>
        <translation>匯出探針資料</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="664"/>
        <source>CSV Files</source>
        <translation>CSV 檔案</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="442"/>
        <source>The load data cannot be matched to the frames: %1</source>
        <translation>載荷資料無法與影格對應：%1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="664"/>
        <location filename="../../gui/panels/analysis/tab.py" line="728"/>
        <source>All Files</source>
        <translation>所有檔案</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="699"/>
        <source>Probe export failed: %1</source>
        <translation>探針匯出失敗：%1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="701"/>
        <source>Probe data written to %1</source>
        <translation>探針資料已寫入 %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="709"/>
        <source>Export Line Data</source>
        <translation>匯出線資料</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="718"/>
        <source>Line export failed: %1</source>
        <translation>線資料匯出失敗：%1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="720"/>
        <source>Line data written to %1</source>
        <translation>線資料已寫入 %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="726"/>
        <source>SVG Images</source>
        <translation>SVG 影像</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="737"/>
        <source>Export Chart</source>
        <translation>匯出圖表</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="756"/>
        <source>Chart copied to the clipboard.</source>
        <translation>圖表已複製到剪貼簿。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="764"/>
        <source>Plotted data copied to the clipboard.</source>
        <translation>所繪資料已複製到剪貼簿。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="725"/>
        <source>PNG Images</source>
        <translation>PNG 影像</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="727"/>
        <source>PDF Documents</source>
        <translation>PDF 文件</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="748"/>
        <source>Chart export failed: %1</source>
        <translation>圖表匯出失敗：%1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="750"/>
        <source>Chart written to %1</source>
        <translation>圖表已寫入 %1</translation>
    </message>
</context>
<context>
    <name>App</name>
    <message>
        <location filename="../../gui/app.py" line="713"/>
        <source>Run DIC first -- no displacement results to post-process.</source>
        <translation>請先執行 DIC —— 目前沒有可後處理的位移結果。</translation>
    </message>
</context>
<context>
    <name>Application</name>
    <message>
        <location filename="../../gui/app.py" line="846"/>
        <source>RAFTcorr has hit an error</source>
        <translation>RAFTcorr 發生錯誤</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="847"/>
        <source>An unexpected error occurred. The application may not behave correctly from here on, so saving your session and restarting is recommended.</source>
        <translation>發生了非預期的錯誤。應用程式之後的行為可能不正常，建議儲存工作階段並重新啟動。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="857"/>
        <source>Details were written to %1</source>
        <translation>詳細資訊已寫入 %1</translation>
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
        <translation>根據每個影格的資料範圍自動縮放顏色範圍</translation>
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
        <translation>所有影格都使用手動設定的最小/最大值</translation>
    </message>
</context>
<context>
    <name>BatchImportDialog</name>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="368"/>
        <source>Batch Import Region of Interest Masks</source>
        <translation>批次匯入感興趣區域遮罩</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="419"/>
        <source>Mask Folder:</source>
        <translation>遮罩資料夾：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="420"/>
        <source>(none)</source>
        <translation>（無）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="425"/>
        <source>Browse...</source>
        <translation>瀏覽…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="444"/>
        <source>Available Masks</source>
        <translation>可用遮罩</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="455"/>
        <source>Auto-Match by Name</source>
        <translation>按檔名自動匹配</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="457"/>
        <source>Match mask files to frames by number in filename</source>
        <translation>根據檔名中的數字把遮罩檔案匹配到對應影格</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="461"/>
        <source>Assign Sequential</source>
        <translation>順序分配</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="463"/>
        <source>Assign masks to frames in order starting from frame 0</source>
        <translation>從第 0 影格起按順序把遮罩分配給各影格</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="471"/>
        <source>Frame Assignments</source>
        <translation>影格分配</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="474"/>
        <source>Frame</source>
        <translation>影格</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="474"/>
        <source>Image</source>
        <translation>影像</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="474"/>
        <source>Mask</source>
        <translation>遮罩</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="485"/>
        <source>Assign Selected -&gt;</source>
        <translation>分配所選 -&gt;</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="487"/>
        <source>Pair selected mask(s) with selected frame(s)</source>
        <translation>將所選遮罩與所選影格配對</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="491"/>
        <source>Clear All</source>
        <translation>全部清除</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="536"/>
        <source>Select Mask Folder</source>
        <translation>選擇遮罩資料夾</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="590"/>
        <source>Failed to read mask file.</source>
        <translation>無法讀取遮罩檔案。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="594"/>
        <source>Mismatched shape: %1×%2 (expected %3×%4)</source>
        <translation>尺寸不符：%1×%2（預期 %3×%4）</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="666"/>
        <source>%n mask(s) have mismatched sizes and are disabled.</source>
        <translation>
            <numerusform>%n 個遮罩尺寸不符，已停用。</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="719"/>
        <source>Invalid assignment</source>
        <translation>無效的分配</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="721"/>
        <source>A frame can only have one mask. Select exactly one mask, or select multiple frames to assign one mask to many.</source>
        <translation>一個影格只能對應一個遮罩。請選擇恰好一個遮罩，或選擇多個影格以將同一個遮罩分配給多個影格。</translation>
    </message>
</context>
<context>
    <name>CanvasArea</name>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1222"/>
        <source>Fit</source>
        <translation>適配</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1223"/>
        <source>Fit image to viewport</source>
        <translation>將影像適配到檢視區</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1227"/>
        <source>100%</source>
        <translation>100%</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1228"/>
        <source>Zoom to 100% (1:1)</source>
        <translation>縮放到 100%（1:1）</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1231"/>
        <source>Zoom in</source>
        <translation>放大</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1236"/>
        <source>–</source>
        <translation>–</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1237"/>
        <source>Zoom out</source>
        <translation>縮小</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1321"/>
        <source>Editing the ROI of frame 1 (the reference)</source>
        <translation>正在編輯第 1 影格（參考影格）的 ROI</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1322"/>
        <source>Editing the mask of frame %1 (regions left out are excluded)</source>
        <translation>正在編輯第 %1 影格的遮罩（未包含的區域將被排除）</translation>
    </message>
</context>
<context>
    <name>ColorRange</name>
    <message>
        <location filename="../../gui/widgets/color_range.py" line="27"/>
        <source>Range</source>
        <translation>範圍</translation>
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
        <translation>位移大小</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="46"/>
        <source>Residual</source>
        <translation>殘差</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="66"/>
        <source>Open this tab to render a preview.</source>
        <translation>開啟此分頁以繪製預覽。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="73"/>
        <location filename="../../gui/dialogs/export_widgets.py" line="368"/>
        <source>Field</source>
        <translation>場變數</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="80"/>
        <source>Frame</source>
        <translation>影格</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="101"/>
        <source>FIELD APPEARANCE</source>
        <translation>場的外觀</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="106"/>
        <location filename="../../gui/dialogs/export_widgets.py" line="368"/>
        <source>Colormap</source>
        <translation>色彩對映</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="110"/>
        <source>Range</source>
        <translation>範圍</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="114"/>
        <location filename="../../gui/dialogs/export_widgets.py" line="368"/>
        <source>Min</source>
        <translation>最小</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="118"/>
        <location filename="../../gui/dialogs/export_widgets.py" line="368"/>
        <source>Max</source>
        <translation>最大</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="125"/>
        <location filename="../../gui/dialogs/export_widgets.py" line="174"/>
        <source>Opacity</source>
        <extracomment>Column widths of the field rows (export, field, colormap, range mode, min / max), shared by the header labels of the Images and Animation tabs. Older names that share a field&apos;s colour state.</extracomment>
        <translation>不透明度</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="127"/>
        <source>Apply to all fields</source>
        <translation>套用到所有場</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="128"/>
        <source>Apply this field&apos;s colormap, opacity and auto-range to every enabled field (each field keeps its own min/max).</source>
        <translation>將該場的色彩對映、不透明度和自動範圍套用到所有已啟用的場（每個場保留各自的最小/最大值）。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="133"/>
        <source>Refresh preview</source>
        <translation>重新整理預覽</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="160"/>
        <source>Preview failed: </source>
        <translation>預覽失敗：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="194"/>
        <source>Enable a field on the Images tab to preview.</source>
        <translation>請在「影像」頁啟用一個場以進行預覽。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="198"/>
        <source>No data for this field/frame.</source>
        <translation>該場/影格沒有資料。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="175"/>
        <source>Field opacity (0 = transparent, 1 = fully opaque)</source>
        <translation>場的不透明度（0 = 透明，1 = 完全不透明）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="282"/>
        <source>All</source>
        <translation>全部</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="284"/>
        <source>None</source>
        <translation>無</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="355"/>
        <source>Select:</source>
        <translation>選擇：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="368"/>
        <source>Export</source>
        <translation>匯出</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="368"/>
        <source>Auto</source>
        <translation>自動</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="385"/>
        <source>PHYSICAL UNITS</source>
        <translation>物理單位</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="389"/>
        <source>Enable physical units</source>
        <translation>啟用物理單位</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="391"/>
        <source>Scale displacement values by pixel size and show physical units on colorbar labels. Strain is dimensionless and unaffected.</source>
        <translation>按像素尺寸縮放位移值，並在色條標籤上顯示物理單位。應變為無因次量，不受影響。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="408"/>
        <source>/ pixel</source>
        <translation>/ 像素</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="410"/>
        <source>Pixel size</source>
        <translation>像素尺寸</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="419"/>
        <source>fps</source>
        <translation>fps</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="421"/>
        <source>Frame rate</source>
        <translation>影格率</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="447"/>
        <source>FRAME RANGE</source>
        <translation>影格範圍</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="449"/>
        <source>All frames</source>
        <translation>所有影格</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="454"/>
        <source>Numbered like the frame slider: frame 1 is the reference image.
File names use the same numbers: frame 1 is written as _0001.</source>
        <translation>與影格滑桿的編號一致：第 1 影格是參考影像。
檔名使用相同的編號：第 1 影格寫為 _0001。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="457"/>
        <source>From</source>
        <comment>Frame range: starting frame</comment>
        <translation>從</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="462"/>
        <source>to</source>
        <comment>Frame range: ending frame</comment>
        <translation>到</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="161"/>
        <source>Export Results</source>
        <extracomment>GUI field name -&gt; variable name in the data files (raftcorr.export.scientific).</extracomment>
        <translation>匯出結果</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="188"/>
        <source>OUTPUT FOLDER</source>
        <translation>輸出資料夾</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="194"/>
        <source>Select output folder…</source>
        <translation>選擇輸出資料夾…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="200"/>
        <source>Browse…</source>
        <translation>瀏覽…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="205"/>
        <source>Open Folder</source>
        <translation>開啟資料夾</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="219"/>
        <source>Data</source>
        <translation>資料</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="220"/>
        <source>Images</source>
        <translation>影像</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="221"/>
        <source>Animation</source>
        <translation>動畫</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="222"/>
        <source>Report</source>
        <translation>報告</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="304"/>
        <source>FORMAT</source>
        <translation>格式</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="306"/>
        <source>NumPy Archive (.npz)</source>
        <translation>NumPy 歸檔 (.npz)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="313"/>
        <source>CSV (per frame)</source>
        <translation>CSV（逐影格）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="323"/>
        <source>DISPLACEMENT</source>
        <translation>位移</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="326"/>
        <source>STRAIN</source>
        <translation>應變</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="271"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="329"/>
        <source>Run Compute Strain first.</source>
        <translation>請先執行“計算應變”。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="223"/>
        <source>Preview</source>
        <translation>預覽</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="252"/>
        <source>Export: images not readable (%1); pictures go on white.</source>
        <translation>匯出：無法讀取影像（%1）；圖片將以白色為背景。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="256"/>
        <source>Export: the images do not match the results&apos; image size; pictures go on white.</source>
        <translation>匯出：影像尺寸與結果的影像尺寸不一致；圖片將以白色為背景。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="270"/>
        <source>This run has no intensity residual.</source>
        <translation>本次執行沒有灰階殘差。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="308"/>
        <source>MATLAB v7.3 (.mat)</source>
        <translation>MATLAB v7.3 (.mat)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="310"/>
        <source>An HDF5-based MAT file: load it in MATLAB with load().</source>
        <translation>以 HDF5 為基礎的 MAT 檔案：在 MATLAB 中以 load() 讀取。</translation>
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
        <translation>每個已計算影格一張表，感興趣區域內每個點一列。
密集場會產生很大的檔案：1000 x 1000 px 的區域每個影格有一百萬列。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="333"/>
        <source>✓ Parameters file (JSON) always exported</source>
        <translation>✓ 參數檔案（JSON）始終匯出</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="334"/>
        <source>✓ Grid coordinates (X, Y), ROI and computed-frame flags always included</source>
        <translation>✓ 一律包含網格座標（X, Y）、ROI 與已計算影格標記</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="335"/>
        <source>Displacements are written in pixels (+u right, +v down); the pixel size and frame rate above are recorded with them.</source>
        <translation>位移以像素為單位寫入（+u 向右，+v 向下）；上方的像素尺寸與影格率會一併記錄。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="341"/>
        <source>Export Data</source>
        <translation>匯出資料</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="377"/>
        <source>IMAGE SETTINGS</source>
        <translation>影像設定</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="387"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="464"/>
        <source>Format</source>
        <translation>格式</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="351"/>
        <source>Deformed frame</source>
        <translation>變形影格</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="352"/>
        <source>Reference frame</source>
        <translation>參考影格</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="358"/>
        <source>Show background image</source>
        <translation>顯示背景影像</translation>
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
        <translation>包含色條</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="421"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="496"/>
        <source>Render as</source>
        <translation>繪製為</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="431"/>
        <source>Export Images</source>
        <translation>匯出影像</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="445"/>
        <source>ANIMATION SETTINGS</source>
        <translation>動畫設定</translation>
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
        <translation>每 N 個影格匯出一個影格（1 = 每個影格都匯出）。越大越快、越小，
但看起來更卡頓。播放時長保持不變（上方 FPS 為抽取影格前的影格率）。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="488"/>
        <source>Frame step</source>
        <translation>影格間隔</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="511"/>
        <source>Export Animation</source>
        <translation>匯出動畫</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="521"/>
        <source>CONTENT</source>
        <translation>內容</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="523"/>
        <source>Parameter summary table</source>
        <translation>參數摘要表</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="525"/>
        <source>Field statistics (min/max/mean/std per frame)</source>
        <translation>場統計（每影格 最小/最大/平均/標準差）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="527"/>
        <source>Sample field images</source>
        <translation>場的範例影像</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="535"/>
        <source>Sample every</source>
        <translation>每隔</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="541"/>
        <source>frames</source>
        <comment>Report: sample every N frames</comment>
        <translation>個影格</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="547"/>
        <source>FIELDS</source>
        <translation>場變數</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="549"/>
        <source>Displacement:</source>
        <translation>位移：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="550"/>
        <source>Strain:</source>
        <translation>應變：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="559"/>
        <source>Format: HTML (self-contained, view in any browser)</source>
        <translation>格式：HTML（自包含，可在任意瀏覽器中檢視）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="563"/>
        <source>Generate Report</source>
        <translation>產生報告</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="706"/>
        <source>Select Output Folder</source>
        <translation>選擇輸出資料夾</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="807"/>
        <source>Writing %1 (%2%)</source>
        <translation>正在寫入 %1 (%2%)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="840"/>
        <source>Error: %1</source>
        <translation>錯誤：%1</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="782"/>
        <source>Starting…</source>
        <translation>開始中…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="355"/>
        <source>Deformed: the field is drawn where each point is in that frame, over the frame&apos;s own photo.
Reference: drawn at the points&apos; positions in frame 1, over frame 1.</source>
        <translation>變形影格：場繪製在各點於該影格中的位置，疊加在該影格自己的照片上。
參考影格：繪製在各點於第 1 影格中的位置，疊加在第 1 影格上。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="362"/>
        <source>Uncheck to export the field on its own, on white, with no speckle image behind it.</source>
        <translation>取消勾選可僅匯出場本身，以白色為背景，其後不含散斑影像。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="363"/>
        <source>The images of this run are not available.</source>
        <translation>本次執行的影像無法使用。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="395"/>
        <source>Resolution of the written figure: about 8.6 x 7 inches at this many dots per inch.</source>
        <translation>輸出圖片的解析度：圖幅約 8.6 x 7 英吋，按此每英吋點數輸出。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="402"/>
        <source>Each frame</source>
        <translation>逐影格</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="403"/>
        <source>Whole sequence</source>
        <translation>整個序列</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="405"/>
        <source>How Auto picks the colour range: from each frame&apos;s own values (like the main window),
or one range from every exported frame, so a colour means the same value in every image.</source>
        <translation>自動範圍的取法：依每個影格自身的數值（與主視窗相同），
或由所有匯出影格共同決定一個範圍，使同一顏色在每張影像中代表同一數值。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="410"/>
        <source>Auto range</source>
        <translation>自動範圍</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="414"/>
        <source>Add a vertical colorbar to the right of each image.</source>
        <translation>在每張影像右側添加一條垂直色條。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="423"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="498"/>
        <source>Crop to ROI</source>
        <translation>裁切至 ROI</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="424"/>
        <source>Cut the picture to the region of interest plus a small margin.</source>
        <translation>將圖片裁切至感興趣區域，並保留少量邊距。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="457"/>
        <source>MP4 needs the imageio and imageio-ffmpeg packages (pip install imageio imageio-ffmpeg). Save as GIF instead.</source>
        <translation>MP4 需要 imageio 與 imageio-ffmpeg 套件（pip install imageio imageio-ffmpeg）。請改存為 GIF。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="492"/>
        <source>Add a vertical colorbar to the right of each frame.</source>
        <translation>在每個影格右側添加一條垂直色條。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="499"/>
        <source>Cut the frames to the region of interest plus a small margin.</source>
        <translation>將各影格裁切至感興趣區域，並保留少量邊距。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="503"/>
        <source>Auto range: one range for the whole animation, so a colour means the same value in every frame.</source>
        <translation>自動範圍：整段動畫使用同一個範圍，使同一顏色在每個影格中代表同一數值。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="530"/>
        <source>Drawn like the Images tab: same colormap, range, opacity and configuration.</source>
        <translation>繪製方式與「影像」頁相同：色彩對映、範圍、不透明度與配置皆一致。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="639"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="687"/>
        <source>Min must be below Max for %1.</source>
        <translation>%1 的最小值必須小於最大值。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="645"/>
        <source>Nothing to export: enable at least one field with computed frames in the frame range.</source>
        <translation>沒有可匯出的內容：請至少啟用一個在影格範圍內有已計算影格的場。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="653"/>
        <source>Select at least one format.</source>
        <translation>請至少選擇一種格式。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="657"/>
        <source>Select at least one variable to export.</source>
        <translation>請至少選擇一個要匯出的變數。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="690"/>
        <source>Nothing to export: select a field or the parameter table.</source>
        <translation>沒有可匯出的內容：請選擇一個場或參數表。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="743"/>
        <source>… and %1 more</source>
        <translation>… 另有 %1 個檔案</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="745"/>
        <source>Replace Existing Files?</source>
        <translation>取代現有檔案？</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="746"/>
        <source>These already exist in the output folder and will be replaced:

%1</source>
        <translation>以下檔案已存在於輸出資料夾中，將被取代：

%1</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="765"/>
        <source>The results changed after this dialog was opened (a new run, strain or project). Close it and open Export again to export them.</source>
        <translation>開啟此對話方塊後結果已變更（新的執行、應變或專案）。請關閉對話方塊並重新開啟「匯出」，再匯出這些結果。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="769"/>
        <source>Nothing exported: the existing files were kept.</source>
        <translation>未匯出任何內容：已保留現有檔案。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="794"/>
        <source>Stopping…</source>
        <translation>正在停止…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="805"/>
        <source>Rendering %1 (%2/%3)</source>
        <translation>正在繪製 %1 (%2/%3)</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/dialogs/export_dialog.py" line="823"/>
        <source>Exported %n file(s) → %1</source>
        <translation>
            <numerusform>已匯出 %n 個檔案 → %1</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/dialogs/export_dialog.py" line="825"/>
        <source>Exported %n image(s) → %1</source>
        <translation>
            <numerusform>已匯出 %n 張影像 → %1</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="841"/>
        <source>Export failed: %1</source>
        <translation>匯出失敗：%1</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/dialogs/export_dialog.py" line="849"/>
        <source>Export cancelled: %n file(s) written before stopping.</source>
        <translation>
            <numerusform>匯出已取消：停止前已寫入 %n 個檔案。</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="829"/>
        <source>No animation was written. See the log for details.</source>
        <translation>未寫入任何動畫。詳情請查看日誌。</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/dialogs/export_dialog.py" line="831"/>
        <source>Exported %n animation(s) → %1</source>
        <translation>
            <numerusform>已匯出 %n 個動畫 → %1</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="833"/>
        <source>Report saved → %1</source>
        <translation>報告已儲存 → %1</translation>
    </message>
</context>
<context>
    <name>FieldSelector</name>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="19"/>
        <source>Horizontal displacement (+ right)</source>
        <translation>水平位移（向右為正）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="20"/>
        <source>Vertical displacement (+ down)</source>
        <translation>垂直位移（向下為正）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="21"/>
        <source>Displacement magnitude</source>
        <translation>位移大小</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="22"/>
        <source>Intensity residual: local RMS difference between the reference image and
the deformed image mapped back by the displacement. Large where the
tracking failed.</source>
        <translation>灰階殘差：參考影像與依位移映射回來的變形影像之間的
局部均方根差。追蹤失敗處數值較大。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="26"/>
        <source>Disp U</source>
        <translation>位移 U</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="26"/>
        <source>Disp V</source>
        <translation>位移 V</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="27"/>
        <source>|U|</source>
        <translation>|U|</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="27"/>
        <source>Residual</source>
        <translation>殘差</translation>
    </message>
</context>
<context>
    <name>FrameNavigator</name>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="59"/>
        <source>Previous frame</source>
        <translation>上一影格</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="69"/>
        <location filename="../../gui/widgets/frame_navigator.py" line="170"/>
        <source>Play animation</source>
        <translation>播放動畫</translation>
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
        <translation>下一影格</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="93"/>
        <source>Playback speed</source>
        <translation>播放速度</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="98"/>
        <location filename="../../gui/widgets/frame_navigator.py" line="194"/>
        <source>FRAME 0/0</source>
        <translation>影格 0/0</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="159"/>
        <source>⏸</source>
        <translation>⏸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="160"/>
        <source>Pause animation</source>
        <translation>暫停動畫</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="191"/>
        <source>FRAME %1/%2</source>
        <translation>影格 %1/%2</translation>
    </message>
</context>
<context>
    <name>ImageCanvas</name>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1066"/>
        <source>Load images first before drawing a Region of Interest.</source>
        <translation>請先載入影像，再繪製感興趣區域。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1085"/>
        <source>The three points are nearly collinear — pick points spread around the circle&apos;s edge.</source>
        <translation>三個點幾乎共線 — 請在圓周上分散地選取三個點。</translation>
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
        <translation>檔名</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="99"/>
        <source>Region</source>
        <comment>Image list column: ROI status</comment>
        <translation>區域</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="193"/>
        <location filename="../../gui/widgets/image_list.py" line="255"/>
        <source>Add</source>
        <translation>新增</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="249"/>
        <source>Edit</source>
        <translation>編輯</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="252"/>
        <source>Need</source>
        <translation>待繪</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/widgets/image_list.py" line="368"/>
        <source>Import Region of Interest for %n frame(s)</source>
        <translation>
            <numerusform>為 %n 個影格匯入感興趣區域</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="379"/>
        <source>Clear Region of Interest (%1 with region)</source>
        <translation>清除感興趣區域（%1 個影格已有區域）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="383"/>
        <source>Clear Region of Interest</source>
        <translation>清除感興趣區域</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/widgets/image_list.py" line="394"/>
        <source>Delete %n image(s)</source>
        <translation>
            <numerusform>刪除 %n 張影像</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/widgets/image_list.py" line="435"/>
        <source>Select %n Mask File(s)</source>
        <translation>
            <numerusform>選擇 %n 個遮罩檔案</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="440"/>
        <source>Images</source>
        <translation>影像</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="441"/>
        <source>All Files</source>
        <translation>所有檔案</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="450"/>
        <source>Selected %1 files for %2 frames — count must match</source>
        <translation>已選擇 %1 個檔案用於 %2 個影格 — 數量必須相符</translation>
    </message>
</context>
<context>
    <name>ImageMessages</name>
    <message>
        <location filename="../../gui/controllers/image_controller.py" line="43"/>
        <source>Cannot open the images</source>
        <translation>無法開啟影像</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/image_controller.py" line="47"/>
        <source>%n image(s), %1 x %2 px, %3</source>
        <translation>
            <numerusform>%n 張影像，%1 x %2 px，%3</numerusform>
        </translation>
    </message>
</context>
<context>
    <name>LeftSidebar</name>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="176"/>
        <source>IMAGES</source>
        <translation>影像</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="183"/>
        <source>Natural Sort (1, 2, …, 10)</source>
        <translation>自然排序（1, 2, …, 10）</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="259"/>
        <source>WORKFLOW TYPE</source>
        <translation>工作流程類型</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="186"/>
        <source>Sort by embedded numbers: image1, image2, ..., image10 (default).
Unchecked: plain alphabetical order, for zero-padded names.</source>
        <translation>按檔名中的數字排序：image1, image2, ..., image10（預設）。
不勾選：按一般字母順序，適合已補零的檔名。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="266"/>
        <source>REGION OF INTEREST</source>
        <translation>感興趣區域</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="278"/>
        <source>PARAMETERS</source>
        <translation>參數</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="284"/>
        <source>ADVANCED</source>
        <translation>進階</translation>
    </message>
</context>
<context>
    <name>LoadDataDialog</name>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="103"/>
        <source>Load Data</source>
        <translation>載荷資料</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="115"/>
        <source>No file chosen.</source>
        <translation>未選擇檔案。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="117"/>
        <source>Choose file…</source>
        <translation>選擇檔案…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="137"/>
        <source>Load column:</source>
        <translation>載荷欄：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="139"/>
        <source>By time</source>
        <translation>按時間</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="140"/>
        <source>By frame number</source>
        <translation>按影格編號</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="144"/>
        <source>Match rows to frames:</source>
        <translation>列與影格的對應方式：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="147"/>
        <source>Time column:</source>
        <translation>時間欄：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="153"/>
        <source>The machine&apos;s time at the reference image. If the camera started 2 s after the machine, enter 2.</source>
        <translation>參考影像時刻對應的試驗機時間。若相機比試驗機晚 2 s 啟動，則輸入 2。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="155"/>
        <source>Offset:</source>
        <translation>偏移：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="163"/>
        <source>Frame column:</source>
        <translation>影格欄：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="167"/>
        <source>The first image is numbered:</source>
        <translation>第一張影像的編號：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="174"/>
        <source>Initial cross-section, for engineering stress F / A0 in MPa. Leave at 0 for load only.</source>
        <translation>初始橫截面積，用於計算工程應力 F / A0（MPa）。僅需載荷時保持為 0。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="176"/>
        <source>Cross-section A0:</source>
        <translation>橫截面積 A0：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="192"/>
        <source>Remove Load Data</source>
        <translation>移除載荷資料</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="201"/>
        <source>Camera frame rate: %1 fps, from Physical Units.</source>
        <translation>相機影格率：%1 fps，取自「物理單位」。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="204"/>
        <source>Set the camera frame rate under Physical Units to match by time.</source>
        <translation>請在「物理單位」中設定相機影格率，才能按時間對應。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="227"/>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="245"/>
        <source>(unnamed)</source>
        <translation>（未命名）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="324"/>
        <source>No frame falls within the record: check the columns and the offset.</source>
        <translation>沒有影格落在記錄範圍內：請檢查所選欄與偏移。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="328"/>
        <source>Frames with a load: %1 of %2.</source>
        <translation>有載荷值的影格：%1 / %2。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="334"/>
        <source>Open Load Data</source>
        <translation>開啟載荷資料</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="335"/>
        <source>CSV Files</source>
        <translation>CSV 檔案</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="335"/>
        <source>All Files</source>
        <translation>所有檔案</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="341"/>
        <source>Could not read %1: %2</source>
        <translation>無法讀取 %1：%2</translation>
    </message>
</context>
<context>
    <name>MainWindow</name>
    <message>
        <location filename="../../gui/app.py" line="136"/>
        <source>&amp;File</source>
        <translation>檔案</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="161"/>
        <source>Quit</source>
        <translation>離開</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="165"/>
        <source>&amp;Settings</source>
        <translation>設定</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="168"/>
        <source>Theme</source>
        <translation>主題</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="172"/>
        <source>Dark</source>
        <comment>Color theme</comment>
        <translation>深色</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="172"/>
        <source>Light</source>
        <comment>Color theme</comment>
        <translation>淺色</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="182"/>
        <source>Language</source>
        <translation>語言</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="423"/>
        <source>Language changed</source>
        <translation>語言已切換</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="594"/>
        <source>All Files</source>
        <translation>所有檔案</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="200"/>
        <source>Select Image Folder</source>
        <translation>選擇影像資料夾</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="289"/>
        <source>File Association Failed</source>
        <translation>檔案關聯失敗</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="292"/>
        <source>File Association</source>
        <translation>檔案關聯</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="146"/>
        <source>Open Image Folder…</source>
        <translation>開啟影像資料夾…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="148"/>
        <source>Open Project…</source>
        <translation>開啟專案…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="149"/>
        <location filename="../../gui/app.py" line="233"/>
        <source>Save Project</source>
        <translation>儲存專案</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="150"/>
        <source>Save Project As…</source>
        <translation>專案另存為…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="152"/>
        <source>Export Results…</source>
        <translation>匯出結果…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="157"/>
        <source>Associate .raftcorr files with RAFTcorr…</source>
        <translation>將 .raftcorr 檔案關聯到 RAFTcorr…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="158"/>
        <source>Register .raftcorr so double-clicking a project file opens RAFTcorr (current user only, no admin rights needed).</source>
        <translation>註冊 .raftcorr，讓雙擊專案檔案即可開啟 RAFTcorr（僅目前使用者，無需系統管理員權限）。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="191"/>
        <source>&amp;Help</source>
        <translation>說明</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="192"/>
        <source>Installed Models…</source>
        <translation>已安裝的模型…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="193"/>
        <location filename="../../gui/app.py" line="359"/>
        <location filename="../../gui/app.py" line="361"/>
        <source>Self-test</source>
        <translation>自我測試</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="194"/>
        <location filename="../../gui/app.py" line="366"/>
        <source>About RAFTcorr</source>
        <translation>關於 RAFTcorr</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="209"/>
        <source>Cancel</source>
        <translation>取消</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="245"/>
        <source>Project saved: %1</source>
        <translation>專案已儲存：%1</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="247"/>
        <source>Saving the project</source>
        <translation>正在儲存專案</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="250"/>
        <source>Open Project</source>
        <translation>開啟專案</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="257"/>
        <source>Opening the project</source>
        <translation>正在開啟專案</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="265"/>
        <source>Images not found</source>
        <translation>找不到影像</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="266"/>
        <source>The image folder of this project was not found. Choose its new location.</source>
        <translation>找不到此專案的影像資料夾。請選擇它的新位置。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="267"/>
        <source>Image folder of the project</source>
        <translation>專案的影像資料夾</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="274"/>
        <source>Cannot open the project</source>
        <translation>無法開啟專案</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="280"/>
        <source>Project opened: %1</source>
        <translation>已開啟專案：%1</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="290"/>
        <source>Could not register .raftcorr files: </source>
        <translation>無法註冊 .raftcorr 檔案：</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="293"/>
        <source>Done. Double-clicking a .raftcorr file will now open RAFTcorr and restore that project.</source>
        <translation>完成。現在雙擊 .raftcorr 檔案即可開啟 RAFTcorr 並還原該專案。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="298"/>
        <source>Load at least two images first.</source>
        <translation>請先載入至少兩張影像。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="301"/>
        <source>Wait for the running task to finish.</source>
        <translation>請等待目前的工作完成。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="330"/>
        <source>Suggested: RAFTcorr-%1. Largest displacement of the ROI about %2 px (99th percentile over frames %3); RAFTcorr-%1 reaches %4 px.</source>
        <translation>建議：RAFTcorr-%1。ROI 內最大位移約 %2 px（在第 %3 影格上取第 99 百分位數）；RAFTcorr-%1 的量程為 %4 px。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="334"/>
        <source>This exceeds even the reach of RAFTcorr-s8: use incremental tracking.</source>
        <translation>這甚至超出了 RAFTcorr-s8 的量程：請使用增量追蹤。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="339"/>
        <source>Measuring the motion with RAFTcorr-s8</source>
        <translation>正在以 RAFTcorr-s8 測量運動</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="348"/>
        <source>%1  (s = %2, reach %3 px): %4</source>
        <translation>%1  （s = %2，量程 %3 px）：%4</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="349"/>
        <source>not installed</source>
        <translation>未安裝</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="351"/>
        <source>Searched:</source>
        <translation>搜尋位置：</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="352"/>
        <source>Installed Models</source>
        <translation>已安裝的模型</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="367"/>
        <source>&lt;b&gt;RAFTcorr %1&lt;/b&gt;&lt;br&gt;Learned digital image correlation with the RAFTcorr-s1, -s2, -s4 and -s8 models.&lt;br&gt;&lt;br&gt;Displacements: reference to deformed, +u right, +v down, in pixels, frame-1 (Lagrangian) coordinates.&lt;br&gt;&lt;br&gt;Interface derived from pyALDIC.&lt;br&gt;Zixiang (Zach) Tong, The University of Texas at Austin.</source>
        <translation>&lt;b&gt;RAFTcorr %1&lt;/b&gt;&lt;br&gt;基於學習的數位影像相關，使用 RAFTcorr-s1、-s2、-s4 與 -s8 模型。&lt;br&gt;&lt;br&gt;位移：從參考到變形，+u 向右，+v 向下，單位為像素，採用第 1 影格（拉格朗日）座標。&lt;br&gt;&lt;br&gt;介面源自 pyALDIC。&lt;br&gt;Zixiang (Zach) Tong, The University of Texas at Austin.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="426"/>
        <source>Language set to %1. Please restart RAFTcorr for all widgets to pick up the new language.</source>
        <translation>語言已切換至 %1。請重新啟動 RAFTcorr 以讓所有介面生效。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="561"/>
        <source>Frame %1 has no Region of Interest of its own — frame 1&apos;s is used for computation. Switch to frame 1 to edit it, or import a mask to give this frame its own.</source>
        <translation>第 %1 影格沒有自己的感興趣區域 — 計算時使用第 1 影格的感興趣區域。請切換到第 1 影格編輯，或匯入遮罩為此影格單獨指定。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="572"/>
        <source>No Region of Interest to save — load images first.</source>
        <translation>沒有可儲存的感興趣區域 — 請先載入影像。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="579"/>
        <source>Region of Interest mask is empty.</source>
        <translation>感興趣區域遮罩為空。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="591"/>
        <source>Save Region of Interest Mask</source>
        <translation>儲存感興趣區域遮罩</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="593"/>
        <source>PNG Images</source>
        <translation>PNG 影像</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="602"/>
        <source>Mask saved to %1</source>
        <translation>遮罩已儲存至 %1</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="611"/>
        <source>No Region of Interest to invert — load images first.</source>
        <translation>沒有可反轉的感興趣區域 — 請先載入影像。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="634"/>
        <source>Load images first.</source>
        <translation>請先載入影像。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="670"/>
        <source>  Imported mask for frame %1</source>
        <translation>  已匯入影格 %1 的遮罩</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/app.py" line="676"/>
        <source>Batch import: %n mask(s) loaded</source>
        <translation>
            <numerusform>批次匯入：已載入 %n 個遮罩</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/app.py" line="703"/>
        <source>Imported Region of Interest for %n frame(s)</source>
        <translation>
            <numerusform>為 %n 個影格匯入了感興趣區域</numerusform>
        </translation>
    </message>
</context>
<context>
    <name>ParamPanel</name>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="65"/>
        <source>Iterations</source>
        <translation>迭代次數</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="66"/>
        <source>Refinement iterations of the network.
The paper evaluates every model with 20.</source>
        <translation>網路的迭代細化次數。
論文中所有模型皆以 20 次進行評估。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="73"/>
        <source>off (raw)</source>
        <translation>關閉（原始）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="75"/>
        <source>Smoothing σ</source>
        <translation>平滑 σ</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="76"/>
        <source>Gaussian smoothing of the displacement (standard deviation).
Off by default, as in the paper: smoothing lengthens the
spatial resolution.</source>
        <translation>位移的高斯平滑（標準差）。
預設關閉，與論文一致：平滑會降低
空間解析度。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="81"/>
        <source>Automatic</source>
        <translation>自動</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="85"/>
        <source>Device</source>
        <translation>裝置</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="86"/>
        <source>Automatic uses the GPU when PyTorch sees one.</source>
        <translation>選擇「自動」時，只要 PyTorch 偵測到 GPU 就使用 GPU。</translation>
    </message>
</context>
<context>
    <name>PhysicalUnitsWidget</name>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="57"/>
        <source>Use physical units</source>
        <translation>使用物理單位</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="68"/>
        <source>Physical size of one image pixel</source>
        <translation>單個影像像素對應的物理尺寸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="83"/>
        <source>/ px</source>
        <translation>/ px</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="86"/>
        <source>Pixel size</source>
        <translation>像素尺寸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="96"/>
        <source>Acquisition frame rate (used for velocity field)</source>
        <translation>採集影格率（用於速度場）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="97"/>
        <source>Frame rate</source>
        <translation>影格率</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="165"/>
        <source>Disp: %1  Velocity: %2/s</source>
        <translation>位移：%1  速度：%2/s</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="170"/>
        <source>Disp: px  Velocity: px/fr</source>
        <translation>位移：px  速度：px/影格</translation>
    </message>
</context>
<context>
    <name>PipelineController</name>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="77"/>
        <source>Stopped.</source>
        <extracomment>Results larger than this are kept in memory-mapped files in a temporary folder.</extracomment>
        <translation>已停止。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="84"/>
        <source>The GPU ran out of memory. Close other GPU programs, choose a model with a larger s, or lower the memory fraction (Advanced).</source>
        <translation>GPU 記憶體不足。請關閉其他使用 GPU 的程式，選擇 s 較大的模型，或調低「進階」中的 GPU 記憶體比例。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="86"/>
        <source>Analysis failed</source>
        <translation>分析失敗</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="164"/>
        <source>Reference frames without their own ROI carry the ROI of frame 1 forward: %1</source>
        <translation>沒有自己 ROI 的參考影格沿用第 1 影格的 ROI：%1</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="183"/>
        <source>Starting...</source>
        <translation>開始中…</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="209"/>
        <source>The images changed: the running analysis is cancelled.</source>
        <translation>影像已變更：正在執行的分析已取消。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="233"/>
        <source>The images changed during the run: its results are discarded.</source>
        <translation>執行期間影像發生了變化：本次結果已捨棄。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="246"/>
        <source>No frame was computed; the previous results are kept.</source>
        <translation>沒有計算任何影格；保留先前的結果。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="260"/>
        <source>Tracking quality</source>
        <translation>追蹤品質</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="261"/>
        <source>Part of the sequence may not be tracked correctly.</source>
        <translation>序列中的部分內容可能未被正確追蹤。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="263"/>
        <source>Show the &apos;Residual&apos; field to see where the images do not match after the displacement is applied.</source>
        <translation>顯示「殘差」場，即可查看套用位移後影像在哪些位置不相符。</translation>
    </message>
</context>
<context>
    <name>PipelineMessages</name>
    <message numerus="yes">
        <location filename="../../gui/controllers/pipeline_messages.py" line="73"/>
        <source>  %n frame(s) with custom ROI masks</source>
        <translation>
            <numerusform>  %n 個影格使用自訂 ROI 遮罩</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/pipeline_messages.py" line="77"/>
        <source>Results received: %n frame(s)</source>
        <translation>
            <numerusform>已收到結果：%n 個影格</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/pipeline_messages.py" line="81"/>
        <source>  %n image(s), %1</source>
        <translation>
            <numerusform>  %n 張影像，%1</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="84"/>
        <source>  ROI range: %1</source>
        <translation>  ROI 範圍：%1</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="87"/>
        <source>  Refinement: %1</source>
        <translation>  網格加密：%1</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="91"/>
        <source>Error: %1</source>
        <translation>錯誤：%1</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="95"/>
        <source>Run canceled. Give the listed reference frames a Region of Interest of their own, or run again and continue with frame 1&apos;s region.</source>
        <translation>已取消執行。請為列出的參考影格指定各自的感興趣區域，或重新執行並沿用第 1 影格的區域繼續。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="102"/>
        <source>Cannot start analysis</source>
        <translation>無法開始分析</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="106"/>
        <source>Need at least 2 images. Load an image folder in the %1 section before running DIC.</source>
        <translation>至少需要 2 張影像。請先在「%1」區段載入影像資料夾，再執行 DIC。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="111"/>
        <source>Define a Region of Interest first: draw or import one for frame 1 in the %1 section.</source>
        <translation>請先定義感興趣區域：在「%1」區段為第 1 影格繪製或匯入一個。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="116"/>
        <source>The Region of Interest of frame 1 is empty (no pixels selected). Draw it again in the %1 section.</source>
        <translation>第 1 影格的感興趣區域為空（未選取任何像素）。請在「%1」區段重新繪製。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="121"/>
        <source>Region of Interest too small</source>
        <translation>感興趣區域過小</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="127"/>
        <source>The Region of Interest is %1×%2 px, but the current %3 (%4) needs at least %5×%5 px. Enlarge the Region of Interest or reduce %3.</source>
        <translation>感興趣區域為 %1×%2 px，但目前的%3（%4）至少需要 %5×%5 px。請擴大感興趣區域或減小%3。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="133"/>
        <source>Starting Points required</source>
        <translation>需要種子點</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="137"/>
        <source>Place at least one Starting Point on the canvas before running: click “%1” or “%2” in the %3 section.</source>
        <translation>執行前請在畫布上放置至少一個種子點：點擊「%3」區段中的「%1」或「%2」。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="143"/>
        <source>Invalid Starting Points</source>
        <translation>種子點無效</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="147"/>
        <source>One or more Starting Points do not lie in any part of the Region of Interest. Click “%1” in the %2 section, then place the Starting Points again.</source>
        <translation>有一個或多個種子點不在感興趣區域的任何部分內。請點擊「%2」區段中的「%1」，然後重新放置種子點。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="155"/>
        <source>DIC analysis failed</source>
        <translation>DIC 分析失敗</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="158"/>
        <source>Failed to start analysis</source>
        <translation>啟動分析失敗</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="161"/>
        <source>Failed to process results</source>
        <translation>處理結果失敗</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="166"/>
        <source>The analysis stopped with a runtime error:

%1

See the console log for the full traceback.</source>
        <translation>分析因執行階段錯誤而停止：

%1

完整的回溯資訊請參閱主控台記錄。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="172"/>
        <source>%1

See the console log for the full traceback.</source>
        <translation>%1

完整的回溯資訊請參閱主控台記錄。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="177"/>
        <source>Incomplete Region of Interest Coverage</source>
        <translation>感興趣區域涵蓋不完整</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/pipeline_messages.py" line="182"/>
        <source>This incremental run has %n reference frame(s) with no Region of Interest of their own: frame(s) %1.</source>
        <translation>
            <numerusform>本次增量追蹤執行中有 %n 個參考影格沒有自己的感興趣區域：影格 %1。</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/pipeline_messages.py" line="188"/>
        <source>%1 and %n more</source>
        <comment>List of frame numbers cut short: %1 = the frame numbers shown, %n = how many more frames there are</comment>
        <translation>
            <numerusform>%1，以及另外 %n 個影格</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="194"/>
        <source>A reference frame without a Region of Interest of its own reuses frame 1&apos;s region unchanged: nothing moves the region with the specimen. Where the specimen has moved out of that region, the part outside it is not measured from such a frame, and the nodes there stop being tracked or have their displacement copied from the nearest measured node. The run reports these nodes when it finishes.</source>
        <translation>沒有自己感興趣區域的參考影格會原樣沿用第 1 影格的區域：該區域不會隨試樣移動。試樣移出該區域後，區域外的部分不會以這類參考影格為基準進行量測，位於其中的節點將不再被追蹤，或其位移從最近的已量測節點複製而來。執行結束時會報告這些節點。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="204"/>
        <source>To measure the whole specimen, give each reference frame its own Region of Interest that covers the specimen in that frame (Batch Import can assign Regions of Interest to many frames at once). A reference frame with a region of its own is measured on a grid that covers that region.</source>
        <translation>要量測整個試樣，請為每個參考影格單獨指定一個涵蓋該影格中試樣的感興趣區域（批次匯入可一次為多個影格指定感興趣區域）。擁有自己區域的參考影格會在涵蓋該區域的量測網格上進行量測。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="211"/>
        <source>Continue with frame 1&apos;s region?</source>
        <translation>是否沿用第 1 影格的區域繼續？</translation>
    </message>
</context>
<context>
    <name>ROIHint</name>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="63"/>
        <source>Load images first, then draw a Region of Interest on frame 1.</source>
        <translation>請先載入影像，再在第 1 影格上繪製感興趣區域。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="70"/>
        <source>&lt;b&gt;Accumulative mode&lt;/b&gt; — only frame 1 needs a Region of Interest. All later frames are compared against it directly.</source>
        <translation>&lt;b&gt;累積模式&lt;/b&gt; — 只有第 1 影格需要感興趣區域。後續影格都直接與其比較。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="83"/>
        <source>&lt;b&gt;Incremental&lt;/b&gt; — only frame 1 needs a Region of Interest. RAFTcorr carries it to the reference frames (&lt;b&gt;%1&lt;/b&gt;) with the measured displacement. Draw on another frame only to replace that region or to leave out cracks and holes.</source>
        <translation>&lt;b&gt;增量模式&lt;/b&gt; — 只有第 1 影格需要感興趣區域。RAFTcorr 會依測得的位移將其帶到各參考影格（&lt;b&gt;%1&lt;/b&gt;）。僅在需要取代該影格的區域、或排除裂紋與孔洞時，才在其他影格上繪製。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="92"/>
        <source>Draw a Region of Interest on frame 1.</source>
        <translation>請在第 1 影格繪製感興趣區域。</translation>
    </message>
</context>
<context>
    <name>ROIToolbar</name>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="72"/>
        <source>+ Add</source>
        <translation>+ 新增</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="74"/>
        <source>Add region to the Region of Interest (Polygon / Rectangle / Circle)</source>
        <translation>向感興趣區域新增形狀（多邊形 / 矩形 / 圓形）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="79"/>
        <source>Cut</source>
        <translation>裁剪</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="81"/>
        <source>Cut region from the Region of Interest (Polygon / Rectangle / Circle)</source>
        <translation>從感興趣區域裁剪形狀（多邊形 / 矩形 / 圓形）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="86"/>
        <source>+ Refine</source>
        <translation>+ 加密</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="90"/>
        <source>Paint extra mesh-refinement zones with a brush
(only on frame 1 — material points auto-warped to later frames)</source>
        <translation>用畫筆繪製額外的網格加密區域
（僅在第 1 影格可用 — 網格點會自動扭曲到後續影格）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="94"/>
        <source>Refine brush is only available on frame 1. Switch to frame 1 to paint refinement zones; they are automatically warped to later frames.</source>
        <translation>加密畫筆僅在第 1 影格可用。切換到第 1 影格後可繪製加密區域；系統會自動將其扭曲到後續影格。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="118"/>
        <source>Import</source>
        <translation>匯入</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="119"/>
        <source>Import mask from image file</source>
        <translation>從影像檔案匯入遮罩</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="124"/>
        <source>Batch Import</source>
        <translation>批次匯入</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="126"/>
        <source>Batch import mask files for multiple frames</source>
        <translation>批次匯入多個影格的遮罩檔案</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="139"/>
        <source>Save</source>
        <translation>儲存</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="140"/>
        <source>Save current mask to PNG file</source>
        <translation>將目前遮罩儲存為 PNG 檔案</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="145"/>
        <source>Invert</source>
        <translation>反轉</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="147"/>
        <source>Invert the Region of Interest mask</source>
        <translation>反轉感興趣區域遮罩</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="152"/>
        <source>Clear</source>
        <translation>清除</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="154"/>
        <source>Clear all Region of Interest masks</source>
        <translation>清除所有感興趣區域遮罩</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="186"/>
        <source>Radius</source>
        <translation>半徑</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="204"/>
        <source>Paint</source>
        <translation>繪製</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="209"/>
        <source>Erase</source>
        <translation>擦除</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="219"/>
        <source>Clear Brush</source>
        <translation>清除畫筆</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="247"/>
        <source>Polygon</source>
        <translation>多邊形</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="251"/>
        <source>Rectangle</source>
        <translation>矩形</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="255"/>
        <source>Circle</source>
        <translation>圓形</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="259"/>
        <source>Circle (3-point)</source>
        <translation>圓形（三點）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="352"/>
        <source>Import Mask Image</source>
        <translation>匯入遮罩影像</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="354"/>
        <source>Images</source>
        <translation>影像</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="355"/>
        <source>All Files</source>
        <translation>所有檔案</translation>
    </message>
</context>
<context>
    <name>RightSidebar</name>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="75"/>
        <source>Run DIC Analysis</source>
        <translation>執行 DIC 分析</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="88"/>
        <source>Cancel</source>
        <translation>取消</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="93"/>
        <source>Cancel the current analysis. Frames already computed are kept so you can review or export the partial run.</source>
        <translation>取消目前的分析。已計算的影格會被保留，你可以檢視或匯出這部分結果。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="101"/>
        <source>Export Results</source>
        <translation>匯出結果</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="109"/>
        <source>Open Strain Window</source>
        <translation>開啟應變視窗</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="112"/>
        <source>Compute and visualize strain in a separate post-processing window. Requires displacement results from a completed Run.</source>
        <translation>在獨立的後處理視窗中計算並視覺化應變。需先完成一次執行以獲得位移結果。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="121"/>
        <source>PROGRESS</source>
        <translation>進度</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="130"/>
        <location filename="../../gui/panels/right_sidebar.py" line="430"/>
        <source>Ready</source>
        <translation>就緒</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="138"/>
        <location filename="../../gui/panels/right_sidebar.py" line="432"/>
        <location filename="../../gui/panels/right_sidebar.py" line="532"/>
        <source>ELAPSED  %1</source>
        <translation>已用時間  %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="143"/>
        <location filename="../../gui/panels/right_sidebar.py" line="434"/>
        <location filename="../../gui/panels/right_sidebar.py" line="540"/>
        <location filename="../../gui/panels/right_sidebar.py" line="544"/>
        <source>REMAINING  %1</source>
        <translation>剩餘時間  %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="151"/>
        <source>FIELD</source>
        <translation>場變數</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="160"/>
        <source>Show on</source>
        <translation>顯示於</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="165"/>
        <source>Deformed frame</source>
        <translation>變形影格</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="166"/>
        <source>Reference frame</source>
        <translation>參考影格</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="168"/>
        <source>Plot the field at the deformed node positions, or at their positions in the reference frame.</source>
        <translation>將場繪製在變形後的節點位置，或繪製在其於參考影格中的位置。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="176"/>
        <source>Show background image</source>
        <translation>顯示背景影像</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="179"/>
        <source>Uncheck to show the field on its own, with no speckle image behind it.</source>
        <translation>取消勾選可僅顯示場本身，其後不顯示散斑影像。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="190"/>
        <source>Hidden background</source>
        <translation>隱藏背景填充</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="194"/>
        <source>White</source>
        <translation>白色</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="195"/>
        <source>Black</source>
        <translation>黑色</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="196"/>
        <source>Transparent</source>
        <translation>透明</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="199"/>
        <source>What replaces the image when it is hidden. Transparency is kept for PNG and TIFF on export; other formats get white.</source>
        <translation>背景隱藏時用什麼替代影像。匯出時 PNG 與 TIFF 會保留透明度，其他格式改用白色。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="208"/>
        <source>Hide untracked regions</source>
        <translation>隱藏未追蹤區域</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="211"/>
        <source>Leave areas without a tracked node transparent, for example material that left the field of view, instead of filling them with values copied from distant nodes. Also used for exported images and animations; data files always keep NaN there.</source>
        <translation>讓沒有已追蹤節點的區域（例如已移出視野的材料）保持透明，而不是以從遠處節點複製的值填滿。匯出的影像與動畫也採用此設定；資料檔中這些區域一律為 NaN。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="221"/>
        <source>VISUALIZATION</source>
        <translation>視覺化</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="226"/>
        <source>Colormap</source>
        <translation>色彩對映</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="245"/>
        <source>Opacity</source>
        <translation>不透明度</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="252"/>
        <source>Overlay opacity (0 = transparent, 100 = opaque)</source>
        <translation>疊加圖不透明度（0 = 透明，100 = 不透明）</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="258"/>
        <source>PHYSICAL UNITS</source>
        <translation>物理單位</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="264"/>
        <source>LOG</source>
        <translation>日誌</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="270"/>
        <source>Clear</source>
        <translation>清除</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="399"/>
        <source>Place at least one Starting Point in each red region before running (red = needs a Starting Point).</source>
        <translation>執行前請在每個紅色區域放置至少一個種子點（紅色 = 需要種子點）。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="515"/>
        <source>%1  —  Frame %2</source>
        <translation>%1  —  第 %2 影格</translation>
    </message>
</context>
<context>
    <name>StrainController</name>
    <message>
        <location filename="../../gui/controllers/strain_controller.py" line="96"/>
        <source>Strain frame %1/%2</source>
        <translation>應變：影格 %1/%2</translation>
    </message>
</context>
<context>
    <name>StrainFieldSelector</name>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="78"/>
        <source>Magnitude</source>
        <extracomment>GUI field name -&gt; FieldProvider key.</extracomment>
        <translation>位移大小</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="79"/>
        <source>Velocity</source>
        <translation>速度</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="200"/>
        <source>DISPLACEMENT</source>
        <translation>位移</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="211"/>
        <source>STRAIN</source>
        <translation>應變</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="265"/>
        <source>Horizontal displacement u (+ right)</source>
        <translation>水平位移 u（向右為正）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="266"/>
        <source>Vertical displacement v (+ down)</source>
        <translation>垂直位移 v（向下為正）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="267"/>
        <source>Displacement magnitude |u|</source>
        <translation>位移大小 |u|</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="269"/>
        <source>Speed |du/dt|: central difference of the displacement over the neighbouring computed frames, times the frame rate</source>
        <translation>速度大小 |du/dt|：位移在相鄰已計算影格之間的中心差分，乘以影格率</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="271"/>
        <source>Normal strain εxx</source>
        <translation>正應變 εxx</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="272"/>
        <source>Normal strain εyy</source>
        <translation>正應變 εyy</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="274"/>
        <source>Shear strain εxy (tensor component: half the engineering shear γxy)</source>
        <translation>剪應變 εxy（張量分量：工程剪應變 γxy 的一半）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="276"/>
        <source>Major principal strain ε₁</source>
        <translation>最大主應變 ε₁</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="277"/>
        <source>Minor principal strain ε₂</source>
        <translation>最小主應變 ε₂</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="279"/>
        <source>Maximum in-plane shear strain (ε₁ − ε₂) / 2 (tensor component: half the engineering shear)</source>
        <translation>最大面內剪應變 (ε₁ − ε₂) / 2（張量分量：工程剪應變的一半）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="282"/>
        <source>Von Mises equivalent strain (incompressible material)</source>
        <translation>von Mises 等效應變（不可壓縮材料）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="284"/>
        <source>Rigid rotation in degrees (polar decomposition), clockwise on screen positive</source>
        <translation>剛體轉動角，單位為度（極分解），螢幕上順時針為正</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="287"/>
        <source>Direction of ε₁ in degrees from +x, clockwise on screen positive</source>
        <translation>ε₁ 的方向，自 +x 軸起算，單位為度，螢幕上順時針為正</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="290"/>
        <source>Relative area change ΔA/A₀ = det F − 1</source>
        <translation>相對面積變化 ΔA/A₀ = det F − 1</translation>
    </message>
</context>
<context>
    <name>StrainNavigator</name>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="73"/>
        <source>Previous frame</source>
        <translation>上一影格</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="84"/>
        <location filename="../../gui/widgets/strain_navigator.py" line="216"/>
        <source>Play animation</source>
        <translation>播放動畫</translation>
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
        <translation>下一影格</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="109"/>
        <source>Playback speed</source>
        <translation>播放速度</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="114"/>
        <location filename="../../gui/widgets/strain_navigator.py" line="228"/>
        <source>FRAME 0/0</source>
        <translation>影格 0/0</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="205"/>
        <source>⏸</source>
        <translation>⏸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="206"/>
        <source>Pause animation</source>
        <translation>暫停動畫</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="223"/>
        <source>FRAME %1/%2</source>
        <translation>影格 %1/%2</translation>
    </message>
</context>
<context>
    <name>StrainParamPanel</name>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="86"/>
        <source>Plane fitting</source>
        <extracomment>Strain measures in combo order (pyALDIC&apos;s order: infinitesimal, Eulerian, Green-Lagrangian). The configuration in the controls changed (by the user or Paper settings).</extracomment>
        <translation>平面擬合</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="95"/>
        <source>Method</source>
        <translation>方法</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="110"/>
        <source>VSG size</source>
        <translation>VSG 尺寸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="152"/>
        <source>Off</source>
        <comment>Strain smoothing preset</comment>
        <translation>關閉</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="87"/>
        <source>Quadratic fitting</source>
        <translation>二次擬合</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="89"/>
        <source>Polynomial fitted to each displacement component over the VSG window. Its slopes at the centre are the displacement gradients.
• Plane fitting (1st order): the standard VSG (paper setting).
• Quadratic fitting (2nd order): follows strain gradients inside the window better, at the cost of more noise.</source>
        <translation>在 VSG 視窗內對每個位移分量擬合多項式，其在中心處的斜率即為位移梯度。
• 平面擬合（一階）：標準 VSG（論文設定）。
• 二次擬合（二階）：能更好地跟隨視窗內的應變梯度，但雜訊較大。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="103"/>
        <source>VSG (Virtual Strain Gauge) size Lg is the side, in pixels, of the square window around each point over which the displacement is fitted. Strain is then taken from the fit&apos;s slopes.

• Larger VSG → smoother strain, lower spatial resolution.
• Smaller VSG → sharper strain, more noise.
• Odd, so that the window has a centre pixel (default: 41 px).</source>
        <translation>VSG（虛擬應變計）尺寸 Lg 是每個點周圍用於擬合位移的正方形視窗的邊長（像素）。應變由擬合的斜率求得。

• VSG 越大 → 應變越平滑，空間解析度越低。
• VSG 越小 → 應變越銳利，雜訊越多。
• 取奇數，使視窗有中心像素（預設：41 px）。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="114"/>
        <source>Uniform</source>
        <translation>均勻</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="115"/>
        <source>Gaussian</source>
        <translation>高斯</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="117"/>
        <source>Weight of the points inside the VSG window: all equal (uniform, paper setting), or decreasing with the distance from the centre (Gaussian).</source>
        <translation>VSG 視窗內各點的權重：全部相等（均勻，論文設定），或隨離中心的距離遞減（高斯）。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="121"/>
        <source>Weighting</source>
        <translation>加權</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="129"/>
        <source>Auto (Lg / 4)</source>
        <translation>自動（Lg / 4）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="131"/>
        <source>Standard deviation of the Gaussian weight. Auto uses a quarter of the VSG size.</source>
        <translation>高斯權重的標準差。「自動」取 VSG 尺寸的四分之一。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="134"/>
        <source>Gaussian σ</source>
        <translation>高斯 σ</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="141"/>
        <source>Spacing of the strain points: 1 computes strain at every pixel; larger values are faster and lighter for long sequences.</source>
        <translation>應變點的間距：1 表示在每個像素上計算應變；較大的值對長序列更快、占用更少。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="144"/>
        <source>Output spacing</source>
        <translation>輸出間距</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="154"/>
        <source>Gaussian smoothing (σ in pixels) of the displacement before the fit. Off (0) differentiates the measured displacement as it is (paper setting).</source>
        <translation>擬合前對位移進行高斯平滑（σ 以像素計）。關閉（0）時直接對測得的位移求導（論文設定）。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="158"/>
        <source>Displacement smoothing</source>
        <translation>位移平滑</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="161"/>
        <source>Never fit across cracks or holes</source>
        <translation>擬合不跨越裂紋或孔洞</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="163"/>
        <source>Fit each connected part of the tracked region separately, so no VSG window ever spans a crack, a hole or the gap between two regions.</source>
        <translation>對追蹤區域的每個連通部分分別擬合，使任何 VSG 視窗都不會跨越裂紋、孔洞或兩個區域之間的間隙。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="172"/>
        <source>Infinitesimal</source>
        <translation>無窮小應變</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="173"/>
        <source>Eulerian</source>
        <translation>歐拉應變</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="174"/>
        <source>Green-Lagrangian</source>
        <translation>格林-拉格朗日應變</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="176"/>
        <source>Strain measure from the deformation gradient F = I + ∇u:
• Infinitesimal: (∇u + ∇uᵀ) / 2 (small strains only).
• Eulerian (Euler-Almansi): (I − (F Fᵀ)⁻¹) / 2.
• Green-Lagrangian: (Fᵀ F − I) / 2 (paper setting).</source>
        <translation>由變形梯度 F = I + ∇u 得到的應變度量：
• 無窮小應變：(∇u + ∇uᵀ) / 2（僅適用於小應變）。
• 歐拉應變（Euler-Almansi）：(I − (F Fᵀ)⁻¹) / 2。
• 格林-拉格朗日應變：(Fᵀ F − I) / 2（論文設定）。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="181"/>
        <source>Strain type</source>
        <translation>應變類型</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="184"/>
        <source>Paper settings</source>
        <translation>論文設定</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="186"/>
        <source>Green-Lagrangian strain, unweighted plane fit, VSG 41 px, every pixel, no smoothing, never across cracks (the settings of the RAFTcorr paper).</source>
        <translation>格林-拉格朗日應變、無加權平面擬合、VSG 41 px、逐像素、不平滑、不跨越裂紋（RAFTcorr 論文中的設定）。</translation>
    </message>
</context>
<context>
    <name>StrainVizPanel</name>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="67"/>
        <source>Deformed frame</source>
        <extracomment>Alias so existing references read unchanged; the list itself lives in raftcorr.gui.support.colormaps, so every chooser offers the same options.</extracomment>
        <translation>變形影格</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="68"/>
        <source>Reference frame</source>
        <translation>參考影格</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="70"/>
        <source>Plot the field where the material points are in the current frame, or at their positions in the reference frame.</source>
        <translation>將場繪製在材料點於目前影格中的位置，或繪製在其於參考影格中的位置。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="73"/>
        <source>Show on</source>
        <translation>顯示於</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="75"/>
        <source>Show background image</source>
        <translation>顯示背景影像</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="78"/>
        <source>Uncheck to show the field on its own, with no speckle image behind it.</source>
        <translation>取消勾選可僅顯示場本身，其後不顯示散斑影像。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="84"/>
        <source>White</source>
        <translation>白色</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="85"/>
        <source>Black</source>
        <translation>黑色</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="86"/>
        <source>Transparent</source>
        <translation>透明</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="89"/>
        <source>What replaces the image when it is hidden. Transparency is kept for PNG and TIFF on export; other formats get white.</source>
        <translation>背景隱藏時用什麼替代影像。匯出時 PNG 與 TIFF 會保留透明度，其他格式改用白色。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="139"/>
        <source>Hide untracked regions</source>
        <translation>隱藏未追蹤區域</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="142"/>
        <source>Leave points without a result transparent, for example material that left the field of view, instead of filling them with the nearest value inside the region of interest (reference frame view). Data files always keep NaN there.</source>
        <translation>讓沒有結果的點（例如已移出視野的材料）保持透明，而不是以感興趣區域內最近的值填滿（參考影格檢視）。資料檔中這些點一律為 NaN。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="81"/>
        <source>Background</source>
        <translation>背景</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="93"/>
        <source>Hidden background</source>
        <translation>隱藏背景填充</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="100"/>
        <source>Colormap</source>
        <translation>色彩對映</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="104"/>
        <source>Range</source>
        <translation>範圍</translation>
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
        <location filename="../../gui/widgets/strain_viz_panel.py" line="147"/>
        <source>Edges</source>
        <translation>邊緣</translation>
    </message>
</context>
<context>
    <name>StrainWindow</name>
    <message>
        <location filename="../../gui/strain_window.py" line="125"/>
        <source>Strain Post-Processing</source>
        <translation>應變後處理</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="210"/>
        <source>Fit</source>
        <translation>適配</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="211"/>
        <source>Fit image to viewport</source>
        <translation>將影像適配到檢視區</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="217"/>
        <source>100%</source>
        <translation>100%</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="218"/>
        <source>Zoom to 100% (1:1)</source>
        <translation>縮放到 100%（1:1）</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="221"/>
        <source>Zoom in</source>
        <translation>放大</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="226"/>
        <source>–</source>
        <translation>–</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="284"/>
        <source>STRAIN PARAMETERS</source>
        <translation>應變參數</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="303"/>
        <source>Cancel</source>
        <translation>取消</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="307"/>
        <source>Cancel the running strain computation. The previous strain result is kept.</source>
        <translation>取消正在進行的應變計算。保留先前的應變結果。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="319"/>
        <source>Export displacement and strain results to NPZ / MAT / CSV / PNG</source>
        <translation>將位移和應變結果匯出為 NPZ / MAT / CSV / PNG</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="349"/>
        <source>FIELD</source>
        <translation>場變數</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="356"/>
        <source>VISUALIZATION</source>
        <translation>視覺化</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="368"/>
        <source>PHYSICAL UNITS</source>
        <translation>物理單位</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="374"/>
        <source>LOG</source>
        <translation>日誌</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="855"/>
        <location filename="../../gui/strain_window.py" line="891"/>
        <source>No displacement results.</source>
        <translation>沒有位移結果。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="893"/>
        <source>Frame %1 was not computed.</source>
        <translation>第 %1 影格未計算。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="901"/>
        <source>Click Compute Strain to show this field.</source>
        <translation>點擊「計算應變」以顯示此場。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="902"/>
        <source>No strain for frame %1: click Compute Strain.</source>
        <translation>第 %1 影格沒有應變：請點擊「計算應變」。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="904"/>
        <source>Velocity needs at least two computed frames.</source>
        <translation>速度至少需要兩個已計算影格。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="905"/>
        <source>No valid values in frame %1.</source>
        <translation>第 %1 影格沒有有效值。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="170"/>
        <source>Strain Field</source>
        <translation>應變場</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="414"/>
        <source>Analysis is not available</source>
        <translation>分析功能無法使用</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="417"/>
        <location filename="../../gui/strain_window.py" line="420"/>
        <source>Analysis</source>
        <translation>分析</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="418"/>
        <source>Analysis is not available (see the application log).</source>
        <translation>分析功能無法使用（詳見應用程式日誌）。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="529"/>
        <source>Strain compute failed: %1: %2</source>
        <translation>應變計算失敗：%1：%2</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="535"/>
        <location filename="../../gui/strain_window.py" line="598"/>
        <source>Strain computation complete.</source>
        <translation>應變計算完成。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="546"/>
        <source>Strain window: no displacement results to post-process.</source>
        <translation>應變視窗：沒有可後處理的位移結果。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="591"/>
        <source>The displacement results changed during the strain computation; its result was discarded.</source>
        <translation>應變計算期間位移結果發生了變化；本次應變結果已捨棄。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="615"/>
        <source>Strain computation canceled.</source>
        <translation>應變計算已取消。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="622"/>
        <source>Strain compute failed: %1</source>
        <translation>應變計算失敗：%1</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="626"/>
        <source>Strain Computation Failed</source>
        <translation>應變計算失敗</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="665"/>
        <location filename="../../gui/strain_window.py" line="667"/>
        <source>Export is not available: %1</source>
        <translation>匯出功能無法使用：%1</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="679"/>
        <source>⚠ Params changed -- click Compute Strain</source>
        <translation>⚠ 參數已變更 — 請點擊“計算應變”</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="227"/>
        <source>Zoom out</source>
        <translation>縮小</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="294"/>
        <source>Compute Strain</source>
        <translation>計算應變</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="316"/>
        <location filename="../../gui/strain_window.py" line="666"/>
        <source>Export Results</source>
        <translation>匯出結果</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="558"/>
        <source>Starting…</source>
        <translation>開始中…</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="576"/>
        <source>Canceling…</source>
        <translation>正在取消…</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="595"/>
        <source>Complete</source>
        <translation>已完成</translation>
    </message>
</context>
<context>
    <name>VelocitySettingsWidget</name>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="47"/>
        <source>Use physical units</source>
        <translation>使用物理單位</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="69"/>
        <source>/ px</source>
        <translation>/ px</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="84"/>
        <source>Unit: px/frame</source>
        <translation>單位：px/影格</translation>
    </message>
</context>
<context>
    <name>WorkflowTypePanel</name>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="56"/>
        <source>Incremental</source>
        <translation>增量式</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="55"/>
        <source>Accumulative</source>
        <translation>累積式</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="70"/>
        <source>Tracking Mode</source>
        <translation>追蹤模式</translation>
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
        <translation>累積式：每個影格都直接與第 1 影格比較。
只要運動不超出模型量程，就優先使用：
誤差不會逐影格累積。

增量式：每個影格與第 1 影格之後的某個參考影格比較，
再將各步結果串接。當相對第 1 影格的運動
超出模型量程（大拉伸或大轉動）時需要使用；每一步
都會引入自己的誤差。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="82"/>
        <source>finest spatial resolution; small displacements</source>
        <translation>空間解析度最高；適合小位移</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="83"/>
        <source>fine resolution, several times faster than s1; small to moderate displacements</source>
        <translation>解析度高，速度是 s1 的數倍；適合小到中等位移</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="84"/>
        <source>balanced resolution and capture range</source>
        <translation>解析度與量程均衡</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="85"/>
        <source>largest capture range, coarsest resolution; large motion and rotation</source>
        <translation>量程最大，解析度最粗；適合大運動與大轉動</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="89"/>
        <source>%1: %2.
Reach (nominal lookup radius 32 s): %3 px.</source>
        <translation>%1：%2。
量程（名義查找半徑 32 s）：%3 px。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="92"/>
        <source>Weights not installed.</source>
        <translation>權重檔案未安裝。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="97"/>
        <source>RAFTcorr model. A smaller s resolves finer detail; a larger s
reaches further (the coarsest correlation level looks 32 s px away).

s1, s2: small displacements, strain concentrations.
s4: balanced.
s8: large motion or rotation, or no prior knowledge.</source>
        <translation>RAFTcorr 模型。s 越小，可分辨的細節越精細；s 越大，
量程越大（最粗相關層級的查找距離為 32 s px）。

s1、s2：小位移、應變集中。
s4：均衡。
s8：大運動或大轉動，或無先驗資訊。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="104"/>
        <source>Model</source>
        <translation>模型</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="110"/>
        <source>Suggest a model</source>
        <translation>建議模型</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="112"/>
        <source>Measure the motion of the sequence quickly with s8 and pick the
finest model whose reach covers it.</source>
        <translation>先以 s8 快速測量序列的運動，再選出量程足以
涵蓋該運動的最精細模型。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="134"/>
        <source>Every Frame</source>
        <translation>每個影格</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="135"/>
        <source>Every N Frames</source>
        <translation>每 N 個影格</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="136"/>
        <source>Custom Frames</source>
        <translation>自訂影格</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="139"/>
        <source>When the reference frame refreshes during incremental tracking.
Every Frame: reset reference every frame (smallest per-step displacement,
most robust for large deformation).
Every N Frames: reset every N frames (balance speed vs robustness).
Custom Frames: user-defined list of reference frame indices.</source>
        <translation>增量追蹤中參考影格的更新策略。
每個影格：每個影格都更新參考（單步位移最小，
對大變形最穩健）。
每 N 個影格：每 N 個影格更新一次（速度與穩健性的折中）。
自訂影格：由使用者指定參考影格索引清單。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="148"/>
        <source>Reference Update</source>
        <translation>參考影格更新</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="158"/>
        <source>Update reference every N frames</source>
        <translation>每 N 個影格更新一次參考影格</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="160"/>
        <source>Interval</source>
        <translation>間隔</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="169"/>
        <source>Comma-separated frame indices to use as reference frames (0-based)</source>
        <translation>用作參考影格的影格索引清單（0 為起始），用逗號分隔</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="173"/>
        <source>Reference Frames</source>
        <translation>參考影格清單</translation>
    </message>
</context>
<context>
    <name>_DropZone</name>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="118"/>
        <source>Drop image folder
or Browse</source>
        <translation>拖入影像資料夾
或點擊瀏覽</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="128"/>
        <source>Select Image Folder</source>
        <translation>選擇影像資料夾</translation>
    </message>
</context>
<context>
    <name>_MaskPreviewPanel</name>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="133"/>
        <source>Preview</source>
        <translation>預覽</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="137"/>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="263"/>
        <source>(no image)</source>
        <translation>（無影像）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="157"/>
        <source>Image only</source>
        <translation>僅影像</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="158"/>
        <source>Image + Mask</source>
        <translation>影像 + 遮罩</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="159"/>
        <source>Mask only</source>
        <translation>僅遮罩</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="162"/>
        <source>View:</source>
        <translation>檢視：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="168"/>
        <source>Alpha:</source>
        <translation>不透明度：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="176"/>
        <source>Blue</source>
        <comment>Mask overlay color</comment>
        <translation>藍色</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="180"/>
        <source>Red</source>
        <comment>Mask overlay color</comment>
        <translation>紅色</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="184"/>
        <source>Green</source>
        <comment>Mask overlay color</comment>
        <translation>綠色</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="188"/>
        <source>Yellow</source>
        <comment>Mask overlay color</comment>
        <translation>黃色</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="192"/>
        <source>Mask color:</source>
        <translation>遮罩顏色：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="269"/>
        <source>No mask assigned</source>
        <translation>未分配遮罩</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="275"/>
        <source>Frame %1 — %2</source>
        <translation>影格 %1 — %2</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="290"/>
        <source>Failed to load image</source>
        <translation>無法載入影像</translation>
    </message>
</context>
<context>
    <name>dialog</name>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="307"/>
        <source>Cancel Export</source>
        <translation>取消匯出</translation>
    </message>
</context>
</TS>
