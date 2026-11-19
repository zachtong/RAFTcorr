<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1" language="zh_CN" sourcelanguage="en_US">
<context>
    <name>AdvancedPanel</name>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="148"/>
        <source>Margin</source>
        <translation>边距</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="167"/>
        <source>Fraction of the free GPU memory one window may use.
Large regions are cut into tiles to fit it.</source>
        <translation>单个窗口可使用的空闲 GPU 显存比例。
较大的区域会被切分成多个分块以满足该限制。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="149"/>
        <source>Must cover the largest displacement of the region of interest.</source>
        <translation>必须覆盖感兴趣区域内的最大位移。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="152"/>
        <source>Use the Model Reach</source>
        <translation>使用模型量程</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="155"/>
        <source>Extra image context around the region of interest so that displaced points stay inside
the window given to the network. Automatic: the model&apos;s reach (32 s px).</source>
        <translation>在感兴趣区域周围额外保留的图像范围，使发生位移的点仍位于
送入网络的窗口内。自动：取模型量程（32 s px）。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="166"/>
        <source>GPU Memory</source>
        <translation>GPU 显存</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="174"/>
        <source>Tile Blending</source>
        <translation>分块融合</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="175"/>
        <source>Width of the feathered overlap between neighbouring tiles.</source>
        <translation>相邻分块之间羽化重叠区的宽度。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="178"/>
        <source>Automatic</source>
        <translation>自动</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="179"/>
        <source>On Demand</source>
        <translation>按需计算</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="180"/>
        <source>Stored Volume</source>
        <translation>预存相关体</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="182"/>
        <source>Correlation</source>
        <translation>相关计算</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="183"/>
        <source>How the correlation lookup is computed; the results are identical.
Automatic: stored volume for s8 (fast), on demand for s1-s4
(memory linear in the image size).</source>
        <translation>相关查找的计算方式；两种方式的结果完全相同。
自动：s8 使用预存相关体（较快），s1-s4 使用按需计算
（内存随图像尺寸线性增长）。</translation>
    </message>
</context>
<context>
    <name>AnalysisChart</name>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="75"/>
        <source>too few valid points</source>
        <translation>有效点过少</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="73"/>
        <source>partly untracked</source>
        <translation>部分未追踪</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="77"/>
        <source>no strain value (sparse strain gauge)</source>
        <translation>无应变值（VSG 内点过少）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="80"/>
        <source>gauge endpoint lost</source>
        <translation>量规端点失效</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="82"/>
        <source>not computed</source>
        <translation>未计算</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="83"/>
        <source>no data</source>
        <translation>无数据</translation>
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
        <translation>线段</translation>
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
        <translation>圆形</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="90"/>
        <source>Polygon</source>
        <comment>Placement tool</comment>
        <translation>多边形</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="102"/>
        <source>Click once to place a point probe.</source>
        <translation>点击一次放置点探针。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="108"/>
        <source>Click twice: opposite corners.</source>
        <translation>点击两次：对角两点。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="112"/>
        <source>Click each vertex, then double-click to close.</source>
        <translation>逐个点击顶点，双击闭合。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="170"/>
        <source>partly untracked from frame %1</source>
        <translation>第 %1 帧起部分未追踪</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="179"/>
        <source>gaps: no strain value</source>
        <translation>有缺口：无应变值</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="201"/>
        <source>Nothing valid along %1: the strain gauge found too few points there (near an edge or a hole). Plot a displacement, or use a smaller gauge on the Strain Field tab.</source>
        <translation>%1 沿线没有有效数据：该处靠近边缘或孔，VSG 内的点过少。请改为绘制位移，或在“应变场”页减小 VSG 尺寸。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="207"/>
        <source>Nothing valid along %1: the material under it is untracked (cut out of the frame&apos;s ROI, or out of view).</source>
        <translation>%1 沿线没有有效数据：其下方的材料未被追踪（已被该帧的 ROI 排除，或已移出视场）。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="89"/>
        <source>Show</source>
        <comment>Probe list column: visibility checkbox</comment>
        <translation>显示</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="91"/>
        <source>Name</source>
        <comment>Probe list column: the probe&apos;s label</comment>
        <translation>名称</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="93"/>
        <source>Type</source>
        <comment>Probe list column: point, line or region</comment>
        <translation>类型</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="95"/>
        <source>Color</source>
        <comment>Probe list column: color swatch</comment>
        <translation>颜色</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="100"/>
        <source>Color…</source>
        <translation>颜色…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="97"/>
        <source>Note</source>
        <comment>Probe list column: why a probe shows gaps</comment>
        <translation>说明</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="101"/>
        <source>Delete</source>
        <comment>Button: delete the selected probe</comment>
        <translation>删除</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="103"/>
        <source>Clear All</source>
        <translation>全部清除</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="183"/>
        <source>Statistic:</source>
        <translation>统计量：</translation>
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
        <translation>中位数</translation>
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
        <translation>标准差</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="62"/>
        <source>Valid fraction</source>
        <translation>有效比例</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="46"/>
        <source>Crack opening</source>
        <translation>裂纹张开位移</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="489"/>
        <source>Strain has not been computed yet. Compute it on the Strain Field tab, or plot a displacement.</source>
        <translation>尚未计算应变。请在“应变场”页中计算，或改为绘制位移。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="533"/>
        <source>Gauge quantities need a line probe.</source>
        <translation>量规类物理量需要线探针。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="530"/>
        <source>No visible probe can show this quantity.</source>
        <translation>没有可见的探针能显示此物理量。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="166"/>
        <source>no valid data: %1</source>
        <translation>无有效数据：%1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="174"/>
        <source>endpoint lost from frame %1</source>
        <translation>第 %1 帧起端点失效</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="177"/>
        <source>gaps: too few valid points</source>
        <translation>有缺口：有效点过少</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="185"/>
        <source>not plotted: gauges need a line</source>
        <translation>未绘制：量规需要线探针</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="188"/>
        <source>not plotted: one point has no spread or coverage</source>
        <translation>未绘制：单个点没有离散度或覆盖率</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="190"/>
        <source>not plotted</source>
        <translation>未绘制</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="513"/>
        <source>Added probe &apos;%1&apos;.</source>
        <translation>已添加探针「%1」。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="599"/>
        <source>Clear All Probes</source>
        <translation>清除所有探针</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="600"/>
        <source>Delete every probe? This cannot be undone.</source>
        <translation>删除所有探针？此操作无法撤销。</translation>
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
        <translation>线段</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="112"/>
        <source>Region</source>
        <comment>Probe type: an enclosed area</comment>
        <translation>区域</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="480"/>
        <source>Run a DIC analysis to plot probes.</source>
        <translation>请先运行 DIC 分析，然后才能绘制探针曲线。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="92"/>
        <source>Virtual extensometer</source>
        <comment>Placement tool</comment>
        <translation>虚拟引伸计</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="94"/>
        <source>Crack gauge</source>
        <comment>Placement tool: a line across a crack</comment>
        <translation>裂纹量规</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="104"/>
        <source>Click twice: start and end. A line is also a virtual extensometer and a crack-opening gauge.</source>
        <translation>点击两次：起点和终点。线探针同时也是虚拟引伸计和裂纹张开量规。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="110"/>
        <source>Click twice: center, then the edge.</source>
        <translation>点击两次：先圆心，后边缘。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="114"/>
        <source>Click the two gauge points. The chart then shows the strain between them.</source>
        <translation>点击两个标距点。图表随后显示两点之间的应变。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="118"/>
        <source>Click one point on each side of the crack. The chart then shows how far it opens.</source>
        <translation>在裂纹两侧各点击一个点。图表随后显示裂纹的张开量。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="124"/>
        <source>Fit</source>
        <comment>Zoom button: fit the image to the view</comment>
        <translation>适配</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="131"/>
        <source>Fit image to viewport</source>
        <translation>将图像适配到视口</translation>
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
        <translation>缩放到 100%（1:1）</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="136"/>
        <source>Zoom in</source>
        <translation>放大</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="137"/>
        <source>Zoom out</source>
        <translation>缩小</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="138"/>
        <source>Show field</source>
        <comment>Analysis canvas: color the image by the field</comment>
        <translation>显示场</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="140"/>
        <source>Color the reference image with the plotted field at the current frame. For a gauge reading, the Strain Field tab&apos;s field is shown.</source>
        <translation>用当前帧所绘物理量的场为参考图像着色。对于量规读数，显示“应变场”页中的场。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="181"/>
        <source>Plot:</source>
        <translation>绘制：</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="184"/>
        <source>X axis:</source>
        <translation>X 轴：</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="185"/>
        <source>Strain as:</source>
        <translation>应变显示为：</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="187"/>
        <source>Min. valid fraction:</source>
        <translation>最小有效比例：</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="188"/>
        <source>A frame is left blank when fewer than this fraction of a line&apos;s or region&apos;s points are reliable. Guards against a curve that stays smooth while its sample shrinks away.</source>
        <translation>当线或区域中可靠点的比例低于此值时，该帧留空。避免样本逐渐缩小而曲线依然平滑的假象。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="195"/>
        <source>Over time</source>
        <comment>Chart view: every frame of each probe</comment>
        <translation>随时间变化</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="197"/>
        <source>Each probe&apos;s reading at every frame.</source>
        <translation>每个探针在各帧的读数。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="200"/>
        <source>Along the line</source>
        <comment>Chart view: a profile</comment>
        <translation>沿线分布</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="207"/>
        <source>Kymograph</source>
        <comment>Chart view: distance against frame</comment>
        <translation>时空图</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="202"/>
        <source>The field along the selected line at the current frame, over the other frames in gray.</source>
        <translation>当前帧沿所选线的场分布，其它帧以灰色显示在下方。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="209"/>
        <source>The field along the selected line at every frame: distance against frame, value as color.</source>
        <translation>所选线在每一帧的场分布：距离对帧，数值以颜色表示。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="214"/>
        <source>Stress–strain</source>
        <comment>Chart view: stress against strain</comment>
        <translation>应力–应变</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="216"/>
        <source>Stress (or load, without A0) against the plotted quantity, one curve per probe: stress-strain with an extensometer, load against opening with a crack gauge. Needs load data.</source>
        <translation>以所绘物理量为横轴的应力（无 A0 时为载荷）曲线，每个探针一条：配合引伸计为应力–应变曲线，配合裂纹量规为载荷–张开量曲线。需要载荷数据。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="227"/>
        <source>Other frames</source>
        <translation>其它帧</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="228"/>
        <source>Draw the other frames&apos; profiles faintly behind the current one (at most twelve, evenly spaced).</source>
        <translation>在当前帧后方淡色绘制其它帧的分布（最多十二帧，均匀间隔）。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="233"/>
        <source>Line data (CSV)…</source>
        <translation>线数据（CSV）…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="234"/>
        <source>Export</source>
        <translation>导出</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="235"/>
        <source>Load data…</source>
        <translation>载荷数据…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="236"/>
        <source>Import a testing machine&apos;s load record (CSV) to plot against load or stress, and to draw stress-strain curves.</source>
        <translation>导入试验机的载荷记录（CSV），以载荷或应力为横轴绘图，并绘制应力–应变曲线。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="241"/>
        <source>Probe data (CSV)…</source>
        <translation>探针数据（CSV）…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="243"/>
        <source>Chart image…</source>
        <translation>图表图像…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="245"/>
        <source>Copy chart</source>
        <translation>复制图表</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="247"/>
        <source>Copy plotted data</source>
        <translation>复制所绘数据</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="559"/>
        <source>Import the testing machine&apos;s load record with Load data… to draw stress-strain curves.</source>
        <translation>请通过“载荷数据…”导入试验机的载荷记录，以绘制应力–应变曲线。</translation>
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
        <translation>引伸计应变</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="43"/>
        <source>Extensometer true strain</source>
        <translation>引伸计真应变</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="45"/>
        <source>Elongation ΔL</source>
        <translation>伸长量 ΔL</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="47"/>
        <source>Crack sliding</source>
        <translation>裂纹滑移</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="48"/>
        <source>Crack opening magnitude</source>
        <translation>裂纹张开位移大小</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="279"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="434"/>
        <source>Frame</source>
        <translation>帧</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="281"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="429"/>
        <source>Time (s)</source>
        <translation>时间 (s)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="283"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="431"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="574"/>
        <source>Load (N)</source>
        <translation>载荷 (N)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="286"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="433"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="571"/>
        <source>Stress (MPa)</source>
        <translation>应力 (MPa)</translation>
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
        <translation>无法绘制场：%1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="130"/>
        <source>Esc cancels placement</source>
        <translation>Esc 取消放置</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="138"/>
        <source>Drag to move the probe, or drag a handle to reshape it. Delete removes it; F2 renames it.</source>
        <translation>拖动可移动探针，拖动控制点可改变其形状。按 Delete 删除，按 F2 重命名。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="485"/>
        <source>Place a probe on the reference image to begin.</source>
        <translation>在参考图像上放置一个探针即可开始。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="212"/>
        <source>Nothing valid along %1: it lies off the measured area.</source>
        <translation>%1 沿线没有有效数据：它不在测量区域内。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="610"/>
        <source>A line view shows a field. Choose a field to plot.</source>
        <translation>沿线视图显示的是场。请选择一个场来绘制。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="614"/>
        <source>Place a line probe, or select one, to see the field along it.</source>
        <translation>放置或选择一个线探针，即可查看沿线的场。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="632"/>
        <source>Distance along %1 (%2)</source>
        <translation>沿 %1 的距离（%2）</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="641"/>
        <source>%1, frame %2</source>
        <translation>%1，第 %2 帧</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="153"/>
        <source>not plotted: off the measured area</source>
        <translation>未绘制：不在测量区域内</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="156"/>
        <source>not plotted: a gauge end is off the measured area</source>
        <translation>未绘制：量规端点不在测量区域内</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="671"/>
        <source>Export Probe Data</source>
        <translation>导出探针数据</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="664"/>
        <source>CSV Files</source>
        <translation>CSV 文件</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="442"/>
        <source>The load data cannot be matched to the frames: %1</source>
        <translation>载荷数据无法与帧对应：%1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="664"/>
        <location filename="../../gui/panels/analysis/tab.py" line="728"/>
        <source>All Files</source>
        <translation>所有文件</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="699"/>
        <source>Probe export failed: %1</source>
        <translation>探针导出失败：%1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="701"/>
        <source>Probe data written to %1</source>
        <translation>探针数据已写入 %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="709"/>
        <source>Export Line Data</source>
        <translation>导出线数据</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="718"/>
        <source>Line export failed: %1</source>
        <translation>线数据导出失败：%1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="720"/>
        <source>Line data written to %1</source>
        <translation>线数据已写入 %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="726"/>
        <source>SVG Images</source>
        <translation>SVG 图像</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="737"/>
        <source>Export Chart</source>
        <translation>导出图表</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="756"/>
        <source>Chart copied to the clipboard.</source>
        <translation>图表已复制到剪贴板。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="764"/>
        <source>Plotted data copied to the clipboard.</source>
        <translation>所绘数据已复制到剪贴板。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="725"/>
        <source>PNG Images</source>
        <translation>PNG 图像</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="727"/>
        <source>PDF Documents</source>
        <translation>PDF 文档</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="748"/>
        <source>Chart export failed: %1</source>
        <translation>图表导出失败：%1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="750"/>
        <source>Chart written to %1</source>
        <translation>图表已写入 %1</translation>
    </message>
</context>
<context>
    <name>App</name>
    <message>
        <location filename="../../gui/app.py" line="753"/>
        <source>Run DIC first -- no displacement results to post-process.</source>
        <translation>请先运行 DIC —— 当前没有可后处理的位移结果。</translation>
    </message>
</context>
<context>
    <name>Application</name>
    <message>
        <location filename="../../gui/app.py" line="886"/>
        <source>RAFTcorr has hit an error</source>
        <translation>RAFTcorr 发生错误</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="887"/>
        <source>An unexpected error occurred. The application may not behave correctly from here on, so saving your project and restarting is recommended.</source>
        <translation>发生了意外错误。程序此后可能无法正常运行，建议保存项目并重新启动。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="897"/>
        <source>Details were written to %1</source>
        <translation>详细信息已写入 %1</translation>
    </message>
</context>
<context>
    <name>AutoFixedSelector</name>
    <message>
        <location filename="../../gui/widgets/range_mode.py" line="33"/>
        <source>Auto</source>
        <comment>Color range mode: rescale to the data range</comment>
        <translation>自动</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/range_mode.py" line="36"/>
        <source>Rescale the color range to each frame&apos;s data range</source>
        <translation>根据每帧的数据范围自动缩放颜色范围</translation>
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
        <translation>所有帧都使用手动设置的最小/最大值</translation>
    </message>
</context>
<context>
    <name>BatchImportDialog</name>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="368"/>
        <source>Batch Import Region of Interest Masks</source>
        <translation>批量导入感兴趣区域掩模</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="419"/>
        <source>Mask Folder:</source>
        <translation>掩模文件夹：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="420"/>
        <source>(none)</source>
        <translation>（无）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="425"/>
        <source>Browse...</source>
        <translation>浏览…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="444"/>
        <source>Available Masks</source>
        <translation>可用掩模</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="455"/>
        <source>Auto-Match by Name</source>
        <translation>按文件名自动匹配</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="457"/>
        <source>Match mask files to frames by number in filename</source>
        <translation>根据文件名中的数字把掩模文件匹配到对应帧</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="461"/>
        <source>Assign Sequential</source>
        <translation>顺序分配</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="463"/>
        <source>Assign masks to frames in order starting from frame 0</source>
        <translation>从第 0 帧起按顺序把掩模分配给各帧</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="471"/>
        <source>Frame Assignments</source>
        <translation>帧分配</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="474"/>
        <source>Frame</source>
        <translation>帧</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="474"/>
        <source>Image</source>
        <translation>图像</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="474"/>
        <source>Mask</source>
        <translation>掩模</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="485"/>
        <source>Assign Selected -&gt;</source>
        <translation>分配所选 -&gt;</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="487"/>
        <source>Pair selected mask(s) with selected frame(s)</source>
        <translation>将所选掩模与所选帧配对</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="491"/>
        <source>Clear All</source>
        <translation>全部清除</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="536"/>
        <source>Select Mask Folder</source>
        <translation>选择掩模文件夹</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="590"/>
        <source>Failed to read mask file.</source>
        <translation>无法读取掩模文件。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="594"/>
        <source>Mismatched shape: %1×%2 (expected %3×%4)</source>
        <translation>尺寸不匹配：%1×%2（期望 %3×%4）</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="666"/>
        <source>%n mask(s) have mismatched sizes and are disabled.</source>
        <translation>
            <numerusform>%n 个掩模尺寸不匹配，已禁用。</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="719"/>
        <source>Invalid assignment</source>
        <translation>无效的分配</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="721"/>
        <source>A frame can only have one mask. Select exactly one mask, or select multiple frames to assign one mask to many.</source>
        <translation>一个帧只能对应一个掩模。请选择恰好一个掩模，或者选择多个帧以把同一个掩模应用到多个帧。</translation>
    </message>
</context>
<context>
    <name>CanvasArea</name>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1223"/>
        <source>Fit</source>
        <translation>适配</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1224"/>
        <source>Fit image to viewport</source>
        <translation>将图像适配到视口</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1228"/>
        <source>100%</source>
        <translation>100%</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1229"/>
        <source>Zoom to 100% (1:1)</source>
        <translation>缩放到 100%（1:1）</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1232"/>
        <source>Zoom in</source>
        <translation>放大</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1237"/>
        <source>–</source>
        <translation>–</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1238"/>
        <source>Zoom out</source>
        <translation>缩小</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1329"/>
        <source>EDITING REGION OF INTEREST FOR FRAME %1</source>
        <translation>正在编辑第 %1 帧的感兴趣区域</translation>
    </message>
</context>
<context>
    <name>CanvasConfigOverlay</name>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="40"/>
        <source>Mode</source>
        <translation>模式</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="41"/>
        <source>Model</source>
        <translation>模型</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="42"/>
        <source>Iterations</source>
        <translation>迭代次数</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="85"/>
        <source>Accumulative</source>
        <translation>累积式</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="87"/>
        <source>Incremental, every %1 frames</source>
        <translation>增量式，每 %1 帧</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="89"/>
        <source>Custom reference frames</source>
        <translation>自定义参考帧</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="90"/>
        <source>Incremental</source>
        <translation>增量式</translation>
    </message>
</context>
<context>
    <name>ColorRange</name>
    <message>
        <location filename="../../gui/widgets/color_range.py" line="27"/>
        <source>Range</source>
        <translation>范围</translation>
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
        <translation>残差</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="194"/>
        <source>Open this tab to render a preview.</source>
        <translation>打开此页以渲染预览。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="201"/>
        <location filename="../../gui/dialogs/export_widgets.py" line="383"/>
        <source>Field</source>
        <translation>场变量</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="208"/>
        <source>Frame</source>
        <translation>帧</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="239"/>
        <source>FIELD APPEARANCE</source>
        <translation>场的外观</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="244"/>
        <location filename="../../gui/dialogs/export_widgets.py" line="383"/>
        <source>Colormap</source>
        <translation>颜色映射</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="248"/>
        <source>Range</source>
        <translation>范围</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="252"/>
        <location filename="../../gui/dialogs/export_widgets.py" line="383"/>
        <source>Min</source>
        <translation>最小</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="259"/>
        <location filename="../../gui/dialogs/export_widgets.py" line="383"/>
        <source>Max</source>
        <translation>最大</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="266"/>
        <location filename="../../gui/dialogs/export_widgets.py" line="179"/>
        <source>Opacity</source>
        <extracomment>Column widths of the field rows (export, field, colormap, range mode, min / max), shared by the header labels of the Images and Animation tabs. Long-edge presets (pixels) of the &quot;Resolution (long edge)&quot; combos, in pyALDIC&apos;s order; the last entry, &quot;Full resolution&quot;, is 0 (no cap). Older names that share a field&apos;s colour state.</extracomment>
        <translation>不透明度</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="268"/>
        <source>Apply to all fields</source>
        <translation>应用到所有场</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="269"/>
        <source>Apply this field&apos;s colormap, opacity and auto-range to every enabled field (each field keeps its own min/max).</source>
        <translation>将该场的颜色映射、不透明度和自动范围应用到所有已启用的场（每个场保留各自的最小/最大值）。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="222"/>
        <source>Refresh preview</source>
        <translation>刷新预览</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="66"/>
        <source>COLORBAR STYLE</source>
        <translation>色条样式</translation>
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
        <translation>字号</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="88"/>
        <source>Font family</source>
        <translation>字体</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="91"/>
        <source>Bar thickness as a fraction of the picture&apos;s width (bar on the left or right) or height (bar on top or at the bottom).</source>
        <translation>色条的粗细，以图片宽度（色条在左侧或右侧）或高度（色条在上方或下方）的比例表示。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="94"/>
        <source>Bar thickness</source>
        <translation>色条粗细</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="97"/>
        <location filename="../../gui/dialogs/export_preview.py" line="114"/>
        <location filename="../../gui/dialogs/export_preview.py" line="120"/>
        <source>Black</source>
        <translation>黑色</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="98"/>
        <location filename="../../gui/dialogs/export_preview.py" line="113"/>
        <location filename="../../gui/dialogs/export_preview.py" line="119"/>
        <source>White</source>
        <translation>白色</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="101"/>
        <source>Colour behind the colorbar and around the picture; text and ticks are drawn in the opposite colour.</source>
        <translation>色条后面和图片四周的颜色；文字和刻度用相反的颜色绘制。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="104"/>
        <source>Background</source>
        <translation>背景</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="107"/>
        <source>Add a blank border around the exported content, as a fraction of the long edge (0 = none).</source>
        <translation>在导出内容外围加一圈空白边框，宽度为长边的比例（0 = 无）。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="110"/>
        <source>Margin</source>
        <translation>边距</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="116"/>
        <source>Margin color</source>
        <translation>边距颜色</translation>
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
        <translation>关闭“显示背景图像”时，原本背景图像所在位置的填充。
PNG 和 TIFF 保留透明；其他格式改为白色。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="128"/>
        <source>Hidden background</source>
        <translation>隐藏背景填充</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="293"/>
        <source>Preview failed: </source>
        <translation>预览失败：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="327"/>
        <source>Enable a field on the Images tab to preview.</source>
        <translation>请在“图像”页启用一个场以进行预览。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="331"/>
        <source>No data for this field/frame.</source>
        <translation>该场/帧没有数据。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="180"/>
        <source>Field opacity (0 = transparent, 1 = fully opaque)</source>
        <translation>场不透明度（0 = 透明，1 = 完全不透明）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="290"/>
        <source>Full resolution</source>
        <translation>原始分辨率</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="297"/>
        <source>All</source>
        <translation>全部</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="299"/>
        <source>None</source>
        <translation>无</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="370"/>
        <source>Select:</source>
        <translation>选择：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="383"/>
        <source>Export</source>
        <translation>导出</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="383"/>
        <source>Auto</source>
        <translation>自动</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="400"/>
        <source>PHYSICAL UNITS</source>
        <translation>物理单位</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="404"/>
        <source>Enable physical units</source>
        <translation>启用物理单位</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="406"/>
        <source>Scale displacement values by pixel size and show physical units on colorbar labels. Strain is dimensionless and unaffected.</source>
        <translation>按像素尺寸缩放位移值，并在色条标签上显示物理单位。应变为无量纲量，不受影响。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="423"/>
        <source>/ pixel</source>
        <translation>/ 像素</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="425"/>
        <source>Pixel size</source>
        <translation>像素尺寸</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="434"/>
        <source>fps</source>
        <translation>fps</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="436"/>
        <source>Frame rate</source>
        <translation>帧率</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="462"/>
        <source>FRAME RANGE</source>
        <translation>帧范围</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="464"/>
        <source>All frames</source>
        <translation>所有帧</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="469"/>
        <source>Numbered like the frame slider: frame 1 is the reference image.
File names use the same numbers: frame 1 is written as _0001.</source>
        <translation>与帧滑块的编号一致：第 1 帧是参考图像。
文件名使用相同的编号：第 1 帧写为 _0001。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="472"/>
        <source>From</source>
        <comment>Frame range: starting frame</comment>
        <translation>从</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="477"/>
        <source>to</source>
        <comment>Frame range: ending frame</comment>
        <translation>到</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="174"/>
        <source>Export Results</source>
        <extracomment>GUI field name -&gt; variable name in the data files (raftcorr.export.scientific). Long-edge caps the tabs start from (pixels, 0 = full resolution): pictures keep the size their DPI gives, animations start at pyALDIC&apos;s 1024 px.</extracomment>
        <translation>导出结果</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="201"/>
        <source>OUTPUT FOLDER</source>
        <translation>输出文件夹</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="207"/>
        <source>Select output folder…</source>
        <translation>选择输出文件夹…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="213"/>
        <source>Browse…</source>
        <translation>浏览…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="218"/>
        <source>Open Folder</source>
        <translation>打开文件夹</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="232"/>
        <source>Data</source>
        <translation>数据</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="233"/>
        <source>Images</source>
        <translation>图像</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="234"/>
        <source>Animation</source>
        <translation>动画</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="235"/>
        <source>Report</source>
        <translation>报告</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="320"/>
        <source>FORMAT</source>
        <translation>格式</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="322"/>
        <source>NumPy Archive (.npz)</source>
        <translation>NumPy 归档 (.npz)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="329"/>
        <source>CSV (per frame)</source>
        <translation>CSV（逐帧）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="345"/>
        <source>DISPLACEMENT</source>
        <translation>位移</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="348"/>
        <source>STRAIN</source>
        <translation>应变</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="287"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="351"/>
        <source>Run Compute Strain first.</source>
        <translation>请先运行“计算应变”。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="268"/>
        <source>Export: images not readable (%1); pictures go on white.</source>
        <translation>导出：无法读取图像（%1）；图片将以白色为背景。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="272"/>
        <source>Export: the images do not match the results&apos; image size; pictures go on white.</source>
        <translation>导出：图像尺寸与结果的图像尺寸不一致；图片将以白色为背景。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="286"/>
        <source>This run has no intensity residual.</source>
        <translation>本次运行没有灰度残差。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="324"/>
        <source>MATLAB v7.3 (.mat)</source>
        <translation>MATLAB v7.3 (.mat)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="326"/>
        <source>An HDF5-based MAT file: load it in MATLAB with load().</source>
        <translation>基于 HDF5 的 MAT 文件：在 MATLAB 中用 load() 读取。</translation>
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
        <translation>每个已计算帧一张表，感兴趣区域内每个点一行。
稠密场会生成很大的文件：1000 x 1000 px 的区域每帧有一百万行。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="355"/>
        <source>✓ Parameters file (JSON) always exported</source>
        <translation>✓ 参数文件（JSON）始终导出</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="356"/>
        <source>✓ Grid coordinates (X, Y), ROI and computed-frame flags always included</source>
        <translation>✓ 始终包含网格坐标（X, Y）、ROI 和已计算帧标记</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="357"/>
        <source>Displacements are written in pixels (+u right, +v down); the pixel size and frame rate above are recorded with them.</source>
        <translation>位移以像素为单位写入（+u 向右，+v 向下）；上方的像素尺寸和帧率会一并记录。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="363"/>
        <source>Export Data</source>
        <translation>导出数据</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="412"/>
        <source>IMAGE SETTINGS</source>
        <translation>图像设置</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="423"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="530"/>
        <source>Format</source>
        <translation>格式</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="376"/>
        <source>Deformed frame</source>
        <translation>变形帧</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="239"/>
        <source>Preview &amp; Colorbar</source>
        <translation>预览与色条</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="334"/>
        <source>NPZ: one file per frame (default: single merged file)</source>
        <translation>NPZ：逐帧一个文件（默认：合并为单个文件）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="337"/>
        <source>One .npz per computed frame, in a folder of its own: that frame&apos;s variables, the grid coordinates, the ROI and the metadata.</source>
        <translation>每个已计算的帧一个 .npz 文件，放在单独的文件夹中：包含该帧的变量、网格坐标、ROI 和元数据。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="377"/>
        <source>Reference frame</source>
        <translation>参考帧</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="383"/>
        <source>Show background image</source>
        <translation>显示背景图像</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="394"/>
        <source>Hide untracked regions</source>
        <translation>隐藏未追踪区域</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="397"/>
        <source>Leave points without a result transparent in the pictures, for example material that left the field of view, instead of filling them with the nearest value inside the region of interest. Data files always keep NaN there.</source>
        <translation>在图片中让没有结果的点保持透明（例如移出视野的材料），而不是用感兴趣区域内最近的值填充。数据文件中这些点始终为 NaN。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="430"/>
        <source>Cap the long edge of each written image (the larger of width/height, in pixels; the aspect ratio and layout are kept).
Lower = faster and much smaller on disk; for PDF and SVG it caps the embedded picture. &apos;Full resolution&apos; writes the size set by DPI.</source>
        <translation>限制每张输出图像的长边（宽、高中较大者，单位像素；保持宽高比和布局）。
越小越快、文件越小；对 PDF 和 SVG 则限制其中嵌入的图片。“原始分辨率”按 DPI 设定的尺寸输出。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="435"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="543"/>
        <source>Resolution (long edge)</source>
        <translation>分辨率（长边）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="443"/>
        <source>JPEG quality (higher = larger file). Ignored for the other formats.</source>
        <translation>JPEG 质量（越高文件越大）。其他格式忽略此项。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="446"/>
        <source>JPEG quality</source>
        <translation>JPEG 质量</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="454"/>
        <source>Resolution of the written figure: about 8.6 x 7 inches at this many dots per inch.
A resolution cap (long edge) lowers it.</source>
        <translation>输出图的分辨率：约 8.6 x 7 英寸，按此每英寸点数输出。
设置分辨率上限（长边）会降低它。</translation>
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
        <translation>包含色条</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="474"/>
        <source>Add a colorbar to each image; its position and style are set on the Preview &amp; Colorbar tab.</source>
        <translation>为每张图像添加色条；其位置和样式在“预览与色条”标签页中设置。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="479"/>
        <source>Uncheck to export the field on its own, with no speckle image behind it. Pick the fill on the Preview &amp; Colorbar tab.</source>
        <translation>取消勾选可仅导出场，其后不含散斑图像。填充色在“预览与色条”页选择。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="487"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="582"/>
        <source>Render as</source>
        <translation>绘制为</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="497"/>
        <source>Export Images</source>
        <translation>导出图像</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="511"/>
        <source>ANIMATION SETTINGS</source>
        <translation>动画设置</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="537"/>
        <source>Cap the animation&apos;s long edge (the larger of width/height).
Lower = faster and much smaller. Strongly recommended for GIF, whose size explodes at native resolution. &apos;Full resolution&apos; draws the images at their own pixel size.</source>
        <translation>限制动画的长边（宽、高中较大者）。
越小越快、文件越小。强烈建议用于 GIF：原始分辨率下 GIF 的体积会急剧膨胀。“原始分辨率”按图像本身的像素尺寸绘制。</translation>
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
        <translation>每 N 帧导出一帧（1 = 每帧都导出）。越大越快、越小，
但看起来更卡顿。播放时长保持不变（上方 FPS 为抽帧前的帧率）。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="567"/>
        <source>Frame step</source>
        <translation>抽帧间隔</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="572"/>
        <source>Add a colorbar to each frame; its position and style are set on the Preview &amp; Colorbar tab.</source>
        <translation>为每一帧添加色条；其位置和样式在“预览与色条”标签页中设置。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="576"/>
        <source>Uncheck to export the field on its own, with no speckle image behind it. GIF and MP4 cannot store transparency, so a transparent fill is written as white.</source>
        <translation>取消勾选可仅导出场，其后不含散斑图像。GIF 和 MP4 无法保存透明度，透明填充将写为白色。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="595"/>
        <source>Export Animation</source>
        <translation>导出动画</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="605"/>
        <source>CONTENT</source>
        <translation>内容</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="607"/>
        <source>Parameter summary table</source>
        <translation>参数摘要表</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="609"/>
        <source>Field statistics (min/max/mean/std per frame)</source>
        <translation>场统计（每帧 最小/最大/平均/标准差）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="611"/>
        <source>Sample field images</source>
        <translation>场图像示例</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="619"/>
        <source>Sample every</source>
        <translation>每隔</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="625"/>
        <source>frames</source>
        <comment>Report: sample every N frames</comment>
        <translation>帧</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="631"/>
        <source>FIELDS</source>
        <translation>场变量</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="633"/>
        <source>Displacement:</source>
        <translation>位移：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="634"/>
        <source>Strain:</source>
        <translation>应变：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="643"/>
        <source>Format: HTML (self-contained, view in any browser)</source>
        <translation>格式：HTML（自包含，可在任意浏览器中查看）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="647"/>
        <source>Generate Report</source>
        <translation>生成报告</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="808"/>
        <source>Select Output Folder</source>
        <translation>选择输出文件夹</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="909"/>
        <source>Writing %1 (%2%)</source>
        <translation>正在写入 %1 (%2%)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="942"/>
        <source>Error: %1</source>
        <translation>错误：%1</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="884"/>
        <source>Starting…</source>
        <translation>开始中…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="380"/>
        <source>Deformed: the field is drawn where each point is in that frame, over the frame&apos;s own photo.
Reference: drawn at the points&apos; positions in frame 1, over frame 1.</source>
        <translation>变形帧：场绘制在各点在该帧中的位置，叠加在该帧自己的照片上。
参考帧：绘制在各点在第 1 帧中的位置，叠加在第 1 帧上。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="387"/>
        <source>The images of this run are not available.</source>
        <translation>本次运行的图像不可用。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="461"/>
        <source>Each frame</source>
        <translation>逐帧</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="462"/>
        <source>Whole sequence</source>
        <translation>整个序列</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="464"/>
        <source>How Auto picks the colour range: from each frame&apos;s own values (like the main window),
or one range from every exported frame, so a colour means the same value in every image.</source>
        <translation>自动范围的取法：按每帧自身的数值（与主窗口相同），
或由所有导出帧共同确定一个范围，使同一颜色在每张图像中代表同一数值。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="459"/>
        <source>Auto range</source>
        <translation>自动范围</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="483"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="579"/>
        <source>Crop to ROI</source>
        <translation>裁剪到 ROI</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="484"/>
        <source>Cut the picture to the region of interest plus a small margin.</source>
        <translation>将图片裁剪到感兴趣区域，并保留少量边距。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="523"/>
        <source>MP4 needs the imageio and imageio-ffmpeg packages (pip install imageio imageio-ffmpeg). Save as GIF instead.</source>
        <translation>MP4 需要 imageio 和 imageio-ffmpeg 包（pip install imageio imageio-ffmpeg）。请改存为 GIF。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="580"/>
        <source>Cut the frames to the region of interest plus a small margin.</source>
        <translation>将各帧裁剪到感兴趣区域，并保留少量边距。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="587"/>
        <source>Auto range: one range for the whole animation, so a colour means the same value in every frame.</source>
        <translation>自动范围：整个动画使用同一个范围，使同一颜色在每一帧中代表同一数值。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="614"/>
        <source>Drawn like the Images tab: same colormap, range, opacity and configuration.</source>
        <translation>绘制方式与“图像”页相同：颜色映射、范围、不透明度和配置均一致。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="737"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="789"/>
        <source>Min must be below Max for %1.</source>
        <translation>%1 的最小值必须小于最大值。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="743"/>
        <source>Nothing to export: enable at least one field with computed frames in the frame range.</source>
        <translation>没有可导出的内容：请至少启用一个在帧范围内有已计算帧的场。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="751"/>
        <source>Select at least one format.</source>
        <translation>请至少选择一种格式。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="755"/>
        <source>Select at least one variable to export.</source>
        <translation>请至少选择一个要导出的变量。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="792"/>
        <source>Nothing to export: select a field or the parameter table.</source>
        <translation>没有可导出的内容：请选择一个场或参数表。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="845"/>
        <source>… and %1 more</source>
        <translation>… 另有 %1 个文件</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="847"/>
        <source>Replace Existing Files?</source>
        <translation>替换现有文件？</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="848"/>
        <source>These already exist in the output folder and will be replaced:

%1</source>
        <translation>以下文件已存在于输出文件夹中，将被替换：

%1</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="867"/>
        <source>The results changed after this dialog was opened (a new run, strain or project). Close it and open Export again to export them.</source>
        <translation>打开此对话框后结果已发生变化（新的运行、应变或项目）。请关闭对话框并重新打开“导出”，再导出这些结果。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="871"/>
        <source>Nothing exported: the existing files were kept.</source>
        <translation>未导出任何内容：已保留现有文件。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="896"/>
        <source>Stopping…</source>
        <translation>正在停止…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="907"/>
        <source>Rendering %1 (%2/%3)</source>
        <translation>正在渲染 %1 (%2/%3)</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/dialogs/export_dialog.py" line="925"/>
        <source>Exported %n file(s) → %1</source>
        <translation>
            <numerusform>已导出 %n 个文件 → %1</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/dialogs/export_dialog.py" line="927"/>
        <source>Exported %n image(s) → %1</source>
        <translation>
            <numerusform>已导出 %n 张图像 → %1</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="943"/>
        <source>Export failed: %1</source>
        <translation>导出失败：%1</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/dialogs/export_dialog.py" line="951"/>
        <source>Export cancelled: %n file(s) written before stopping.</source>
        <translation>
            <numerusform>导出已取消：停止前已写入 %n 个文件。</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="931"/>
        <source>No animation was written. See the log for details.</source>
        <translation>未写入任何动画。详情请查看日志。</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/dialogs/export_dialog.py" line="933"/>
        <source>Exported %n animation(s) → %1</source>
        <translation>
            <numerusform>已导出 %n 个动画 → %1</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="935"/>
        <source>Report saved → %1</source>
        <translation>报告已保存 → %1</translation>
    </message>
</context>
<context>
    <name>FieldSelector</name>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="19"/>
        <source>Horizontal displacement (+ right)</source>
        <translation>水平位移（向右为正）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="20"/>
        <source>Vertical displacement (+ down)</source>
        <translation>垂直位移（向下为正）</translation>
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
        <translation>灰度残差：参考图像与按位移映射回来的变形图像之间的
局部均方根差。追踪失败处数值较大。</translation>
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
        <source>Magnitude</source>
        <translation>位移大小</translation>
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
        <translation>上一帧</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="69"/>
        <location filename="../../gui/widgets/frame_navigator.py" line="170"/>
        <source>Play animation</source>
        <translation>播放动画</translation>
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
        <translation>下一帧</translation>
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
        <translation>帧 0/0</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="159"/>
        <source>⏸</source>
        <translation>⏸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="160"/>
        <source>Pause animation</source>
        <translation>暂停动画</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="191"/>
        <source>FRAME %1/%2</source>
        <translation>帧 %1/%2</translation>
    </message>
</context>
<context>
    <name>ImageCanvas</name>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1067"/>
        <source>Load images first before drawing a Region of Interest.</source>
        <translation>请先加载图像，再绘制感兴趣区域。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1086"/>
        <source>The three points are nearly collinear — pick points spread around the circle&apos;s edge.</source>
        <translation>三个点几乎共线 — 请在圆周上分散地选取三个点。</translation>
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
        <translation>文件名</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="99"/>
        <source>Region</source>
        <comment>Image list column: ROI status</comment>
        <translation>区域</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="193"/>
        <location filename="../../gui/widgets/image_list.py" line="255"/>
        <source>Add</source>
        <translation>添加</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="249"/>
        <source>Edit</source>
        <translation>编辑</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="252"/>
        <source>Need</source>
        <translation>待绘</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/widgets/image_list.py" line="368"/>
        <source>Import Region of Interest for %n frame(s)</source>
        <translation>
            <numerusform>为 %n 帧导入感兴趣区域</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="379"/>
        <source>Clear Region of Interest (%1 with region)</source>
        <translation>清除感兴趣区域（%1 帧已有区域）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="383"/>
        <source>Clear Region of Interest</source>
        <translation>清除感兴趣区域</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/widgets/image_list.py" line="394"/>
        <source>Delete %n image(s)</source>
        <translation>
            <numerusform>删除 %n 张图像</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/widgets/image_list.py" line="435"/>
        <source>Select %n Mask File(s)</source>
        <translation>
            <numerusform>选择 %n 个掩模文件</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="440"/>
        <source>Images</source>
        <translation>图像</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="441"/>
        <source>All Files</source>
        <translation>所有文件</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="450"/>
        <source>Selected %1 files for %2 frames — count must match</source>
        <translation>已选择 %1 个文件用于 %2 帧 — 数量必须匹配</translation>
    </message>
</context>
<context>
    <name>ImageMessages</name>
    <message>
        <location filename="../../gui/controllers/image_controller.py" line="43"/>
        <source>Cannot open the images</source>
        <translation>无法打开图像</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/image_controller.py" line="47"/>
        <source>%n image(s), %1 x %2 px, %3</source>
        <translation>
            <numerusform>%n 张图像，%1 x %2 px，%3</numerusform>
        </translation>
    </message>
</context>
<context>
    <name>LeftSidebar</name>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="176"/>
        <source>IMAGES</source>
        <translation>图像</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="183"/>
        <source>Natural Sort (1, 2, …, 10)</source>
        <translation>自然排序（1, 2, …, 10）</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="259"/>
        <source>WORKFLOW TYPE</source>
        <translation>工作流类型</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="186"/>
        <source>Sort by embedded numbers: image1, image2, ..., image10 (default).
Unchecked: plain alphabetical order, for zero-padded names.</source>
        <translation>按文件名中的数字排序：image1, image2, ..., image10（默认）。
不勾选：按普通字母顺序，适合已补零的文件名。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="266"/>
        <source>REGION OF INTEREST</source>
        <translation>感兴趣区域</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="278"/>
        <source>PARAMETERS</source>
        <translation>参数</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="284"/>
        <source>ADVANCED</source>
        <translation>高级</translation>
    </message>
</context>
<context>
    <name>LoadDataDialog</name>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="103"/>
        <source>Load Data</source>
        <translation>载荷数据</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="115"/>
        <source>No file chosen.</source>
        <translation>未选择文件。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="117"/>
        <source>Choose file…</source>
        <translation>选择文件…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="137"/>
        <source>Load column:</source>
        <translation>载荷列：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="139"/>
        <source>By time</source>
        <translation>按时间</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="140"/>
        <source>By frame number</source>
        <translation>按帧号</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="144"/>
        <source>Match rows to frames:</source>
        <translation>行与帧的对应方式：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="147"/>
        <source>Time column:</source>
        <translation>时间列：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="153"/>
        <source>The machine&apos;s time at the reference image. If the camera started 2 s after the machine, enter 2.</source>
        <translation>参考图像时刻对应的试验机时间。若相机比试验机晚 2 s 启动，则输入 2。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="155"/>
        <source>Offset:</source>
        <translation>偏移：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="163"/>
        <source>Frame column:</source>
        <translation>帧号列：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="167"/>
        <source>The first image is numbered:</source>
        <translation>第一张图像的编号：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="174"/>
        <source>Initial cross-section, for engineering stress F / A0 in MPa. Leave at 0 for load only.</source>
        <translation>初始横截面积，用于计算工程应力 F / A0（MPa）。仅需载荷时保持为 0。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="176"/>
        <source>Cross-section A0:</source>
        <translation>横截面积 A0：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="192"/>
        <source>Remove Load Data</source>
        <translation>移除载荷数据</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="201"/>
        <source>Camera frame rate: %1 fps, from Physical Units.</source>
        <translation>相机帧率：%1 fps，取自“物理单位”。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="204"/>
        <source>Set the camera frame rate under Physical Units to match by time.</source>
        <translation>请在“物理单位”中设置相机帧率，才能按时间对应。</translation>
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
        <translation>没有帧落在记录范围内：请检查所选列和偏移。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="328"/>
        <source>Frames with a load: %1 of %2.</source>
        <translation>有载荷值的帧：%1 / %2。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="334"/>
        <source>Open Load Data</source>
        <translation>打开载荷数据</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="335"/>
        <source>CSV Files</source>
        <translation>CSV 文件</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="335"/>
        <source>All Files</source>
        <translation>所有文件</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="341"/>
        <source>Could not read %1: %2</source>
        <translation>无法读取 %1：%2</translation>
    </message>
</context>
<context>
    <name>MainWindow</name>
    <message>
        <location filename="../../gui/app.py" line="146"/>
        <source>&amp;File</source>
        <translation>文件</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="171"/>
        <source>Quit</source>
        <translation>退出</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="175"/>
        <source>&amp;Settings</source>
        <translation>设置</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="178"/>
        <source>Theme</source>
        <translation>主题</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="182"/>
        <source>Dark</source>
        <comment>Color theme</comment>
        <translation>深色</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="182"/>
        <source>Light</source>
        <comment>Color theme</comment>
        <translation>浅色</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="192"/>
        <source>Language</source>
        <translation>语言</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="202"/>
        <source>Models…</source>
        <translation>模型…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="376"/>
        <source>Suggested: RAFTcorr-%1. Largest displacement of the region of interest about %2 px (99th percentile over frames %3); RAFTcorr-%1 reaches %4 px.</source>
        <translation>推荐：RAFTcorr-%1。感兴趣区域内最大位移约 %2 px（在第 %3 帧上取第 99 百分位数）；RAFTcorr-%1 的量程为 %4 px。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="463"/>
        <source>Language changed</source>
        <translation>语言已切换</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="634"/>
        <source>All Files</source>
        <translation>所有文件</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="210"/>
        <source>Select Image Folder</source>
        <translation>选择图像文件夹</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="334"/>
        <source>File Association Failed</source>
        <translation>文件关联失败</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="337"/>
        <source>File Association</source>
        <translation>文件关联</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="156"/>
        <source>Open Image Folder…</source>
        <translation>打开图像文件夹…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="158"/>
        <source>Open Project…</source>
        <translation>打开项目…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="159"/>
        <location filename="../../gui/app.py" line="242"/>
        <location filename="../../gui/app.py" line="249"/>
        <source>Save Project</source>
        <translation>保存项目</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="160"/>
        <source>Save Project As…</source>
        <translation>项目另存为…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="162"/>
        <source>Export Results…</source>
        <translation>导出结果…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="167"/>
        <source>Associate .raftcorr files with RAFTcorr…</source>
        <translation>将 .raftcorr 文件关联到 RAFTcorr…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="168"/>
        <source>Register .raftcorr so double-clicking a project file opens RAFTcorr (current user only, no admin rights needed).</source>
        <translation>注册 .raftcorr，双击项目文件即可打开 RAFTcorr（仅当前用户，无需管理员权限）。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="201"/>
        <source>&amp;Help</source>
        <translation>帮助</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="203"/>
        <location filename="../../gui/app.py" line="399"/>
        <location filename="../../gui/app.py" line="401"/>
        <source>Self-test</source>
        <translation>自检</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="204"/>
        <location filename="../../gui/app.py" line="406"/>
        <source>About RAFTcorr</source>
        <translation>关于 RAFTcorr</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="219"/>
        <source>Cancel</source>
        <translation>取消</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="243"/>
        <source>Load the images first: a project keeps the images, the regions of interest, the settings and the results together.</source>
        <translation>请先加载图像：项目会把图像、感兴趣区域、设置和结果保存在一起。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="264"/>
        <source>Project saved: %1</source>
        <translation>项目已保存：%1</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="266"/>
        <source>Saving the project</source>
        <translation>正在保存项目</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="280"/>
        <source>Include Results?</source>
        <translation>包含结果？</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="281"/>
        <source>Include the computed results in this project?</source>
        <translation>在此项目中包含计算结果吗？</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="283"/>
        <source>Including results (about %1 uncompressed) lets you reopen the project without recomputing. Choose No to save a small settings-only file for sharing.</source>
        <translation>包含结果（未压缩约 %1）后，重新打开项目时无需重新计算。选择“否”则只保存设置，文件小、便于分享。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="295"/>
        <source>Open Project</source>
        <translation>打开项目</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="302"/>
        <source>Opening the project</source>
        <translation>正在打开项目</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="310"/>
        <source>Images not found</source>
        <translation>未找到图像</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="311"/>
        <source>The image folder of this project was not found. Choose its new location.</source>
        <translation>未找到此项目的图像文件夹。请选择它的新位置。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="312"/>
        <source>Image folder of the project</source>
        <translation>项目的图像文件夹</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="319"/>
        <source>Cannot open the project</source>
        <translation>无法打开项目</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="325"/>
        <source>Project opened: %1</source>
        <translation>已打开项目：%1</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="335"/>
        <source>Could not register .raftcorr files: </source>
        <translation>无法注册 .raftcorr 文件：</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="338"/>
        <source>Done. Double-clicking a .raftcorr file will now open RAFTcorr and restore that project.</source>
        <translation>完成。现在双击 .raftcorr 文件即可打开 RAFTcorr 并恢复该项目。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="343"/>
        <source>Load at least two images first.</source>
        <translation>请先加载至少两张图像。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="346"/>
        <source>Wait for the running task to finish.</source>
        <translation>请等待当前任务完成。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="380"/>
        <source>This exceeds even the reach of RAFTcorr-s8: use incremental tracking.</source>
        <translation>这甚至超出了 RAFTcorr-s8 的量程：请使用增量追踪。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="385"/>
        <source>Measuring the motion with RAFTcorr-s8</source>
        <translation>正在用 RAFTcorr-s8 测量运动</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="407"/>
        <source>&lt;b&gt;RAFTcorr %1&lt;/b&gt;&lt;br&gt;Learned digital image correlation with the RAFTcorr-s1, -s2, -s4 and -s8 models.&lt;br&gt;&lt;br&gt;Displacements: reference to deformed, +u right, +v down, in pixels, frame-1 (Lagrangian) coordinates.&lt;br&gt;&lt;br&gt;Interface derived from pyALDIC.&lt;br&gt;Zixiang (Zach) Tong, The University of Texas at Austin.</source>
        <translation>&lt;b&gt;RAFTcorr %1&lt;/b&gt;&lt;br&gt;基于学习的数字图像相关，使用 RAFTcorr-s1、-s2、-s4 和 -s8 模型。&lt;br&gt;&lt;br&gt;位移：从参考到变形，+u 向右，+v 向下，单位为像素，采用第 1 帧（拉格朗日）坐标。&lt;br&gt;&lt;br&gt;界面源自 pyALDIC。&lt;br&gt;Zixiang (Zach) Tong, The University of Texas at Austin.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="466"/>
        <source>Language set to %1. Please restart RAFTcorr for all widgets to pick up the new language.</source>
        <translation>语言已切换至 %1。请重启 RAFTcorr 以让所有界面生效。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="601"/>
        <source>Frame %1 has no Region of Interest of its own — frame 1&apos;s is used for computation. Switch to frame 1 to edit it, or import a mask to give this frame its own.</source>
        <translation>帧 %1 没有自己的感兴趣区域 — 计算时使用帧 1 的感兴趣区域。请切换到帧 1 编辑，或导入掩模为此帧单独指定。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="612"/>
        <source>No Region of Interest to save — load images first.</source>
        <translation>没有可保存的感兴趣区域 — 请先加载图像。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="619"/>
        <source>Region of Interest mask is empty.</source>
        <translation>感兴趣区域掩模为空。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="631"/>
        <source>Save Region of Interest Mask</source>
        <translation>保存感兴趣区域掩模</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="633"/>
        <source>PNG Images</source>
        <translation>PNG 图像</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="642"/>
        <source>Mask saved to %1</source>
        <translation>掩模已保存至 %1</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="651"/>
        <source>No Region of Interest to invert — load images first.</source>
        <translation>没有可反选的感兴趣区域 — 请先加载图像。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="674"/>
        <source>Load images first.</source>
        <translation>请先加载图像。</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="710"/>
        <source>  Imported mask for frame %1</source>
        <translation>  已导入帧 %1 的掩模</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/app.py" line="716"/>
        <source>Batch import: %n mask(s) loaded</source>
        <translation>
            <numerusform>批量导入：已加载 %n 个掩模</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/app.py" line="743"/>
        <source>Imported Region of Interest for %n frame(s)</source>
        <translation>
            <numerusform>为 %n 帧导入了感兴趣区域</numerusform>
        </translation>
    </message>
</context>
<context>
    <name>ModelWeights</name>
    <message>
        <location filename="../../gui/support/model_weights.py" line="27"/>
        <source>Downloading model %1: %2 of %3 MB</source>
        <extracomment>``status(fraction, text)``; an empty text ends the download status.</extracomment>
        <translation>正在下载模型 %1：%2 / %3 MB</translation>
    </message>
    <message>
        <location filename="../../gui/support/model_weights.py" line="29"/>
        <source>Downloading model %1…</source>
        <translation>正在下载模型 %1…</translation>
    </message>
</context>
<context>
    <name>ModelsDialog</name>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="115"/>
        <source>Models</source>
        <translation>模型</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="126"/>
        <source>RAFTcorr downloads a model the first time an analysis uses it and keeps it for later runs. Each model is about 20 MB.</source>
        <translation>RAFTcorr 会在分析第一次用到某个模型时自动下载它，并保留供以后使用。每个模型约 20 MB。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="136"/>
        <source>s = %1, reach %2 px</source>
        <translation>s = %1，量程 %2 px</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="139"/>
        <source>Download</source>
        <translation>下载</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="164"/>
        <source>Model folder: %1</source>
        <translation>模型文件夹：%1</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="171"/>
        <source>No internet on this computer? Download the .pth files from the &lt;a href=&quot;%1&quot;&gt;RAFTcorr model release&lt;/a&gt; on another computer, then import them here.</source>
        <translation>这台电脑无法上网？请在另一台电脑上从&lt;a href=&quot;%1&quot;&gt;RAFTcorr 模型发布页&lt;/a&gt;下载 .pth 文件，再在这里导入。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="180"/>
        <source>Download All</source>
        <translation>全部下载</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="182"/>
        <source>Import Model Files…</source>
        <translation>导入模型文件…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="184"/>
        <source>Open Model Folder</source>
        <translation>打开模型文件夹</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="186"/>
        <source>Cancel Download</source>
        <translation>取消下载</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="189"/>
        <source>Close</source>
        <translation>关闭</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="222"/>
        <source>Installed</source>
        <translation>已安装</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="226"/>
        <source>Not installed (%1 MB)</source>
        <translation>未安装（%1 MB）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="227"/>
        <source>Not installed</source>
        <translation>未安装</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="238"/>
        <source>&lt;b&gt;%1 is installed now.&lt;/b&gt; Close this window and start again.</source>
        <translation>&lt;b&gt;%1 现已安装。&lt;/b&gt;关闭此窗口后重新开始即可。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="242"/>
        <source>&lt;b&gt;%1 is needed, but it is not installed and could not be downloaded.&lt;/b&gt; Download it again below, or import the file.</source>
        <translation>&lt;b&gt;需要 %1，但它尚未安装，且下载失败。&lt;/b&gt;请在下方重新下载，或导入模型文件。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="286"/>
        <source>Download cancelled.</source>
        <translation>下载已取消。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="280"/>
        <source>Downloaded: %1.</source>
        <translation>已下载：%1。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="283"/>
        <source>%1 could not be downloaded: %2</source>
        <translation>%1 下载失败：%2</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="310"/>
        <source>Import Model Files</source>
        <translation>导入模型文件</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="311"/>
        <source>RAFTcorr models (*.pth)</source>
        <translation>RAFTcorr 模型 (*.pth)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="325"/>
        <source>Imported: %1.</source>
        <translation>已导入：%1。</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="327"/>
        <source>Not imported: %1. Only the model files of the release can be imported.</source>
        <translation>未导入：%1。只能导入发布页上的模型文件。</translation>
    </message>
</context>
<context>
    <name>ParamPanel</name>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="90"/>
        <source>Iterations</source>
        <extracomment>The label column is as wide as its longest label in the current language, never narrower than the English layout and, past about 28 characters of the label font, wrapped rather than squeezing the inputs (pyALDIC&apos;s ParamPanel).</extracomment>
        <translation>迭代次数</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="91"/>
        <source>Refinement iterations of the network (default: 20).</source>
        <translation>网络的迭代细化次数（默认：20）。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="98"/>
        <source>Off</source>
        <comment>Displacement smoothing</comment>
        <translation>关闭</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="101"/>
        <source>Smoothing</source>
        <translation>平滑</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="102"/>
        <source>Gaussian smoothing (σ in pixels) of the tracked displacement, kept in the results.
Off by default: smoothing coarsens the spatial resolution.</source>
        <translation>对追踪得到的位移进行高斯平滑（σ 以像素计），平滑后的位移保存在结果中。
默认关闭：平滑会降低空间分辨率。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="106"/>
        <source>Automatic</source>
        <translation>自动</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="110"/>
        <source>Device</source>
        <translation>设备</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="111"/>
        <source>Automatic uses the GPU when PyTorch sees one.</source>
        <translation>选择“自动”时，只要 PyTorch 检测到 GPU 就使用 GPU。</translation>
    </message>
</context>
<context>
    <name>PhysicalUnitsWidget</name>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="56"/>
        <source>Use physical units</source>
        <translation>使用物理单位</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="67"/>
        <source>Physical size of one image pixel</source>
        <translation>单个图像像素对应的物理尺寸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="85"/>
        <source>/ px</source>
        <translation>/ px</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="88"/>
        <source>Pixel size</source>
        <translation>像素尺寸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="98"/>
        <source>Acquisition frame rate (used for velocity field)</source>
        <translation>采集帧率（用于速度场）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="100"/>
        <source>Frame rate</source>
        <translation>帧率</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="168"/>
        <source>Disp: %1  Velocity: %2/s</source>
        <translation>位移：%1  速度：%2/s</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="173"/>
        <source>Disp: px  Velocity: px/fr</source>
        <translation>位移：px  速度：px/帧</translation>
    </message>
</context>
<context>
    <name>PipelineController</name>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="80"/>
        <source>Stopped.</source>
        <extracomment>Results larger than this are kept in memory-mapped files in a temporary folder.</extracomment>
        <translation>已停止。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="92"/>
        <source>The GPU ran out of memory. Close other GPU programs, choose a model with a larger s, or lower the memory fraction (Advanced).</source>
        <translation>GPU 显存不足。请关闭其他占用 GPU 的程序，选择 s 更大的模型，或调低“高级”中的 GPU 显存比例。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="94"/>
        <source>Analysis failed</source>
        <translation>分析失败</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="173"/>
        <source>Reference frames without a Region of Interest of their own carry the one of frame 1 forward: %1</source>
        <translation>没有自己感兴趣区域的参考帧沿用第 1 帧的感兴趣区域：%1</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="196"/>
        <source>Starting...</source>
        <translation>开始中…</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="222"/>
        <source>The images changed: the running analysis is cancelled.</source>
        <translation>图像已更改：正在运行的分析已取消。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="252"/>
        <source>The images changed during the run: its results are discarded.</source>
        <translation>运行期间图像发生了变化：本次结果已丢弃。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="265"/>
        <source>No frame was computed; the previous results are kept.</source>
        <translation>没有计算任何帧；保留之前的结果。</translation>
    </message>
</context>
<context>
    <name>PipelineMessages</name>
    <message numerus="yes">
        <location filename="../../gui/controllers/pipeline_messages.py" line="73"/>
        <source>  %n frame(s) with custom ROI masks</source>
        <translation>
            <numerusform>  %n 帧使用自定义 ROI 掩模</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/pipeline_messages.py" line="77"/>
        <source>Results received: %n frame(s)</source>
        <translation>
            <numerusform>已收到结果：%n 帧</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/pipeline_messages.py" line="81"/>
        <source>  %n image(s), %1</source>
        <translation>
            <numerusform>  %n 张图像，%1</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="84"/>
        <source>  ROI range: %1</source>
        <translation>  ROI 范围：%1</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="87"/>
        <source>  Refinement: %1</source>
        <translation>  网格加密：%1</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="91"/>
        <source>Error: %1</source>
        <translation>错误：%1</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="95"/>
        <source>Run canceled. Give the listed reference frames a Region of Interest of their own, or run again and continue with frame 1&apos;s region.</source>
        <translation>已取消运行。请为列出的参考帧指定各自的感兴趣区域，或重新运行并沿用第 1 帧的区域继续。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="102"/>
        <source>Tracking Quality Warning</source>
        <translation>追踪质量警告</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="105"/>
        <source>Part of the sequence may not be tracked correctly.</source>
        <translation>序列中的部分内容可能未被正确追踪。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="110"/>
        <source>The tracking quality drops from frame %1 on: the intensity residual rises to %2 grey levels, against %3 before.</source>
        <translation>从第 %1 帧起追踪质量下降：灰度残差升至 %2 个灰度级，此前为 %3。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="116"/>
        <source>The displacement from frame 1 may exceed the reach of the model. Try incremental tracking, or a model with a larger s.</source>
        <translation>相对第 1 帧的位移可能超出了模型的量程。请尝试增量式追踪，或换用 s 更大的模型。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="119"/>
        <source>The step between reference frames may exceed the reach of the model. Use more reference frames, or a model with a larger s.</source>
        <translation>参考帧之间的位移步长可能超出了模型的量程。请使用更多参考帧，或换用 s 更大的模型。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="123"/>
        <source>Frames with a high residual: %1</source>
        <translation>残差偏高的帧：%1</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="126"/>
        <source>Show the Residual field to see where the images do not match after the displacement is applied.</source>
        <translation>显示“残差”场，即可查看施加位移后图像在哪些位置不匹配。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="132"/>
        <source>Cannot start analysis</source>
        <translation>无法开始分析</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="136"/>
        <source>Need at least 2 images. Load an image folder in the %1 section before running DIC.</source>
        <translation>至少需要 2 张图像。请先在“%1”部分加载图像文件夹，再运行 DIC。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="141"/>
        <source>Define a Region of Interest first: draw or import one for frame 1 in the %1 section.</source>
        <translation>请先定义感兴趣区域：在“%1”部分为第 1 帧绘制或导入一个。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="146"/>
        <source>The Region of Interest of frame 1 is empty (no pixels selected). Draw it again in the %1 section.</source>
        <translation>第 1 帧的感兴趣区域为空（未选中任何像素）。请在“%1”部分重新绘制。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="151"/>
        <source>Region of Interest too small</source>
        <translation>感兴趣区域过小</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="157"/>
        <source>The Region of Interest is %1×%2 px, but the current %3 (%4) needs at least %5×%5 px. Enlarge the Region of Interest or reduce %3.</source>
        <translation>感兴趣区域为 %1×%2 px，但当前的%3（%4）至少需要 %5×%5 px。请扩大感兴趣区域或减小%3。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="163"/>
        <source>Starting Points required</source>
        <translation>需要种子点</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="167"/>
        <source>Place at least one Starting Point on the canvas before running: click “%1” or “%2” in the %3 section.</source>
        <translation>运行前请在画布上放置至少一个种子点：点击“%3”部分中的“%1”或“%2”。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="173"/>
        <source>Invalid Starting Points</source>
        <translation>种子点无效</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="177"/>
        <source>One or more Starting Points do not lie in any part of the Region of Interest. Click “%1” in the %2 section, then place the Starting Points again.</source>
        <translation>有一个或多个种子点不在感兴趣区域的任何部分内。请点击“%2”部分中的“%1”，然后重新放置种子点。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="185"/>
        <source>DIC analysis failed</source>
        <translation>DIC 分析失败</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="188"/>
        <source>Failed to start analysis</source>
        <translation>启动分析失败</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="191"/>
        <source>Failed to process results</source>
        <translation>处理结果失败</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="196"/>
        <source>The analysis stopped with a runtime error:

%1

See the console log for the full traceback.</source>
        <translation>分析因运行时错误而停止：

%1

完整的回溯信息请参见控制台日志。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="202"/>
        <source>%1

See the console log for the full traceback.</source>
        <translation>%1

完整的回溯信息请参见控制台日志。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="207"/>
        <source>Incomplete Region of Interest Coverage</source>
        <translation>感兴趣区域覆盖不完整</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/pipeline_messages.py" line="212"/>
        <source>This incremental run has %n reference frame(s) with no Region of Interest of their own: frame(s) %1.</source>
        <translation>
            <numerusform>本次增量追踪运行中有 %n 个参考帧没有自己的感兴趣区域：帧 %1。</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/pipeline_messages.py" line="218"/>
        <source>%1 and %n more</source>
        <comment>List of frame numbers cut short: %1 = the frame numbers shown, %n = how many more frames there are</comment>
        <translation>
            <numerusform>%1 以及另外 %n 帧</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="224"/>
        <source>A reference frame without a Region of Interest of its own reuses frame 1&apos;s region unchanged: nothing moves the region with the specimen. Where the specimen has moved out of that region, the part outside it is not measured from such a frame, and the nodes there stop being tracked or have their displacement copied from the nearest measured node. The run reports these nodes when it finishes.</source>
        <translation>没有自己感兴趣区域的参考帧会原样沿用第 1 帧的区域：该区域不会随试样移动。试样移出该区域后，区域外的部分不会以这类参考帧为基准进行测量，位于其中的节点将不再被追踪，或其位移从最近的已测量节点复制。运行结束时会报告这些节点。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="234"/>
        <source>To measure the whole specimen, give each reference frame its own Region of Interest that covers the specimen in that frame (Batch Import can assign Regions of Interest to many frames at once). A reference frame with a region of its own is measured on a grid that covers that region.</source>
        <translation>要测量整个试样，请为每个参考帧单独指定一个覆盖该帧中试样的感兴趣区域（批量导入可一次为多帧指定感兴趣区域）。拥有自己区域的参考帧会在覆盖该区域的测量网格上进行测量。</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="241"/>
        <source>Continue with frame 1&apos;s region?</source>
        <translation>是否沿用第 1 帧的区域继续？</translation>
    </message>
</context>
<context>
    <name>ROIHint</name>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="67"/>
        <source>Load images first, then draw a Region of Interest on frame 1.</source>
        <extracomment>The same card for other hints of the left sidebar (the model advice).</extracomment>
        <translation>请先加载图像，再在第 1 帧上绘制感兴趣区域。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="74"/>
        <source>&lt;b&gt;Accumulative mode&lt;/b&gt; — only frame 1 needs a Region of Interest. All later frames are compared against it directly.</source>
        <translation>&lt;b&gt;累积模式&lt;/b&gt; — 只有第 1 帧需要感兴趣区域。后续帧都直接与其比较。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="87"/>
        <source>&lt;b&gt;Incremental&lt;/b&gt; — only frame 1 needs a Region of Interest. RAFTcorr carries it to the reference frames (&lt;b&gt;%1&lt;/b&gt;) with the measured displacement. Draw on another frame only to replace that region or to leave out cracks and holes.</source>
        <translation>&lt;b&gt;增量模式&lt;/b&gt; — 只有第 1 帧需要感兴趣区域。RAFTcorr 会按测得的位移将其带到各参考帧（&lt;b&gt;%1&lt;/b&gt;）。仅在需要替换该帧的区域、或排除裂纹和孔洞时，才在其他帧上绘制。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="96"/>
        <source>Draw a Region of Interest on frame 1.</source>
        <translation>请在第 1 帧绘制感兴趣区域。</translation>
    </message>
</context>
<context>
    <name>ROIToolbar</name>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="72"/>
        <source>+ Add</source>
        <translation>+ 添加</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="74"/>
        <source>Add region to the Region of Interest (Polygon / Rectangle / Circle)</source>
        <translation>向感兴趣区域添加形状（多边形 / 矩形 / 圆形）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="79"/>
        <source>Cut</source>
        <translation>裁剪</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="81"/>
        <source>Cut region from the Region of Interest (Polygon / Rectangle / Circle)</source>
        <translation>从感兴趣区域裁剪形状（多边形 / 矩形 / 圆形）</translation>
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
        <translation>用画笔绘制额外的网格加密区域
（仅在第 1 帧可用 — 网格点会自动扭曲到后续帧）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="94"/>
        <source>Refine brush is only available on frame 1. Switch to frame 1 to paint refinement zones; they are automatically warped to later frames.</source>
        <translation>加密画笔仅在第 1 帧可用。切换到第 1 帧后可绘制加密区域；系统会自动将其扭曲到后续帧。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="118"/>
        <source>Import</source>
        <translation>导入</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="119"/>
        <source>Import mask from image file</source>
        <translation>从图像文件导入掩模</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="124"/>
        <source>Batch Import</source>
        <translation>批量导入</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="126"/>
        <source>Batch import mask files for multiple frames</source>
        <translation>批量导入多帧的掩模文件</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="139"/>
        <source>Save</source>
        <translation>保存</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="140"/>
        <source>Save current mask to PNG file</source>
        <translation>将当前掩模保存为 PNG 文件</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="145"/>
        <source>Invert</source>
        <translation>反选</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="147"/>
        <source>Invert the Region of Interest mask</source>
        <translation>反选感兴趣区域掩模</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="152"/>
        <source>Clear</source>
        <translation>清除</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="154"/>
        <source>Clear all Region of Interest masks</source>
        <translation>清除所有感兴趣区域掩模</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="186"/>
        <source>Radius</source>
        <translation>半径</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="204"/>
        <source>Paint</source>
        <translation>绘制</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="209"/>
        <source>Erase</source>
        <translation>擦除</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="219"/>
        <source>Clear Brush</source>
        <translation>清除画笔</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="247"/>
        <source>Polygon</source>
        <translation>多边形</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="251"/>
        <source>Rectangle</source>
        <translation>矩形</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="255"/>
        <source>Circle</source>
        <translation>圆形</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="259"/>
        <source>Circle (3-point)</source>
        <translation>圆形（三点）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="352"/>
        <source>Import Mask Image</source>
        <translation>导入掩模图像</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="354"/>
        <source>Images</source>
        <translation>图像</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="355"/>
        <source>All Files</source>
        <translation>所有文件</translation>
    </message>
</context>
<context>
    <name>RightSidebar</name>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="76"/>
        <source>Run DIC Analysis</source>
        <translation>运行 DIC 分析</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="89"/>
        <source>Cancel</source>
        <translation>取消</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="94"/>
        <source>Cancel the current analysis. Frames already computed are kept so you can review or export the partial run.</source>
        <translation>取消当前分析。已计算的帧会被保留，你可以查看或导出这部分结果。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="102"/>
        <source>Export Results</source>
        <translation>导出结果</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="110"/>
        <source>Open Strain Window</source>
        <translation>打开应变窗口</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="113"/>
        <source>Compute and visualize strain in a separate post-processing window. Requires displacement results from a completed Run.</source>
        <translation>在独立的后处理窗口中计算并可视化应变。需先完成一次运行以获得位移结果。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="122"/>
        <source>PROGRESS</source>
        <translation>进度</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="131"/>
        <location filename="../../gui/panels/right_sidebar.py" line="437"/>
        <source>Ready</source>
        <translation>就绪</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="139"/>
        <location filename="../../gui/panels/right_sidebar.py" line="439"/>
        <location filename="../../gui/panels/right_sidebar.py" line="547"/>
        <source>ELAPSED  %1</source>
        <translation>已用时间  %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="144"/>
        <location filename="../../gui/panels/right_sidebar.py" line="441"/>
        <location filename="../../gui/panels/right_sidebar.py" line="555"/>
        <location filename="../../gui/panels/right_sidebar.py" line="559"/>
        <source>REMAINING  %1</source>
        <translation>剩余时间  %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="152"/>
        <source>FIELD</source>
        <translation>场变量</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="161"/>
        <source>Show on</source>
        <translation>显示于</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="166"/>
        <source>Deformed frame</source>
        <translation>变形帧</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="167"/>
        <source>Reference frame</source>
        <translation>参考帧</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="169"/>
        <source>Plot the field where the material points are in the current frame, or at their positions in the reference frame.</source>
        <translation>将场绘制在材料点在当前帧中的位置，或绘制在其在参考帧中的位置。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="215"/>
        <source>Leave points without a result transparent, for example material that left the field of view, instead of filling them with the nearest value inside the region of interest. Also used for exported images and animations; data files always keep NaN there.</source>
        <translation>让没有结果的点保持透明（例如移出视野的材料），而不是用感兴趣区域内最近的值填充。导出的图像和动画同样遵守此设置；数据文件中这些点始终为 NaN。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="177"/>
        <source>Show background image</source>
        <translation>显示背景图像</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="180"/>
        <source>Uncheck to show the field on its own, with no speckle image behind it.</source>
        <translation>取消勾选可仅显示场，其后不显示散斑图像。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="196"/>
        <source>Hidden background</source>
        <translation>隐藏背景填充</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="198"/>
        <source>White</source>
        <translation>白色</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="199"/>
        <source>Black</source>
        <translation>黑色</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="200"/>
        <source>Transparent</source>
        <translation>透明</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="203"/>
        <source>What replaces the image when it is hidden. Transparency is kept for PNG and TIFF on export; other formats get white.</source>
        <translation>背景隐藏时用什么替代图像。导出时 PNG 和 TIFF 会保留透明度，其他格式改用白色。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="212"/>
        <source>Hide untracked regions</source>
        <translation>隐藏未追踪区域</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="225"/>
        <source>VISUALIZATION</source>
        <translation>可视化</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="230"/>
        <source>Colormap</source>
        <translation>颜色映射</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="249"/>
        <source>Opacity</source>
        <translation>不透明度</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="256"/>
        <source>Overlay opacity (0 = transparent, 100 = opaque)</source>
        <translation>叠加图不透明度（0 = 透明，100 = 不透明）</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="262"/>
        <source>PHYSICAL UNITS</source>
        <translation>物理单位</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="268"/>
        <source>LOG</source>
        <translation>日志</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="274"/>
        <source>Clear</source>
        <translation>清除</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="406"/>
        <source>Place at least one Starting Point in each red region before running (red = needs a Starting Point).</source>
        <translation>运行前请在每个红色区域放置至少一个种子点（红色 = 需要种子点）。</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="530"/>
        <source>%1  —  Frame %2</source>
        <translation>%1  —  第 %2 帧</translation>
    </message>
</context>
<context>
    <name>StrainController</name>
    <message>
        <location filename="../../gui/controllers/strain_controller.py" line="96"/>
        <source>Strain frame %1/%2</source>
        <translation>应变：帧 %1/%2</translation>
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
        <translation>应变</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="265"/>
        <source>Horizontal displacement u (+ right)</source>
        <translation>水平位移 u（向右为正）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="266"/>
        <source>Vertical displacement v (+ down)</source>
        <translation>垂直位移 v（向下为正）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="267"/>
        <source>Displacement magnitude |u|</source>
        <translation>位移大小 |u|</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="269"/>
        <source>Speed |du/dt|: central difference of the displacement over the neighbouring computed frames, times the frame rate</source>
        <translation>速度大小 |du/dt|：位移在相邻已计算帧之间的中心差分，乘以帧率</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="271"/>
        <source>Normal strain εxx</source>
        <translation>正应变 εxx</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="272"/>
        <source>Normal strain εyy</source>
        <translation>正应变 εyy</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="274"/>
        <source>Shear strain εxy (tensor component: half the engineering shear γxy)</source>
        <translation>剪应变 εxy（张量分量：工程剪应变 γxy 的一半）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="276"/>
        <source>Major principal strain ε₁</source>
        <translation>最大主应变 ε₁</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="277"/>
        <source>Minor principal strain ε₂</source>
        <translation>最小主应变 ε₂</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="279"/>
        <source>Maximum in-plane shear strain (ε₁ − ε₂) / 2 (tensor component: half the engineering shear)</source>
        <translation>最大面内剪应变 (ε₁ − ε₂) / 2（张量分量：工程剪应变的一半）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="282"/>
        <source>Von Mises equivalent strain (incompressible material)</source>
        <translation>von Mises 等效应变（不可压缩材料）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="284"/>
        <source>Rigid rotation in degrees (polar decomposition), clockwise on screen positive</source>
        <translation>刚体转动角，单位为度（极分解），屏幕上顺时针为正</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="287"/>
        <source>Direction of ε₁ in degrees from +x, clockwise on screen positive</source>
        <translation>ε₁ 的方向，从 +x 轴起算，单位为度，屏幕上顺时针为正</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="290"/>
        <source>Relative area change ΔA/A₀ = det F − 1</source>
        <translation>相对面积变化 ΔA/A₀ = det F − 1</translation>
    </message>
</context>
<context>
    <name>StrainNavigator</name>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="73"/>
        <source>Previous frame</source>
        <translation>上一帧</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="84"/>
        <location filename="../../gui/widgets/strain_navigator.py" line="216"/>
        <source>Play animation</source>
        <translation>播放动画</translation>
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
        <translation>下一帧</translation>
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
        <translation>帧 0/0</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="205"/>
        <source>⏸</source>
        <translation>⏸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="206"/>
        <source>Pause animation</source>
        <translation>暂停动画</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="223"/>
        <source>FRAME %1/%2</source>
        <translation>帧 %1/%2</translation>
    </message>
</context>
<context>
    <name>StrainParamPanel</name>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="88"/>
        <source>Plane fitting</source>
        <extracomment>Strain measures in combo order (pyALDIC&apos;s order: infinitesimal, Eulerian, Green-Lagrange). The configuration in the controls changed (edited by the user).</extracomment>
        <translation>平面拟合</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="91"/>
        <source>Polynomial fitted to each displacement component over the VSG window. Its slopes at the centre are the displacement gradients.
• Plane fitting (1st order): the standard VSG (default).
• Quadratic fitting (2nd order): follows strain gradients inside the window better, at the cost of more noise.</source>
        <translation>在 VSG 窗口内对每个位移分量拟合多项式，其在中心处的斜率即为位移梯度。
• 平面拟合（一阶）：标准 VSG（默认）。
• 二次拟合（二阶）：能更好地跟随窗口内的应变梯度，但噪声更大。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="97"/>
        <source>Method</source>
        <translation>方法</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="112"/>
        <source>VSG size</source>
        <translation>VSG 尺寸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="123"/>
        <source>Hides low-confidence strain along the edges of the region of interest, holes and cracks, where the edge cuts the VSG window and the local fit becomes one-sided and unreliable.

• Coefficient × VSG half-width (Lg − 1) / 2 = width of the trimmed band.
• 0.00 = keep every point (no trimming).
• 0.70 = default.
• 1.00 = strictest (trim every point whose window touches an edge).

Trimmed points are empty (NaN) in exported data files.</source>
        <translation>隐藏感兴趣区域、孔洞和裂纹边缘处低置信度的应变：那里边缘截断了 VSG 窗口，局部拟合变成单边、不可靠。

• 系数 × VSG 半宽 (Lg − 1) / 2 = 裁剪带的宽度。
• 0.00 = 保留所有点（不裁剪）。
• 0.70 = 默认值。
• 1.00 = 最严格（窗口一旦触及边缘即裁剪）。

被裁剪的点在导出的数据文件中为空 (NaN)。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="154"/>
        <source>Weight of the points inside the VSG window: all equal (uniform, default), or decreasing with the distance from the centre (Gaussian).</source>
        <translation>VSG 窗口内各点的权重：全部相等（均匀，默认），或随离中心的距离递减（高斯）。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="189"/>
        <source>Off</source>
        <comment>Strain smoothing preset</comment>
        <translation>关闭</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="89"/>
        <source>Quadratic fitting</source>
        <translation>二次拟合</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="105"/>
        <source>VSG (Virtual Strain Gauge) size Lg is the side, in pixels, of the square window around each point over which the displacement is fitted. Strain is then taken from the fit&apos;s slopes.

• Larger VSG → smoother strain, lower spatial resolution.
• Smaller VSG → sharper strain, more noise.
• Odd, so that the window has a centre pixel (default: 41 px).</source>
        <translation>VSG（虚拟应变计）尺寸 Lg 是每个点周围用于拟合位移的正方形窗口的边长（像素）。应变由拟合的斜率得到。

• VSG 越大 → 应变越平滑，空间分辨率越低。
• VSG 越小 → 应变越锐利，噪声越大。
• 取奇数，使窗口有中心像素（默认：41 px）。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="133"/>
        <source>Trim low-confidence edges</source>
        <translation>裁剪低置信度边缘</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="151"/>
        <source>Uniform</source>
        <translation>均匀</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="152"/>
        <source>Gaussian</source>
        <translation>高斯</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="158"/>
        <source>Weighting</source>
        <translation>加权</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="166"/>
        <source>Auto (Lg / 4)</source>
        <translation>自动（Lg / 4）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="168"/>
        <source>Standard deviation of the Gaussian weight. Auto uses a quarter of the VSG size.</source>
        <translation>高斯权重的标准差。“自动”取 VSG 尺寸的四分之一。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="171"/>
        <source>Gaussian σ</source>
        <translation>高斯 σ</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="178"/>
        <source>Spacing of the strain points: 1 computes strain at every pixel; larger values are faster and lighter for long sequences.</source>
        <translation>应变点的间距：1 表示在每个像素上计算应变；更大的值对长序列更快、占用更少。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="181"/>
        <source>Output spacing</source>
        <translation>输出间距</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="191"/>
        <source>Gaussian smoothing (σ in pixels) of the displacement used by the strain fit only; the displacement results are not changed. Off (0) differentiates the measured displacement as it is (default).</source>
        <translation>仅对用于应变拟合的位移做高斯平滑（σ 以像素计），位移结果本身不变。关闭（0）时直接对测得的位移求导（默认）。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="195"/>
        <source>Pre-fit smoothing</source>
        <translation>拟合前平滑</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="198"/>
        <source>Never fit across cracks or holes</source>
        <translation>拟合不跨越裂纹或孔洞</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="200"/>
        <source>Fit each connected part of the tracked region separately, so no VSG window ever spans a crack, a hole or the gap between two regions.</source>
        <translation>对追踪区域的每个连通部分分别拟合，使任何 VSG 窗口都不会跨越裂纹、孔洞或两个区域之间的间隙。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="209"/>
        <source>Infinitesimal</source>
        <translation>无穷小应变</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="210"/>
        <source>Eulerian</source>
        <translation>欧拉应变</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="211"/>
        <source>Green-Lagrange</source>
        <translation>格林-拉格朗日应变</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="213"/>
        <source>Strain measure from the deformation gradient F = I + ∇u:
• Infinitesimal: (∇u + ∇uᵀ) / 2 (small strains only).
• Eulerian (Euler-Almansi): (I − (F Fᵀ)⁻¹) / 2.
• Green-Lagrange: (Fᵀ F − I) / 2 (default).</source>
        <translation>由变形梯度 F = I + ∇u 得到的应变度量：
• 无穷小应变：(∇u + ∇uᵀ) / 2（仅适用于小应变）。
• 欧拉应变（Euler-Almansi）：(I − (F Fᵀ)⁻¹) / 2。
• 格林-拉格朗日应变：(Fᵀ F − I) / 2（默认）。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="218"/>
        <source>Strain type</source>
        <translation>应变类型</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="272"/>
        <source>Trimmed: %1 points (%2%)</source>
        <translation>已裁剪：%1 个点 (%2%)</translation>
    </message>
</context>
<context>
    <name>StrainVizPanel</name>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="67"/>
        <source>Deformed frame</source>
        <extracomment>Alias so existing references read unchanged; the list itself lives in raftcorr.gui.support.colormaps, so every chooser offers the same options.</extracomment>
        <translation>变形帧</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="68"/>
        <source>Reference frame</source>
        <translation>参考帧</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="70"/>
        <source>Plot the field where the material points are in the current frame, or at their positions in the reference frame.</source>
        <translation>将场绘制在材料点在当前帧中的位置，或绘制在其在参考帧中的位置。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="73"/>
        <source>Show on</source>
        <translation>显示于</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="75"/>
        <source>Show background image</source>
        <translation>显示背景图像</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="78"/>
        <source>Uncheck to show the field on its own, with no speckle image behind it.</source>
        <translation>取消勾选可仅显示场，其后不显示散斑图像。</translation>
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
        <translation>背景隐藏时用什么替代图像。导出时 PNG 和 TIFF 会保留透明度，其他格式改用白色。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="141"/>
        <source>Fill trimmed edges (display only)</source>
        <translation>填充裁剪的边缘（仅显示）</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="144"/>
        <source>Fill the edge-trimmed strain band from the nearest reliable point of the same region. Affects the view and exported images and animations; exported data files always keep the trimmed band as NaN.</source>
        <translation>用同一区域内最近的可靠点填充边缘被裁剪的应变带。影响显示以及导出的图像和动画；导出的数据文件始终将裁剪带保留为 NaN。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="151"/>
        <source>Hide untracked regions</source>
        <translation>隐藏未追踪区域</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="154"/>
        <source>Leave points without a result transparent, for example material that left the field of view, instead of filling them with the nearest value inside the region of interest. Also used for exported images and animations; data files always keep NaN there.</source>
        <translation>让没有结果的点保持透明（例如移出视野的材料），而不是用感兴趣区域内最近的值填充。导出的图像和动画同样遵守此设置；数据文件中这些点始终为 NaN。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="81"/>
        <source>Background</source>
        <translation>背景</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="93"/>
        <source>Hidden background</source>
        <translation>隐藏背景填充</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="100"/>
        <source>Colormap</source>
        <translation>颜色映射</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="104"/>
        <source>Range</source>
        <translation>范围</translation>
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
        <translation>边缘</translation>
    </message>
</context>
<context>
    <name>StrainWindow</name>
    <message>
        <location filename="../../gui/strain_window.py" line="125"/>
        <source>Strain Post-Processing</source>
        <translation>应变后处理</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="210"/>
        <source>Fit</source>
        <translation>适配</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="211"/>
        <source>Fit image to viewport</source>
        <translation>将图像适配到视口</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="217"/>
        <source>100%</source>
        <translation>100%</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="218"/>
        <source>Zoom to 100% (1:1)</source>
        <translation>缩放到 100%（1:1）</translation>
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
        <translation>应变参数</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="293"/>
        <source>ADVANCED</source>
        <translation>高级</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="308"/>
        <source>Cancel</source>
        <translation>取消</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="312"/>
        <source>Cancel the running strain computation. The previous strain result is kept.</source>
        <translation>取消正在进行的应变计算。保留之前的应变结果。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="324"/>
        <source>Export displacement and strain results to NPZ / MAT / CSV / PNG</source>
        <translation>将位移和应变结果导出为 NPZ / MAT / CSV / PNG</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="354"/>
        <source>FIELD</source>
        <translation>场变量</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="361"/>
        <source>VISUALIZATION</source>
        <translation>可视化</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="373"/>
        <source>PHYSICAL UNITS</source>
        <translation>物理单位</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="379"/>
        <source>LOG</source>
        <translation>日志</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="861"/>
        <location filename="../../gui/strain_window.py" line="899"/>
        <source>No displacement results.</source>
        <translation>没有位移结果。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="901"/>
        <source>Frame %1 was not computed.</source>
        <translation>第 %1 帧未计算。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="909"/>
        <source>Click Compute Strain to show this field.</source>
        <translation>点击“计算应变”以显示此场。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="910"/>
        <source>No strain for frame %1: click Compute Strain.</source>
        <translation>第 %1 帧没有应变：请点击“计算应变”。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="912"/>
        <source>Velocity needs at least two computed frames.</source>
        <translation>速度至少需要两个已计算帧。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="913"/>
        <source>No valid values in frame %1.</source>
        <translation>第 %1 帧没有有效值。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="170"/>
        <source>Strain Field</source>
        <translation>应变场</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="419"/>
        <source>Analysis is not available</source>
        <translation>分析功能不可用</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="422"/>
        <location filename="../../gui/strain_window.py" line="425"/>
        <source>Analysis</source>
        <translation>分析</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="423"/>
        <source>Analysis is not available (see the application log).</source>
        <translation>分析功能不可用（详见应用程序日志）。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="534"/>
        <source>Strain compute failed: %1: %2</source>
        <translation>应变计算失败：%1：%2</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="540"/>
        <location filename="../../gui/strain_window.py" line="603"/>
        <source>Strain computation complete.</source>
        <translation>应变计算完成。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="551"/>
        <source>Strain window: no displacement results to post-process.</source>
        <translation>应变窗口：没有可后处理的位移结果。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="596"/>
        <source>The displacement results changed during the strain computation; its result was discarded.</source>
        <translation>应变计算期间位移结果发生了变化；本次应变结果已丢弃。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="620"/>
        <source>Strain computation canceled.</source>
        <translation>应变计算已取消。</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="627"/>
        <source>Strain compute failed: %1</source>
        <translation>应变计算失败：%1</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="631"/>
        <source>Strain Computation Failed</source>
        <translation>应变计算失败</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="671"/>
        <location filename="../../gui/strain_window.py" line="673"/>
        <source>Export is not available: %1</source>
        <translation>导出功能不可用：%1</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="685"/>
        <source>⚠ Params changed -- click Compute Strain</source>
        <translation>⚠ 参数已变更 — 请点击“计算应变”</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="227"/>
        <source>Zoom out</source>
        <translation>缩小</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="299"/>
        <source>Compute Strain</source>
        <translation>计算应变</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="321"/>
        <location filename="../../gui/strain_window.py" line="672"/>
        <source>Export Results</source>
        <translation>导出结果</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="563"/>
        <source>Starting…</source>
        <translation>开始中…</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="581"/>
        <source>Canceling…</source>
        <translation>正在取消…</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="600"/>
        <source>Complete</source>
        <translation>已完成</translation>
    </message>
</context>
<context>
    <name>VelocitySettingsWidget</name>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="47"/>
        <source>Use physical units</source>
        <translation>使用物理单位</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="69"/>
        <source>/ px</source>
        <translation>/ px</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/velocity_settings.py" line="84"/>
        <source>Unit: px/frame</source>
        <translation>单位：px/帧</translation>
    </message>
</context>
<context>
    <name>WorkflowTypePanel</name>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="59"/>
        <source>Incremental</source>
        <translation>增量式</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="58"/>
        <source>Accumulative</source>
        <translation>累积式</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="73"/>
        <source>Tracking Mode</source>
        <translation>追踪模式</translation>
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
        <translation>累积式：每帧都直接与第 1 帧比较。
只要运动不超出模型量程，就优先使用：
误差不会逐帧累积。

增量式：每帧与第 1 帧之后的某个参考帧比较，
再将各步结果串接。当相对第 1 帧的运动
超出模型量程（大拉伸或大转动）时需要使用；每一步
都会引入自己的误差。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="85"/>
        <source>finest spatial resolution; small displacements</source>
        <translation>空间分辨率最高；适合小位移</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="86"/>
        <source>fine resolution, several times faster than s1; small to moderate displacements</source>
        <translation>分辨率高，速度是 s1 的数倍；适合小到中等位移</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="87"/>
        <source>balanced resolution and capture range</source>
        <translation>分辨率与量程均衡</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="88"/>
        <source>largest capture range, coarsest resolution; large motion and rotation</source>
        <translation>量程最大，分辨率最粗；适合大运动和大转动</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="92"/>
        <source>%1: %2.
Reach (nominal lookup radius 32 s): %3 px.</source>
        <translation>%1：%2。
量程（名义查找半径 32 s）：%3 px。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="95"/>
        <source>Weights not installed.</source>
        <translation>权重文件未安装。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="100"/>
        <source>RAFTcorr model. A smaller s resolves finer detail; a larger s
reaches further (the coarsest correlation level looks 32 s px away).

s1, s2: small displacements, strain concentrations.
s4: balanced.
s8: large motion or rotation, or no prior knowledge.</source>
        <translation>RAFTcorr 模型。s 越小，可分辨的细节越精细；s 越大，
量程越大（最粗相关层级的查找距离为 32 s px）。

s1、s2：小位移、应变集中。
s4：均衡。
s8：大运动或大转动，或无先验信息。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="107"/>
        <source>Model</source>
        <translation>模型</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="115"/>
        <source>Suggest a Model</source>
        <translation>推荐模型</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="173"/>
        <source>e.g. 5, 10, 20</source>
        <translation>例如 5, 10, 20</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="175"/>
        <source>Frame numbers (as in the image list) to use as reference frames, separated by commas. Frame 1 is always the first reference.</source>
        <translation>用作参考帧的帧号（与图像列表一致），以逗号分隔。第 1 帧始终是第一个参考帧。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="272"/>
        <source>Invalid reference frames: use frame numbers from 1 on, separated by commas.</source>
        <translation>参考帧无效：请使用从 1 开始的帧号，以逗号分隔。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="118"/>
        <source>Measure the motion of the sequence quickly with s8 and pick the
finest model whose reach covers it.</source>
        <translation>先用 s8 快速测量序列的运动，再选出量程足以
覆盖该运动的最精细模型。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="140"/>
        <source>Every Frame</source>
        <translation>每帧</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="141"/>
        <source>Every N Frames</source>
        <translation>每 N 帧</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="142"/>
        <source>Custom Frames</source>
        <translation>自定义帧</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="145"/>
        <source>When the reference frame refreshes during incremental tracking.
Every Frame: reset reference every frame (smallest per-step displacement,
most robust for large deformation).
Every N Frames: reset every N frames (balance speed vs robustness).
Custom Frames: user-defined list of reference frame indices.</source>
        <translation>增量追踪中参考帧的刷新策略。
每帧：每帧都更新参考（单步位移最小，
对大变形最稳健）。
每 N 帧：每 N 帧更新一次（速度与稳健性的折中）。
自定义帧：由用户指定参考帧索引列表。</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="154"/>
        <source>Reference Update</source>
        <translation>参考帧更新</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="164"/>
        <source>Update reference every N frames</source>
        <translation>每 N 帧更新一次参考帧</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="166"/>
        <source>Interval</source>
        <translation>间隔</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="179"/>
        <source>Reference Frames</source>
        <translation>参考帧列表</translation>
    </message>
</context>
<context>
    <name>_DropZone</name>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="118"/>
        <source>Drop image folder
or Browse</source>
        <translation>拖入图像文件夹
或点击浏览</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="128"/>
        <source>Select Image Folder</source>
        <translation>选择图像文件夹</translation>
    </message>
</context>
<context>
    <name>_MaskPreviewPanel</name>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="133"/>
        <source>Preview</source>
        <translation>预览</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="137"/>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="263"/>
        <source>(no image)</source>
        <translation>（无图像）</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="157"/>
        <source>Image only</source>
        <translation>仅图像</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="158"/>
        <source>Image + Mask</source>
        <translation>图像 + 掩模</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="159"/>
        <source>Mask only</source>
        <translation>仅掩模</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="162"/>
        <source>View:</source>
        <translation>视图：</translation>
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
        <translation>蓝色</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="180"/>
        <source>Red</source>
        <comment>Mask overlay color</comment>
        <translation>红色</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="184"/>
        <source>Green</source>
        <comment>Mask overlay color</comment>
        <translation>绿色</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="188"/>
        <source>Yellow</source>
        <comment>Mask overlay color</comment>
        <translation>黄色</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="192"/>
        <source>Mask color:</source>
        <translation>掩模颜色：</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="269"/>
        <source>No mask assigned</source>
        <translation>未分配掩模</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="275"/>
        <source>Frame %1 — %2</source>
        <translation>帧 %1 — %2</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="290"/>
        <source>Failed to load image</source>
        <translation>无法加载图像</translation>
    </message>
</context>
<context>
    <name>dialog</name>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="322"/>
        <source>Cancel Export</source>
        <translation>取消导出</translation>
    </message>
</context>
</TS>
