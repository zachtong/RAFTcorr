<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1" language="ko" sourcelanguage="en_US">
<context>
    <name>AdvancedPanel</name>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="148"/>
        <source>Margin</source>
        <translation>마진</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="167"/>
        <source>Fraction of the free GPU memory one window may use.
Large regions are cut into tiles to fit it.</source>
        <translation>창 하나가 사용할 수 있는 여유 GPU 메모리 비율입니다.
큰 영역은 이에 맞도록 타일로 나뉩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="149"/>
        <source>Must cover the largest displacement of the region of interest.</source>
        <translation>관심 영역 내 최대 변위를 포함해야 합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="152"/>
        <source>Use the Model Reach</source>
        <translation>모델 도달 범위 사용</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="155"/>
        <source>Extra image context around the region of interest so that displaced points stay inside
the window given to the network. Automatic: the model&apos;s reach (32 s px).</source>
        <translation>변위된 점이 네트워크에 입력되는 창 안에 머물도록 관심 영역 주위에
추가로 포함하는 이미지 영역입니다. 자동: 모델 도달 범위(32 s px).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="166"/>
        <source>GPU Memory</source>
        <translation>GPU 메모리</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="174"/>
        <source>Tile Blending</source>
        <translation>타일 블렌딩</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="175"/>
        <source>Width of the feathered overlap between neighbouring tiles.</source>
        <translation>인접 타일 사이에서 페더링되는 겹침 영역의 너비입니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="178"/>
        <source>Automatic</source>
        <translation>자동</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="179"/>
        <source>On Demand</source>
        <translation>필요 시 계산</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="180"/>
        <source>Stored Volume</source>
        <translation>볼륨 저장</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="182"/>
        <source>Correlation</source>
        <translation>상관 계산</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="183"/>
        <source>How the correlation lookup is computed; the results are identical.
Automatic: stored volume for s8 (fast), on demand for s1-s4
(memory linear in the image size).</source>
        <translation>상관 조회를 계산하는 방식이며, 결과는 동일합니다.
자동: s8은 볼륨 저장(빠름), s1-s4는 필요 시 계산
(메모리가 이미지 크기에 비례).</translation>
    </message>
</context>
<context>
    <name>AnalysisChart</name>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="75"/>
        <source>too few valid points</source>
        <translation>유효한 점이 너무 적음</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="73"/>
        <source>partly untracked</source>
        <translation>일부 추적 안 됨</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="77"/>
        <source>no strain value (sparse strain gauge)</source>
        <translation>변형률 값 없음(VSG 내 점 부족)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="80"/>
        <source>gauge endpoint lost</source>
        <translation>게이지 끝점 상실</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="82"/>
        <source>not computed</source>
        <translation>계산되지 않음</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/mpl_chart.py" line="83"/>
        <source>no data</source>
        <translation>데이터 없음</translation>
    </message>
</context>
<context>
    <name>AnalysisTab</name>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="82"/>
        <source>Point</source>
        <comment>Placement tool: a single location</comment>
        <translation>점</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="84"/>
        <source>Line</source>
        <comment>Placement tool: a two-point gauge</comment>
        <translation>선분</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="86"/>
        <source>Rectangle</source>
        <comment>Placement tool</comment>
        <translation>사각형</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="88"/>
        <source>Circle</source>
        <comment>Placement tool</comment>
        <translation>원</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="90"/>
        <source>Polygon</source>
        <comment>Placement tool</comment>
        <translation>다각형</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="102"/>
        <source>Click once to place a point probe.</source>
        <translation>한 번 클릭하여 점 프로브를 배치합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="108"/>
        <source>Click twice: opposite corners.</source>
        <translation>두 번 클릭: 마주 보는 두 모서리.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="112"/>
        <source>Click each vertex, then double-click to close.</source>
        <translation>각 꼭짓점을 클릭한 뒤 두 번 클릭하여 닫습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="170"/>
        <source>partly untracked from frame %1</source>
        <translation>%1 프레임부터 일부 추적 안 됨</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="179"/>
        <source>gaps: no strain value</source>
        <translation>공백: 변형률 값 없음</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="201"/>
        <source>Nothing valid along %1: the strain gauge found too few points there (near an edge or a hole). Plot a displacement, or use a smaller gauge on the Strain Field tab.</source>
        <translation>%1을(를) 따라 유효한 데이터가 없습니다: 가장자리나 구멍 근처라 VSG 안의 점이 너무 적습니다. 변위를 표시하거나 &apos;변형률장&apos; 탭에서 VSG 크기를 줄이십시오.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="207"/>
        <source>Nothing valid along %1: the material under it is untracked (cut out of the frame&apos;s ROI, or out of view).</source>
        <translation>%1을(를) 따라 유효한 데이터가 없습니다: 그 아래의 재료가 추적되지 않습니다(해당 프레임의 ROI에서 제외되었거나 시야 밖으로 나갔습니다).</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="89"/>
        <source>Show</source>
        <comment>Probe list column: visibility checkbox</comment>
        <translation>표시</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="91"/>
        <source>Name</source>
        <comment>Probe list column: the probe&apos;s label</comment>
        <translation>이름</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="93"/>
        <source>Type</source>
        <comment>Probe list column: point, line or region</comment>
        <translation>종류</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="95"/>
        <source>Color</source>
        <comment>Probe list column: color swatch</comment>
        <translation>색상</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="100"/>
        <source>Color…</source>
        <translation>색상…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="97"/>
        <source>Note</source>
        <comment>Probe list column: why a probe shows gaps</comment>
        <translation>비고</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="101"/>
        <source>Delete</source>
        <comment>Button: delete the selected probe</comment>
        <translation>삭제</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="103"/>
        <source>Clear All</source>
        <translation>모두 지우기</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="183"/>
        <source>Statistic:</source>
        <translation>통계량:</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="56"/>
        <source>Mean</source>
        <comment>Statistic: arithmetic mean</comment>
        <translation>평균</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="58"/>
        <source>Median</source>
        <comment>Statistic</comment>
        <translation>중앙값</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="59"/>
        <source>Maximum</source>
        <comment>Statistic</comment>
        <translation>최댓값</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="60"/>
        <source>Minimum</source>
        <comment>Statistic</comment>
        <translation>최솟값</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="61"/>
        <source>Standard deviation</source>
        <translation>표준편차</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="62"/>
        <source>Valid fraction</source>
        <translation>유효 비율</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="46"/>
        <source>Crack opening</source>
        <translation>균열 개구 변위</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="489"/>
        <source>Strain has not been computed yet. Compute it on the Strain Field tab, or plot a displacement.</source>
        <translation>변형률이 아직 계산되지 않았습니다. &apos;변형률장&apos; 탭에서 계산하거나 변위를 표시하십시오.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="533"/>
        <source>Gauge quantities need a line probe.</source>
        <translation>게이지 양에는 선 프로브가 필요합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="530"/>
        <source>No visible probe can show this quantity.</source>
        <translation>이 양을 표시할 수 있는 보이는 프로브가 없습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="166"/>
        <source>no valid data: %1</source>
        <translation>유효한 데이터 없음: %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="174"/>
        <source>endpoint lost from frame %1</source>
        <translation>%1 프레임부터 끝점 상실</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="177"/>
        <source>gaps: too few valid points</source>
        <translation>공백: 유효 점이 너무 적음</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="185"/>
        <source>not plotted: gauges need a line</source>
        <translation>표시 안 됨: 게이지에는 선이 필요</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="188"/>
        <source>not plotted: one point has no spread or coverage</source>
        <translation>표시 안 됨: 한 점에는 분산이나 적용 범위가 없음</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="190"/>
        <source>not plotted</source>
        <translation>표시 안 됨</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="513"/>
        <source>Added probe &apos;%1&apos;.</source>
        <translation>프로브 &apos;%1&apos;을(를) 추가했습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="599"/>
        <source>Clear All Probes</source>
        <translation>모든 프로브 지우기</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="600"/>
        <source>Delete every probe? This cannot be undone.</source>
        <translation>모든 프로브를 삭제하시겠습니까? 되돌릴 수 없습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="110"/>
        <source>Point</source>
        <comment>Probe type</comment>
        <translation>점</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="111"/>
        <source>Line</source>
        <comment>Probe type</comment>
        <translation>선분</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/probe_table.py" line="112"/>
        <source>Region</source>
        <comment>Probe type: an enclosed area</comment>
        <translation>영역</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="480"/>
        <source>Run a DIC analysis to plot probes.</source>
        <translation>프로브를 그리려면 먼저 DIC 분석을 실행하세요.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="92"/>
        <source>Virtual extensometer</source>
        <comment>Placement tool</comment>
        <translation>가상 신율계</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="94"/>
        <source>Crack gauge</source>
        <comment>Placement tool: a line across a crack</comment>
        <translation>균열 게이지</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="104"/>
        <source>Click twice: start and end. A line is also a virtual extensometer and a crack-opening gauge.</source>
        <translation>두 번 클릭: 시작점과 끝점. 선은 가상 신율계이자 균열 개구 게이지이기도 합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="110"/>
        <source>Click twice: center, then the edge.</source>
        <translation>두 번 클릭: 중심, 그다음 가장자리.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="114"/>
        <source>Click the two gauge points. The chart then shows the strain between them.</source>
        <translation>두 개의 표점을 클릭하십시오. 차트에 두 점 사이의 변형률이 표시됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="118"/>
        <source>Click one point on each side of the crack. The chart then shows how far it opens.</source>
        <translation>균열 양쪽에 한 점씩 클릭하십시오. 차트에 균열이 벌어진 정도가 표시됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="124"/>
        <source>Fit</source>
        <comment>Zoom button: fit the image to the view</comment>
        <translation>맞춤</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="131"/>
        <source>Fit image to viewport</source>
        <translation>이미지를 뷰포트에 맞춤</translation>
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
        <translation>100%(1:1) 확대</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="136"/>
        <source>Zoom in</source>
        <translation>확대</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="137"/>
        <source>Zoom out</source>
        <translation>축소</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="138"/>
        <source>Show field</source>
        <comment>Analysis canvas: color the image by the field</comment>
        <translation>필드 표시</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/canvas_panel.py" line="140"/>
        <source>Color the reference image with the plotted field at the current frame. For a gauge reading, the Strain Field tab&apos;s field is shown.</source>
        <translation>현재 프레임에서 표시 중인 양의 필드로 참조 이미지를 색칠합니다. 게이지 판독값의 경우 &apos;변형률장&apos; 탭의 필드를 표시합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="181"/>
        <source>Plot:</source>
        <translation>표시:</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="184"/>
        <source>X axis:</source>
        <translation>X 축:</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="185"/>
        <source>Strain as:</source>
        <translation>변형률 표시:</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="187"/>
        <source>Min. valid fraction:</source>
        <translation>최소 유효 비율:</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="188"/>
        <source>A frame is left blank when fewer than this fraction of a line&apos;s or region&apos;s points are reliable. Guards against a curve that stays smooth while its sample shrinks away.</source>
        <translation>선이나 영역에서 신뢰할 수 있는 점의 비율이 이 값보다 낮으면 해당 프레임은 비워 둡니다. 표본이 줄어드는데도 곡선이 매끄럽게 유지되는 것을 방지합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="195"/>
        <source>Over time</source>
        <comment>Chart view: every frame of each probe</comment>
        <translation>시간 변화</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="197"/>
        <source>Each probe&apos;s reading at every frame.</source>
        <translation>각 프로브의 모든 프레임에서의 판독값.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="200"/>
        <source>Along the line</source>
        <comment>Chart view: a profile</comment>
        <translation>선을 따른 분포</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="207"/>
        <source>Kymograph</source>
        <comment>Chart view: distance against frame</comment>
        <translation>키모그래프</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="202"/>
        <source>The field along the selected line at the current frame, over the other frames in gray.</source>
        <translation>현재 프레임에서 선택한 선을 따른 필드이며, 다른 프레임은 뒤에 회색으로 표시됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="209"/>
        <source>The field along the selected line at every frame: distance against frame, value as color.</source>
        <translation>모든 프레임에서 선택한 선을 따른 필드: 거리 대 프레임, 값은 색으로 표시.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="214"/>
        <source>Stress–strain</source>
        <comment>Chart view: stress against strain</comment>
        <translation>응력–변형률</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="216"/>
        <source>Stress (or load, without A0) against the plotted quantity, one curve per probe: stress-strain with an extensometer, load against opening with a crack gauge. Needs load data.</source>
        <translation>표시 중인 양에 대한 응력(A0가 없으면 하중) 곡선을 프로브마다 그립니다: 신율계면 응력–변형률, 균열 게이지면 하중–개구 곡선입니다. 하중 데이터가 필요합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="227"/>
        <source>Other frames</source>
        <translation>다른 프레임</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="228"/>
        <source>Draw the other frames&apos; profiles faintly behind the current one (at most twelve, evenly spaced).</source>
        <translation>현재 프레임 뒤에 다른 프레임의 분포를 흐리게 그립니다(최대 12개, 균등 간격).</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="233"/>
        <source>Line data (CSV)…</source>
        <translation>선 데이터(CSV)…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="234"/>
        <source>Export</source>
        <translation>내보내기</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="235"/>
        <source>Load data…</source>
        <translation>하중 데이터…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="236"/>
        <source>Import a testing machine&apos;s load record (CSV) to plot against load or stress, and to draw stress-strain curves.</source>
        <translation>시험기의 하중 기록(CSV)을 가져와 하중 또는 응력에 대해 그래프를 그리고 응력–변형률 곡선을 그립니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="241"/>
        <source>Probe data (CSV)…</source>
        <translation>프로브 데이터(CSV)…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="243"/>
        <source>Chart image…</source>
        <translation>차트 이미지…</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="245"/>
        <source>Copy chart</source>
        <translation>차트 복사</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="247"/>
        <source>Copy plotted data</source>
        <translation>표시된 데이터 복사</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="559"/>
        <source>Import the testing machine&apos;s load record with Load data… to draw stress-strain curves.</source>
        <translation>응력–변형률 곡선을 그리려면 &apos;하중 데이터…&apos;로 시험기의 하중 기록을 가져오십시오.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="31"/>
        <source>Displacement U</source>
        <translation>변위 U</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="32"/>
        <source>Displacement V</source>
        <translation>변위 V</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="33"/>
        <source>Displacement magnitude</source>
        <translation>변위 크기</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="42"/>
        <source>Extensometer strain</source>
        <translation>신율계 변형률</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="43"/>
        <source>Extensometer true strain</source>
        <translation>신율계 진변형률</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="45"/>
        <source>Elongation ΔL</source>
        <translation>신장량 ΔL</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="47"/>
        <source>Crack sliding</source>
        <translation>균열 미끄럼</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="48"/>
        <source>Crack opening magnitude</source>
        <translation>균열 개구 변위 크기</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="279"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="434"/>
        <source>Frame</source>
        <translation>프레임</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="281"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="429"/>
        <source>Time (s)</source>
        <translation>시간 (s)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="283"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="431"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="574"/>
        <source>Load (N)</source>
        <translation>하중 (N)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="286"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="433"/>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="571"/>
        <source>Stress (MPa)</source>
        <translation>응력 (MPa)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="295"/>
        <source>ratio</source>
        <comment>Strain display unit: plain number</comment>
        <translation>비율</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="420"/>
        <source>Could not draw the field: %1</source>
        <translation>필드를 그릴 수 없습니다: %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="130"/>
        <source>Esc cancels placement</source>
        <translation>Esc 키로 배치 취소</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="138"/>
        <source>Drag to move the probe, or drag a handle to reshape it. Delete removes it; F2 renames it.</source>
        <translation>드래그하여 프로브를 이동하거나 핸들을 드래그하여 모양을 바꿉니다. Delete로 삭제하고 F2로 이름을 바꿉니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="485"/>
        <source>Place a probe on the reference image to begin.</source>
        <translation>참조 이미지에 프로브를 배치하면 시작됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="212"/>
        <source>Nothing valid along %1: it lies off the measured area.</source>
        <translation>%1을(를) 따라 유효한 데이터가 없습니다: 측정 영역 밖에 있습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="610"/>
        <source>A line view shows a field. Choose a field to plot.</source>
        <translation>선 보기는 필드를 표시합니다. 표시할 필드를 선택하십시오.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="614"/>
        <source>Place a line probe, or select one, to see the field along it.</source>
        <translation>선 프로브를 배치하거나 선택하면 선을 따른 필드를 볼 수 있습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="632"/>
        <source>Distance along %1 (%2)</source>
        <translation>%1을(를) 따른 거리(%2)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/chart_panel.py" line="641"/>
        <source>%1, frame %2</source>
        <translation>%1, %2 프레임</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="153"/>
        <source>not plotted: off the measured area</source>
        <translation>표시 안 됨: 측정 영역 밖</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/text.py" line="156"/>
        <source>not plotted: a gauge end is off the measured area</source>
        <translation>표시 안 됨: 게이지 끝점이 측정 영역 밖</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="671"/>
        <source>Export Probe Data</source>
        <translation>프로브 데이터 내보내기</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="664"/>
        <source>CSV Files</source>
        <translation>CSV 파일</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="442"/>
        <source>The load data cannot be matched to the frames: %1</source>
        <translation>하중 데이터를 프레임에 맞출 수 없습니다: %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="664"/>
        <location filename="../../gui/panels/analysis/tab.py" line="728"/>
        <source>All Files</source>
        <translation>모든 파일</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="699"/>
        <source>Probe export failed: %1</source>
        <translation>프로브 내보내기 실패: %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="701"/>
        <source>Probe data written to %1</source>
        <translation>프로브 데이터를 %1에 기록했습니다</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="709"/>
        <source>Export Line Data</source>
        <translation>선 데이터 내보내기</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="718"/>
        <source>Line export failed: %1</source>
        <translation>선 데이터 내보내기 실패: %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="720"/>
        <source>Line data written to %1</source>
        <translation>선 데이터를 %1에 기록했습니다</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="726"/>
        <source>SVG Images</source>
        <translation>SVG 이미지</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="737"/>
        <source>Export Chart</source>
        <translation>차트 내보내기</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="756"/>
        <source>Chart copied to the clipboard.</source>
        <translation>차트를 클립보드에 복사했습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="764"/>
        <source>Plotted data copied to the clipboard.</source>
        <translation>표시된 데이터를 클립보드에 복사했습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="725"/>
        <source>PNG Images</source>
        <translation>PNG 이미지</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="727"/>
        <source>PDF Documents</source>
        <translation>PDF 문서</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="748"/>
        <source>Chart export failed: %1</source>
        <translation>차트 내보내기 실패: %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/analysis/tab.py" line="750"/>
        <source>Chart written to %1</source>
        <translation>차트를 %1에 기록했습니다</translation>
    </message>
</context>
<context>
    <name>App</name>
    <message>
        <location filename="../../gui/app.py" line="811"/>
        <source>Run DIC first -- no displacement results to post-process.</source>
        <translation>DIC를 먼저 실행하세요 —— 후처리할 변위 결과가 없습니다.</translation>
    </message>
</context>
<context>
    <name>Application</name>
    <message>
        <location filename="../../gui/app.py" line="976"/>
        <source>RAFTcorr has hit an error</source>
        <translation>RAFTcorr에서 오류가 발생했습니다</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="977"/>
        <source>An unexpected error occurred. The application may not behave correctly from here on, so saving your project and restarting is recommended.</source>
        <translation>예기치 않은 오류가 발생했습니다. 이후 애플리케이션이 올바르게 동작하지 않을 수 있으므로 프로젝트를 저장하고 다시 시작하는 것을 권장합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="987"/>
        <source>Details were written to %1</source>
        <translation>자세한 내용을 %1에 기록했습니다</translation>
    </message>
</context>
<context>
    <name>AutoFixedSelector</name>
    <message>
        <location filename="../../gui/widgets/range_mode.py" line="33"/>
        <source>Auto</source>
        <comment>Color range mode: rescale to the data range</comment>
        <translation>자동</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/range_mode.py" line="36"/>
        <source>Rescale the color range to each frame&apos;s data range</source>
        <translation>각 프레임의 데이터 범위에 맞춰 색상 범위를 다시 조정합니다</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/range_mode.py" line="39"/>
        <source>Fixed</source>
        <comment>Color range mode: manual min/max bounds</comment>
        <translation>고정</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/range_mode.py" line="42"/>
        <source>Keep the manual Min/Max bounds for every frame</source>
        <translation>모든 프레임에서 수동 최소/최대 값을 유지합니다</translation>
    </message>
</context>
<context>
    <name>BatchImportDialog</name>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="369"/>
        <source>Batch Import Region of Interest Masks</source>
        <translation>관심 영역 마스크 일괄 가져오기</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="420"/>
        <source>Mask Folder:</source>
        <translation>마스크 폴더:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="421"/>
        <source>(none)</source>
        <translation>(없음)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="426"/>
        <source>Browse...</source>
        <translation>찾아보기…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="445"/>
        <source>Available Masks</source>
        <translation>사용 가능한 마스크</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="456"/>
        <source>Auto-Match by Name</source>
        <translation>이름으로 자동 매칭</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="466"/>
        <source>Assign Sequential</source>
        <translation>순차 할당</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="458"/>
        <source>Give each image the mask whose file name ends in the same number: mask6.png goes to image6.tif.
If the image names carry no number, the mask&apos;s number is read as the frame position counted from 0: mask0 goes to frame 1.</source>
        <translation>각 이미지에 파일 이름 끝의 숫자가 같은 마스크를 할당합니다: mask6.png는 image6.tif에 할당됩니다.
이미지 이름에 숫자가 없으면 마스크의 숫자를 0부터 센 프레임 위치로 읽습니다: mask0은 프레임 1에 할당됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="468"/>
        <source>Assign masks to frames in order, starting from frame 1</source>
        <translation>프레임 1부터 순서대로 마스크를 프레임에 할당합니다</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="476"/>
        <source>Frame Assignments</source>
        <translation>프레임 할당</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="479"/>
        <source>Frame</source>
        <translation>프레임</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="479"/>
        <source>Image</source>
        <translation>이미지</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="479"/>
        <source>Mask</source>
        <translation>마스크</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="490"/>
        <source>Assign Selected -&gt;</source>
        <translation>선택 항목 할당 -&gt;</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="492"/>
        <source>Pair selected mask(s) with selected frame(s)</source>
        <translation>선택한 마스크와 선택한 프레임을 짝지웁니다</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="496"/>
        <source>Clear All</source>
        <translation>모두 지우기</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="541"/>
        <source>Select Mask Folder</source>
        <translation>마스크 폴더 선택</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="599"/>
        <source>Failed to read mask file.</source>
        <translation>마스크 파일을 읽지 못했습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="603"/>
        <source>Mismatched shape: %1×%2 (expected %3×%4)</source>
        <translation>형상 불일치: %1×%2 (예상 %3×%4)</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="675"/>
        <source>%n mask(s) have mismatched sizes and are disabled.</source>
        <translation>
            <numerusform>크기가 일치하지 않는 마스크 %n개가 비활성화되었습니다.</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="745"/>
        <source>Invalid assignment</source>
        <translation>잘못된 할당</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="747"/>
        <source>A frame can only have one mask. Select exactly one mask, or select multiple frames to assign one mask to many.</source>
        <translation>한 프레임에는 마스크를 하나만 지정할 수 있습니다. 마스크를 정확히 하나 선택하거나, 여러 프레임을 선택해 하나의 마스크를 여러 프레임에 지정하세요.</translation>
    </message>
</context>
<context>
    <name>CanvasArea</name>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1239"/>
        <source>Fit</source>
        <translation>맞춤</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1240"/>
        <source>Fit image to viewport</source>
        <translation>이미지를 뷰포트에 맞춤</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1244"/>
        <source>100%</source>
        <translation>100%</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1245"/>
        <source>Zoom to 100% (1:1)</source>
        <translation>100%(1:1) 확대</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1248"/>
        <source>Zoom in</source>
        <translation>확대</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1253"/>
        <source>–</source>
        <translation>–</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1254"/>
        <source>Zoom out</source>
        <translation>축소</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1345"/>
        <source>EDITING REGION OF INTEREST FOR FRAME %1</source>
        <translation>프레임 %1의 관심 영역 편집 중</translation>
    </message>
</context>
<context>
    <name>CanvasConfigOverlay</name>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="40"/>
        <source>Mode</source>
        <translation>모드</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="41"/>
        <source>Model</source>
        <translation>모델</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="42"/>
        <source>Iterations</source>
        <translation>반복 횟수</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="85"/>
        <source>Accumulative</source>
        <translation>누적형</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="87"/>
        <source>Incremental, every %1 frames</source>
        <translation>증분형, %1 프레임마다</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="89"/>
        <source>Custom reference frames</source>
        <translation>사용자 지정 참조 프레임</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/canvas_config_overlay.py" line="90"/>
        <source>Incremental</source>
        <translation>증분형</translation>
    </message>
</context>
<context>
    <name>ColorRange</name>
    <message>
        <location filename="../../gui/widgets/color_range.py" line="27"/>
        <source>Range</source>
        <translation>범위</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/color_range.py" line="37"/>
        <source>Min</source>
        <translation>최소</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/color_range.py" line="47"/>
        <source>Max</source>
        <translation>최대</translation>
    </message>
</context>
<context>
    <name>ExportDialog</name>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="45"/>
        <source>Magnitude</source>
        <translation>변위 크기</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="46"/>
        <source>Residual</source>
        <translation>잔차</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="202"/>
        <source>Open this tab to render a preview.</source>
        <translation>이 탭을 열면 미리보기가 렌더링됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="209"/>
        <location filename="../../gui/dialogs/export_widgets.py" line="383"/>
        <source>Field</source>
        <translation>필드</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="216"/>
        <source>Frame</source>
        <translation>프레임</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="247"/>
        <source>FIELD APPEARANCE</source>
        <translation>필드 모양</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="252"/>
        <location filename="../../gui/dialogs/export_widgets.py" line="383"/>
        <source>Colormap</source>
        <translation>색상 맵</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="256"/>
        <source>Range</source>
        <translation>범위</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="260"/>
        <location filename="../../gui/dialogs/export_widgets.py" line="383"/>
        <source>Min</source>
        <translation>최소</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="267"/>
        <location filename="../../gui/dialogs/export_widgets.py" line="383"/>
        <source>Max</source>
        <translation>최대</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="274"/>
        <location filename="../../gui/dialogs/export_widgets.py" line="179"/>
        <source>Opacity</source>
        <extracomment>Column widths of the field rows (export, field, colormap, range mode, min / max), shared by the header labels of the Images and Animation tabs. Long-edge presets (pixels) of the &quot;Resolution (long edge)&quot; combos, in pyALDIC&apos;s order; the last entry, &quot;Full resolution&quot;, is 0 (no cap). Older names that share a field&apos;s colour state.</extracomment>
        <translation>불투명도</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="276"/>
        <source>Apply to all fields</source>
        <translation>모든 필드에 적용</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="277"/>
        <source>Apply this field&apos;s colormap, opacity and auto-range to every enabled field (each field keeps its own min/max).</source>
        <translation>이 필드의 색상 맵, 불투명도, 자동 범위를 활성화된 모든 필드에 적용합니다(각 필드의 최소/최대는 유지).</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="230"/>
        <source>Refresh preview</source>
        <translation>미리보기 새로고침</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="74"/>
        <source>COLORBAR STYLE</source>
        <translation>컬러바 스타일</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="79"/>
        <source>Right</source>
        <translation>오른쪽</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="80"/>
        <source>Left</source>
        <translation>왼쪽</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="81"/>
        <source>Top</source>
        <translation>위</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="82"/>
        <source>Bottom</source>
        <translation>아래</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="85"/>
        <source>Position</source>
        <translation>위치</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="90"/>
        <source>Font size</source>
        <translation>글꼴 크기</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="96"/>
        <source>Font family</source>
        <translation>글꼴</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="99"/>
        <source>Bar thickness as a fraction of the picture&apos;s width (bar on the left or right) or height (bar on top or at the bottom).</source>
        <translation>컬러바 두께. 그림 너비(왼쪽이나 오른쪽에 둘 때) 또는 높이(위나 아래에 둘 때)에 대한 비율입니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="102"/>
        <source>Bar thickness</source>
        <translation>막대 두께</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="105"/>
        <location filename="../../gui/dialogs/export_preview.py" line="122"/>
        <location filename="../../gui/dialogs/export_preview.py" line="128"/>
        <source>Black</source>
        <translation>검정</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="106"/>
        <location filename="../../gui/dialogs/export_preview.py" line="121"/>
        <location filename="../../gui/dialogs/export_preview.py" line="127"/>
        <source>White</source>
        <translation>흰색</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="109"/>
        <source>Colour behind the colorbar and around the picture; text and ticks are drawn in the opposite colour.</source>
        <translation>컬러바 뒤와 그림 주변의 색입니다. 글자와 눈금은 반대 색으로 그립니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="112"/>
        <source>Background</source>
        <translation>배경</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="115"/>
        <source>Add a blank border around the exported content, as a fraction of the long edge (0 = none).</source>
        <translation>내보내는 콘텐츠 주위에 여백 테두리를 추가합니다. 너비는 긴 변에 대한 비율입니다(0 = 없음).</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="118"/>
        <source>Margin</source>
        <translation>여백</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="124"/>
        <source>Margin color</source>
        <translation>여백 색상</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="129"/>
        <source>Transparent</source>
        <translation>투명</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="132"/>
        <source>Fill used where the background image would have been, when &apos;Show background image&apos; is off.
Transparency is kept for PNG and TIFF; the other formats get white instead.</source>
        <translation>&apos;배경 이미지 표시&apos;가 꺼져 있을 때 배경 이미지 자리를 채우는 색입니다.
PNG와 TIFF는 투명을 유지하고, 다른 형식은 흰색이 됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="136"/>
        <source>Hidden background</source>
        <translation>숨김 시 배경</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="301"/>
        <source>Preview failed: </source>
        <translation>미리보기 실패: </translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="335"/>
        <source>Enable a field on the Images tab to preview.</source>
        <translation>미리보려면 &apos;이미지&apos; 탭에서 필드를 활성화하세요.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_preview.py" line="339"/>
        <source>No data for this field/frame.</source>
        <translation>이 필드/프레임에 데이터가 없습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="180"/>
        <source>Field opacity (0 = transparent, 1 = fully opaque)</source>
        <translation>필드 불투명도 (0 = 투명, 1 = 완전 불투명)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="290"/>
        <source>Full resolution</source>
        <translation>전체 해상도</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="297"/>
        <source>All</source>
        <translation>모두 선택</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="299"/>
        <source>None</source>
        <translation>모두 해제</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="370"/>
        <source>Select:</source>
        <translation>선택:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="383"/>
        <source>Export</source>
        <translation>내보내기</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="383"/>
        <source>Auto</source>
        <translation>자동</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="400"/>
        <source>PHYSICAL UNITS</source>
        <translation>물리 단위</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="404"/>
        <source>Enable physical units</source>
        <translation>물리 단위 활성화</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="406"/>
        <source>Scale displacement values by pixel size and show physical units on colorbar labels. Strain is dimensionless and unaffected.</source>
        <translation>변위 값을 픽셀 크기로 스케일링하고 컬러바 레이블에 물리 단위를 표시합니다. 변형률은 무차원이므로 영향받지 않습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="423"/>
        <source>/ pixel</source>
        <translation>/ 픽셀</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="425"/>
        <source>Pixel size</source>
        <translation>픽셀 크기</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="434"/>
        <source>fps</source>
        <translation>fps</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="436"/>
        <source>Frame rate</source>
        <translation>프레임 속도</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="462"/>
        <source>FRAME RANGE</source>
        <translation>프레임 범위</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="464"/>
        <source>All frames</source>
        <translation>모든 프레임</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="469"/>
        <source>Numbered like the frame slider: frame 1 is the reference image.
File names use the same numbers: frame 1 is written as _0001.</source>
        <translation>프레임 슬라이더와 같은 번호입니다. 프레임 1은 참조 이미지입니다.
파일 이름도 같은 번호를 사용하며, 프레임 1은 _0001로 기록됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="472"/>
        <source>From</source>
        <comment>Frame range: starting frame</comment>
        <translation>시작</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="477"/>
        <source>to</source>
        <comment>Frame range: ending frame</comment>
        <translation>끝</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="174"/>
        <source>Export Results</source>
        <extracomment>GUI field name -&gt; variable name in the data files (raftcorr.export.scientific). Long-edge caps the tabs start from (pixels, 0 = full resolution): pictures keep the size their DPI gives, animations start at pyALDIC&apos;s 1024 px.</extracomment>
        <translation>결과 내보내기</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="201"/>
        <source>OUTPUT FOLDER</source>
        <translation>출력 폴더</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="207"/>
        <source>Select output folder…</source>
        <translation>출력 폴더 선택…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="213"/>
        <source>Browse…</source>
        <translation>찾아보기…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="218"/>
        <source>Open Folder</source>
        <translation>폴더 열기</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="232"/>
        <source>Data</source>
        <translation>데이터</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="233"/>
        <source>Images</source>
        <translation>이미지</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="234"/>
        <source>Animation</source>
        <translation>애니메이션</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="235"/>
        <source>Report</source>
        <translation>보고서</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="320"/>
        <source>FORMAT</source>
        <translation>형식</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="322"/>
        <source>NumPy Archive (.npz)</source>
        <translation>NumPy 아카이브 (.npz)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="329"/>
        <source>CSV (per frame)</source>
        <translation>CSV(프레임별)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="345"/>
        <source>DISPLACEMENT</source>
        <translation>변위</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="348"/>
        <source>STRAIN</source>
        <translation>변형률</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="287"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="351"/>
        <source>Run Compute Strain first.</source>
        <translation>먼저 「변형률 계산」을 실행하세요.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="268"/>
        <source>Export: images not readable (%1); pictures go on white.</source>
        <translation>내보내기: 이미지를 읽을 수 없습니다(%1). 그림은 흰색 배경으로 출력됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="272"/>
        <source>Export: the images do not match the results&apos; image size; pictures go on white.</source>
        <translation>내보내기: 이미지 크기가 결과의 이미지 크기와 일치하지 않습니다. 그림은 흰색 배경으로 출력됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="286"/>
        <source>This run has no intensity residual.</source>
        <translation>이번 실행에는 밝기 잔차가 없습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="324"/>
        <source>MATLAB v7.3 (.mat)</source>
        <translation>MATLAB v7.3 (.mat)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="326"/>
        <source>An HDF5-based MAT file: load it in MATLAB with load().</source>
        <translation>HDF5 기반 MAT 파일입니다. MATLAB에서 load()로 불러오세요.</translation>
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
        <translation>계산된 프레임마다 표 하나, 관심 영역의 점마다 한 행입니다.
조밀한 필드는 파일이 커집니다: 1000 x 1000 px 영역은 프레임당 백만 행입니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="355"/>
        <source>✓ Parameters file (JSON) always exported</source>
        <translation>✓ 매개변수 파일(JSON)은 항상 내보내집니다</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="356"/>
        <source>✓ Grid coordinates (X, Y), ROI and computed-frame flags always included</source>
        <translation>✓ 격자 좌표(X, Y), ROI 및 계산된 프레임 플래그는 항상 포함됩니다</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="357"/>
        <source>Displacements are written in pixels (+u right, +v down); the pixel size and frame rate above are recorded with them.</source>
        <translation>변위는 픽셀 단위로 기록됩니다(+u 오른쪽, +v 아래쪽). 위의 픽셀 크기와 프레임 속도도 함께 기록됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="363"/>
        <source>Export Data</source>
        <translation>데이터 내보내기</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="413"/>
        <source>IMAGE SETTINGS</source>
        <translation>이미지 설정</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="424"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="531"/>
        <source>Format</source>
        <translation>형식</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="376"/>
        <source>Deformed frame</source>
        <translation>변형된 프레임</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="239"/>
        <source>Preview &amp; Colorbar</source>
        <translation>미리보기 및 컬러바</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="334"/>
        <source>NPZ: one file per frame (default: single merged file)</source>
        <translation>NPZ: 프레임별 1 파일(기본값: 통합 단일 파일)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="337"/>
        <source>One .npz per computed frame, in a folder of its own: that frame&apos;s variables, the grid coordinates, the ROI and the metadata.</source>
        <translation>계산된 프레임마다 .npz 파일 하나를 별도 폴더에 저장합니다. 해당 프레임의 변수, 격자 좌표, ROI, 메타데이터가 들어 있습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="377"/>
        <source>Reference frame</source>
        <translation>참조 프레임</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="383"/>
        <source>Show background image</source>
        <translation>배경 이미지 표시</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="394"/>
        <source>Hide untracked regions</source>
        <translation>추적되지 않은 영역 숨기기</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="431"/>
        <source>Cap the long edge of each written image (the larger of width/height, in pixels; the aspect ratio and layout are kept).
Lower = faster and much smaller on disk; for PDF and SVG it caps the embedded picture. &apos;Full resolution&apos; writes the size set by DPI.</source>
        <translation>내보내는 각 이미지의 긴 변(너비와 높이 중 큰 값, 픽셀 단위. 가로세로 비율과 배치는 유지)을 제한합니다.
작을수록 빠르고 파일도 훨씬 작아집니다. PDF와 SVG에서는 포함된 그림을 제한합니다. &apos;전체 해상도&apos;는 DPI로 정한 크기로 저장합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="436"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="544"/>
        <source>Resolution (long edge)</source>
        <translation>해상도(긴 변)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="444"/>
        <source>JPEG quality (higher = larger file). Ignored for the other formats.</source>
        <translation>JPEG 품질(높을수록 파일이 커집니다). 다른 형식에서는 무시됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="447"/>
        <source>JPEG quality</source>
        <translation>JPEG 품질</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="455"/>
        <source>Resolution of the written figure: about 8.6 x 7 inches at this many dots per inch.
A resolution cap (long edge) lowers it.</source>
        <translation>저장되는 그림의 해상도: 약 8.6 x 7인치를 이 DPI로 저장합니다.
해상도 상한(긴 변)을 정하면 낮아집니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="470"/>
        <source>DPI</source>
        <translation>DPI</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="472"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="570"/>
        <source>Include colorbar</source>
        <translation>컬러바 포함</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="475"/>
        <source>Add a colorbar to each image; its position and style are set on the Preview &amp; Colorbar tab.</source>
        <translation>각 이미지에 컬러바를 추가합니다. 위치와 스타일은 &apos;미리보기 및 컬러바&apos; 탭에서 정합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="480"/>
        <source>Uncheck to export the field on its own, with no speckle image behind it. Pick the fill on the Preview &amp; Colorbar tab.</source>
        <translation>선택을 해제하면 뒤에 스페클 이미지 없이 필드만 내보냅니다. 채우기는 &apos;미리보기 및 컬러바&apos; 탭에서 선택합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="488"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="583"/>
        <source>Render as</source>
        <translation>렌더링 방식</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="498"/>
        <source>Export Images</source>
        <translation>이미지 내보내기</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="512"/>
        <source>ANIMATION SETTINGS</source>
        <translation>애니메이션 설정</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="538"/>
        <source>Cap the animation&apos;s long edge (the larger of width/height).
Lower = faster and much smaller. Strongly recommended for GIF, whose size explodes at native resolution. &apos;Full resolution&apos; draws the images at their own pixel size.</source>
        <translation>애니메이션의 긴 변(너비와 높이 중 큰 값)을 제한합니다.
작을수록 빠르고 훨씬 작아집니다. GIF에는 강력히 권장합니다(원래 해상도에서는 크기가 급격히 커집니다). &apos;전체 해상도&apos;는 이미지를 원래 픽셀 크기로 그립니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="553"/>
        <source>FPS</source>
        <translation>FPS</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="563"/>
        <source>Export every Nth frame (1 = every frame). Higher is faster and smaller
but looks choppier. Playback duration is preserved (the FPS above is the pre-decimation rate).</source>
        <translation>N 프레임마다 하나씩 내보냅니다(1 = 모든 프레임). 클수록 빠르고 작지만,
더 끊겨 보입니다. 재생 시간은 유지됩니다(위의 FPS는 추출 전 프레임 속도).</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="568"/>
        <source>Frame step</source>
        <translation>프레임 간격</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="573"/>
        <source>Add a colorbar to each frame; its position and style are set on the Preview &amp; Colorbar tab.</source>
        <translation>각 프레임에 컬러바를 추가합니다. 위치와 스타일은 &apos;미리보기 및 컬러바&apos; 탭에서 정합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="577"/>
        <source>Uncheck to export the field on its own, with no speckle image behind it. GIF and MP4 cannot store transparency, so a transparent fill is written as white.</source>
        <translation>선택을 해제하면 뒤에 스페클 이미지 없이 필드만 내보냅니다. GIF와 MP4는 투명도를 저장할 수 없으므로 투명 채우기는 흰색으로 기록됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="596"/>
        <source>Export Animation</source>
        <translation>애니메이션 내보내기</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="606"/>
        <source>CONTENT</source>
        <translation>내용</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="608"/>
        <source>Parameter summary table</source>
        <translation>매개변수 요약 표</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="610"/>
        <source>Field statistics (min/max/mean/std per frame)</source>
        <translation>필드 통계 (프레임별 최소/최대/평균/표준편차)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="612"/>
        <source>Sample field images</source>
        <translation>필드 이미지 샘플</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="620"/>
        <source>Sample every</source>
        <translation>샘플 간격</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="626"/>
        <source>frames</source>
        <comment>Report: sample every N frames</comment>
        <translation>프레임</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="632"/>
        <source>FIELDS</source>
        <translation>필드</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="634"/>
        <source>Displacement:</source>
        <translation>변위:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="635"/>
        <source>Strain:</source>
        <translation>변형률:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="644"/>
        <source>Format: HTML (self-contained, view in any browser)</source>
        <translation>형식: HTML (자체 포함, 모든 브라우저에서 볼 수 있음)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="648"/>
        <source>Generate Report</source>
        <translation>보고서 생성</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="809"/>
        <source>Select Output Folder</source>
        <translation>출력 폴더 선택</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="910"/>
        <source>Writing %1 (%2%)</source>
        <translation>%1 쓰는 중 (%2%)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="943"/>
        <source>Error: %1</source>
        <translation>오류: %1</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="885"/>
        <source>Starting…</source>
        <translation>시작 중…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="380"/>
        <source>Deformed: the field is drawn where each point is in that frame, over the frame&apos;s own photo.
Reference: drawn at the points&apos; positions in frame 1, over frame 1.</source>
        <translation>변형된 프레임: 각 점이 해당 프레임에서 있는 위치에 필드를 그려 그 프레임 자체의 사진 위에 겹칩니다.
참조 프레임: 프레임 1에서의 점 위치에 그려 프레임 1 위에 겹칩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="387"/>
        <source>The images of this run are not available.</source>
        <translation>이번 실행의 이미지를 사용할 수 없습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="462"/>
        <source>Each frame</source>
        <translation>프레임별</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="463"/>
        <source>Whole sequence</source>
        <translation>전체 시퀀스</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="465"/>
        <source>How Auto picks the colour range: from each frame&apos;s own values (like the main window),
or one range from every exported frame, so a colour means the same value in every image.</source>
        <translation>자동 범위를 정하는 방식: 각 프레임 자체의 값으로 정하거나(메인 창과 동일),
내보내는 모든 프레임에서 하나의 범위를 정해 모든 이미지에서 같은 색이 같은 값을 뜻하도록 합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="460"/>
        <source>Auto range</source>
        <translation>자동 범위</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="397"/>
        <source>Leave points without a result transparent in the pictures, for example material that left the field of view, instead of filling them with the nearest value inside the region of interest. Applies to the Reference frame view only: the Deformed frame view always leaves them empty. Data files always keep NaN there.</source>
        <translation>결과가 없는 점(시야 밖으로 나간 재료 등)을 그림에서 관심 영역 안의 가장 가까운 값으로 채우지 않고 투명하게 둡니다. &apos;참조 프레임&apos; 보기에만 적용되며, &apos;변형된 프레임&apos; 보기에서는 항상 비워 둡니다. 데이터 파일에서는 항상 NaN입니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="484"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="580"/>
        <source>Crop to ROI</source>
        <translation>ROI로 자르기</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="485"/>
        <source>Cut the picture to the region of interest plus a small margin.</source>
        <translation>그림을 관심 영역에 약간의 여백을 더한 범위로 자릅니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="524"/>
        <source>MP4 needs the imageio and imageio-ffmpeg packages (pip install imageio imageio-ffmpeg). Save as GIF instead.</source>
        <translation>MP4에는 imageio 및 imageio-ffmpeg 패키지가 필요합니다(pip install imageio imageio-ffmpeg). 대신 GIF로 저장하세요.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="581"/>
        <source>Cut the frames to the region of interest plus a small margin.</source>
        <translation>각 프레임을 관심 영역에 약간의 여백을 더한 범위로 자릅니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="588"/>
        <source>Auto range: one range for the whole animation, so a colour means the same value in every frame.</source>
        <translation>자동 범위: 애니메이션 전체에 하나의 범위를 사용하므로 모든 프레임에서 같은 색이 같은 값을 뜻합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="615"/>
        <source>Drawn like the Images tab: same colormap, range, opacity and configuration.</source>
        <translation>&apos;이미지&apos; 탭과 같은 방식으로 그립니다: 색상 맵, 범위, 불투명도, 구성이 같습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="738"/>
        <location filename="../../gui/dialogs/export_dialog.py" line="790"/>
        <source>Min must be below Max for %1.</source>
        <translation>%1의 최솟값은 최댓값보다 작아야 합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="744"/>
        <source>Nothing to export: enable at least one field with computed frames in the frame range.</source>
        <translation>내보낼 항목이 없습니다: 프레임 범위 안에 계산된 프레임이 있는 필드를 하나 이상 활성화하세요.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="752"/>
        <source>Select at least one format.</source>
        <translation>형식을 하나 이상 선택하세요.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="756"/>
        <source>Select at least one variable to export.</source>
        <translation>내보낼 변수를 하나 이상 선택하세요.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="793"/>
        <source>Nothing to export: select a field or the parameter table.</source>
        <translation>내보낼 항목이 없습니다: 필드 또는 매개변수 표를 선택하세요.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="846"/>
        <source>… and %1 more</source>
        <translation>… 외 %1개</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="848"/>
        <source>Replace Existing Files?</source>
        <translation>기존 파일을 바꾸시겠습니까?</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="849"/>
        <source>These already exist in the output folder and will be replaced:

%1</source>
        <translation>다음 파일이 출력 폴더에 이미 있으며 교체됩니다:

%1</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="868"/>
        <source>The results changed after this dialog was opened (a new run, strain or project). Close it and open Export again to export them.</source>
        <translation>이 대화 상자를 연 뒤 결과가 바뀌었습니다(새 실행, 변형률 또는 프로젝트). 대화 상자를 닫고 &apos;내보내기&apos;를 다시 열어 내보내세요.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="872"/>
        <source>Nothing exported: the existing files were kept.</source>
        <translation>아무것도 내보내지 않았습니다: 기존 파일을 유지했습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="897"/>
        <source>Stopping…</source>
        <translation>중지 중…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="908"/>
        <source>Rendering %1 (%2/%3)</source>
        <translation>%1 렌더링 중 (%2/%3)</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/dialogs/export_dialog.py" line="926"/>
        <source>Exported %n file(s) → %1</source>
        <translation>
            <numerusform>%n 개 파일 내보냄 → %1</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/dialogs/export_dialog.py" line="928"/>
        <source>Exported %n image(s) → %1</source>
        <translation>
            <numerusform>%n 개 이미지 내보냄 → %1</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="944"/>
        <source>Export failed: %1</source>
        <translation>내보내기 실패: %1</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/dialogs/export_dialog.py" line="952"/>
        <source>Export cancelled: %n file(s) written before stopping.</source>
        <translation>
            <numerusform>내보내기가 취소되었습니다: 중지 전에 파일 %n개를 기록했습니다.</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="932"/>
        <source>No animation was written. See the log for details.</source>
        <translation>애니메이션이 기록되지 않았습니다. 자세한 내용은 로그를 확인하세요.</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/dialogs/export_dialog.py" line="934"/>
        <source>Exported %n animation(s) → %1</source>
        <translation>
            <numerusform>%n 개 애니메이션 내보냄 → %1</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="936"/>
        <source>Report saved → %1</source>
        <translation>보고서 저장됨 → %1</translation>
    </message>
</context>
<context>
    <name>FieldSelector</name>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="19"/>
        <source>Horizontal displacement (+ right)</source>
        <translation>수평 변위(오른쪽이 +)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="20"/>
        <source>Vertical displacement (+ down)</source>
        <translation>수직 변위(아래쪽이 +)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="21"/>
        <source>Displacement magnitude</source>
        <translation>변위 크기</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="22"/>
        <source>Intensity residual: local RMS difference between the reference image and
the deformed image mapped back by the displacement. Large where the
tracking failed.</source>
        <translation>밝기 잔차: 참조 이미지와 변위로 되돌려 매핑한 변형 이미지 사이의
국소 RMS 차이입니다. 추적에 실패한 곳에서 큽니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="26"/>
        <source>Disp U</source>
        <translation>변위 U</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="26"/>
        <source>Disp V</source>
        <translation>변위 V</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/field_selector.py" line="27"/>
        <source>Magnitude</source>
        <translation>변위 크기</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/viz_controller.py" line="138"/>
        <location filename="../../gui/widgets/field_selector.py" line="27"/>
        <source>Residual</source>
        <translation>잔차</translation>
    </message>
</context>
<context>
    <name>FrameNavigator</name>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="60"/>
        <source>Previous frame</source>
        <translation>이전 프레임</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="70"/>
        <location filename="../../gui/widgets/frame_navigator.py" line="174"/>
        <source>Play animation</source>
        <translation>애니메이션 재생</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="74"/>
        <location filename="../../gui/widgets/frame_navigator.py" line="173"/>
        <source>▶</source>
        <translation>▶</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="81"/>
        <source>Next frame</source>
        <translation>다음 프레임</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="97"/>
        <source>Playback speed</source>
        <translation>재생 속도</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="102"/>
        <location filename="../../gui/widgets/frame_navigator.py" line="198"/>
        <source>FRAME 0/0</source>
        <translation>프레임 0/0</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="163"/>
        <source>⏸</source>
        <translation>⏸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="164"/>
        <source>Pause animation</source>
        <translation>애니메이션 일시정지</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/frame_navigator.py" line="195"/>
        <source>FRAME %1/%2</source>
        <translation>프레임 %1/%2</translation>
    </message>
</context>
<context>
    <name>ImageCanvas</name>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1078"/>
        <source>Load images first before drawing a Region of Interest.</source>
        <translation>관심 영역을 그리기 전에 먼저 이미지를 불러오세요.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/canvas_area.py" line="1097"/>
        <source>The three points are nearly collinear — pick points spread around the circle&apos;s edge.</source>
        <translation>세 점이 거의 일직선입니다 — 원의 가장자리에 고르게 세 점을 찍으세요.</translation>
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
        <translation>파일 이름</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="99"/>
        <source>Region</source>
        <comment>Image list column: ROI status</comment>
        <translation>영역</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="193"/>
        <location filename="../../gui/widgets/image_list.py" line="255"/>
        <source>Add</source>
        <translation>추가</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="249"/>
        <source>Edit</source>
        <translation>편집</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="252"/>
        <source>Need</source>
        <translation>필요</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/widgets/image_list.py" line="368"/>
        <source>Import Region of Interest for %n frame(s)</source>
        <translation>
            <numerusform>%n 프레임에 관심 영역 가져오기</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="379"/>
        <source>Clear Region of Interest (%1 with region)</source>
        <translation>관심 영역 지우기 (%1개 프레임에 영역 있음)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="383"/>
        <source>Clear Region of Interest</source>
        <translation>관심 영역 지우기</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/widgets/image_list.py" line="394"/>
        <source>Delete %n image(s)</source>
        <translation>
            <numerusform>%n 개 이미지 삭제</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/widgets/image_list.py" line="435"/>
        <source>Select %n Mask File(s)</source>
        <translation>
            <numerusform>%n 개 마스크 파일 선택</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="440"/>
        <source>Images</source>
        <translation>이미지</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="441"/>
        <source>All Files</source>
        <translation>모든 파일</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/image_list.py" line="450"/>
        <source>Selected %1 files for %2 frames — count must match</source>
        <translation>%2 프레임에 대해 %1 개 파일 선택됨 — 개수가 일치해야 합니다</translation>
    </message>
</context>
<context>
    <name>ImageMessages</name>
    <message>
        <location filename="../../gui/controllers/image_controller.py" line="43"/>
        <source>Cannot open the images</source>
        <translation>이미지를 열 수 없습니다</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/image_controller.py" line="47"/>
        <source>%n image(s), %1 x %2 px, %3</source>
        <translation>
            <numerusform>이미지 %n개, %1 x %2 px, %3</numerusform>
        </translation>
    </message>
</context>
<context>
    <name>LeftSidebar</name>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="177"/>
        <source>IMAGES</source>
        <translation>이미지</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="184"/>
        <source>Natural Sort (1, 2, …, 10)</source>
        <translation>자연 정렬 (1, 2, …, 10)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="260"/>
        <source>WORKFLOW TYPE</source>
        <translation>워크플로 유형</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="187"/>
        <source>Sort by embedded numbers: image1, image2, ..., image10 (default).
Unchecked: plain alphabetical order, for zero-padded names.</source>
        <translation>파일명 속 숫자 순 정렬: image1, image2, ..., image10(기본값).
체크 해제: 단순 알파벳순, 0으로 채운 파일명에 적합.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="267"/>
        <source>REGION OF INTEREST</source>
        <translation>관심 영역</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="279"/>
        <source>PARAMETERS</source>
        <translation>매개변수</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="285"/>
        <source>ADVANCED</source>
        <translation>고급 설정</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="378"/>
        <source>Change the image order?</source>
        <translation>이미지 순서를 바꾸시겠습니까?</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="380"/>
        <source>Sorting the images differently renumbers the frames. The Regions of Interest, results and probes of the current frames will be discarded.

Sort the images again?</source>
        <translation>이미지를 다르게 정렬하면 프레임 번호가 다시 매겨집니다. 현재 프레임의 관심 영역, 결과, 프로브는 폐기됩니다.

이미지를 다시 정렬하시겠습니까?</translation>
    </message>
</context>
<context>
    <name>LoadDataDialog</name>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="103"/>
        <source>Load Data</source>
        <translation>하중 데이터</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="115"/>
        <source>No file chosen.</source>
        <translation>선택한 파일이 없습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="117"/>
        <source>Choose file…</source>
        <translation>파일 선택…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="137"/>
        <source>Load column:</source>
        <translation>하중 열:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="139"/>
        <source>By time</source>
        <translation>시간으로</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="140"/>
        <source>By frame number</source>
        <translation>프레임 번호로</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="144"/>
        <source>Match rows to frames:</source>
        <translation>행과 프레임 연결:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="147"/>
        <source>Time column:</source>
        <translation>시간 열:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="153"/>
        <source>The machine&apos;s time at the reference image. If the camera started 2 s after the machine, enter 2.</source>
        <translation>참조 이미지 시점의 시험기 시간입니다. 카메라가 시험기보다 2 s 늦게 시작했다면 2를 입력하십시오.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="155"/>
        <source>Offset:</source>
        <translation>오프셋:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="163"/>
        <source>Frame column:</source>
        <translation>프레임 열:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="167"/>
        <source>The first image is numbered:</source>
        <translation>첫 이미지의 번호:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="174"/>
        <source>Initial cross-section, for engineering stress F / A0 in MPa. Leave at 0 for load only.</source>
        <translation>초기 단면적으로, 공칭 응력 F / A0(MPa) 계산에 사용합니다. 하중만 필요하면 0으로 두십시오.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="176"/>
        <source>Cross-section A0:</source>
        <translation>단면적 A0:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="192"/>
        <source>Remove Load Data</source>
        <translation>하중 데이터 제거</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="201"/>
        <source>Camera frame rate: %1 fps, from Physical Units.</source>
        <translation>카메라 프레임 속도: %1 fps, &apos;물리 단위&apos;에서 설정.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="204"/>
        <source>Set the camera frame rate under Physical Units to match by time.</source>
        <translation>시간으로 맞추려면 &apos;물리 단위&apos;에서 카메라 프레임 속도를 설정하십시오.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="227"/>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="245"/>
        <source>(unnamed)</source>
        <translation>(이름 없음)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="324"/>
        <source>No frame falls within the record: check the columns and the offset.</source>
        <translation>기록 범위에 드는 프레임이 없습니다: 열과 오프셋을 확인하십시오.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="328"/>
        <source>Frames with a load: %1 of %2.</source>
        <translation>하중이 있는 프레임: %1 / %2.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="334"/>
        <source>Open Load Data</source>
        <translation>하중 데이터 열기</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="335"/>
        <source>CSV Files</source>
        <translation>CSV 파일</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="335"/>
        <source>All Files</source>
        <translation>모든 파일</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/load_data_dialog.py" line="341"/>
        <source>Could not read %1: %2</source>
        <translation>%1을(를) 읽을 수 없습니다: %2</translation>
    </message>
</context>
<context>
    <name>MainWindow</name>
    <message>
        <location filename="../../gui/app.py" line="146"/>
        <source>&amp;File</source>
        <translation>파일</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="171"/>
        <source>Quit</source>
        <translation>종료</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="175"/>
        <source>&amp;Settings</source>
        <translation>설정</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="178"/>
        <source>Theme</source>
        <translation>테마</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="182"/>
        <source>Dark</source>
        <comment>Color theme</comment>
        <translation>다크</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="182"/>
        <source>Light</source>
        <comment>Color theme</comment>
        <translation>라이트</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="192"/>
        <source>Language</source>
        <translation>언어</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="205"/>
        <source>Models…</source>
        <translation>모델…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="401"/>
        <source>Suggested: RAFTcorr-%1. Largest displacement of the region of interest about %2 px (99th percentile over frames %3); RAFTcorr-%1 reaches %4 px.</source>
        <translation>추천: RAFTcorr-%1. 관심 영역의 최대 변위는 약 %2 px(프레임 %3에서의 99번째 백분위수)이며, RAFTcorr-%1의 도달 범위는 %4 px입니다.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="493"/>
        <source>Language changed</source>
        <translation>언어가 변경되었습니다</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="688"/>
        <source>All Files</source>
        <translation>모든 파일</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="213"/>
        <source>Select Image Folder</source>
        <translation>이미지 폴더 선택</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="338"/>
        <source>File Association Failed</source>
        <translation>파일 연결 실패</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="341"/>
        <source>File Association</source>
        <translation>파일 연결</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="156"/>
        <source>Open Image Folder…</source>
        <translation>이미지 폴더 열기…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="158"/>
        <source>Open Project…</source>
        <translation>프로젝트 열기…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="159"/>
        <location filename="../../gui/app.py" line="245"/>
        <location filename="../../gui/app.py" line="252"/>
        <source>Save Project</source>
        <translation>프로젝트 저장</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="160"/>
        <source>Save Project As…</source>
        <translation>다른 이름으로 프로젝트 저장…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="162"/>
        <source>Export Results…</source>
        <translation>결과 내보내기…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="167"/>
        <source>Associate .raftcorr files with RAFTcorr…</source>
        <translation>.raftcorr 파일을 RAFTcorr에 연결…</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="168"/>
        <source>Register .raftcorr so double-clicking a project file opens RAFTcorr (current user only, no admin rights needed).</source>
        <translation>.raftcorr를 등록하여 프로젝트 파일을 두 번 클릭하면 RAFTcorr가 열리도록 합니다(현재 사용자만, 관리자 권한 불필요).</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="204"/>
        <source>&amp;Help</source>
        <translation>도움말</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="206"/>
        <location filename="../../gui/app.py" line="429"/>
        <location filename="../../gui/app.py" line="431"/>
        <source>Self-test</source>
        <translation>자체 테스트</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="207"/>
        <location filename="../../gui/app.py" line="436"/>
        <source>About RAFTcorr</source>
        <translation>RAFTcorr 정보</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="222"/>
        <source>Cancel</source>
        <translation>취소</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="246"/>
        <source>Load the images first: a project keeps the images, the regions of interest, the settings and the results together.</source>
        <translation>먼저 이미지를 불러오세요. 프로젝트는 이미지, 관심 영역, 설정, 결과를 함께 저장합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="267"/>
        <source>Project saved: %1</source>
        <translation>프로젝트 저장됨: %1</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="269"/>
        <source>Saving the project</source>
        <translation>프로젝트 저장 중</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="283"/>
        <source>Include Results?</source>
        <translation>결과를 포함할까요?</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="284"/>
        <source>Include the computed results in this project?</source>
        <translation>이 프로젝트에 계산 결과를 포함할까요?</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="286"/>
        <source>Including results (about %1 uncompressed) lets you reopen the project without recomputing. Choose No to save a small settings-only file for sharing.</source>
        <translation>결과(압축하지 않으면 약 %1)를 포함하면 다시 계산하지 않고 프로젝트를 다시 열 수 있습니다. 공유용으로 설정만 담은 작은 파일을 저장하려면 &apos;아니요&apos;를 선택하세요.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="298"/>
        <source>Open Project</source>
        <translation>프로젝트 열기</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="306"/>
        <source>Opening the project</source>
        <translation>프로젝트 여는 중</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="314"/>
        <source>Images not found</source>
        <translation>이미지를 찾을 수 없음</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="315"/>
        <source>The image folder of this project was not found. Choose its new location.</source>
        <translation>이 프로젝트의 이미지 폴더를 찾을 수 없습니다. 새 위치를 선택하세요.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="316"/>
        <source>Image folder of the project</source>
        <translation>프로젝트의 이미지 폴더</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="323"/>
        <source>Cannot open the project</source>
        <translation>프로젝트를 열 수 없습니다</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="329"/>
        <source>Project opened: %1</source>
        <translation>프로젝트 열림: %1</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="339"/>
        <source>Could not register .raftcorr files: </source>
        <translation>.raftcorr 파일을 등록할 수 없습니다: </translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="342"/>
        <source>Done. Double-clicking a .raftcorr file will now open RAFTcorr and restore that project.</source>
        <translation>완료되었습니다. 이제 .raftcorr 파일을 두 번 클릭하면 RAFTcorr가 열리고 해당 프로젝트가 복원됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="347"/>
        <source>Load at least two images first.</source>
        <translation>먼저 이미지를 두 장 이상 불러오세요.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="350"/>
        <source>Wait for the running task to finish.</source>
        <translation>실행 중인 작업이 끝날 때까지 기다리세요.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="368"/>
        <source>Survey: frame %1 against frame %2</source>
        <translation>움직임 측정: 프레임 %1 대 프레임 %2</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="395"/>
        <source>Suggested: RAFTcorr-%1. Largest displacement of the region of interest between a frame and its reference frame about %2 px (99th percentile over the pairs reference → frame %3); RAFTcorr-%1 reaches %4 px.</source>
        <translation>추천: RAFTcorr-%1. 각 프레임과 그 참조 프레임 사이의 관심 영역 최대 변위는 약 %2 px(참조 → 프레임 쌍 %3에서의 99번째 백분위수)이며, RAFTcorr-%1의 도달 범위는 %4 px입니다.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="404"/>
        <source>This exceeds even the reach of RAFTcorr-s8: use incremental tracking.</source>
        <translation>이는 RAFTcorr-s8의 도달 범위조차 넘습니다: 증분 추적을 사용하세요.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="406"/>
        <source>This exceeds even the reach of RAFTcorr-s8: the motion between consecutive frames is too large to track.</source>
        <translation>이는 RAFTcorr-s8의 도달 범위조차 넘습니다: 연속한 프레임 사이의 움직임이 너무 커서 추적할 수 없습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="409"/>
        <source>This exceeds even the reach of RAFTcorr-s8: update the reference frame more often.</source>
        <translation>이는 RAFTcorr-s8의 도달 범위조차 넘습니다: 참조 프레임을 더 자주 갱신하세요.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="415"/>
        <source>Measuring the motion with RAFTcorr-s8</source>
        <translation>RAFTcorr-s8로 움직임 측정 중</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="437"/>
        <source>&lt;b&gt;RAFTcorr %1&lt;/b&gt;&lt;br&gt;Learned digital image correlation with the RAFTcorr-s1, -s2, -s4 and -s8 models.&lt;br&gt;&lt;br&gt;Displacements: reference to deformed, +u right, +v down, in pixels, frame-1 (Lagrangian) coordinates.&lt;br&gt;&lt;br&gt;Interface derived from pyALDIC.&lt;br&gt;Zixiang (Zach) Tong, The University of Texas at Austin.</source>
        <translation>&lt;b&gt;RAFTcorr %1&lt;/b&gt;&lt;br&gt;RAFTcorr-s1, -s2, -s4, -s8 모델을 사용하는 학습 기반 디지털 이미지 상관법.&lt;br&gt;&lt;br&gt;변위: 참조에서 변형으로, +u 오른쪽, +v 아래쪽, 픽셀 단위, 프레임 1(라그랑주) 좌표.&lt;br&gt;&lt;br&gt;인터페이스는 pyALDIC에서 파생되었습니다.&lt;br&gt;Zixiang (Zach) Tong, The University of Texas at Austin.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="496"/>
        <source>Language set to %1. Please restart RAFTcorr for all widgets to pick up the new language.</source>
        <translation>언어가 %1(으)로 설정되었습니다. 모든 화면에 반영하려면 RAFTcorr를 재시작하세요.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="613"/>
        <source>Could not read the mask file %1: %2</source>
        <translation>마스크 파일 %1을(를) 읽을 수 없습니다: %2</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="640"/>
        <source>Frame %1 has no Region of Interest of its own — frame 1&apos;s is used for computation. Switch to frame 1 to edit it, or import a mask to give this frame its own.</source>
        <translation>프레임 %1에는 자체 관심 영역이 없습니다 — 계산에는 프레임 1의 관심 영역을 사용합니다. 프레임 1로 전환하여 편집하거나 마스크를 가져와 이 프레임 전용으로 지정하세요.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="668"/>
        <source>No Region of Interest to save — load images first.</source>
        <translation>저장할 관심 영역이 없습니다 — 먼저 이미지를 불러오세요.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="678"/>
        <source>Region of Interest mask is empty.</source>
        <translation>관심 영역 마스크가 비어 있습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="685"/>
        <source>Save Region of Interest Mask</source>
        <translation>관심 영역 마스크 저장</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="687"/>
        <source>PNG Images</source>
        <translation>PNG 이미지</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="696"/>
        <source>Mask saved to %1</source>
        <translation>마스크를 %1에 저장했습니다</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="705"/>
        <source>No Region of Interest to invert — load images first.</source>
        <translation>반전할 관심 영역이 없습니다 — 먼저 이미지를 불러오세요.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="732"/>
        <source>Load images first.</source>
        <translation>먼저 이미지를 불러오세요.</translation>
    </message>
    <message>
        <location filename="../../gui/app.py" line="768"/>
        <source>  Imported mask for frame %1</source>
        <translation>  프레임 %1의 마스크를 가져왔습니다</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/app.py" line="774"/>
        <source>Batch import: %n mask(s) loaded</source>
        <translation>
            <numerusform>일괄 가져오기: 마스크 %n개를 불러왔습니다</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/app.py" line="801"/>
        <source>Imported Region of Interest for %n frame(s)</source>
        <translation>
            <numerusform>%n 프레임에 관심 영역 가져옴</numerusform>
        </translation>
    </message>
</context>
<context>
    <name>ModelWeights</name>
    <message>
        <location filename="../../gui/support/model_weights.py" line="27"/>
        <source>Downloading model %1: %2 of %3 MB</source>
        <extracomment>``status(fraction, text)``; an empty text ends the download status.</extracomment>
        <translation>모델 %1 다운로드 중: %2 / %3 MB</translation>
    </message>
    <message>
        <location filename="../../gui/support/model_weights.py" line="29"/>
        <source>Downloading model %1…</source>
        <translation>모델 %1 다운로드 중…</translation>
    </message>
</context>
<context>
    <name>ModelsDialog</name>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="115"/>
        <source>Models</source>
        <translation>모델</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="126"/>
        <source>RAFTcorr downloads a model the first time an analysis uses it and keeps it for later runs. Each model is about 20 MB.</source>
        <translation>RAFTcorr는 분석에서 모델을 처음 사용할 때 내려받아 이후 실행을 위해 보관합니다. 모델 하나는 약 20 MB입니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="136"/>
        <source>s = %1, reach %2 px</source>
        <translation>s = %1, 도달 범위 %2 px</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="139"/>
        <source>Download</source>
        <translation>다운로드</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="164"/>
        <source>Model folder: %1</source>
        <translation>모델 폴더: %1</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="171"/>
        <source>No internet on this computer? Download the .pth files from the &lt;a href=&quot;%1&quot;&gt;RAFTcorr model release&lt;/a&gt; on another computer, then import them here.</source>
        <translation>이 컴퓨터가 인터넷에 연결되지 않나요? 다른 컴퓨터에서 &lt;a href=&quot;%1&quot;&gt;RAFTcorr 모델 릴리스 페이지&lt;/a&gt;의 .pth 파일을 내려받은 뒤 여기서 가져오세요.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="180"/>
        <source>Download All</source>
        <translation>모두 다운로드</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="182"/>
        <source>Import Model Files…</source>
        <translation>모델 파일 가져오기…</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="184"/>
        <source>Open Model Folder</source>
        <translation>모델 폴더 열기</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="186"/>
        <source>Cancel Download</source>
        <translation>다운로드 취소</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="189"/>
        <source>Close</source>
        <translation>닫기</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="222"/>
        <source>Installed</source>
        <translation>설치됨</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="226"/>
        <source>Not installed (%1 MB)</source>
        <translation>설치되지 않음 (%1 MB)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="227"/>
        <source>Not installed</source>
        <translation>설치되지 않음</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="238"/>
        <source>&lt;b&gt;%1 is installed now.&lt;/b&gt; Close this window and start again.</source>
        <translation>&lt;b&gt;%1이(가) 이제 설치되었습니다.&lt;/b&gt; 이 창을 닫고 다시 시작하세요.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="242"/>
        <source>&lt;b&gt;%1 is needed, but it is not installed and could not be downloaded.&lt;/b&gt; Download it again below, or import the file.</source>
        <translation>&lt;b&gt;%1이(가) 필요하지만 설치되어 있지 않고 다운로드하지 못했습니다.&lt;/b&gt; 아래에서 다시 다운로드하거나 파일을 가져오세요.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="286"/>
        <source>Download cancelled.</source>
        <translation>다운로드를 취소했습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="280"/>
        <source>Downloaded: %1.</source>
        <translation>다운로드함: %1.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="283"/>
        <source>%1 could not be downloaded: %2</source>
        <translation>%1을(를) 다운로드하지 못했습니다: %2</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="310"/>
        <source>Import Model Files</source>
        <translation>모델 파일 가져오기</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="311"/>
        <source>RAFTcorr models (*.pth)</source>
        <translation>RAFTcorr 모델 (*.pth)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="325"/>
        <source>Imported: %1.</source>
        <translation>가져옴: %1.</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/models_dialog.py" line="327"/>
        <source>Not imported: %1. Only the model files of the release can be imported.</source>
        <translation>가져오지 않음: %1. 릴리스의 모델 파일만 가져올 수 있습니다.</translation>
    </message>
</context>
<context>
    <name>ParamPanel</name>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="90"/>
        <source>Iterations</source>
        <extracomment>The label column is as wide as its longest label in the current language, never narrower than the English layout and, past about 28 characters of the label font, wrapped rather than squeezing the inputs (pyALDIC&apos;s ParamPanel).</extracomment>
        <translation>반복 횟수</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="91"/>
        <source>Refinement iterations of the network (default: 20).</source>
        <translation>네트워크의 반복 정제 횟수입니다(기본값: 20).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="98"/>
        <source>Off</source>
        <comment>Displacement smoothing</comment>
        <translation>끔</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="101"/>
        <source>Smoothing</source>
        <translation>평활화</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="102"/>
        <source>Gaussian smoothing (σ in pixels) of the tracked displacement, kept in the results.
Off by default: smoothing coarsens the spatial resolution.</source>
        <translation>추적한 변위의 가우스 평활화(σ는 픽셀 단위)이며, 결과에 저장됩니다.
기본값은 끔: 평활화하면 공간 분해능이 낮아집니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="106"/>
        <source>Automatic</source>
        <translation>자동</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="110"/>
        <source>Device</source>
        <translation>장치</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="111"/>
        <source>Automatic uses the GPU when PyTorch sees one.</source>
        <translation>&apos;자동&apos;은 PyTorch가 GPU를 인식하면 GPU를 사용합니다.</translation>
    </message>
</context>
<context>
    <name>PhysicalUnitsWidget</name>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="56"/>
        <source>Use physical units</source>
        <translation>물리 단위 사용</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="67"/>
        <source>Physical size of one image pixel</source>
        <translation>이미지 픽셀 1개의 물리 크기</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="85"/>
        <source>/ px</source>
        <translation>/ px</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="88"/>
        <source>Pixel size</source>
        <translation>픽셀 크기</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="98"/>
        <source>Acquisition frame rate (used for velocity field)</source>
        <translation>획득 프레임 속도(속도장에 사용)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="100"/>
        <source>Frame rate</source>
        <translation>프레임 속도</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="168"/>
        <source>Disp: %1  Velocity: %2/s</source>
        <translation>변위: %1  속도: %2/s</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/physical_units_widget.py" line="173"/>
        <source>Disp: px  Velocity: px/fr</source>
        <translation>변위: px  속도: px/fr</translation>
    </message>
</context>
<context>
    <name>PipelineController</name>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="80"/>
        <source>Stopped.</source>
        <extracomment>Results larger than this are kept in memory-mapped files in a temporary folder.</extracomment>
        <translation>중지되었습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="94"/>
        <source>Analysis failed</source>
        <translation>분석 실패</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="176"/>
        <source>Reference frames without a Region of Interest of their own carry the one of frame 1 forward: %1</source>
        <translation>자체 관심 영역이 없는 참조 프레임은 프레임 1의 관심 영역을 이어서 사용합니다: %1</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="199"/>
        <source>Starting...</source>
        <translation>시작 중…</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="225"/>
        <source>The images changed: the running analysis is cancelled.</source>
        <translation>이미지가 바뀌어 실행 중인 분석을 취소했습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="255"/>
        <source>The images changed during the run: its results are discarded.</source>
        <translation>실행 중에 이미지가 바뀌어 그 결과를 폐기했습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_controller.py" line="268"/>
        <source>No frame was computed; the previous results are kept.</source>
        <translation>계산된 프레임이 없습니다. 이전 결과를 유지합니다.</translation>
    </message>
</context>
<context>
    <name>PipelineMessages</name>
    <message numerus="yes">
        <location filename="../../gui/controllers/pipeline_messages.py" line="75"/>
        <source>  %n frame(s) with custom ROI masks</source>
        <translation>
            <numerusform>  %n 개 프레임에서 사용자 지정 ROI 마스크 사용</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/pipeline_messages.py" line="79"/>
        <source>Results received: %n frame(s)</source>
        <translation>
            <numerusform>결과 수신: %n 프레임</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/pipeline_messages.py" line="83"/>
        <source>  %n image(s), %1</source>
        <translation>
            <numerusform>  이미지 %n개, %1</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="86"/>
        <source>  ROI range: %1</source>
        <translation>  ROI 범위: %1</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="89"/>
        <source>  Refinement: %1</source>
        <translation>  메시 세분화: %1</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="93"/>
        <source>Error: %1</source>
        <translation>오류: %1</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="97"/>
        <source>Run canceled. Give the listed reference frames a Region of Interest of their own, or run again and continue with frame 1&apos;s region.</source>
        <translation>실행이 취소되었습니다. 나열된 참조 프레임에 각각 자체 관심 영역을 지정하거나, 다시 실행하여 프레임 1의 영역으로 계속하세요.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="104"/>
        <source>Tracking Quality Warning</source>
        <translation>추적 품질 경고</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="107"/>
        <source>Part of the sequence may not be tracked correctly.</source>
        <translation>시퀀스의 일부가 올바르게 추적되지 않았을 수 있습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="112"/>
        <source>The tracking quality drops from frame %1 on: the intensity residual rises to %2 grey levels, against %3 before.</source>
        <translation>프레임 %1부터 추적 품질이 떨어집니다. 밝기 잔차가 %2 회색 단계로 올라갔습니다(이전에는 %3).</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="118"/>
        <source>The displacement from frame 1 may exceed the reach of the model. Try incremental tracking, or a model with a larger s.</source>
        <translation>프레임 1로부터의 변위가 모델의 도달 범위를 넘었을 수 있습니다. 증분형 추적이나 s가 더 큰 모델을 사용해 보세요.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="121"/>
        <source>The step between reference frames may exceed the reach of the model. Use more reference frames, or a model with a larger s.</source>
        <translation>참조 프레임 사이의 변위가 모델의 도달 범위를 넘었을 수 있습니다. 참조 프레임을 늘리거나 s가 더 큰 모델을 사용하세요.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="125"/>
        <source>Frames with a high residual: %1</source>
        <translation>잔차가 큰 프레임: %1</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="128"/>
        <source>Show the Residual field to see where the images do not match after the displacement is applied.</source>
        <translation>&apos;잔차&apos; 필드를 표시하면 변위를 적용한 뒤 이미지가 일치하지 않는 곳을 볼 수 있습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="134"/>
        <source>Cannot start analysis</source>
        <translation>분석을 시작할 수 없음</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="138"/>
        <source>Need at least 2 images. Load an image folder in the %1 section before running DIC.</source>
        <translation>이미지가 2개 이상 필요합니다. DIC를 실행하기 전에 %1 섹션에서 이미지 폴더를 불러오세요.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="143"/>
        <source>Define a Region of Interest first: draw or import one for frame 1 in the %1 section.</source>
        <translation>먼저 관심 영역을 정의하세요. %1 섹션에서 프레임 1의 관심 영역을 그리거나 가져옵니다.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="148"/>
        <source>The Region of Interest of frame 1 is empty (no pixels selected). Draw it again in the %1 section.</source>
        <translation>프레임 1의 관심 영역이 비어 있습니다(선택된 픽셀 없음). %1 섹션에서 다시 그리세요.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="153"/>
        <source>Region of Interest too small</source>
        <translation>관심 영역이 너무 작음</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="159"/>
        <source>The Region of Interest is %1×%2 px, but the current %3 (%4) needs at least %5×%5 px. Enlarge the Region of Interest or reduce %3.</source>
        <translation>관심 영역이 %1×%2 px이지만 현재 %3(%4)에는 최소 %5×%5 px가 필요합니다. 관심 영역을 넓히거나 %3을(를) 줄이세요.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="165"/>
        <source>Starting Points required</source>
        <translation>시드점 필요</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="169"/>
        <source>Place at least one Starting Point on the canvas before running: click “%1” or “%2” in the %3 section.</source>
        <translation>실행하기 전에 캔버스에 시드점을 하나 이상 배치하세요. %3 섹션에서 “%1” 또는 “%2”를 클릭합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="175"/>
        <source>Invalid Starting Points</source>
        <translation>잘못된 시드점</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="179"/>
        <source>One or more Starting Points do not lie in any part of the Region of Interest. Click “%1” in the %2 section, then place the Starting Points again.</source>
        <translation>하나 이상의 시드점이 관심 영역의 어느 부분에도 있지 않습니다. %2 섹션에서 “%1”을(를) 클릭한 다음 시드점을 다시 배치하세요.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="187"/>
        <source>DIC analysis failed</source>
        <translation>DIC 분석 실패</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="190"/>
        <source>Failed to start analysis</source>
        <translation>분석 시작 실패</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="193"/>
        <source>Failed to process results</source>
        <translation>결과 처리 실패</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="198"/>
        <source>The analysis stopped with a runtime error:

%1

See the console log for the full traceback.</source>
        <translation>런타임 오류로 분석이 중지되었습니다:

%1

전체 트레이스백은 콘솔 로그를 참조하세요.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="204"/>
        <source>%1

See the console log for the full traceback.</source>
        <translation>%1

전체 트레이스백은 콘솔 로그를 참조하세요.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="209"/>
        <source>The GPU ran out of memory. Close other GPU programs, choose a model with a larger s, or lower %1 in %2.</source>
        <translation>GPU 메모리가 부족합니다. GPU를 사용하는 다른 프로그램을 닫거나, s가 더 큰 모델을 선택하거나, &apos;%2&apos;에서 &apos;%1&apos;을(를) 낮추세요.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="217"/>
        <source>There is not enough free GPU memory for the smallest window this model accepts. Close other GPU programs, or raise %1 in %2 if it is set low.</source>
        <translation>이 모델이 받아들이는 가장 작은 창에 필요한 여유 GPU 메모리가 부족합니다. GPU를 사용하는 다른 프로그램을 닫거나, &apos;%2&apos;의 &apos;%1&apos;이(가) 낮게 설정되어 있으면 높이세요.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="224"/>
        <source>Incomplete Region of Interest Coverage</source>
        <translation>관심 영역 적용 범위 불완전</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/pipeline_messages.py" line="229"/>
        <source>This incremental run has %n reference frame(s) with no Region of Interest of their own: frame(s) %1.</source>
        <translation>
            <numerusform>이번 증분 추적 실행에서 참조 프레임 %n개에 자체 관심 영역이 없습니다: 프레임 %1.</numerusform>
        </translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/controllers/pipeline_messages.py" line="235"/>
        <source>%1 and %n more</source>
        <comment>List of frame numbers cut short: %1 = the frame numbers shown, %n = how many more frames there are</comment>
        <translation>
            <numerusform>%1 외 %n개</numerusform>
        </translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="241"/>
        <source>A reference frame without a Region of Interest of its own reuses frame 1&apos;s region unchanged: nothing moves the region with the specimen. Where the specimen has moved out of that region, the part outside it is not measured from such a frame, and the nodes there stop being tracked or have their displacement copied from the nearest measured node. The run reports these nodes when it finishes.</source>
        <translation>자체 관심 영역이 없는 참조 프레임은 프레임 1의 영역을 그대로 재사용합니다. 이 영역은 시편을 따라 움직이지 않습니다. 시편이 이 영역 밖으로 이동하면 영역 밖의 부분은 그러한 참조 프레임에서 측정되지 않으며, 그곳의 노드는 더 이상 추적되지 않거나 가장 가까운 측정된 노드에서 변위가 복사됩니다. 실행이 끝나면 이러한 노드가 보고됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="251"/>
        <source>To measure the whole specimen, give each reference frame its own Region of Interest that covers the specimen in that frame (Batch Import can assign Regions of Interest to many frames at once). A reference frame with a region of its own is measured on a grid that covers that region.</source>
        <translation>시편 전체를 측정하려면 각 참조 프레임에 해당 프레임의 시편을 덮는 자체 관심 영역을 지정하세요(일괄 가져오기로 여러 프레임에 관심 영역을 한 번에 지정할 수 있습니다). 자체 영역이 있는 참조 프레임은 그 영역을 덮는 측정 격자에서 측정됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/controllers/pipeline_messages.py" line="258"/>
        <source>Continue with frame 1&apos;s region?</source>
        <translation>프레임 1의 영역으로 계속하시겠습니까?</translation>
    </message>
</context>
<context>
    <name>ROIHint</name>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="67"/>
        <source>Load images first, then draw a Region of Interest on frame 1.</source>
        <extracomment>The same card for other hints of the left sidebar (the model advice).</extracomment>
        <translation>먼저 이미지를 불러온 후, 프레임 1에 관심 영역을 그리세요.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="74"/>
        <source>&lt;b&gt;Accumulative mode&lt;/b&gt; — only frame 1 needs a Region of Interest. All later frames are compared against it directly.</source>
        <translation>&lt;b&gt;누적 모드&lt;/b&gt; — 관심 영역은 프레임 1에만 필요합니다. 이후 프레임은 모두 직접 비교됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="87"/>
        <source>&lt;b&gt;Incremental&lt;/b&gt; — only frame 1 needs a Region of Interest. RAFTcorr carries it to the reference frames (&lt;b&gt;%1&lt;/b&gt;) with the measured displacement. Draw on another frame only to replace that region or to leave out cracks and holes.</source>
        <translation>&lt;b&gt;증분&lt;/b&gt; — 관심 영역은 프레임 1에만 필요합니다. RAFTcorr가 측정된 변위로 이를 참조 프레임(&lt;b&gt;%1&lt;/b&gt;)으로 옮깁니다. 다른 프레임에는 해당 영역을 대체하거나 균열과 구멍을 제외할 때만 그리세요.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_hint.py" line="96"/>
        <source>Draw a Region of Interest on frame 1.</source>
        <translation>프레임 1에 관심 영역을 그리세요.</translation>
    </message>
</context>
<context>
    <name>ROIToolbar</name>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="72"/>
        <source>+ Add</source>
        <translation>+ 추가</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="74"/>
        <source>Add region to the Region of Interest (Polygon / Rectangle / Circle)</source>
        <translation>관심 영역에 도형을 추가합니다(다각형 / 사각형 / 원)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="79"/>
        <source>Cut</source>
        <translation>잘라내기</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="81"/>
        <source>Cut region from the Region of Interest (Polygon / Rectangle / Circle)</source>
        <translation>관심 영역에서 도형을 잘라냅니다(다각형 / 사각형 / 원)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="86"/>
        <source>+ Refine</source>
        <translation>+ 세분화</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="90"/>
        <source>Paint extra mesh-refinement zones with a brush
(only on frame 1 — material points auto-warped to later frames)</source>
        <translation>브러시로 추가 메시 세분화 영역을 칠합니다
(프레임 1에서만 — 후속 프레임으로 자동 워프됨)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="94"/>
        <source>Refine brush is only available on frame 1. Switch to frame 1 to paint refinement zones; they are automatically warped to later frames.</source>
        <translation>세분화 브러시는 프레임 1에서만 사용할 수 있습니다. 프레임 1로 전환해 영역을 칠하세요. 후속 프레임으로 자동 워프됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="118"/>
        <source>Import</source>
        <translation>가져오기</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="119"/>
        <source>Import mask from image file</source>
        <translation>이미지 파일에서 마스크 가져오기</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="124"/>
        <source>Batch Import</source>
        <translation>일괄 가져오기</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="126"/>
        <source>Batch import mask files for multiple frames</source>
        <translation>여러 프레임의 마스크 파일을 일괄 가져오기</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="139"/>
        <source>Save</source>
        <translation>저장</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="140"/>
        <source>Save current mask to PNG file</source>
        <translation>현재 마스크를 PNG 파일로 저장</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="145"/>
        <source>Invert</source>
        <translation>반전</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="147"/>
        <source>Invert the Region of Interest mask</source>
        <translation>관심 영역 마스크 반전</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="152"/>
        <source>Clear</source>
        <translation>지우기</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="154"/>
        <source>Clear the Region of Interest of the current frame</source>
        <translation>현재 프레임의 관심 영역 지우기</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="186"/>
        <source>Radius</source>
        <translation>반경</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="204"/>
        <source>Paint</source>
        <translation>칠하기</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="209"/>
        <source>Erase</source>
        <translation>지우개</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="219"/>
        <source>Clear Brush</source>
        <translation>브러시 지우기</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="247"/>
        <source>Polygon</source>
        <translation>다각형</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="251"/>
        <source>Rectangle</source>
        <translation>사각형</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="255"/>
        <source>Circle</source>
        <translation>원</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="259"/>
        <source>Circle (3-point)</source>
        <translation>원(3점)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="352"/>
        <source>Import Mask Image</source>
        <translation>마스크 이미지 가져오기</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="354"/>
        <source>Images</source>
        <translation>이미지</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/roi_toolbar.py" line="355"/>
        <source>All Files</source>
        <translation>모든 파일</translation>
    </message>
</context>
<context>
    <name>RightSidebar</name>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="76"/>
        <source>Run DIC Analysis</source>
        <translation>DIC 분석 실행</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="89"/>
        <source>Cancel</source>
        <translation>취소</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="94"/>
        <source>Cancel the current analysis. Frames already computed are kept so you can review or export the partial run.</source>
        <translation>현재 분석을 취소합니다. 이미 계산된 프레임은 유지되므로 부분 결과를 검토하거나 내보낼 수 있습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="102"/>
        <source>Export Results</source>
        <translation>결과 내보내기</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="110"/>
        <source>Open Strain Window</source>
        <translation>변형률 창 열기</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="113"/>
        <source>Compute and visualize strain in a separate post-processing window. Requires displacement results from a completed Run.</source>
        <translation>별도의 후처리 창에서 변형률을 계산·시각화합니다. 완료된 실행의 변위 결과가 필요합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="122"/>
        <source>PROGRESS</source>
        <translation>진행률</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="131"/>
        <location filename="../../gui/panels/right_sidebar.py" line="445"/>
        <source>Ready</source>
        <translation>준비됨</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="139"/>
        <location filename="../../gui/panels/right_sidebar.py" line="447"/>
        <location filename="../../gui/panels/right_sidebar.py" line="561"/>
        <source>ELAPSED  %1</source>
        <translation>경과 시간  %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="144"/>
        <location filename="../../gui/panels/right_sidebar.py" line="449"/>
        <location filename="../../gui/panels/right_sidebar.py" line="569"/>
        <location filename="../../gui/panels/right_sidebar.py" line="573"/>
        <source>REMAINING  %1</source>
        <translation>남은 시간  %1</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="152"/>
        <source>FIELD</source>
        <translation>필드</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="161"/>
        <source>Show on</source>
        <translation>표시 기준</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="166"/>
        <source>Deformed frame</source>
        <translation>변형된 프레임</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="167"/>
        <source>Reference frame</source>
        <translation>참조 프레임</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="169"/>
        <source>Plot the field where the material points are in the current frame, or at their positions in the reference frame.</source>
        <translation>필드를 재료점의 현재 프레임 위치에 그리거나 참조 프레임에서의 위치에 그립니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="177"/>
        <source>Show background image</source>
        <translation>배경 이미지 표시</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="180"/>
        <source>Uncheck to show the field on its own, with no speckle image behind it.</source>
        <translation>선택을 해제하면 뒤에 스페클 이미지 없이 필드만 표시합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="196"/>
        <source>Hidden background</source>
        <translation>숨김 시 배경</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="198"/>
        <source>White</source>
        <translation>흰색</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="199"/>
        <source>Black</source>
        <translation>검정</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="200"/>
        <source>Transparent</source>
        <translation>투명</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="203"/>
        <source>What replaces the image when it is hidden. Transparency is kept for PNG and TIFF on export; other formats get white.</source>
        <translation>배경을 숨겼을 때 이미지를 대신할 채우기입니다. 내보낼 때 PNG와 TIFF는 투명도를 유지하며, 다른 형식은 흰색이 됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="212"/>
        <source>Hide untracked regions</source>
        <translation>추적되지 않은 영역 숨기기</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="215"/>
        <source>Leave points without a result transparent, for example material that left the field of view, instead of filling them with the nearest value inside the region of interest. Applies to the Reference frame view only: the Deformed frame view always leaves them empty. Also used for exported images and animations; data files always keep NaN there.</source>
        <translation>결과가 없는 점(시야 밖으로 나간 재료 등)을 관심 영역 안의 가장 가까운 값으로 채우지 않고 투명하게 둡니다. &apos;참조 프레임&apos; 보기에만 적용되며, &apos;변형된 프레임&apos; 보기에서는 항상 비워 둡니다. 내보낸 이미지와 애니메이션에도 적용되며, 데이터 파일에서는 항상 NaN입니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="227"/>
        <source>VISUALIZATION</source>
        <translation>시각화</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="232"/>
        <source>Colormap</source>
        <translation>색상 맵</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="251"/>
        <source>Opacity</source>
        <translation>불투명도</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="258"/>
        <source>Overlay opacity (0 = transparent, 100 = opaque)</source>
        <translation>오버레이 불투명도(0 = 투명, 100 = 불투명)</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="264"/>
        <source>PHYSICAL UNITS</source>
        <translation>물리 단위</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="270"/>
        <source>LOG</source>
        <translation>로그</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="276"/>
        <source>Clear</source>
        <translation>지우기</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="353"/>
        <source>Run DIC first -- no displacement results to export.</source>
        <translation>DIC를 먼저 실행하세요 —— 내보낼 변위 결과가 없습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="412"/>
        <source>Place at least one Starting Point in each red region before running (red = needs a Starting Point).</source>
        <translation>실행 전에 각 빨간 영역에 시드점을 하나 이상 배치하세요(빨강 = 시드점 필요).</translation>
    </message>
    <message>
        <location filename="../../gui/panels/right_sidebar.py" line="544"/>
        <source>%1  —  Frame %2</source>
        <translation>%1  —  프레임 %2</translation>
    </message>
</context>
<context>
    <name>StrainController</name>
    <message>
        <location filename="../../gui/controllers/strain_controller.py" line="96"/>
        <source>Strain frame %1/%2</source>
        <translation>변형률: 프레임 %1/%2</translation>
    </message>
</context>
<context>
    <name>StrainFieldSelector</name>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="78"/>
        <source>Magnitude</source>
        <extracomment>GUI field name -&gt; FieldProvider key.</extracomment>
        <translation>변위 크기</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="79"/>
        <source>Velocity</source>
        <translation>속도</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="200"/>
        <source>DISPLACEMENT</source>
        <translation>변위</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="211"/>
        <source>STRAIN</source>
        <translation>변형률</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="265"/>
        <source>Horizontal displacement u (+ right)</source>
        <translation>수평 변위 u(오른쪽이 +)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="266"/>
        <source>Vertical displacement v (+ down)</source>
        <translation>수직 변위 v(아래쪽이 +)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="267"/>
        <source>Displacement magnitude |u|</source>
        <translation>변위 크기 |u|</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="269"/>
        <source>Speed |du/dt|: central difference of the displacement over the neighbouring computed frames, times the frame rate</source>
        <translation>속력 |du/dt|: 인접한 계산된 프레임 간 변위의 중심 차분에 프레임 속도를 곱한 값</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="271"/>
        <source>Normal strain εxx</source>
        <translation>수직 변형률 εxx</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="272"/>
        <source>Normal strain εyy</source>
        <translation>수직 변형률 εyy</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="274"/>
        <source>Shear strain εxy (tensor component: half the engineering shear γxy)</source>
        <translation>전단 변형률 εxy(텐서 성분: 공학 전단 변형률 γxy의 절반)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="276"/>
        <source>Major principal strain ε₁</source>
        <translation>최대 주변형률 ε₁</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="277"/>
        <source>Minor principal strain ε₂</source>
        <translation>최소 주변형률 ε₂</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="279"/>
        <source>Maximum in-plane shear strain (ε₁ − ε₂) / 2 (tensor component: half the engineering shear)</source>
        <translation>최대 면내 전단 변형률 (ε₁ − ε₂) / 2(텐서 성분: 공학 전단 변형률의 절반)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="282"/>
        <source>Von Mises equivalent strain (incompressible material)</source>
        <translation>von Mises 등가 변형률(비압축성 재료)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="284"/>
        <source>Rigid rotation in degrees (polar decomposition), clockwise on screen positive</source>
        <translation>강체 회전각(도, 극분해), 화면에서 시계 방향이 양수</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="287"/>
        <source>Direction of ε₁ in degrees from +x, clockwise on screen positive</source>
        <translation>+x에서 잰 ε₁의 방향(도), 화면에서 시계 방향이 양수</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_field_selector.py" line="290"/>
        <source>Relative area change ΔA/A₀ = det F − 1</source>
        <translation>상대 면적 변화 ΔA/A₀ = det F − 1</translation>
    </message>
</context>
<context>
    <name>StrainNavigator</name>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="74"/>
        <source>Previous frame</source>
        <translation>이전 프레임</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="85"/>
        <location filename="../../gui/widgets/strain_navigator.py" line="220"/>
        <source>Play animation</source>
        <translation>애니메이션 재생</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="89"/>
        <location filename="../../gui/widgets/strain_navigator.py" line="219"/>
        <source>▶</source>
        <translation>▶</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="96"/>
        <source>Next frame</source>
        <translation>다음 프레임</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="113"/>
        <source>Playback speed</source>
        <translation>재생 속도</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="118"/>
        <location filename="../../gui/widgets/strain_navigator.py" line="232"/>
        <source>FRAME 0/0</source>
        <translation>프레임 0/0</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="209"/>
        <source>⏸</source>
        <translation>⏸</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="210"/>
        <source>Pause animation</source>
        <translation>애니메이션 일시정지</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_navigator.py" line="227"/>
        <source>FRAME %1/%2</source>
        <translation>프레임 %1/%2</translation>
    </message>
</context>
<context>
    <name>StrainParamPanel</name>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="88"/>
        <source>Plane fitting</source>
        <extracomment>Strain measures in combo order (pyALDIC&apos;s order: infinitesimal, Eulerian, Green-Lagrange). The configuration in the controls changed (edited by the user).</extracomment>
        <translation>평면 피팅</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="91"/>
        <source>Polynomial fitted to each displacement component over the VSG window. Its slopes at the centre are the displacement gradients.
• Plane fitting (1st order): the standard VSG (default).
• Quadratic fitting (2nd order): follows strain gradients inside the window better, at the cost of more noise.</source>
        <translation>VSG 창 안에서 각 변위 성분에 다항식을 피팅합니다. 중심에서의 기울기가 변위 구배입니다.
• 평면 피팅(1차): 표준 VSG(기본값).
• 2차 피팅(2차): 창 내부의 변형률 구배를 더 잘 따르지만 노이즈가 커집니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="97"/>
        <source>Method</source>
        <translation>방법</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="112"/>
        <source>VSG size</source>
        <translation>VSG 크기</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="123"/>
        <source>Hides low-confidence strain along the edges of the region of interest, holes and cracks, where the edge cuts the VSG window and the local fit becomes one-sided and unreliable.

• Coefficient × VSG half-width (Lg − 1) / 2 = width of the trimmed band.
• 0.00 = keep every point (no trimming).
• 0.70 = default.
• 1.00 = strictest (trim every point whose window touches an edge).

Trimmed points are empty (NaN) in exported data files.</source>
        <translation>관심 영역, 구멍, 균열의 가장자리에서 가장자리가 VSG 창을 잘라 국소 피팅이 한쪽으로 치우쳐 신뢰할 수 없게 되는 곳의 저신뢰도 변형률을 숨깁니다.

• 계수 × VSG 반폭 (Lg − 1) / 2 = 잘라내는 띠의 폭.
• 0.00 = 모든 점 유지(잘라내기 없음).
• 0.70 = 기본값.
• 1.00 = 가장 엄격(창이 가장자리에 닿는 모든 점을 잘라냄).

잘라낸 점은 내보낸 데이터 파일에서 비어 있습니다(NaN).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="154"/>
        <source>Weight of the points inside the VSG window: all equal (uniform, default), or decreasing with the distance from the centre (Gaussian).</source>
        <translation>VSG 창 안 점들의 가중치: 모두 같음(균일, 기본값) 또는 중심에서 멀어질수록 감소(가우스).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="178"/>
        <source>Spacing of the strain points: 1 keeps strain at every pixel; larger values keep every n-th point, so the results take less memory and smaller files for long sequences. The gauge fit still runs at every pixel, so the computation takes about as long.</source>
        <translation>변형률 점의 간격: 1이면 모든 픽셀의 변형률을 유지합니다. 값이 크면 n번째 점마다 하나씩 유지하므로, 긴 시퀀스에서 결과가 메모리를 덜 차지하고 파일도 작아집니다. VSG 피팅은 여전히 모든 픽셀에서 수행되므로 계산 시간은 거의 같습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="191"/>
        <source>Off</source>
        <comment>Strain smoothing preset</comment>
        <translation>끔</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="89"/>
        <source>Quadratic fitting</source>
        <translation>2차 피팅</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="105"/>
        <source>VSG (Virtual Strain Gauge) size Lg is the side, in pixels, of the square window around each point over which the displacement is fitted. Strain is then taken from the fit&apos;s slopes.

• Larger VSG → smoother strain, lower spatial resolution.
• Smaller VSG → sharper strain, more noise.
• Odd, so that the window has a centre pixel (default: 41 px).</source>
        <translation>VSG(가상 변형률 게이지) 크기 Lg는 각 점 주위에서 변위를 피팅하는 정사각형 창의 한 변 길이(픽셀)입니다. 변형률은 피팅의 기울기에서 구합니다.

• VSG가 클수록 → 변형률이 매끄럽고 공간 분해능이 낮아집니다.
• VSG가 작을수록 → 변형률이 선명하고 노이즈가 많아집니다.
• 창에 중심 픽셀이 있도록 홀수로 합니다(기본값: 41 px).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="133"/>
        <source>Trim low-confidence edges</source>
        <translation>저신뢰도 가장자리 잘라내기</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="151"/>
        <source>Uniform</source>
        <translation>균일</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="152"/>
        <source>Gaussian</source>
        <translation>가우스</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="158"/>
        <source>Weighting</source>
        <translation>가중치</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="166"/>
        <source>Auto (Lg / 4)</source>
        <translation>자동(Lg / 4)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="168"/>
        <source>Standard deviation of the Gaussian weight. Auto uses a quarter of the VSG size.</source>
        <translation>가우스 가중치의 표준편차입니다. &apos;자동&apos;은 VSG 크기의 4분의 1을 사용합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="171"/>
        <source>Gaussian σ</source>
        <translation>가우스 σ</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="183"/>
        <source>Output spacing</source>
        <translation>출력 간격</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="193"/>
        <source>Gaussian smoothing (σ in pixels) of the displacement used by the strain fit only; the displacement results are not changed. Off (0) differentiates the measured displacement as it is (default).</source>
        <translation>변형률 피팅에 쓰는 변위만 가우스 평활화합니다(σ는 픽셀 단위). 변위 결과는 바뀌지 않습니다. 끔(0)이면 측정된 변위를 그대로 미분합니다(기본값).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="197"/>
        <source>Pre-fit smoothing</source>
        <translation>피팅 전 평활화</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="200"/>
        <source>Never fit across cracks or holes</source>
        <translation>균열이나 구멍을 가로질러 피팅하지 않음</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="202"/>
        <source>Fit each connected part of the tracked region separately, so no VSG window ever spans a crack, a hole or the gap between two regions.</source>
        <translation>추적 영역의 연결된 부분마다 따로 피팅하여, 어떤 VSG 창도 균열, 구멍 또는 두 영역 사이의 틈을 가로지르지 않도록 합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="211"/>
        <source>Infinitesimal</source>
        <translation>미소 변형률</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="212"/>
        <source>Eulerian</source>
        <translation>오일러 변형률</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="213"/>
        <source>Green-Lagrange</source>
        <translation>그린-라그랑주 변형률</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="215"/>
        <source>Strain measure from the deformation gradient F = I + ∇u:
• Infinitesimal: (∇u + ∇uᵀ) / 2 (small strains only).
• Eulerian (Euler-Almansi): (I − (F Fᵀ)⁻¹) / 2.
• Green-Lagrange: (Fᵀ F − I) / 2 (default).</source>
        <translation>변형 구배 F = I + ∇u로부터의 변형률 척도:
• 미소 변형률: (∇u + ∇uᵀ) / 2(작은 변형률에만).
• 오일러 변형률(Euler-Almansi): (I − (F Fᵀ)⁻¹) / 2.
• 그린-라그랑주 변형률: (Fᵀ F − I) / 2(기본값).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="220"/>
        <source>Strain type</source>
        <translation>변형률 종류</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_param_panel.py" line="274"/>
        <source>Trimmed: %1 points (%2%)</source>
        <translation>잘라냄: 점 %1개 (%2%)</translation>
    </message>
</context>
<context>
    <name>StrainVizPanel</name>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="67"/>
        <source>Deformed frame</source>
        <extracomment>Alias so existing references read unchanged; the list itself lives in raftcorr.gui.support.colormaps, so every chooser offers the same options.</extracomment>
        <translation>변형된 프레임</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="68"/>
        <source>Reference frame</source>
        <translation>참조 프레임</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="70"/>
        <source>Plot the field where the material points are in the current frame, or at their positions in the reference frame.</source>
        <translation>필드를 재료점의 현재 프레임 위치에 그리거나 참조 프레임에서의 위치에 그립니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="73"/>
        <source>Show on</source>
        <translation>표시 기준</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="75"/>
        <source>Show background image</source>
        <translation>배경 이미지 표시</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="78"/>
        <source>Uncheck to show the field on its own, with no speckle image behind it.</source>
        <translation>선택을 해제하면 뒤에 스페클 이미지 없이 필드만 표시합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="84"/>
        <source>White</source>
        <translation>흰색</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="85"/>
        <source>Black</source>
        <translation>검정</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="86"/>
        <source>Transparent</source>
        <translation>투명</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="89"/>
        <source>What replaces the image when it is hidden. Transparency is kept for PNG and TIFF on export; other formats get white.</source>
        <translation>배경을 숨겼을 때 이미지를 대신할 채우기입니다. 내보낼 때 PNG와 TIFF는 투명도를 유지하며, 다른 형식은 흰색이 됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="141"/>
        <source>Fill trimmed edges (display only)</source>
        <translation>잘라낸 가장자리 채우기 (표시 전용)</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="144"/>
        <source>Fill the edge-trimmed strain band from the nearest reliable point of the same region. Affects the view and exported images and animations; exported data files always keep the trimmed band as NaN.</source>
        <translation>가장자리가 잘린 변형률 띠를 같은 영역에서 가장 가까운 신뢰할 수 있는 점의 값으로 채웁니다. 화면 표시와 내보낸 이미지 및 애니메이션에 영향을 줍니다. 내보낸 데이터 파일은 잘라낸 띠를 항상 NaN으로 유지합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="151"/>
        <source>Hide untracked regions</source>
        <translation>추적되지 않은 영역 숨기기</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="154"/>
        <source>Leave points without a result transparent, for example material that left the field of view, instead of filling them with the nearest value inside the region of interest. Applies to the Reference frame view only: the Deformed frame view always leaves them empty. Also used for exported images and animations; data files always keep NaN there.</source>
        <translation>결과가 없는 점(시야 밖으로 나간 재료 등)을 관심 영역 안의 가장 가까운 값으로 채우지 않고 투명하게 둡니다. &apos;참조 프레임&apos; 보기에만 적용되며, &apos;변형된 프레임&apos; 보기에서는 항상 비워 둡니다. 내보낸 이미지와 애니메이션에도 적용되며, 데이터 파일에서는 항상 NaN입니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="81"/>
        <source>Background</source>
        <translation>배경</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="93"/>
        <source>Hidden background</source>
        <translation>숨김 시 배경</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="100"/>
        <source>Colormap</source>
        <translation>색상 맵</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="104"/>
        <source>Range</source>
        <translation>범위</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="124"/>
        <source>Min</source>
        <translation>최소</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="126"/>
        <source>Max</source>
        <translation>최대</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="136"/>
        <source>Opacity</source>
        <translation>불투명도</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/strain_viz_panel.py" line="148"/>
        <source>Edges</source>
        <translation>가장자리</translation>
    </message>
</context>
<context>
    <name>StrainWindow</name>
    <message>
        <location filename="../../gui/strain_window.py" line="125"/>
        <source>Strain Post-Processing</source>
        <translation>변형률 후처리</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="210"/>
        <source>Fit</source>
        <translation>맞춤</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="211"/>
        <source>Fit image to viewport</source>
        <translation>이미지를 뷰포트에 맞춤</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="217"/>
        <source>100%</source>
        <translation>100%</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="218"/>
        <source>Zoom to 100% (1:1)</source>
        <translation>100%(1:1) 확대</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="221"/>
        <source>Zoom in</source>
        <translation>확대</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="226"/>
        <source>–</source>
        <translation>–</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="284"/>
        <source>STRAIN PARAMETERS</source>
        <translation>변형률 매개변수</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="293"/>
        <source>ADVANCED</source>
        <translation>고급 설정</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="308"/>
        <source>Cancel</source>
        <translation>취소</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="312"/>
        <source>Cancel the running strain computation. The previous strain result is kept.</source>
        <translation>실행 중인 변형률 계산을 취소합니다. 이전 변형률 결과는 유지됩니다.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="356"/>
        <source>FIELD</source>
        <translation>필드</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="363"/>
        <source>VISUALIZATION</source>
        <translation>시각화</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="375"/>
        <source>PHYSICAL UNITS</source>
        <translation>물리 단위</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="381"/>
        <source>LOG</source>
        <translation>로그</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="863"/>
        <location filename="../../gui/strain_window.py" line="901"/>
        <source>No displacement results.</source>
        <translation>변위 결과가 없습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="903"/>
        <source>Frame %1 was not computed.</source>
        <translation>프레임 %1은(는) 계산되지 않았습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="911"/>
        <source>Click Compute Strain to show this field.</source>
        <translation>&apos;변형률 계산&apos;을 클릭하면 이 필드를 표시합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="912"/>
        <source>No strain for frame %1: click Compute Strain.</source>
        <translation>프레임 %1의 변형률이 없습니다: &apos;변형률 계산&apos;을 클릭하세요.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="914"/>
        <source>Velocity needs at least two computed frames.</source>
        <translation>속도에는 계산된 프레임이 두 개 이상 필요합니다.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="915"/>
        <source>No valid values in frame %1.</source>
        <translation>프레임 %1에 유효한 값이 없습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="170"/>
        <source>Strain Field</source>
        <translation>변형률장</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="324"/>
        <source>Export displacement and strain results: data files (NPZ, MAT, HDF5, CSV), pictures (PNG, TIFF, JPEG, PDF, SVG), animations (GIF, MP4) and an HTML report</source>
        <translation>변위 및 변형률 결과 내보내기: 데이터 파일(NPZ, MAT, HDF5, CSV), 그림(PNG, TIFF, JPEG, PDF, SVG), 애니메이션(GIF, MP4), HTML 보고서</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="421"/>
        <source>Analysis is not available</source>
        <translation>분석을 사용할 수 없습니다</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="424"/>
        <location filename="../../gui/strain_window.py" line="427"/>
        <source>Analysis</source>
        <translation>분석</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="425"/>
        <source>Analysis is not available (see the application log).</source>
        <translation>분석을 사용할 수 없습니다(애플리케이션 로그를 확인하세요).</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="536"/>
        <source>Strain compute failed: %1: %2</source>
        <translation>변형률 계산 실패: %1: %2</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="542"/>
        <location filename="../../gui/strain_window.py" line="605"/>
        <source>Strain computation complete.</source>
        <translation>변형률 계산 완료.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="553"/>
        <source>Strain window: no displacement results to post-process.</source>
        <translation>변형률 창: 후처리할 변위 결과가 없습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="598"/>
        <source>The displacement results changed during the strain computation; its result was discarded.</source>
        <translation>변형률 계산 중에 변위 결과가 바뀌어 그 결과를 폐기했습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="622"/>
        <source>Strain computation canceled.</source>
        <translation>변형률 계산 취소됨.</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="629"/>
        <source>Strain compute failed: %1</source>
        <translation>변형률 계산 실패: %1</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="633"/>
        <source>Strain Computation Failed</source>
        <translation>변형률 계산 실패</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="673"/>
        <location filename="../../gui/strain_window.py" line="675"/>
        <source>Export is not available: %1</source>
        <translation>내보내기를 사용할 수 없습니다: %1</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="687"/>
        <source>⚠ Params changed -- click Compute Strain</source>
        <translation>⚠ 매개변수가 변경됨 — 「변형률 계산」을 클릭하세요</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="227"/>
        <source>Zoom out</source>
        <translation>축소</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="299"/>
        <source>Compute Strain</source>
        <translation>변형률 계산</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="321"/>
        <location filename="../../gui/strain_window.py" line="674"/>
        <source>Export Results</source>
        <translation>결과 내보내기</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="565"/>
        <source>Starting…</source>
        <translation>시작 중…</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="583"/>
        <source>Canceling…</source>
        <translation>취소 중…</translation>
    </message>
    <message>
        <location filename="../../gui/strain_window.py" line="602"/>
        <source>Complete</source>
        <translation>완료됨</translation>
    </message>
</context>
<context>
    <name>WorkflowTypePanel</name>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="72"/>
        <source>Incremental</source>
        <translation>증분형</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="71"/>
        <source>Accumulative</source>
        <extracomment>Border of a field whose text was refused (the run keeps the last valid value).</extracomment>
        <translation>누적형</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="86"/>
        <source>Tracking Mode</source>
        <translation>추적 모드</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="232"/>
        <source>finest spatial resolution; small displacements</source>
        <translation>가장 높은 공간 분해능; 작은 변위</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="233"/>
        <source>fine resolution, several times faster than s1; small to moderate displacements</source>
        <translation>높은 분해능, s1보다 몇 배 빠름; 작거나 중간 정도의 변위</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="234"/>
        <source>balanced resolution and capture range</source>
        <translation>분해능과 도달 범위의 균형</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="235"/>
        <source>largest capture range, coarsest resolution; large motion and rotation</source>
        <translation>가장 넓은 도달 범위, 가장 거친 분해능; 큰 움직임과 회전</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="240"/>
        <source>%1: %2.
Reach (nominal lookup radius 32 s): %3 px.</source>
        <translation>%1: %2.
도달 범위(공칭 조회 반경 32 s): %3 px.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="243"/>
        <source>Weights not installed.</source>
        <translation>가중치 파일이 설치되지 않았습니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="105"/>
        <source>RAFTcorr model. A smaller s resolves finer detail; a larger s
reaches further (the coarsest correlation level looks 32 s px away).

s1, s2: small displacements, strain concentrations.
s4: balanced.
s8: large motion or rotation, or no prior knowledge.</source>
        <translation>RAFTcorr 모델. s가 작을수록 더 미세한 세부를 분해하고, s가 클수록
더 멀리 도달합니다(가장 거친 상관 단계는 32 s px 떨어진 곳까지 봄).

s1, s2: 작은 변위, 변형률 집중.
s4: 균형.
s8: 큰 움직임이나 회전, 또는 사전 정보가 없을 때.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="112"/>
        <source>Model</source>
        <translation>모델</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="120"/>
        <source>Suggest a Model</source>
        <translation>모델 추천</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="178"/>
        <source>e.g. 5, 10, 20</source>
        <translation>예: 5, 10, 20</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="180"/>
        <source>Frame numbers (as in the image list) to use as reference frames, separated by commas. Frame 1 is always the first reference.</source>
        <translation>참조 프레임으로 쓸 프레임 번호(이미지 목록과 같음)를 쉼표로 구분해 입력합니다. 프레임 1은 항상 첫 번째 참조입니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="317"/>
        <source>Reference frames %1 dropped: a reference must be one of the frames 1 to %2 (the last frame cannot be a reference).</source>
        <translation>참조 프레임 %1을(를) 제외했습니다: 참조 프레임은 프레임 1부터 %2까지 중 하나여야 합니다(마지막 프레임은 참조가 될 수 없습니다).</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="339"/>
        <source>Invalid reference frames: use frame numbers from 1 on, separated by commas.</source>
        <translation>참조 프레임이 잘못되었습니다. 1부터 시작하는 프레임 번호를 쉼표로 구분해 입력하세요.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="123"/>
        <source>Measure the motion of the sequence quickly with s8 and pick the
finest model whose reach covers it.</source>
        <translation>s8로 시퀀스의 움직임을 빠르게 측정한 뒤, 도달 범위가 그 움직임을
포함하는 가장 세밀한 모델을 고릅니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="77"/>
        <source>Accumulative: every frame is compared directly with frame 1.
Preferred whenever the motion stays within the model&apos;s reach:
no error accumulates from frame to frame.

Incremental: each frame is compared with an earlier reference frame
and the steps are composed. Needed when the motion from frame 1
exceeds the model&apos;s reach (large stretch or rotation); every step
adds its error.</source>
        <translation>누적형: 모든 프레임을 프레임 1과 직접 비교합니다.
움직임이 모델의 도달 범위 안에 있으면 이 방식을 권장합니다:
프레임 간에 오차가 누적되지 않습니다.

증분형: 각 프레임을 그보다 앞선 참조 프레임과 비교하고
단계들을 합성합니다. 프레임 1로부터의 움직임이
모델의 도달 범위를 넘을 때(큰 신장이나 회전) 필요하며, 단계마다
오차가 더해집니다.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="145"/>
        <source>Every Frame</source>
        <translation>매 프레임</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="146"/>
        <source>Every N Frames</source>
        <translation>N 프레임마다</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="147"/>
        <source>Custom Frames</source>
        <translation>사용자 지정 프레임</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="150"/>
        <source>When the reference frame refreshes during incremental tracking.
Every Frame: reset reference every frame (smallest per-step displacement,
most robust for large deformation).
Every N Frames: reset every N frames (balance speed vs robustness).
Custom Frames: user-defined list of reference frame indices.</source>
        <translation>증분 추적 중 참조 프레임 갱신 시점.
매 프레임: 매 프레임마다 참조 프레임 리셋(단계 변위 최소,
큰 변형에 가장 견고).
N 프레임마다: N 프레임마다 리셋(속도-견고성 균형).
사용자 지정: 사용자 정의 참조 프레임 인덱스 목록.</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="159"/>
        <source>Reference Update</source>
        <translation>참조 프레임 갱신</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="169"/>
        <source>Update reference every N frames</source>
        <translation>N 프레임마다 참조 프레임 갱신</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="171"/>
        <source>Interval</source>
        <translation>간격</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="185"/>
        <source>Reference Frames</source>
        <translation>참조 프레임</translation>
    </message>
    <message>
        <location filename="../../gui/widgets/workflow_type_panel.py" line="343"/>
        <source>Invalid reference frames %1: use frames 1 to %2, separated by commas (the last frame cannot be a reference).</source>
        <translation>참조 프레임 %1이(가) 잘못되었습니다. 프레임 1부터 %2까지 중에서 쉼표로 구분해 입력하세요(마지막 프레임은 참조가 될 수 없습니다).</translation>
    </message>
</context>
<context>
    <name>_DropZone</name>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="119"/>
        <source>Drop image folder
or Browse</source>
        <translation>이미지 폴더를 드롭하거나
찾아보기</translation>
    </message>
    <message>
        <location filename="../../gui/panels/left_sidebar.py" line="129"/>
        <source>Select Image Folder</source>
        <translation>이미지 폴더 선택</translation>
    </message>
</context>
<context>
    <name>_MaskPreviewPanel</name>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="134"/>
        <source>Preview</source>
        <translation>미리보기</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="138"/>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="264"/>
        <source>(no image)</source>
        <translation>(이미지 없음)</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="158"/>
        <source>Image only</source>
        <translation>이미지만</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="159"/>
        <source>Image + Mask</source>
        <translation>이미지 + 마스크</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="160"/>
        <source>Mask only</source>
        <translation>마스크만</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="163"/>
        <source>View:</source>
        <translation>보기:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="169"/>
        <source>Alpha:</source>
        <translation>알파:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="177"/>
        <source>Blue</source>
        <comment>Mask overlay color</comment>
        <translation>파랑</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="181"/>
        <source>Red</source>
        <comment>Mask overlay color</comment>
        <translation>빨강</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="185"/>
        <source>Green</source>
        <comment>Mask overlay color</comment>
        <translation>초록</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="189"/>
        <source>Yellow</source>
        <comment>Mask overlay color</comment>
        <translation>노랑</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="193"/>
        <source>Mask color:</source>
        <translation>마스크 색상:</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="270"/>
        <source>No mask assigned</source>
        <translation>지정된 마스크 없음</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="276"/>
        <source>Frame %1 — %2</source>
        <translation>프레임 %1 — %2</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/batch_import_dialog.py" line="291"/>
        <source>Failed to load image</source>
        <translation>이미지를 불러오지 못했습니다</translation>
    </message>
</context>
<context>
    <name>dialog</name>
    <message>
        <location filename="../../gui/dialogs/export_widgets.py" line="322"/>
        <source>Cancel Export</source>
        <translation>내보내기 취소</translation>
    </message>
</context>
</TS>
