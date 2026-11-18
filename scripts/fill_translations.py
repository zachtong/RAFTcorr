"""Fill the current batch of unfinished translations (ported from pyALDIC).

Usage:
    python scripts/fill_translations.py

Reads src/raftcorr/i18n/source/raftcorr_<lang>.ts for every language in
LANGUAGES, finds each ``<translation type="unfinished">`` entry and rewrites it
as a finished ``<translation>`` from the tables below. Afterwards run
``python scripts/i18n.py compile`` to rebuild the .qm runtime catalogs.

Lookup order for one message (the first table that has it wins):

1. ``CONTEXTUAL[(context, source)]`` -- the same English text means something
   else in another class ("Margin" of the tiling context here, a blank border
   around an exported picture in pyALDIC's export dialog).
2. ``DISAMBIGUATED[(source, comment)]`` -- the call passed a disambiguation
   comment, ``self.tr("Light", "Color theme")``.
3. ``TRANSLATIONS[source]`` -- everything else.

Plural messages (``self.tr("%n file(s)", "", n)``, marked ``numerus="yes"`` by
lupdate) take their forms from ``NUMERUS_TRANSLATIONS``: one form for the CJK
languages, two (singular, plural) for de / fr / es.

The script is idempotent: finished entries are never touched, so a second run
changes nothing. It writes the catalogs byte-compatible with pyside6-lupdate
(same XML escaping, LF line endings), so the next ``i18n.py extract`` produces
no diff.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape
from xml.sax.saxutils import unescape as _xml_unescape

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TS_DIR = PROJECT_ROOT / "src" / "raftcorr" / "i18n" / "source"

LANGUAGES: tuple[str, ...] = ("zh_CN", "zh_TW", "ja", "ko", "de", "fr", "es")
#: Plural forms per language, as Qt's plural rules define them.
NUMERUS_FORMS: dict[str, int] = {
    "zh_CN": 1, "zh_TW": 1, "ja": 1, "ko": 1, "de": 2, "fr": 2, "es": 2,
}

# pyside6-lupdate serialises .ts text with the five XML predefined entities,
# i.e. it also escapes ' -> &apos; and " -> &quot;. The stdlib helpers only
# handle & < > by default; without the extra pairs every French apostrophe
# (l'image, d'abord) would drift on the next extract.
_XML_EXTRA_ESCAPE = {"'": "&apos;", '"': "&quot;"}
_XML_EXTRA_UNESCAPE = {"&apos;": "'", "&quot;": '"'}


def escape(text: str) -> str:
    """XML-escape exactly like pyside6-lupdate (includes &apos; and &quot;)."""
    return _xml_escape(text, _XML_EXTRA_ESCAPE)


def unescape(text: str) -> str:
    """Inverse of :func:`escape`."""
    return _xml_unescape(text, _XML_EXTRA_UNESCAPE)


# ---------------------------------------------------------------------------
# Translation tables. Keys are the English source exactly as written in the
# code; keep %1 %2 %n, HTML tags, units, symbols and product names literal.
# Terminology: docs/i18n/glossary.md.
# ---------------------------------------------------------------------------

TRANSLATIONS: dict[str, dict[str, str]] = {
    # ===== AdvancedPanel (ADVANCED: tiling and correlation implementation) =====
    "Margin = model reach": {
        "zh_CN": "边距 = 模型量程",
        "zh_TW": "邊距 = 模型量程",
        "ja":    "マージン = モデルの到達範囲",
        "ko":    "마진 = 모델 도달 범위",
        "de":    "Rand = Modellreichweite",
        "fr":    "Marge = portée du modèle",
        "es":    "Margen = alcance del modelo",
    },
    "Extra image context around the ROI so that displaced points stay inside\n"
    "the window given to the network. Automatic: the model's reach (32 s px).": {
        "zh_CN": "在 ROI 周围额外保留的图像范围，使发生位移的点仍位于\n送入网络的窗口内。自动：取模型量程（32 s px）。",
        "zh_TW": "在 ROI 周圍額外保留的影像範圍，使發生位移的點仍位於\n送入網路的視窗內。自動：取模型量程（32 s px）。",
        "ja":    "変位した点がネットワークに渡すウィンドウ内に収まるよう、ROI の周囲に\n追加で含める画像範囲です。自動：モデルの到達範囲（32 s px）。",
        "ko":    "변위된 점이 네트워크에 입력되는 창 안에 머물도록 ROI 주위에\n추가로 포함하는 이미지 영역입니다. 자동: 모델 도달 범위(32 s px).",
        "de":    "Zusätzlicher Bildbereich um die ROI, damit verschobene Punkte innerhalb\ndes Fensters bleiben, das das Netzwerk erhält. Automatisch: die Modellreichweite (32 s px).",
        "fr":    "Contexte d'image supplémentaire autour de la ROI, pour que les points déplacés restent\ndans la fenêtre fournie au réseau. Automatique : la portée du modèle (32 s px).",
        "es":    "Contexto de imagen adicional alrededor de la ROI para que los puntos desplazados queden\ndentro de la ventana que recibe la red. Automático: el alcance del modelo (32 s px).",
    },
    "Must cover the largest displacement of the ROI.": {
        "zh_CN": "必须覆盖 ROI 内的最大位移。",
        "zh_TW": "必須涵蓋 ROI 內的最大位移。",
        "ja":    "ROI 内の最大変位をカバーする必要があります。",
        "ko":    "ROI 내 최대 변위를 포함해야 합니다.",
        "de":    "Muss die größte Verschiebung in der ROI abdecken.",
        "fr":    "Doit couvrir le plus grand déplacement de la ROI.",
        "es":    "Debe cubrir el mayor desplazamiento de la ROI.",
    },
    "GPU memory": {
        "zh_CN": "GPU 显存",
        "zh_TW": "GPU 記憶體",
        "ja":    "GPU メモリ",
        "ko":    "GPU 메모리",
        "de":    "GPU-Speicher",
        "fr":    "Mémoire GPU",
        "es":    "Memoria GPU",
    },
    "Fraction of the free GPU memory one window may use.\nLarge regions are cut into tiles to fit it.": {
        "zh_CN": "单个窗口可使用的空闲 GPU 显存比例。\n较大的区域会被切分成多个分块以满足该限制。",
        "zh_TW": "單一視窗可使用的閒置 GPU 記憶體比例。\n較大的區域會被切分成多個分塊以符合此限制。",
        "ja":    "1 つのウィンドウが使用できる空き GPU メモリの割合です。\n大きな領域は、これに収まるようタイルに分割されます。",
        "ko":    "창 하나가 사용할 수 있는 여유 GPU 메모리 비율입니다.\n큰 영역은 이에 맞도록 타일로 나뉩니다.",
        "de":    "Anteil des freien GPU-Speichers, den ein Fenster nutzen darf.\nGroße Bereiche werden passend dazu in Kacheln zerlegt.",
        "fr":    "Fraction de la mémoire GPU libre qu'une fenêtre peut utiliser.\nLes grandes régions sont découpées en tuiles pour s'y conformer.",
        "es":    "Fracción de la memoria GPU libre que puede usar una ventana.\nLas regiones grandes se dividen en teselas para ajustarse a ella.",
    },
    "Tile blending": {
        "zh_CN": "分块融合",
        "zh_TW": "分塊融合",
        "ja":    "タイルのブレンド",
        "ko":    "타일 블렌딩",
        "de":    "Kachelüberblendung",
        "fr":    "Fondu des tuiles",
        "es":    "Fusión de teselas",
    },
    "Width of the feathered overlap between neighbouring tiles.": {
        "zh_CN": "相邻分块之间羽化重叠区的宽度。",
        "zh_TW": "相鄰分塊之間羽化重疊區的寬度。",
        "ja":    "隣接タイル間でフェザー処理する重なり領域の幅です。",
        "ko":    "인접 타일 사이에서 페더링되는 겹침 영역의 너비입니다.",
        "de":    "Breite der weich überblendeten Überlappung zwischen benachbarten Kacheln.",
        "fr":    "Largeur du chevauchement en fondu entre tuiles voisines.",
        "es":    "Anchura del solapamiento difuminado entre teselas vecinas.",
    },
    # Also the Device combo of ParamPanel (same meaning).
    "Automatic": {
        "zh_CN": "自动",
        "zh_TW": "自動",
        "ja":    "自動",
        "ko":    "자동",
        "de":    "Automatisch",
        "fr":    "Automatique",
        "es":    "Automático",
    },
    "On demand": {
        "zh_CN": "按需计算",
        "zh_TW": "隨需計算",
        "ja":    "オンデマンド",
        "ko":    "필요 시 계산",
        "de":    "Bei Bedarf",
        "fr":    "À la demande",
        "es":    "Bajo demanda",
    },
    "Stored volume": {
        "zh_CN": "预存相关体",
        "zh_TW": "預存相關體",
        "ja":    "ボリューム保持",
        "ko":    "볼륨 저장",
        "de":    "Gespeichertes Volumen",
        "fr":    "Volume stocké",
        "es":    "Volumen almacenado",
    },
    "How the correlation lookup is computed; the results are identical.\n"
    "Automatic: stored volume for s8 (fast), on demand for s1-s4\n"
    "(memory linear in the image size).": {
        "zh_CN": "相关查找的计算方式；两种方式的结果完全相同。\n自动：s8 使用预存相关体（较快），s1-s4 使用按需计算\n（内存随图像尺寸线性增长）。",
        "zh_TW": "相關查找的計算方式；兩種方式的結果完全相同。\n自動：s8 使用預存相關體（較快），s1-s4 使用隨需計算\n（記憶體隨影像尺寸線性增長）。",
        "ja":    "相関ルックアップの計算方法です。結果はどちらも同一です。\n自動：s8 ではボリューム保持（高速）、s1-s4 ではオンデマンド\n（メモリは画像サイズに比例）。",
        "ko":    "상관 조회를 계산하는 방식이며, 결과는 동일합니다.\n자동: s8은 볼륨 저장(빠름), s1-s4는 필요 시 계산\n(메모리가 이미지 크기에 비례).",
        "de":    "Wie die Korrelationsabfrage berechnet wird; die Ergebnisse sind identisch.\nAutomatisch: gespeichertes Volumen für s8 (schnell), bei Bedarf für s1-s4\n(Speicher linear in der Bildgröße).",
        "fr":    "Mode de calcul de la recherche de corrélation ; les résultats sont identiques.\nAutomatique : volume stocké pour s8 (rapide), à la demande pour s1-s4\n(mémoire proportionnelle à la taille de l'image).",
        "es":    "Cómo se calcula la consulta de correlación; los resultados son idénticos.\nAutomático: volumen almacenado para s8 (rápido), bajo demanda para s1-s4\n(memoria proporcional al tamaño de la imagen).",
    },

    # ===== AnalysisChart / AnalysisTab (probe notes, short and lowercase) =====
    "partly untracked": {
        "zh_CN": "部分未追踪",
        "zh_TW": "部分未追蹤",
        "ja":    "一部未追跡",
        "ko":    "일부 추적 안 됨",
        "de":    "teilweise nicht verfolgt",
        "fr":    "en partie hors suivi",
        "es":    "parcialmente sin seguimiento",
    },
    "no strain value (sparse strain gauge)": {
        "zh_CN": "无应变值（VSG 内点过少）",
        "zh_TW": "無應變值（VSG 內點過少）",
        "ja":    "ひずみ値なし（VSG 内の点が不足）",
        "ko":    "변형률 값 없음(VSG 내 점 부족)",
        "de":    "kein Dehnungswert (zu wenige Punkte im VSG)",
        "fr":    "aucune valeur de déformation (trop peu de points dans le VSG)",
        "es":    "sin valor de deformación (muy pocos puntos en el VSG)",
    },
    "partly untracked from frame %1": {
        "zh_CN": "第 %1 帧起部分未追踪",
        "zh_TW": "自第 %1 影格起部分未追蹤",
        "ja":    "フレーム %1 から一部未追跡",
        "ko":    "%1 프레임부터 일부 추적 안 됨",
        "de":    "ab Bild %1 teilweise nicht verfolgt",
        "fr":    "en partie hors suivi dès l'image %1",
        "es":    "parcialmente sin seguimiento desde el fotograma %1",
    },
    "gaps: no strain value": {
        "zh_CN": "有缺口：无应变值",
        "zh_TW": "有缺口：無應變值",
        "ja":    "欠損あり：ひずみ値なし",
        "ko":    "공백: 변형률 값 없음",
        "de":    "Lücken: kein Dehnungswert",
        "fr":    "lacunes : aucune valeur de déformation",
        "es":    "huecos: sin valor de deformación",
    },
    "Nothing valid along %1: the strain gauge found too few points there (near an edge or a hole). "
    "Plot a displacement, or use a smaller gauge on the Strain Field tab.": {
        "zh_CN": "%1 沿线没有有效数据：该处靠近边缘或孔，VSG 内的点过少。请改为绘制位移，或在“应变场”页减小 VSG 尺寸。",
        "zh_TW": "%1 沿線沒有有效資料：該處靠近邊緣或孔，VSG 內的點過少。請改為繪製位移，或在「應變場」頁減小 VSG 尺寸。",
        "ja":    "%1 に沿って有効なデータがありません：縁や穴の近くで、VSG 内の点が不足しています。変位を表示するか、「ひずみ場」タブで VSG サイズを小さくしてください。",
        "ko":    "%1을(를) 따라 유효한 데이터가 없습니다: 가장자리나 구멍 근처라 VSG 안의 점이 너무 적습니다. 변위를 표시하거나 '변형률장' 탭에서 VSG 크기를 줄이십시오.",
        "de":    "Keine gültigen Daten entlang %1: Das VSG findet dort zu wenige Punkte (nahe einem Rand oder Loch). Stellen Sie eine Verschiebung dar oder verwenden Sie im Reiter „Dehnungsfeld“ ein kleineres VSG.",
        "fr":    "Aucune donnée valide le long de %1 : le VSG y trouve trop peu de points (près d'un bord ou d'un trou). Tracez un déplacement, ou utilisez un VSG plus petit dans l'onglet « Champ de déformation ».",
        "es":    "No hay datos válidos a lo largo de %1: el VSG encuentra allí muy pocos puntos (cerca de un borde o un agujero). Represente un desplazamiento, o use un VSG más pequeño en la pestaña «Campo de deformación».",
    },
    "Nothing valid along %1: the material under it is untracked (cut out of the frame's ROI, or out of view).": {
        "zh_CN": "%1 沿线没有有效数据：其下方的材料未被追踪（已被该帧的 ROI 排除，或已移出视场）。",
        "zh_TW": "%1 沿線沒有有效資料：其下方的材料未被追蹤（已被該影格的 ROI 排除，或已移出視野）。",
        "ja":    "%1 に沿って有効なデータがありません：その下の材料が追跡されていません（そのフレームの ROI から除外されたか、視野の外に出ています）。",
        "ko":    "%1을(를) 따라 유효한 데이터가 없습니다: 그 아래의 재료가 추적되지 않습니다(해당 프레임의 ROI에서 제외되었거나 시야 밖으로 나갔습니다).",
        "de":    "Keine gültigen Daten entlang %1: Das Material darunter wird nicht verfolgt (aus der ROI des Bildes ausgeschnitten oder außerhalb des Sichtfelds).",
        "fr":    "Aucune donnée valide le long de %1 : le matériau en dessous n'est pas suivi (exclu de la ROI de l'image, ou hors du champ de vision).",
        "es":    "No hay datos válidos a lo largo de %1: el material que hay debajo no se sigue (recortado de la ROI del fotograma, o fuera de la vista).",
    },

    # ===== Application (crash dialog) ==========================================
    "RAFTcorr has hit an error": {
        "zh_CN": "RAFTcorr 发生错误",
        "zh_TW": "RAFTcorr 發生錯誤",
        "ja":    "RAFTcorr でエラーが発生しました",
        "ko":    "RAFTcorr에서 오류가 발생했습니다",
        "de":    "In RAFTcorr ist ein Fehler aufgetreten",
        "fr":    "RAFTcorr a rencontré une erreur",
        "es":    "RAFTcorr ha encontrado un error",
    },

    # ===== CanvasArea (ROI editing banner) =====================================
    "Editing the ROI of frame 1 (the reference)": {
        "zh_CN": "正在编辑第 1 帧（参考帧）的 ROI",
        "zh_TW": "正在編輯第 1 影格（參考影格）的 ROI",
        "ja":    "フレーム 1（参照）の ROI を編集中",
        "ko":    "프레임 1(참조)의 ROI 편집 중",
        "de":    "ROI von Bild 1 (Referenz) wird bearbeitet",
        "fr":    "Modification de la ROI de l'image 1 (la référence)",
        "es":    "Editando la ROI del fotograma 1 (la referencia)",
    },
    "Editing the mask of frame %1 (regions left out are excluded)": {
        "zh_CN": "正在编辑第 %1 帧的掩模（未包含的区域将被排除）",
        "zh_TW": "正在編輯第 %1 影格的遮罩（未包含的區域將被排除）",
        "ja":    "フレーム %1 のマスクを編集中（含まれない領域は除外されます）",
        "ko":    "프레임 %1의 마스크 편집 중(포함되지 않은 영역은 제외됨)",
        "de":    "Maske von Bild %1 wird bearbeitet (ausgesparte Bereiche werden ausgeschlossen)",
        "fr":    "Modification du masque de l'image %1 (les zones laissées de côté sont exclues)",
        "es":    "Editando la máscara del fotograma %1 (las zonas que quedan fuera se excluyen)",
    },

    # ===== ExportDialog ========================================================
    "Preview": {
        "zh_CN": "预览",
        "zh_TW": "預覽",
        "ja":    "プレビュー",
        "ko":    "미리보기",
        "de":    "Vorschau",
        "fr":    "Aperçu",
        "es":    "Vista previa",
    },
    "Export: images not readable (%1); pictures go on white.": {
        "zh_CN": "导出：无法读取图像（%1）；图片将以白色为背景。",
        "zh_TW": "匯出：無法讀取影像（%1）；圖片將以白色為背景。",
        "ja":    "エクスポート：画像を読み込めません（%1）。図は白背景で出力されます。",
        "ko":    "내보내기: 이미지를 읽을 수 없습니다(%1). 그림은 흰색 배경으로 출력됩니다.",
        "de":    "Export: Bilder nicht lesbar (%1); Abbildungen werden auf Weiß gezeichnet.",
        "fr":    "Export : images illisibles (%1) ; les figures sont tracées sur fond blanc.",
        "es":    "Exportación: imágenes ilegibles (%1); las figuras se dibujan sobre blanco.",
    },
    "Export: the images do not match the results' image size; pictures go on white.": {
        "zh_CN": "导出：图像尺寸与结果的图像尺寸不一致；图片将以白色为背景。",
        "zh_TW": "匯出：影像尺寸與結果的影像尺寸不一致；圖片將以白色為背景。",
        "ja":    "エクスポート：画像のサイズが結果の画像サイズと一致しません。図は白背景で出力されます。",
        "ko":    "내보내기: 이미지 크기가 결과의 이미지 크기와 일치하지 않습니다. 그림은 흰색 배경으로 출력됩니다.",
        "de":    "Export: Die Bilder passen nicht zur Bildgröße der Ergebnisse; Abbildungen werden auf Weiß gezeichnet.",
        "fr":    "Export : les images ne correspondent pas à la taille d'image des résultats ; les figures sont tracées sur fond blanc.",
        "es":    "Exportación: las imágenes no coinciden con el tamaño de imagen de los resultados; las figuras se dibujan sobre blanco.",
    },
    "This run has no intensity residual.": {
        "zh_CN": "本次运行没有灰度残差。",
        "zh_TW": "本次執行沒有灰階殘差。",
        "ja":    "この実行には輝度残差がありません。",
        "ko":    "이번 실행에는 밝기 잔차가 없습니다.",
        "de":    "Dieser Lauf hat kein Intensitätsresiduum.",
        "fr":    "Cette exécution n'a pas de résidu d'intensité.",
        "es":    "Esta ejecución no tiene residuo de intensidad.",
    },
    "MATLAB v7.3 (.mat)": {lang: "MATLAB v7.3 (.mat)" for lang in LANGUAGES},
    "An HDF5-based MAT file: load it in MATLAB with load().": {
        "zh_CN": "基于 HDF5 的 MAT 文件：在 MATLAB 中用 load() 读取。",
        "zh_TW": "以 HDF5 為基礎的 MAT 檔案：在 MATLAB 中以 load() 讀取。",
        "ja":    "HDF5 ベースの MAT ファイルです。MATLAB で load() により読み込めます。",
        "ko":    "HDF5 기반 MAT 파일입니다. MATLAB에서 load()로 불러오세요.",
        "de":    "Eine HDF5-basierte MAT-Datei: in MATLAB mit load() laden.",
        "fr":    "Un fichier MAT basé sur HDF5 : chargez-le dans MATLAB avec load().",
        "es":    "Un archivo MAT basado en HDF5: cárguelo en MATLAB con load().",
    },
    "HDF5 (.h5)": {lang: "HDF5 (.h5)" for lang in LANGUAGES},
    "One table per computed frame, one row per point of the region of interest.\n"
    "Dense fields make large files: a 1000 x 1000 px region is a million rows per frame.": {
        "zh_CN": "每个已计算帧一张表，感兴趣区域内每个点一行。\n稠密场会生成很大的文件：1000 x 1000 px 的区域每帧有一百万行。",
        "zh_TW": "每個已計算影格一張表，感興趣區域內每個點一列。\n密集場會產生很大的檔案：1000 x 1000 px 的區域每個影格有一百萬列。",
        "ja":    "計算済みフレームごとに 1 つの表、関心領域の点ごとに 1 行です。\n密な場はファイルが大きくなります：1000 x 1000 px の領域はフレームあたり 100 万行です。",
        "ko":    "계산된 프레임마다 표 하나, 관심 영역의 점마다 한 행입니다.\n조밀한 필드는 파일이 커집니다: 1000 x 1000 px 영역은 프레임당 백만 행입니다.",
        "de":    "Eine Tabelle pro berechnetem Bild, eine Zeile pro Punkt der Region of Interest.\nDichte Felder ergeben große Dateien: Ein Bereich von 1000 x 1000 px sind eine Million Zeilen pro Bild.",
        "fr":    "Une table par image calculée, une ligne par point de la région d'intérêt.\nLes champs denses produisent de gros fichiers : une région de 1000 x 1000 px fait un million de lignes par image.",
        "es":    "Una tabla por fotograma calculado, una fila por punto de la región de interés.\nLos campos densos generan archivos grandes: una región de 1000 x 1000 px son un millón de filas por fotograma.",
    },
    "✓ Grid coordinates (X, Y), ROI and computed-frame flags always included": {
        "zh_CN": "✓ 始终包含网格坐标（X, Y）、ROI 和已计算帧标记",
        "zh_TW": "✓ 一律包含網格座標（X, Y）、ROI 與已計算影格標記",
        "ja":    "✓ グリッド座標 (X, Y)、ROI、計算済みフレームのフラグは常に含まれます",
        "ko":    "✓ 격자 좌표(X, Y), ROI 및 계산된 프레임 플래그는 항상 포함됩니다",
        "de":    "✓ Gitterkoordinaten (X, Y), ROI und Kennzeichnung berechneter Bilder immer enthalten",
        "fr":    "✓ Coordonnées de grille (X, Y), ROI et indicateurs d'images calculées toujours inclus",
        "es":    "✓ Coordenadas de la cuadrícula (X, Y), ROI e indicadores de fotogramas calculados siempre incluidos",
    },
    "Displacements are written in pixels (+u right, +v down); the pixel size and frame rate above are recorded with them.": {
        "zh_CN": "位移以像素为单位写入（+u 向右，+v 向下）；上方的像素尺寸和帧率会一并记录。",
        "zh_TW": "位移以像素為單位寫入（+u 向右，+v 向下）；上方的像素尺寸與影格率會一併記錄。",
        "ja":    "変位はピクセル単位で書き出されます（+u 右、+v 下）。上のピクセルサイズとフレームレートも併せて記録されます。",
        "ko":    "변위는 픽셀 단위로 기록됩니다(+u 오른쪽, +v 아래쪽). 위의 픽셀 크기와 프레임 속도도 함께 기록됩니다.",
        "de":    "Verschiebungen werden in Pixeln geschrieben (+u nach rechts, +v nach unten); Pixelgröße und Bildrate von oben werden mit abgelegt.",
        "fr":    "Les déplacements sont écrits en pixels (+u vers la droite, +v vers le bas) ; la taille de pixel et la fréquence d'images ci-dessus sont enregistrées avec eux.",
        "es":    "Los desplazamientos se escriben en píxeles (+u a la derecha, +v hacia abajo); el tamaño de píxel y la velocidad de fotogramas de arriba se registran con ellos.",
    },
    "Writing %1 (%2%)": {
        "zh_CN": "正在写入 %1 (%2%)",
        "zh_TW": "正在寫入 %1 (%2%)",
        "ja":    "%1 を書き込み中 (%2%)",
        "ko":    "%1 쓰는 중 (%2%)",
        "de":    "Schreibe %1 (%2%)",
        "fr":    "Écriture de %1 (%2%)",
        "es":    "Escribiendo %1 (%2%)",
    },
    "Deformed: the field is drawn where each point is in that frame, over the frame's own photo.\n"
    "Reference: drawn at the points' positions in frame 1, over frame 1.": {
        "zh_CN": "变形帧：场绘制在各点在该帧中的位置，叠加在该帧自己的照片上。\n参考帧：绘制在各点在第 1 帧中的位置，叠加在第 1 帧上。",
        "zh_TW": "變形影格：場繪製在各點於該影格中的位置，疊加在該影格自己的照片上。\n參考影格：繪製在各點於第 1 影格中的位置，疊加在第 1 影格上。",
        "ja":    "変形フレーム：各点がそのフレームで位置する場所に場を描画し、そのフレーム自身の写真に重ねます。\n参照フレーム：フレーム 1 での点の位置に描画し、フレーム 1 に重ねます。",
        "ko":    "변형된 프레임: 각 점이 해당 프레임에서 있는 위치에 필드를 그려 그 프레임 자체의 사진 위에 겹칩니다.\n참조 프레임: 프레임 1에서의 점 위치에 그려 프레임 1 위에 겹칩니다.",
        "de":    "Verformtes Bild: Das Feld wird dort gezeichnet, wo sich jeder Punkt in diesem Bild befindet, über dem eigenen Foto des Bildes.\nReferenzbild: an den Positionen der Punkte in Bild 1 gezeichnet, über Bild 1.",
        "fr":    "Image déformée : le champ est tracé là où se trouve chaque point dans cette image, par-dessus la photo propre à l'image.\nImage de référence : tracé aux positions des points dans l'image 1, par-dessus l'image 1.",
        "es":    "Fotograma deformado: el campo se dibuja donde está cada punto en ese fotograma, sobre la foto propia del fotograma.\nFotograma de referencia: se dibuja en las posiciones de los puntos en el fotograma 1, sobre el fotograma 1.",
    },
    "Uncheck to export the field on its own, on white, with no speckle image behind it.": {
        "zh_CN": "取消勾选可仅导出场本身，以白色为背景，其后不含散斑图像。",
        "zh_TW": "取消勾選可僅匯出場本身，以白色為背景，其後不含散斑影像。",
        "ja":    "チェックを外すと、背後にスペックル画像を含めず、白背景で場のみをエクスポートします。",
        "ko":    "선택을 해제하면 뒤에 스페클 이미지 없이 흰색 배경에 필드만 내보냅니다.",
        "de":    "Deaktivieren, um nur das Feld auf Weiß ohne Speckle-Bild dahinter zu exportieren.",
        "fr":    "Décochez pour n'exporter que le champ, sur fond blanc, sans image de mouchetis derrière.",
        "es":    "Desmarque para exportar solo el campo, sobre blanco, sin imagen de moteado detrás.",
    },
    "The images of this run are not available.": {
        "zh_CN": "本次运行的图像不可用。",
        "zh_TW": "本次執行的影像無法使用。",
        "ja":    "この実行の画像は利用できません。",
        "ko":    "이번 실행의 이미지를 사용할 수 없습니다.",
        "de":    "Die Bilder dieses Laufs sind nicht verfügbar.",
        "fr":    "Les images de cette exécution ne sont pas disponibles.",
        "es":    "Las imágenes de esta ejecución no están disponibles.",
    },
    "Resolution of the written figure: about 8.6 x 7 inches at this many dots per inch.": {
        "zh_CN": "输出图片的分辨率：图幅约 8.6 x 7 英寸，按此每英寸点数输出。",
        "zh_TW": "輸出圖片的解析度：圖幅約 8.6 x 7 英吋，按此每英吋點數輸出。",
        "ja":    "書き出す図の解像度です：約 8.6 x 7 インチの図を、この 1 インチあたりのドット数で出力します。",
        "ko":    "출력할 그림의 해상도: 약 8.6 x 7인치 크기를 인치당 이 도트 수로 출력합니다.",
        "de":    "Auflösung der geschriebenen Abbildung: etwa 8,6 x 7 Zoll bei dieser Anzahl Punkte pro Zoll.",
        "fr":    "Résolution de la figure écrite : environ 8,6 x 7 pouces à ce nombre de points par pouce.",
        "es":    "Resolución de la figura escrita: unas 8,6 x 7 pulgadas a esta cantidad de puntos por pulgada.",
    },
    "Each frame": {
        "zh_CN": "逐帧",
        "zh_TW": "逐影格",
        "ja":    "フレームごと",
        "ko":    "프레임별",
        "de":    "Pro Bild",
        "fr":    "Par image",
        "es":    "Por fotograma",
    },
    "Whole sequence": {
        "zh_CN": "整个序列",
        "zh_TW": "整個序列",
        "ja":    "シーケンス全体",
        "ko":    "전체 시퀀스",
        "de":    "Gesamte Sequenz",
        "fr":    "Séquence entière",
        "es":    "Secuencia completa",
    },
    "How Auto picks the colour range: from each frame's own values (like the main window),\n"
    "or one range from every exported frame, so a colour means the same value in every image.": {
        "zh_CN": "自动范围的取法：按每帧自身的数值（与主窗口相同），\n或由所有导出帧共同确定一个范围，使同一颜色在每张图像中代表同一数值。",
        "zh_TW": "自動範圍的取法：依每個影格自身的數值（與主視窗相同），\n或由所有匯出影格共同決定一個範圍，使同一顏色在每張影像中代表同一數值。",
        "ja":    "自動で色範囲を決める方法：各フレーム自身の値から決める（メインウィンドウと同じ）か、\nエクスポートする全フレームから 1 つの範囲を決め、どの画像でも同じ色が同じ値を表すようにします。",
        "ko":    "자동 범위를 정하는 방식: 각 프레임 자체의 값으로 정하거나(메인 창과 동일),\n내보내는 모든 프레임에서 하나의 범위를 정해 모든 이미지에서 같은 색이 같은 값을 뜻하도록 합니다.",
        "de":    "Wie Auto den Farbbereich wählt: aus den Werten jedes einzelnen Bildes (wie im Hauptfenster)\noder ein Bereich aus allen exportierten Bildern, sodass eine Farbe in jedem Bild denselben Wert bedeutet.",
        "fr":    "Comment Auto choisit la plage de couleurs : à partir des valeurs propres à chaque image (comme la fenêtre principale),\nou une seule plage pour toutes les images exportées, pour qu'une couleur ait la même valeur dans chaque image.",
        "es":    "Cómo elige Auto el rango de colores: a partir de los valores de cada fotograma (como la ventana principal),\no un único rango a partir de todos los fotogramas exportados, para que un color signifique el mismo valor en todas las imágenes.",
    },
    "Auto range": {
        "zh_CN": "自动范围",
        "zh_TW": "自動範圍",
        "ja":    "自動範囲",
        "ko":    "자동 범위",
        "de":    "Auto-Bereich",
        "fr":    "Plage auto",
        "es":    "Rango auto",
    },
    "Add a vertical colorbar to the right of each image.": {
        "zh_CN": "在每张图像右侧添加一条垂直色条。",
        "zh_TW": "在每張影像右側添加一條垂直色條。",
        "ja":    "各画像の右側に垂直カラーバーを追加します。",
        "ko":    "각 이미지 오른쪽에 수직 컬러바를 추가합니다.",
        "de":    "Fügt rechts neben jedem Bild eine vertikale Farbleiste hinzu.",
        "fr":    "Ajoute une barre de couleur verticale à droite de chaque image.",
        "es":    "Añade una barra de color vertical a la derecha de cada imagen.",
    },
    "Crop to ROI": {
        "zh_CN": "裁剪到 ROI",
        "zh_TW": "裁切至 ROI",
        "ja":    "ROI で切り抜く",
        "ko":    "ROI로 자르기",
        "de":    "Auf ROI zuschneiden",
        "fr":    "Rogner à la ROI",
        "es":    "Recortar a la ROI",
    },
    "Cut the picture to the region of interest plus a small margin.": {
        "zh_CN": "将图片裁剪到感兴趣区域，并保留少量边距。",
        "zh_TW": "將圖片裁切至感興趣區域，並保留少量邊距。",
        "ja":    "図を関心領域に少し余白を加えた範囲で切り抜きます。",
        "ko":    "그림을 관심 영역에 약간의 여백을 더한 범위로 자릅니다.",
        "de":    "Schneidet die Abbildung auf die Region of Interest plus einen kleinen Rand zu.",
        "fr":    "Rogne la figure à la région d'intérêt plus une petite marge.",
        "es":    "Recorta la figura a la región de interés más un pequeño margen.",
    },
    "MP4 needs the imageio and imageio-ffmpeg packages (pip install imageio imageio-ffmpeg). Save as GIF instead.": {
        "zh_CN": "MP4 需要 imageio 和 imageio-ffmpeg 包（pip install imageio imageio-ffmpeg）。请改存为 GIF。",
        "zh_TW": "MP4 需要 imageio 與 imageio-ffmpeg 套件（pip install imageio imageio-ffmpeg）。請改存為 GIF。",
        "ja":    "MP4 には imageio と imageio-ffmpeg パッケージが必要です（pip install imageio imageio-ffmpeg）。代わりに GIF で保存してください。",
        "ko":    "MP4에는 imageio 및 imageio-ffmpeg 패키지가 필요합니다(pip install imageio imageio-ffmpeg). 대신 GIF로 저장하세요.",
        "de":    "MP4 benötigt die Pakete imageio und imageio-ffmpeg (pip install imageio imageio-ffmpeg). Speichern Sie stattdessen als GIF.",
        "fr":    "Le MP4 nécessite les paquets imageio et imageio-ffmpeg (pip install imageio imageio-ffmpeg). Enregistrez plutôt en GIF.",
        "es":    "MP4 necesita los paquetes imageio e imageio-ffmpeg (pip install imageio imageio-ffmpeg). Guarde como GIF en su lugar.",
    },
    "Add a vertical colorbar to the right of each frame.": {
        "zh_CN": "在每一帧右侧添加一条垂直色条。",
        "zh_TW": "在每個影格右側添加一條垂直色條。",
        "ja":    "各フレームの右側に垂直カラーバーを追加します。",
        "ko":    "각 프레임 오른쪽에 수직 컬러바를 추가합니다.",
        "de":    "Fügt rechts neben jedem Bild eine vertikale Farbleiste hinzu.",
        "fr":    "Ajoute une barre de couleur verticale à droite de chaque image.",
        "es":    "Añade una barra de color vertical a la derecha de cada fotograma.",
    },
    "Cut the frames to the region of interest plus a small margin.": {
        "zh_CN": "将各帧裁剪到感兴趣区域，并保留少量边距。",
        "zh_TW": "將各影格裁切至感興趣區域，並保留少量邊距。",
        "ja":    "各フレームを関心領域に少し余白を加えた範囲で切り抜きます。",
        "ko":    "각 프레임을 관심 영역에 약간의 여백을 더한 범위로 자릅니다.",
        "de":    "Schneidet die Bilder auf die Region of Interest plus einen kleinen Rand zu.",
        "fr":    "Rogne les images à la région d'intérêt plus une petite marge.",
        "es":    "Recorta los fotogramas a la región de interés más un pequeño margen.",
    },
    "Auto range: one range for the whole animation, so a colour means the same value in every frame.": {
        "zh_CN": "自动范围：整个动画使用同一个范围，使同一颜色在每一帧中代表同一数值。",
        "zh_TW": "自動範圍：整段動畫使用同一個範圍，使同一顏色在每個影格中代表同一數值。",
        "ja":    "自動範囲：アニメーション全体で 1 つの範囲を使うため、どのフレームでも同じ色が同じ値を表します。",
        "ko":    "자동 범위: 애니메이션 전체에 하나의 범위를 사용하므로 모든 프레임에서 같은 색이 같은 값을 뜻합니다.",
        "de":    "Auto-Bereich: ein Bereich für die ganze Animation, sodass eine Farbe in jedem Bild denselben Wert bedeutet.",
        "fr":    "Plage auto : une seule plage pour toute l'animation, pour qu'une couleur ait la même valeur dans chaque image.",
        "es":    "Rango auto: un único rango para toda la animación, para que un color signifique el mismo valor en todos los fotogramas.",
    },
    "Drawn like the Images tab: same colormap, range, opacity and configuration.": {
        "zh_CN": "绘制方式与“图像”页相同：颜色映射、范围、不透明度和配置均一致。",
        "zh_TW": "繪製方式與「影像」頁相同：色彩對映、範圍、不透明度與配置皆一致。",
        "ja":    "「画像」タブと同じように描画します：カラーマップ、範囲、不透明度、構成が同じです。",
        "ko":    "'이미지' 탭과 같은 방식으로 그립니다: 색상 맵, 범위, 불투명도, 구성이 같습니다.",
        "de":    "Wie im Reiter „Bilder“ gezeichnet: gleiche Farbskala, gleicher Bereich, gleiche Deckkraft und Konfiguration.",
        "fr":    "Tracé comme dans l'onglet « Images » : même palette, plage, opacité et configuration.",
        "es":    "Se dibuja como en la pestaña «Imágenes»: mismo mapa de colores, rango, opacidad y configuración.",
    },
    "Min must be below Max for %1.": {
        "zh_CN": "%1 的最小值必须小于最大值。",
        "zh_TW": "%1 的最小值必須小於最大值。",
        "ja":    "%1 の最小値は最大値より小さくする必要があります。",
        "ko":    "%1의 최솟값은 최댓값보다 작아야 합니다.",
        "de":    "Min muss für %1 kleiner als Max sein.",
        "fr":    "Min doit être inférieur à Max pour %1.",
        "es":    "Mín debe ser menor que Máx para %1.",
    },
    "Nothing to export: enable at least one field with computed frames in the frame range.": {
        "zh_CN": "没有可导出的内容：请至少启用一个在帧范围内有已计算帧的场。",
        "zh_TW": "沒有可匯出的內容：請至少啟用一個在影格範圍內有已計算影格的場。",
        "ja":    "エクスポートするものがありません：フレーム範囲内に計算済みフレームがある場を少なくとも 1 つ有効にしてください。",
        "ko":    "내보낼 항목이 없습니다: 프레임 범위 안에 계산된 프레임이 있는 필드를 하나 이상 활성화하세요.",
        "de":    "Nichts zu exportieren: Aktivieren Sie mindestens ein Feld mit berechneten Bildern im Bildbereich.",
        "fr":    "Rien à exporter : activez au moins un champ ayant des images calculées dans la plage d'images.",
        "es":    "Nada que exportar: active al menos un campo con fotogramas calculados en el rango de fotogramas.",
    },
    "Select at least one format.": {
        "zh_CN": "请至少选择一种格式。",
        "zh_TW": "請至少選擇一種格式。",
        "ja":    "形式を少なくとも 1 つ選択してください。",
        "ko":    "형식을 하나 이상 선택하세요.",
        "de":    "Wählen Sie mindestens ein Format.",
        "fr":    "Sélectionnez au moins un format.",
        "es":    "Seleccione al menos un formato.",
    },
    "Select at least one variable to export.": {
        "zh_CN": "请至少选择一个要导出的变量。",
        "zh_TW": "請至少選擇一個要匯出的變數。",
        "ja":    "エクスポートする変数を少なくとも 1 つ選択してください。",
        "ko":    "내보낼 변수를 하나 이상 선택하세요.",
        "de":    "Wählen Sie mindestens eine zu exportierende Variable.",
        "fr":    "Sélectionnez au moins une variable à exporter.",
        "es":    "Seleccione al menos una variable para exportar.",
    },
    "Nothing to export: select a field or the parameter table.": {
        "zh_CN": "没有可导出的内容：请选择一个场或参数表。",
        "zh_TW": "沒有可匯出的內容：請選擇一個場或參數表。",
        "ja":    "エクスポートするものがありません：場またはパラメータ表を選択してください。",
        "ko":    "내보낼 항목이 없습니다: 필드 또는 매개변수 표를 선택하세요.",
        "de":    "Nichts zu exportieren: Wählen Sie ein Feld oder die Parametertabelle.",
        "fr":    "Rien à exporter : sélectionnez un champ ou le tableau des paramètres.",
        "es":    "Nada que exportar: seleccione un campo o la tabla de parámetros.",
    },
    # Last line of a cut-short file list; %1 = how many files are not listed.
    "… and %1 more": {
        "zh_CN": "… 另有 %1 个文件",
        "zh_TW": "… 另有 %1 個檔案",
        "ja":    "… ほか %1 件",
        "ko":    "… 외 %1개",
        "de":    "… und %1 weitere",
        "fr":    "… et %1 de plus",
        "es":    "… y %1 más",
    },
    "Replace Existing Files?": {
        "zh_CN": "替换现有文件？",
        "zh_TW": "取代現有檔案？",
        "ja":    "既存のファイルを置き換えますか？",
        "ko":    "기존 파일을 바꾸시겠습니까?",
        "de":    "Vorhandene Dateien ersetzen?",
        "fr":    "Remplacer les fichiers existants ?",
        "es":    "¿Reemplazar los archivos existentes?",
    },
    "These already exist in the output folder and will be replaced:\n\n%1": {
        "zh_CN": "以下文件已存在于输出文件夹中，将被替换：\n\n%1",
        "zh_TW": "以下檔案已存在於輸出資料夾中，將被取代：\n\n%1",
        "ja":    "次のファイルは出力フォルダに既に存在し、置き換えられます：\n\n%1",
        "ko":    "다음 파일이 출력 폴더에 이미 있으며 교체됩니다:\n\n%1",
        "de":    "Diese Dateien existieren bereits im Ausgabeordner und werden ersetzt:\n\n%1",
        "fr":    "Ces fichiers existent déjà dans le dossier de sortie et seront remplacés :\n\n%1",
        "es":    "Estos archivos ya existen en la carpeta de salida y se reemplazarán:\n\n%1",
    },
    "The results changed after this dialog was opened (a new run, strain or project). "
    "Close it and open Export again to export them.": {
        "zh_CN": "打开此对话框后结果已发生变化（新的运行、应变或项目）。请关闭对话框并重新打开“导出”，再导出这些结果。",
        "zh_TW": "開啟此對話方塊後結果已變更（新的執行、應變或專案）。請關閉對話方塊並重新開啟「匯出」，再匯出這些結果。",
        "ja":    "このダイアログを開いた後に結果が変わりました（新しい実行、ひずみ、またはプロジェクト）。ダイアログを閉じ、もう一度「エクスポート」を開いてからエクスポートしてください。",
        "ko":    "이 대화 상자를 연 뒤 결과가 바뀌었습니다(새 실행, 변형률 또는 프로젝트). 대화 상자를 닫고 '내보내기'를 다시 열어 내보내세요.",
        "de":    "Die Ergebnisse haben sich geändert, seit dieser Dialog geöffnet wurde (neuer Lauf, neue Dehnung oder neues Projekt). Schließen Sie ihn und öffnen Sie den Export erneut, um sie zu exportieren.",
        "fr":    "Les résultats ont changé depuis l'ouverture de cette boîte de dialogue (nouvelle exécution, déformation ou projet). Fermez-la et rouvrez l'export pour les exporter.",
        "es":    "Los resultados cambiaron después de abrir este cuadro de diálogo (una nueva ejecución, deformación o proyecto). Ciérrelo y vuelva a abrir Exportar para exportarlos.",
    },
    "Nothing exported: the existing files were kept.": {
        "zh_CN": "未导出任何内容：已保留现有文件。",
        "zh_TW": "未匯出任何內容：已保留現有檔案。",
        "ja":    "何もエクスポートされませんでした：既存のファイルは保持されています。",
        "ko":    "아무것도 내보내지 않았습니다: 기존 파일을 유지했습니다.",
        "de":    "Nichts exportiert: Die vorhandenen Dateien wurden beibehalten.",
        "fr":    "Rien n'a été exporté : les fichiers existants ont été conservés.",
        "es":    "No se exportó nada: se conservaron los archivos existentes.",
    },
    "Stopping…": {
        "zh_CN": "正在停止…",
        "zh_TW": "正在停止…",
        "ja":    "停止中…",
        "ko":    "중지 중…",
        "de":    "Wird gestoppt…",
        "fr":    "Arrêt en cours…",
        "es":    "Deteniendo…",
    },
    "Export failed: %1": {
        "zh_CN": "导出失败：%1",
        "zh_TW": "匯出失敗：%1",
        "ja":    "エクスポートに失敗しました：%1",
        "ko":    "내보내기 실패: %1",
        "de":    "Export fehlgeschlagen: %1",
        "fr":    "Échec de l'export : %1",
        "es":    "Error al exportar: %1",
    },
    # dialog (export_widgets.RunControls)
    "Cancel Export": {
        "zh_CN": "取消导出",
        "zh_TW": "取消匯出",
        "ja":    "エクスポートをキャンセル",
        "ko":    "내보내기 취소",
        "de":    "Export abbrechen",
        "fr":    "Annuler l'export",
        "es":    "Cancelar exportación",
    },

    # ===== FieldSelector (main window: Disp U / Disp V / |U| / Residual) =======
    "Horizontal displacement (+ right)": {
        "zh_CN": "水平位移（向右为正）",
        "zh_TW": "水平位移（向右為正）",
        "ja":    "水平変位（右向きが正）",
        "ko":    "수평 변위(오른쪽이 +)",
        "de":    "Horizontale Verschiebung (+ nach rechts)",
        "fr":    "Déplacement horizontal (+ vers la droite)",
        "es":    "Desplazamiento horizontal (+ a la derecha)",
    },
    "Vertical displacement (+ down)": {
        "zh_CN": "垂直位移（向下为正）",
        "zh_TW": "垂直位移（向下為正）",
        "ja":    "垂直変位（下向きが正）",
        "ko":    "수직 변위(아래쪽이 +)",
        "de":    "Vertikale Verschiebung (+ nach unten)",
        "fr":    "Déplacement vertical (+ vers le bas)",
        "es":    "Desplazamiento vertical (+ hacia abajo)",
    },
    "Displacement magnitude": {
        "zh_CN": "位移大小",
        "zh_TW": "位移大小",
        "ja":    "変位の大きさ",
        "ko":    "변위 크기",
        "de":    "Verschiebungsbetrag",
        "fr":    "Norme du déplacement",
        "es":    "Magnitud del desplazamiento",
    },
    "Intensity residual: local RMS difference between the reference image and\n"
    "the deformed image mapped back by the displacement. Large where the\n"
    "tracking failed.": {
        "zh_CN": "灰度残差：参考图像与按位移映射回来的变形图像之间的\n局部均方根差。追踪失败处数值较大。",
        "zh_TW": "灰階殘差：參考影像與依位移映射回來的變形影像之間的\n局部均方根差。追蹤失敗處數值較大。",
        "ja":    "輝度残差：参照画像と、変位で逆写像した変形画像との\n局所 RMS 差です。追跡に失敗した箇所で大きくなります。",
        "ko":    "밝기 잔차: 참조 이미지와 변위로 되돌려 매핑한 변형 이미지 사이의\n국소 RMS 차이입니다. 추적에 실패한 곳에서 큽니다.",
        "de":    "Intensitätsresiduum: lokale RMS-Differenz zwischen dem Referenzbild und\ndem mit der Verschiebung zurückabgebildeten verformten Bild. Groß, wo das\nTracking fehlgeschlagen ist.",
        "fr":    "Résidu d'intensité : différence RMS locale entre l'image de référence et\nl'image déformée ramenée par le déplacement. Élevé là où le\nsuivi a échoué.",
        "es":    "Residuo de intensidad: diferencia RMS local entre la imagen de referencia y\nla imagen deformada devuelta mediante el desplazamiento. Grande donde\nel seguimiento falló.",
    },
    "|U|": {lang: "|U|" for lang in LANGUAGES},
    "Residual": {
        "zh_CN": "残差",
        "zh_TW": "殘差",
        "ja":    "残差",
        "ko":    "잔차",
        "de":    "Residuum",
        "fr":    "Résidu",
        "es":    "Residuo",
    },

    # ===== LeftSidebar =========================================================
    "Sort by embedded numbers: image1, image2, ..., image10 (default).\n"
    "Unchecked: plain alphabetical order, for zero-padded names.": {
        "zh_CN": "按文件名中的数字排序：image1, image2, ..., image10（默认）。\n不勾选：按普通字母顺序，适合已补零的文件名。",
        "zh_TW": "按檔名中的數字排序：image1, image2, ..., image10（預設）。\n不勾選：按一般字母順序，適合已補零的檔名。",
        "ja":    "ファイル名中の数字順でソート：image1, image2, ..., image10（デフォルト）。\nオフ：単純なアルファベット順。ゼロ埋めされた名前向けです。",
        "ko":    "파일명 속 숫자 순 정렬: image1, image2, ..., image10(기본값).\n체크 해제: 단순 알파벳순, 0으로 채운 파일명에 적합.",
        "de":    "Nach eingebetteten Zahlen sortieren: image1, image2, ..., image10 (Standard).\nNicht aktiviert: rein alphabetisch, für nullgefüllte Namen.",
        "fr":    "Tri par numéros intégrés : image1, image2, ..., image10 (par défaut).\nNon coché : ordre alphabétique simple, pour les noms avec zéros de remplissage.",
        "es":    "Ordenar por números incrustados: image1, image2, ..., image10 (predeterminado).\nDesmarcado: orden alfabético simple, para nombres con ceros a la izquierda.",
    },

    # ===== MainWindow (menus, project files, models) ===========================
    "Open Image Folder…": {
        "zh_CN": "打开图像文件夹…",
        "zh_TW": "開啟影像資料夾…",
        "ja":    "画像フォルダを開く…",
        "ko":    "이미지 폴더 열기…",
        "de":    "Bildordner öffnen…",
        "fr":    "Ouvrir un dossier d'images…",
        "es":    "Abrir carpeta de imágenes…",
    },
    "Open Project…": {
        "zh_CN": "打开项目…",
        "zh_TW": "開啟專案…",
        "ja":    "プロジェクトを開く…",
        "ko":    "프로젝트 열기…",
        "de":    "Projekt öffnen…",
        "fr":    "Ouvrir un projet…",
        "es":    "Abrir proyecto…",
    },
    "Save Project": {
        "zh_CN": "保存项目",
        "zh_TW": "儲存專案",
        "ja":    "プロジェクトを保存",
        "ko":    "프로젝트 저장",
        "de":    "Projekt speichern",
        "fr":    "Enregistrer le projet",
        "es":    "Guardar proyecto",
    },
    "Save Project As…": {
        "zh_CN": "项目另存为…",
        "zh_TW": "專案另存為…",
        "ja":    "名前を付けてプロジェクトを保存…",
        "ko":    "다른 이름으로 프로젝트 저장…",
        "de":    "Projekt speichern unter…",
        "fr":    "Enregistrer le projet sous…",
        "es":    "Guardar proyecto como…",
    },
    "Export Results…": {
        "zh_CN": "导出结果…",
        "zh_TW": "匯出結果…",
        "ja":    "結果をエクスポート…",
        "ko":    "결과 내보내기…",
        "de":    "Ergebnisse exportieren…",
        "fr":    "Exporter les résultats…",
        "es":    "Exportar resultados…",
    },
    "Associate .raftcorr files with RAFTcorr…": {
        "zh_CN": "将 .raftcorr 文件关联到 RAFTcorr…",
        "zh_TW": "將 .raftcorr 檔案關聯到 RAFTcorr…",
        "ja":    ".raftcorr ファイルを RAFTcorr に関連付け…",
        "ko":    ".raftcorr 파일을 RAFTcorr에 연결…",
        "de":    ".raftcorr-Dateien mit RAFTcorr verknüpfen…",
        "fr":    "Associer les fichiers .raftcorr à RAFTcorr…",
        "es":    "Asociar archivos .raftcorr con RAFTcorr…",
    },
    "Register .raftcorr so double-clicking a project file opens RAFTcorr (current user only, no admin rights needed).": {
        "zh_CN": "注册 .raftcorr，双击项目文件即可打开 RAFTcorr（仅当前用户，无需管理员权限）。",
        "zh_TW": "註冊 .raftcorr，讓雙擊專案檔案即可開啟 RAFTcorr（僅目前使用者，無需系統管理員權限）。",
        "ja":    ".raftcorr を登録し、プロジェクトファイルをダブルクリックすると RAFTcorr が開くようにします（現在のユーザーのみ、管理者権限は不要）。",
        "ko":    ".raftcorr를 등록하여 프로젝트 파일을 두 번 클릭하면 RAFTcorr가 열리도록 합니다(현재 사용자만, 관리자 권한 불필요).",
        "de":    "Registriert .raftcorr, sodass ein Doppelklick auf eine Projektdatei RAFTcorr öffnet (nur aktueller Benutzer, keine Administratorrechte nötig).",
        "fr":    "Enregistre .raftcorr pour qu'un double-clic sur un fichier de projet ouvre RAFTcorr (utilisateur actuel uniquement, sans droits administrateur).",
        "es":    "Registra .raftcorr para que hacer doble clic en un archivo de proyecto abra RAFTcorr (solo el usuario actual, sin permisos de administrador).",
    },
    # Mnemonic dropped, as for &File and &Settings (glossary QA item 5).
    "&Help": {
        "zh_CN": "帮助",
        "zh_TW": "說明",
        "ja":    "ヘルプ",
        "ko":    "도움말",
        "de":    "Hilfe",
        "fr":    "Aide",
        "es":    "Ayuda",
    },
    "Installed Models…": {
        "zh_CN": "已安装的模型…",
        "zh_TW": "已安裝的模型…",
        "ja":    "インストール済みモデル…",
        "ko":    "설치된 모델…",
        "de":    "Installierte Modelle…",
        "fr":    "Modèles installés…",
        "es":    "Modelos instalados…",
    },
    # Menu item, progress title ("Self-test…") and result dialog title.
    "Self-test": {
        "zh_CN": "自检",
        "zh_TW": "自我測試",
        "ja":    "セルフテスト",
        "ko":    "자체 테스트",
        "de":    "Selbsttest",
        "fr":    "Autotest",
        "es":    "Autoprueba",
    },
    "About RAFTcorr": {
        "zh_CN": "关于 RAFTcorr",
        "zh_TW": "關於 RAFTcorr",
        "ja":    "RAFTcorr について",
        "ko":    "RAFTcorr 정보",
        "de":    "Über RAFTcorr",
        "fr":    "À propos de RAFTcorr",
        "es":    "Acerca de RAFTcorr",
    },
    "Cancel": {
        "zh_CN": "取消",
        "zh_TW": "取消",
        "ja":    "キャンセル",
        "ko":    "취소",
        "de":    "Abbrechen",
        "fr":    "Annuler",
        "es":    "Cancelar",
    },
    # Progress-dialog titles: the code appends "…".
    "Saving the project": {
        "zh_CN": "正在保存项目",
        "zh_TW": "正在儲存專案",
        "ja":    "プロジェクトを保存中",
        "ko":    "프로젝트 저장 중",
        "de":    "Projekt wird gespeichert",
        "fr":    "Enregistrement du projet",
        "es":    "Guardando el proyecto",
    },
    "Open Project": {
        "zh_CN": "打开项目",
        "zh_TW": "開啟專案",
        "ja":    "プロジェクトを開く",
        "ko":    "프로젝트 열기",
        "de":    "Projekt öffnen",
        "fr":    "Ouvrir un projet",
        "es":    "Abrir proyecto",
    },
    "Opening the project": {
        "zh_CN": "正在打开项目",
        "zh_TW": "正在開啟專案",
        "ja":    "プロジェクトを開いています",
        "ko":    "프로젝트 여는 중",
        "de":    "Projekt wird geöffnet",
        "fr":    "Ouverture du projet",
        "es":    "Abriendo el proyecto",
    },
    "Images not found": {
        "zh_CN": "未找到图像",
        "zh_TW": "找不到影像",
        "ja":    "画像が見つかりません",
        "ko":    "이미지를 찾을 수 없음",
        "de":    "Bilder nicht gefunden",
        "fr":    "Images introuvables",
        "es":    "No se encontraron las imágenes",
    },
    "The image folder of this project was not found. Choose its new location.": {
        "zh_CN": "未找到此项目的图像文件夹。请选择它的新位置。",
        "zh_TW": "找不到此專案的影像資料夾。請選擇它的新位置。",
        "ja":    "このプロジェクトの画像フォルダが見つかりませんでした。新しい場所を選択してください。",
        "ko":    "이 프로젝트의 이미지 폴더를 찾을 수 없습니다. 새 위치를 선택하세요.",
        "de":    "Der Bildordner dieses Projekts wurde nicht gefunden. Wählen Sie seinen neuen Speicherort.",
        "fr":    "Le dossier d'images de ce projet est introuvable. Choisissez son nouvel emplacement.",
        "es":    "No se encontró la carpeta de imágenes de este proyecto. Elija su nueva ubicación.",
    },
    "Image folder of the project": {
        "zh_CN": "项目的图像文件夹",
        "zh_TW": "專案的影像資料夾",
        "ja":    "プロジェクトの画像フォルダ",
        "ko":    "프로젝트의 이미지 폴더",
        "de":    "Bildordner des Projekts",
        "fr":    "Dossier d'images du projet",
        "es":    "Carpeta de imágenes del proyecto",
    },
    "Cannot open the project": {
        "zh_CN": "无法打开项目",
        "zh_TW": "無法開啟專案",
        "ja":    "プロジェクトを開けません",
        "ko":    "프로젝트를 열 수 없습니다",
        "de":    "Projekt kann nicht geöffnet werden",
        "fr":    "Impossible d'ouvrir le projet",
        "es":    "No se puede abrir el proyecto",
    },
    # The error text is appended directly: keep the trailing separator.
    "Could not register .raftcorr files: ": {
        "zh_CN": "无法注册 .raftcorr 文件：",
        "zh_TW": "無法註冊 .raftcorr 檔案：",
        "ja":    ".raftcorr ファイルを登録できませんでした：",
        "ko":    ".raftcorr 파일을 등록할 수 없습니다: ",
        "de":    ".raftcorr-Dateien konnten nicht registriert werden: ",
        "fr":    "Impossible d'enregistrer les fichiers .raftcorr : ",
        "es":    "No se pudieron registrar los archivos .raftcorr: ",
    },
    "Done. Double-clicking a .raftcorr file will now open RAFTcorr and restore that project.": {
        "zh_CN": "完成。现在双击 .raftcorr 文件即可打开 RAFTcorr 并恢复该项目。",
        "zh_TW": "完成。現在雙擊 .raftcorr 檔案即可開啟 RAFTcorr 並還原該專案。",
        "ja":    "完了しました。これで .raftcorr ファイルをダブルクリックすると RAFTcorr が開き、そのプロジェクトが復元されます。",
        "ko":    "완료되었습니다. 이제 .raftcorr 파일을 두 번 클릭하면 RAFTcorr가 열리고 해당 프로젝트가 복원됩니다.",
        "de":    "Fertig. Ein Doppelklick auf eine .raftcorr-Datei öffnet nun RAFTcorr und stellt dieses Projekt wieder her.",
        "fr":    "Terminé. Un double-clic sur un fichier .raftcorr ouvrira désormais RAFTcorr et restaurera ce projet.",
        "es":    "Listo. Ahora, hacer doble clic en un archivo .raftcorr abrirá RAFTcorr y restaurará ese proyecto.",
    },
    "Load at least two images first.": {
        "zh_CN": "请先加载至少两张图像。",
        "zh_TW": "請先載入至少兩張影像。",
        "ja":    "先に画像を 2 枚以上読み込んでください。",
        "ko":    "먼저 이미지를 두 장 이상 불러오세요.",
        "de":    "Laden Sie zuerst mindestens zwei Bilder.",
        "fr":    "Chargez d'abord au moins deux images.",
        "es":    "Cargue primero al menos dos imágenes.",
    },
    "Wait for the running task to finish.": {
        "zh_CN": "请等待当前任务完成。",
        "zh_TW": "請等待目前的工作完成。",
        "ja":    "実行中のタスクが終わるまでお待ちください。",
        "ko":    "실행 중인 작업이 끝날 때까지 기다리세요.",
        "de":    "Warten Sie, bis die laufende Aufgabe beendet ist.",
        "fr":    "Attendez la fin de la tâche en cours.",
        "es":    "Espere a que termine la tarea en curso.",
    },
    "Measuring the motion with RAFTcorr-s8": {
        "zh_CN": "正在用 RAFTcorr-s8 测量运动",
        "zh_TW": "正在以 RAFTcorr-s8 測量運動",
        "ja":    "RAFTcorr-s8 で動きを計測中",
        "ko":    "RAFTcorr-s8로 움직임 측정 중",
        "de":    "Bewegung wird mit RAFTcorr-s8 gemessen",
        "fr":    "Mesure du mouvement avec RAFTcorr-s8",
        "es":    "Midiendo el movimiento con RAFTcorr-s8",
    },
    "not installed": {
        "zh_CN": "未安装",
        "zh_TW": "未安裝",
        "ja":    "未インストール",
        "ko":    "설치되지 않음",
        "de":    "nicht installiert",
        "fr":    "non installé",
        "es":    "no instalado",
    },
    "Searched:": {
        "zh_CN": "搜索位置：",
        "zh_TW": "搜尋位置：",
        "ja":    "検索した場所：",
        "ko":    "검색한 위치:",
        "de":    "Durchsucht:",
        "fr":    "Emplacements recherchés :",
        "es":    "Ubicaciones buscadas:",
    },
    "Installed Models": {
        "zh_CN": "已安装的模型",
        "zh_TW": "已安裝的模型",
        "ja":    "インストール済みモデル",
        "ko":    "설치된 모델",
        "de":    "Installierte Modelle",
        "fr":    "Modèles installés",
        "es":    "Modelos instalados",
    },
    "Language set to %1. Please restart RAFTcorr for all widgets to pick up the new language.": {
        "zh_CN": "语言已切换至 %1。请重启 RAFTcorr 以让所有界面生效。",
        "zh_TW": "語言已切換至 %1。請重新啟動 RAFTcorr 以讓所有介面生效。",
        "ja":    "言語を %1 に変更しました。すべての画面に反映するには RAFTcorr を再起動してください。",
        "ko":    "언어가 %1(으)로 설정되었습니다. 모든 화면에 반영하려면 RAFTcorr를 재시작하세요.",
        "de":    "Sprache auf %1 eingestellt. Bitte starten Sie RAFTcorr neu, damit alle Elemente die neue Sprache übernehmen.",
        "fr":    "Langue définie sur %1. Veuillez redémarrer RAFTcorr pour que tous les éléments prennent en compte la nouvelle langue.",
        "es":    "Idioma establecido en %1. Reinicie RAFTcorr para que todos los elementos adopten el nuevo idioma.",
    },

    # ===== ParamPanel (PARAMETERS) =============================================
    "Iterations": {
        "zh_CN": "迭代次数",
        "zh_TW": "迭代次數",
        "ja":    "反復回数",
        "ko":    "반복 횟수",
        "de":    "Iterationen",
        "fr":    "Itérations",
        "es":    "Iteraciones",
    },
    "Refinement iterations of the network.\nThe paper evaluates every model with 20.": {
        "zh_CN": "网络的迭代细化次数。\n论文中所有模型均以 20 次进行评估。",
        "zh_TW": "網路的迭代細化次數。\n論文中所有模型皆以 20 次進行評估。",
        "ja":    "ネットワークの反復改良回数です。\n論文ではすべてのモデルを 20 回で評価しています。",
        "ko":    "네트워크의 반복 정제 횟수입니다.\n논문에서는 모든 모델을 20회로 평가합니다.",
        "de":    "Verfeinerungsiterationen des Netzwerks.\nIm Artikel wird jedes Modell mit 20 ausgewertet.",
        "fr":    "Itérations de raffinement du réseau.\nL'article évalue chaque modèle avec 20.",
        "es":    "Iteraciones de refinamiento de la red.\nEl artículo evalúa todos los modelos con 20.",
    },
    # Spin-box special value (0 px): no smoothing, the raw network output.
    "off (raw)": {
        "zh_CN": "关闭（原始）",
        "zh_TW": "關閉（原始）",
        "ja":    "オフ（生データ）",
        "ko":    "끔(원본)",
        "de":    "aus (roh)",
        "fr":    "désactivé (brut)",
        "es":    "desactivado (bruto)",
    },
    "Smoothing σ": {
        "zh_CN": "平滑 σ",
        "zh_TW": "平滑 σ",
        "ja":    "平滑化 σ",
        "ko":    "평활화 σ",
        "de":    "Glättung σ",
        "fr":    "Lissage σ",
        "es":    "Suavizado σ",
    },
    "Gaussian smoothing of the displacement (standard deviation).\n"
    "Off by default, as in the paper: smoothing lengthens the\n"
    "spatial resolution.": {
        "zh_CN": "位移的高斯平滑（标准差）。\n默认关闭，与论文一致：平滑会降低\n空间分辨率。",
        "zh_TW": "位移的高斯平滑（標準差）。\n預設關閉，與論文一致：平滑會降低\n空間解析度。",
        "ja":    "変位のガウス平滑化（標準偏差）です。\n論文と同じく既定ではオフ：平滑化すると\n空間分解能が粗くなります。",
        "ko":    "변위의 가우스 평활화(표준편차)입니다.\n논문과 같이 기본값은 끔: 평활화하면\n공간 분해능이 낮아집니다.",
        "de":    "Gaußsche Glättung der Verschiebung (Standardabweichung).\nStandardmäßig aus, wie im Artikel: Glättung vergröbert die\nräumliche Auflösung.",
        "fr":    "Lissage gaussien du déplacement (écart-type).\nDésactivé par défaut, comme dans l'article : le lissage dégrade la\nrésolution spatiale.",
        "es":    "Suavizado gaussiano del desplazamiento (desviación típica).\nDesactivado por defecto, como en el artículo: el suavizado empeora la\nresolución espacial.",
    },
    "Device": {
        "zh_CN": "设备",
        "zh_TW": "裝置",
        "ja":    "デバイス",
        "ko":    "장치",
        "de":    "Gerät",
        "fr":    "Périphérique",
        "es":    "Dispositivo",
    },
    "Automatic uses the GPU when PyTorch sees one.": {
        "zh_CN": "选择“自动”时，只要 PyTorch 检测到 GPU 就使用 GPU。",
        "zh_TW": "選擇「自動」時，只要 PyTorch 偵測到 GPU 就使用 GPU。",
        "ja":    "「自動」では、PyTorch が GPU を検出すると GPU を使用します。",
        "ko":    "'자동'은 PyTorch가 GPU를 인식하면 GPU를 사용합니다.",
        "de":    "„Automatisch“ verwendet die GPU, wenn PyTorch eine erkennt.",
        "fr":    "« Automatique » utilise le GPU lorsque PyTorch en détecte un.",
        "es":    "«Automático» usa la GPU cuando PyTorch detecta una.",
    },

    # ===== ROIHint =============================================================
    "<b>Incremental</b> — only frame 1 needs a Region of Interest. RAFTcorr carries it to the "
    "reference frames (<b>%1</b>) with the measured displacement. Draw on another frame only to "
    "replace that region or to leave out cracks and holes.": {
        "zh_CN": "<b>增量模式</b> — 只有第 1 帧需要感兴趣区域。RAFTcorr 会按测得的位移将其带到各参考帧（<b>%1</b>）。仅在需要替换该帧的区域、或排除裂纹和孔洞时，才在其他帧上绘制。",
        "zh_TW": "<b>增量模式</b> — 只有第 1 影格需要感興趣區域。RAFTcorr 會依測得的位移將其帶到各參考影格（<b>%1</b>）。僅在需要取代該影格的區域、或排除裂紋與孔洞時，才在其他影格上繪製。",
        "ja":    "<b>逐次</b> — 関心領域はフレーム 1 にのみ必要です。RAFTcorr が計測した変位でそれを参照フレーム（<b>%1</b>）へ移します。他のフレームに描画するのは、その領域を置き換える場合や、き裂や穴を除外する場合だけです。",
        "ko":    "<b>증분</b> — 관심 영역은 프레임 1에만 필요합니다. RAFTcorr가 측정된 변위로 이를 참조 프레임(<b>%1</b>)으로 옮깁니다. 다른 프레임에는 해당 영역을 대체하거나 균열과 구멍을 제외할 때만 그리세요.",
        "de":    "<b>Inkrementell</b> — nur Bild 1 benötigt eine Region of Interest. RAFTcorr überträgt sie mit der gemessenen Verschiebung auf die Referenzbilder (<b>%1</b>). Zeichnen Sie auf einem anderen Bild nur, um diesen Bereich zu ersetzen oder Risse und Löcher auszusparen.",
        "fr":    "<b>Incrémental</b> — seule l'image 1 a besoin d'une région d'intérêt. RAFTcorr la reporte sur les images de référence (<b>%1</b>) avec le déplacement mesuré. Ne dessinez sur une autre image que pour y remplacer cette région ou pour exclure des fissures et des trous.",
        "es":    "<b>Incremental</b> — solo el fotograma 1 requiere una región de interés. RAFTcorr la traslada a los fotogramas de referencia (<b>%1</b>) con el desplazamiento medido. Dibuje en otro fotograma solo para reemplazar esa región o para excluir grietas y agujeros.",
    },

    # ===== StrainFieldSelector (strain window field tooltips) ==================
    "Horizontal displacement u (+ right)": {
        "zh_CN": "水平位移 u（向右为正）",
        "zh_TW": "水平位移 u（向右為正）",
        "ja":    "水平変位 u（右向きが正）",
        "ko":    "수평 변위 u(오른쪽이 +)",
        "de":    "Horizontale Verschiebung u (+ nach rechts)",
        "fr":    "Déplacement horizontal u (+ vers la droite)",
        "es":    "Desplazamiento horizontal u (+ a la derecha)",
    },
    "Vertical displacement v (+ down)": {
        "zh_CN": "垂直位移 v（向下为正）",
        "zh_TW": "垂直位移 v（向下為正）",
        "ja":    "垂直変位 v（下向きが正）",
        "ko":    "수직 변위 v(아래쪽이 +)",
        "de":    "Vertikale Verschiebung v (+ nach unten)",
        "fr":    "Déplacement vertical v (+ vers le bas)",
        "es":    "Desplazamiento vertical v (+ hacia abajo)",
    },
    "Displacement magnitude |u|": {
        "zh_CN": "位移大小 |u|",
        "zh_TW": "位移大小 |u|",
        "ja":    "変位の大きさ |u|",
        "ko":    "변위 크기 |u|",
        "de":    "Verschiebungsbetrag |u|",
        "fr":    "Norme du déplacement |u|",
        "es":    "Magnitud del desplazamiento |u|",
    },
    "Speed |du/dt|: central difference of the displacement over the neighbouring computed frames, times the frame rate": {
        "zh_CN": "速度大小 |du/dt|：位移在相邻已计算帧之间的中心差分，乘以帧率",
        "zh_TW": "速度大小 |du/dt|：位移在相鄰已計算影格之間的中心差分，乘以影格率",
        "ja":    "速さ |du/dt|：隣接する計算済みフレーム間の変位の中心差分にフレームレートを掛けたもの",
        "ko":    "속력 |du/dt|: 인접한 계산된 프레임 간 변위의 중심 차분에 프레임 속도를 곱한 값",
        "de":    "Geschwindigkeitsbetrag |du/dt|: zentrale Differenz der Verschiebung über die benachbarten berechneten Bilder, mal Bildrate",
        "fr":    "Vitesse |du/dt| : différence centrée du déplacement sur les images calculées voisines, multipliée par la fréquence d'images",
        "es":    "Rapidez |du/dt|: diferencia central del desplazamiento entre los fotogramas calculados vecinos, multiplicada por la velocidad de fotogramas",
    },
    "Normal strain εxx": {
        "zh_CN": "正应变 εxx",
        "zh_TW": "正應變 εxx",
        "ja":    "垂直ひずみ εxx",
        "ko":    "수직 변형률 εxx",
        "de":    "Normaldehnung εxx",
        "fr":    "Déformation normale εxx",
        "es":    "Deformación normal εxx",
    },
    "Normal strain εyy": {
        "zh_CN": "正应变 εyy",
        "zh_TW": "正應變 εyy",
        "ja":    "垂直ひずみ εyy",
        "ko":    "수직 변형률 εyy",
        "de":    "Normaldehnung εyy",
        "fr":    "Déformation normale εyy",
        "es":    "Deformación normal εyy",
    },
    "Shear strain εxy (tensor component: half the engineering shear γxy)": {
        "zh_CN": "剪应变 εxy（张量分量：工程剪应变 γxy 的一半）",
        "zh_TW": "剪應變 εxy（張量分量：工程剪應變 γxy 的一半）",
        "ja":    "せん断ひずみ εxy（テンソル成分：工学せん断ひずみ γxy の半分）",
        "ko":    "전단 변형률 εxy(텐서 성분: 공학 전단 변형률 γxy의 절반)",
        "de":    "Scherdehnung εxy (Tensorkomponente: die Hälfte der technischen Scherung γxy)",
        "fr":    "Déformation de cisaillement εxy (composante du tenseur : moitié du cisaillement de l'ingénieur γxy)",
        "es":    "Deformación cortante εxy (componente del tensor: la mitad de la deformación cortante ingenieril γxy)",
    },
    "Major principal strain ε₁": {
        "zh_CN": "最大主应变 ε₁",
        "zh_TW": "最大主應變 ε₁",
        "ja":    "最大主ひずみ ε₁",
        "ko":    "최대 주변형률 ε₁",
        "de":    "Größte Hauptdehnung ε₁",
        "fr":    "Déformation principale majeure ε₁",
        "es":    "Deformación principal mayor ε₁",
    },
    "Minor principal strain ε₂": {
        "zh_CN": "最小主应变 ε₂",
        "zh_TW": "最小主應變 ε₂",
        "ja":    "最小主ひずみ ε₂",
        "ko":    "최소 주변형률 ε₂",
        "de":    "Kleinste Hauptdehnung ε₂",
        "fr":    "Déformation principale mineure ε₂",
        "es":    "Deformación principal menor ε₂",
    },
    "Maximum in-plane shear strain (ε₁ − ε₂) / 2 (tensor component: half the engineering shear)": {
        "zh_CN": "最大面内剪应变 (ε₁ − ε₂) / 2（张量分量：工程剪应变的一半）",
        "zh_TW": "最大面內剪應變 (ε₁ − ε₂) / 2（張量分量：工程剪應變的一半）",
        "ja":    "最大面内せん断ひずみ (ε₁ − ε₂) / 2（テンソル成分：工学せん断ひずみの半分）",
        "ko":    "최대 면내 전단 변형률 (ε₁ − ε₂) / 2(텐서 성분: 공학 전단 변형률의 절반)",
        "de":    "Maximale Scherdehnung in der Ebene (ε₁ − ε₂) / 2 (Tensorkomponente: die Hälfte der technischen Scherung)",
        "fr":    "Déformation de cisaillement maximale dans le plan (ε₁ − ε₂) / 2 (composante du tenseur : moitié du cisaillement de l'ingénieur)",
        "es":    "Deformación cortante máxima en el plano (ε₁ − ε₂) / 2 (componente del tensor: la mitad de la deformación cortante ingenieril)",
    },
    "Von Mises equivalent strain (incompressible material)": {
        "zh_CN": "von Mises 等效应变（不可压缩材料）",
        "zh_TW": "von Mises 等效應變（不可壓縮材料）",
        "ja":    "von Mises 相当ひずみ（非圧縮性材料）",
        "ko":    "von Mises 등가 변형률(비압축성 재료)",
        "de":    "Von-Mises-Vergleichsdehnung (inkompressibles Material)",
        "fr":    "Déformation équivalente de von Mises (matériau incompressible)",
        "es":    "Deformación equivalente de von Mises (material incompresible)",
    },
    "Rigid rotation in degrees (polar decomposition), clockwise on screen positive": {
        "zh_CN": "刚体转动角，单位为度（极分解），屏幕上顺时针为正",
        "zh_TW": "剛體轉動角，單位為度（極分解），螢幕上順時針為正",
        "ja":    "剛体回転角（度、極分解）、画面上で時計回りが正",
        "ko":    "강체 회전각(도, 극분해), 화면에서 시계 방향이 양수",
        "de":    "Starrkörperdrehung in Grad (polare Zerlegung), im Uhrzeigersinn auf dem Bildschirm positiv",
        "fr":    "Rotation de corps rigide en degrés (décomposition polaire), positive dans le sens horaire à l'écran",
        "es":    "Rotación de sólido rígido en grados (descomposición polar), positiva en sentido horario en pantalla",
    },
    "Direction of ε₁ in degrees from +x, clockwise on screen positive": {
        "zh_CN": "ε₁ 的方向，从 +x 轴起算，单位为度，屏幕上顺时针为正",
        "zh_TW": "ε₁ 的方向，自 +x 軸起算，單位為度，螢幕上順時針為正",
        "ja":    "ε₁ の方向（+x からの角度、度）、画面上で時計回りが正",
        "ko":    "+x에서 잰 ε₁의 방향(도), 화면에서 시계 방향이 양수",
        "de":    "Richtung von ε₁ in Grad ab +x, im Uhrzeigersinn auf dem Bildschirm positiv",
        "fr":    "Direction de ε₁ en degrés depuis +x, positive dans le sens horaire à l'écran",
        "es":    "Dirección de ε₁ en grados desde +x, positiva en sentido horario en pantalla",
    },
    "Relative area change ΔA/A₀ = det F − 1": {
        "zh_CN": "相对面积变化 ΔA/A₀ = det F − 1",
        "zh_TW": "相對面積變化 ΔA/A₀ = det F − 1",
        "ja":    "相対面積変化 ΔA/A₀ = det F − 1",
        "ko":    "상대 면적 변화 ΔA/A₀ = det F − 1",
        "de":    "Relative Flächenänderung ΔA/A₀ = det F − 1",
        "fr":    "Variation relative d'aire ΔA/A₀ = det F − 1",
        "es":    "Cambio relativo de área ΔA/A₀ = det F − 1",
    },

    # ===== StrainParamPanel (STRAIN PARAMETERS) ================================
    "Quadratic fitting": {
        "zh_CN": "二次拟合",
        "zh_TW": "二次擬合",
        "ja":    "二次フィット",
        "ko":    "2차 피팅",
        "de":    "Quadratische Anpassung",
        "fr":    "Ajustement quadratique",
        "es":    "Ajuste cuadrático",
    },
    "Polynomial fitted to each displacement component over the VSG window. Its slopes at the centre are the "
    "displacement gradients.\n• Plane fitting (1st order): the standard VSG (paper setting).\n"
    "• Quadratic fitting (2nd order): follows strain gradients inside the window better, at the cost of more noise.": {
        "zh_CN": "在 VSG 窗口内对每个位移分量拟合多项式，其在中心处的斜率即为位移梯度。\n• 平面拟合（一阶）：标准 VSG（论文设置）。\n• 二次拟合（二阶）：能更好地跟随窗口内的应变梯度，但噪声更大。",
        "zh_TW": "在 VSG 視窗內對每個位移分量擬合多項式，其在中心處的斜率即為位移梯度。\n• 平面擬合（一階）：標準 VSG（論文設定）。\n• 二次擬合（二階）：能更好地跟隨視窗內的應變梯度，但雜訊較大。",
        "ja":    "VSG ウィンドウ内で各変位成分に多項式をフィットします。中心での傾きが変位勾配です。\n• 平面フィット（1 次）：標準的な VSG（論文の設定）。\n• 二次フィット（2 次）：ウィンドウ内のひずみ勾配によく追従しますが、ノイズが増えます。",
        "ko":    "VSG 창 안에서 각 변위 성분에 다항식을 피팅합니다. 중심에서의 기울기가 변위 구배입니다.\n• 평면 피팅(1차): 표준 VSG(논문 설정).\n• 2차 피팅(2차): 창 내부의 변형률 구배를 더 잘 따르지만 노이즈가 커집니다.",
        "de":    "Polynom, das über das VSG-Fenster an jede Verschiebungskomponente angepasst wird. Seine Steigungen im Zentrum sind die Verschiebungsgradienten.\n• Ebenenanpassung (1. Ordnung): das Standard-VSG (Einstellung des Artikels).\n• Quadratische Anpassung (2. Ordnung): folgt Dehnungsgradienten im Fenster besser, auf Kosten von mehr Rauschen.",
        "fr":    "Polynôme ajusté à chaque composante du déplacement sur la fenêtre VSG. Ses pentes au centre sont les gradients de déplacement.\n• Ajustement de plan (ordre 1) : le VSG standard (réglage de l'article).\n• Ajustement quadratique (ordre 2) : suit mieux les gradients de déformation dans la fenêtre, au prix de plus de bruit.",
        "es":    "Polinomio ajustado a cada componente del desplazamiento sobre la ventana VSG. Sus pendientes en el centro son los gradientes de desplazamiento.\n• Ajuste de plano (1.er orden): el VSG estándar (ajuste del artículo).\n• Ajuste cuadrático (2.º orden): sigue mejor los gradientes de deformación dentro de la ventana, a costa de más ruido.",
    },
    "VSG (Virtual Strain Gauge) size Lg is the side, in pixels, of the square window around each point over "
    "which the displacement is fitted. Strain is then taken from the fit's slopes.\n\n"
    "• Larger VSG → smoother strain, lower spatial resolution.\n"
    "• Smaller VSG → sharper strain, more noise.\n"
    "• Odd, so that the window has a centre pixel (default: 41 px).": {
        "zh_CN": "VSG（虚拟应变计）尺寸 Lg 是每个点周围用于拟合位移的正方形窗口的边长（像素）。应变由拟合的斜率得到。\n\n• VSG 越大 → 应变越平滑，空间分辨率越低。\n• VSG 越小 → 应变越锐利，噪声越大。\n• 取奇数，使窗口有中心像素（默认：41 px）。",
        "zh_TW": "VSG（虛擬應變計）尺寸 Lg 是每個點周圍用於擬合位移的正方形視窗的邊長（像素）。應變由擬合的斜率求得。\n\n• VSG 越大 → 應變越平滑，空間解析度越低。\n• VSG 越小 → 應變越銳利，雜訊越多。\n• 取奇數，使視窗有中心像素（預設：41 px）。",
        "ja":    "VSG（仮想ひずみゲージ）サイズ Lg は、各点の周囲で変位をフィットする正方形ウィンドウの一辺の長さ（ピクセル）です。ひずみはフィットの傾きから求めます。\n\n• VSG が大きい → ひずみは滑らか、空間分解能は低い。\n• VSG が小さい → ひずみはシャープ、ノイズが多い。\n• ウィンドウに中心ピクセルがあるよう奇数にします（既定値：41 px）。",
        "ko":    "VSG(가상 변형률 게이지) 크기 Lg는 각 점 주위에서 변위를 피팅하는 정사각형 창의 한 변 길이(픽셀)입니다. 변형률은 피팅의 기울기에서 구합니다.\n\n• VSG가 클수록 → 변형률이 매끄럽고 공간 분해능이 낮아집니다.\n• VSG가 작을수록 → 변형률이 선명하고 노이즈가 많아집니다.\n• 창에 중심 픽셀이 있도록 홀수로 합니다(기본값: 41 px).",
        "de":    "Die VSG-Größe Lg (Virtual Strain Gauge, virtueller Dehnungsmessstreifen) ist die Seitenlänge in Pixeln des quadratischen Fensters um jeden Punkt, über das die Verschiebung angepasst wird. Die Dehnung ergibt sich dann aus den Steigungen der Anpassung.\n\n• Größeres VSG → glattere Dehnung, geringere räumliche Auflösung.\n• Kleineres VSG → schärfere Dehnung, mehr Rauschen.\n• Ungerade, damit das Fenster ein Mittelpixel hat (Standard: 41 px).",
        "fr":    "La taille VSG (Virtual Strain Gauge, jauge de déformation virtuelle) Lg est le côté, en pixels, de la fenêtre carrée autour de chaque point sur laquelle le déplacement est ajusté. La déformation est ensuite tirée des pentes de l'ajustement.\n\n• VSG plus grand → déformation plus lisse, résolution spatiale plus faible.\n• VSG plus petit → déformation plus nette, plus de bruit.\n• Impair, pour que la fenêtre ait un pixel central (par défaut : 41 px).",
        "es":    "El tamaño VSG (Virtual Strain Gauge, galga extensométrica virtual) Lg es el lado, en píxeles, de la ventana cuadrada alrededor de cada punto sobre la que se ajusta el desplazamiento. La deformación se obtiene después de las pendientes del ajuste.\n\n• VSG mayor → deformación más suave, menor resolución espacial.\n• VSG menor → deformación más nítida, más ruido.\n• Impar, para que la ventana tenga un píxel central (predeterminado: 41 px).",
    },
    "Uniform": {
        "zh_CN": "均匀",
        "zh_TW": "均勻",
        "ja":    "一様",
        "ko":    "균일",
        "de":    "Gleichmäßig",
        "fr":    "Uniforme",
        "es":    "Uniforme",
    },
    "Gaussian": {
        "zh_CN": "高斯",
        "zh_TW": "高斯",
        "ja":    "ガウス",
        "ko":    "가우스",
        "de":    "Gauß",
        "fr":    "Gaussienne",
        "es":    "Gaussiana",
    },
    "Weight of the points inside the VSG window: all equal (uniform, paper setting), or decreasing with the "
    "distance from the centre (Gaussian).": {
        "zh_CN": "VSG 窗口内各点的权重：全部相等（均匀，论文设置），或随离中心的距离递减（高斯）。",
        "zh_TW": "VSG 視窗內各點的權重：全部相等（均勻，論文設定），或隨離中心的距離遞減（高斯）。",
        "ja":    "VSG ウィンドウ内の点の重みです：すべて等しい（一様、論文の設定）か、中心からの距離とともに減少する（ガウス）。",
        "ko":    "VSG 창 안 점들의 가중치: 모두 같음(균일, 논문 설정) 또는 중심에서 멀어질수록 감소(가우스).",
        "de":    "Gewicht der Punkte im VSG-Fenster: alle gleich (gleichmäßig, Einstellung des Artikels) oder mit dem Abstand vom Zentrum abnehmend (Gauß).",
        "fr":    "Poids des points dans la fenêtre VSG : tous égaux (uniforme, réglage de l'article), ou décroissant avec la distance au centre (gaussienne).",
        "es":    "Peso de los puntos dentro de la ventana VSG: todos iguales (uniforme, ajuste del artículo) o decreciente con la distancia al centro (gaussiana).",
    },
    "Weighting": {
        "zh_CN": "加权",
        "zh_TW": "加權",
        "ja":    "重み付け",
        "ko":    "가중치",
        "de":    "Gewichtung",
        "fr":    "Pondération",
        "es":    "Ponderación",
    },
    "Auto (Lg / 4)": {
        "zh_CN": "自动（Lg / 4）",
        "zh_TW": "自動（Lg / 4）",
        "ja":    "自動（Lg / 4）",
        "ko":    "자동(Lg / 4)",
        "de":    "Auto (Lg / 4)",
        "fr":    "Auto (Lg / 4)",
        "es":    "Auto (Lg / 4)",
    },
    "Standard deviation of the Gaussian weight. Auto uses a quarter of the VSG size.": {
        "zh_CN": "高斯权重的标准差。“自动”取 VSG 尺寸的四分之一。",
        "zh_TW": "高斯權重的標準差。「自動」取 VSG 尺寸的四分之一。",
        "ja":    "ガウス重みの標準偏差です。「自動」では VSG サイズの 4 分の 1 を使います。",
        "ko":    "가우스 가중치의 표준편차입니다. '자동'은 VSG 크기의 4분의 1을 사용합니다.",
        "de":    "Standardabweichung der Gauß-Gewichtung. Auto verwendet ein Viertel der VSG-Größe.",
        "fr":    "Écart-type de la pondération gaussienne. Auto utilise un quart de la taille VSG.",
        "es":    "Desviación típica del peso gaussiano. Auto usa un cuarto del tamaño VSG.",
    },
    "Gaussian σ": {
        "zh_CN": "高斯 σ",
        "zh_TW": "高斯 σ",
        "ja":    "ガウス σ",
        "ko":    "가우스 σ",
        "de":    "Gauß-σ",
        "fr":    "σ gaussien",
        "es":    "σ gaussiana",
    },
    "Spacing of the strain points: 1 computes strain at every pixel; larger values are faster and lighter for "
    "long sequences.": {
        "zh_CN": "应变点的间距：1 表示在每个像素上计算应变；更大的值对长序列更快、占用更少。",
        "zh_TW": "應變點的間距：1 表示在每個像素上計算應變；較大的值對長序列更快、占用更少。",
        "ja":    "ひずみ点の間隔です：1 ですべてのピクセルでひずみを計算します。大きい値ほど高速・軽量で、長いシーケンスに向きます。",
        "ko":    "변형률 점의 간격: 1이면 모든 픽셀에서 변형률을 계산합니다. 값이 클수록 긴 시퀀스에서 더 빠르고 가볍습니다.",
        "de":    "Abstand der Dehnungspunkte: 1 berechnet die Dehnung an jedem Pixel; größere Werte sind für lange Sequenzen schneller und sparsamer.",
        "fr":    "Espacement des points de déformation : 1 calcule la déformation à chaque pixel ; des valeurs plus grandes sont plus rapides et plus légères pour les longues séquences.",
        "es":    "Espaciado de los puntos de deformación: 1 calcula la deformación en cada píxel; valores mayores son más rápidos y ligeros para secuencias largas.",
    },
    "Output spacing": {
        "zh_CN": "输出间距",
        "zh_TW": "輸出間距",
        "ja":    "出力間隔",
        "ko":    "출력 간격",
        "de":    "Ausgabeabstand",
        "fr":    "Espacement de sortie",
        "es":    "Espaciado de salida",
    },
    "Gaussian smoothing (σ in pixels) of the displacement before the fit. Off (0) differentiates the measured "
    "displacement as it is (paper setting).": {
        "zh_CN": "拟合前对位移进行高斯平滑（σ 以像素计）。关闭（0）时直接对测得的位移求导（论文设置）。",
        "zh_TW": "擬合前對位移進行高斯平滑（σ 以像素計）。關閉（0）時直接對測得的位移求導（論文設定）。",
        "ja":    "フィット前の変位のガウス平滑化（σ はピクセル単位）です。オフ（0）では計測した変位をそのまま微分します（論文の設定）。",
        "ko":    "피팅 전 변위의 가우스 평활화(σ는 픽셀 단위)입니다. 끔(0)이면 측정된 변위를 그대로 미분합니다(논문 설정).",
        "de":    "Gaußsche Glättung (σ in Pixeln) der Verschiebung vor der Anpassung. Aus (0) differenziert die gemessene Verschiebung unverändert (Einstellung des Artikels).",
        "fr":    "Lissage gaussien (σ en pixels) du déplacement avant l'ajustement. Désactivé (0) dérive le déplacement mesuré tel quel (réglage de l'article).",
        "es":    "Suavizado gaussiano (σ en píxeles) del desplazamiento antes del ajuste. Desactivado (0) deriva el desplazamiento medido tal cual (ajuste del artículo).",
    },
    "Displacement smoothing": {
        "zh_CN": "位移平滑",
        "zh_TW": "位移平滑",
        "ja":    "変位の平滑化",
        "ko":    "변위 평활화",
        "de":    "Verschiebungsglättung",
        "fr":    "Lissage du déplacement",
        "es":    "Suavizado del desplazamiento",
    },
    "Never fit across cracks or holes": {
        "zh_CN": "拟合不跨越裂纹或孔洞",
        "zh_TW": "擬合不跨越裂紋或孔洞",
        "ja":    "き裂や穴をまたいでフィットしない",
        "ko":    "균열이나 구멍을 가로질러 피팅하지 않음",
        "de":    "Nie über Risse oder Löcher hinweg anpassen",
        "fr":    "Ne jamais ajuster à travers fissures ou trous",
        "es":    "No ajustar nunca a través de grietas o agujeros",
    },
    "Fit each connected part of the tracked region separately, so no VSG window ever spans a crack, a hole or "
    "the gap between two regions.": {
        "zh_CN": "对追踪区域的每个连通部分分别拟合，使任何 VSG 窗口都不会跨越裂纹、孔洞或两个区域之间的间隙。",
        "zh_TW": "對追蹤區域的每個連通部分分別擬合，使任何 VSG 視窗都不會跨越裂紋、孔洞或兩個區域之間的間隙。",
        "ja":    "追跡領域の連結した部分ごとに別々にフィットし、どの VSG ウィンドウもき裂、穴、2 つの領域の間の隙間をまたがないようにします。",
        "ko":    "추적 영역의 연결된 부분마다 따로 피팅하여, 어떤 VSG 창도 균열, 구멍 또는 두 영역 사이의 틈을 가로지르지 않도록 합니다.",
        "de":    "Jeden zusammenhängenden Teil des verfolgten Bereichs getrennt anpassen, sodass kein VSG-Fenster je einen Riss, ein Loch oder die Lücke zwischen zwei Bereichen überspannt.",
        "fr":    "Ajuste séparément chaque partie connexe de la région suivie, pour qu'aucune fenêtre VSG ne recouvre jamais une fissure, un trou ou l'espace entre deux régions.",
        "es":    "Ajusta por separado cada parte conexa de la región seguida, para que ninguna ventana VSG abarque nunca una grieta, un agujero o el hueco entre dos regiones.",
    },
    "Strain measure from the deformation gradient F = I + ∇u:\n"
    "• Infinitesimal: (∇u + ∇uᵀ) / 2 (small strains only).\n"
    "• Eulerian (Euler-Almansi): (I − (F Fᵀ)⁻¹) / 2.\n"
    "• Green-Lagrangian: (Fᵀ F − I) / 2 (paper setting).": {
        "zh_CN": "由变形梯度 F = I + ∇u 得到的应变度量：\n• 无穷小应变：(∇u + ∇uᵀ) / 2（仅适用于小应变）。\n• 欧拉应变（Euler-Almansi）：(I − (F Fᵀ)⁻¹) / 2。\n• 格林-拉格朗日应变：(Fᵀ F − I) / 2（论文设置）。",
        "zh_TW": "由變形梯度 F = I + ∇u 得到的應變度量：\n• 無窮小應變：(∇u + ∇uᵀ) / 2（僅適用於小應變）。\n• 歐拉應變（Euler-Almansi）：(I − (F Fᵀ)⁻¹) / 2。\n• 格林-拉格朗日應變：(Fᵀ F − I) / 2（論文設定）。",
        "ja":    "変形勾配 F = I + ∇u から求めるひずみ尺度：\n• 微小ひずみ：(∇u + ∇uᵀ) / 2（小ひずみのみ）。\n• オイラーひずみ（Euler-Almansi）：(I − (F Fᵀ)⁻¹) / 2。\n• グリーン-ラグランジュひずみ：(Fᵀ F − I) / 2（論文の設定）。",
        "ko":    "변형 구배 F = I + ∇u로부터의 변형률 척도:\n• 미소 변형률: (∇u + ∇uᵀ) / 2(작은 변형률에만).\n• 오일러 변형률(Euler-Almansi): (I − (F Fᵀ)⁻¹) / 2.\n• 그린-라그랑주 변형률: (Fᵀ F − I) / 2(논문 설정).",
        "de":    "Dehnungsmaß aus dem Deformationsgradienten F = I + ∇u:\n• Infinitesimal: (∇u + ∇uᵀ) / 2 (nur kleine Dehnungen).\n• Euler (Euler-Almansi): (I − (F Fᵀ)⁻¹) / 2.\n• Green-Lagrange: (Fᵀ F − I) / 2 (Einstellung des Artikels).",
        "fr":    "Mesure de déformation à partir du gradient de transformation F = I + ∇u :\n• Infinitésimal : (∇u + ∇uᵀ) / 2 (petites déformations uniquement).\n• Eulérien (Euler-Almansi) : (I − (F Fᵀ)⁻¹) / 2.\n• Green-Lagrange : (Fᵀ F − I) / 2 (réglage de l'article).",
        "es":    "Medida de deformación a partir del gradiente de deformación F = I + ∇u:\n• Infinitesimal: (∇u + ∇uᵀ) / 2 (solo deformaciones pequeñas).\n• Euleriana (Euler-Almansi): (I − (F Fᵀ)⁻¹) / 2.\n• Green-Lagrange: (Fᵀ F − I) / 2 (ajuste del artículo).",
    },
    # Button: restore the settings the RAFTcorr paper uses.
    "Paper settings": {
        "zh_CN": "论文设置",
        "zh_TW": "論文設定",
        "ja":    "論文の設定",
        "ko":    "논문 설정",
        "de":    "Artikel-Einstellungen",
        "fr":    "Réglages de l'article",
        "es":    "Ajustes del artículo",
    },
    "Green-Lagrangian strain, unweighted plane fit, VSG 41 px, every pixel, no smoothing, never across cracks "
    "(the settings of the RAFTcorr paper).": {
        "zh_CN": "格林-拉格朗日应变、无加权平面拟合、VSG 41 px、逐像素、不平滑、不跨越裂纹（RAFTcorr 论文中的设置）。",
        "zh_TW": "格林-拉格朗日應變、無加權平面擬合、VSG 41 px、逐像素、不平滑、不跨越裂紋（RAFTcorr 論文中的設定）。",
        "ja":    "グリーン-ラグランジュひずみ、重みなし平面フィット、VSG 41 px、全ピクセル、平滑化なし、き裂をまたがない（RAFTcorr 論文の設定）。",
        "ko":    "그린-라그랑주 변형률, 가중치 없는 평면 피팅, VSG 41 px, 모든 픽셀, 평활화 없음, 균열을 가로지르지 않음(RAFTcorr 논문의 설정).",
        "de":    "Green-Lagrange-Dehnung, ungewichtete Ebenenanpassung, VSG 41 px, jedes Pixel, keine Glättung, nie über Risse hinweg (die Einstellungen des RAFTcorr-Artikels).",
        "fr":    "Déformation de Green-Lagrange, ajustement de plan non pondéré, VSG 41 px, chaque pixel, sans lissage, jamais à travers les fissures (les réglages de l'article RAFTcorr).",
        "es":    "Deformación de Green-Lagrange, ajuste de plano sin ponderar, VSG 41 px, cada píxel, sin suavizado, nunca a través de grietas (los ajustes del artículo de RAFTcorr).",
    },

    # ===== StrainVizPanel ======================================================
    "Plot the field where the material points are in the current frame, or at their positions in the reference frame.": {
        "zh_CN": "将场绘制在材料点在当前帧中的位置，或绘制在其在参考帧中的位置。",
        "zh_TW": "將場繪製在材料點於目前影格中的位置，或繪製在其於參考影格中的位置。",
        "ja":    "場を材料点の現在フレームでの位置に描画するか、参照フレームでの位置に描画します。",
        "ko":    "필드를 재료점의 현재 프레임 위치에 그리거나 참조 프레임에서의 위치에 그립니다.",
        "de":    "Das Feld dort zeichnen, wo sich die Materialpunkte im aktuellen Bild befinden, oder an ihren Positionen im Referenzbild.",
        "fr":    "Tracer le champ là où se trouvent les points matériels dans l'image courante, ou à leurs positions dans l'image de référence.",
        "es":    "Dibujar el campo donde están los puntos materiales en el fotograma actual, o en sus posiciones en el fotograma de referencia.",
    },
    "Leave points without a result transparent, for example material that left the field of view, instead of "
    "filling them with the nearest value inside the region of interest (reference frame view). Data files always "
    "keep NaN there.": {
        "zh_CN": "让没有结果的点（例如已移出视场的材料）保持透明，而不是用感兴趣区域内最近的值填充（参考帧视图）。数据文件中这些点始终为 NaN。",
        "zh_TW": "讓沒有結果的點（例如已移出視野的材料）保持透明，而不是以感興趣區域內最近的值填滿（參考影格檢視）。資料檔中這些點一律為 NaN。",
        "ja":    "結果のない点（視野の外に出た材料など）を、関心領域内の最も近い値で埋めずに透明のままにします（参照フレーム表示）。データファイルでは、これらの点は常に NaN のままです。",
        "ko":    "결과가 없는 점(예: 시야 밖으로 나간 재료)을 관심 영역 안의 가장 가까운 값으로 채우지 않고 투명하게 둡니다(참조 프레임 보기). 데이터 파일에서는 이 점들이 항상 NaN으로 유지됩니다.",
        "de":    "Punkte ohne Ergebnis, etwa Material, das das Sichtfeld verlassen hat, bleiben transparent, statt mit dem nächstgelegenen Wert innerhalb der Region of Interest gefüllt zu werden (Ansicht im Referenzbild). In Datendateien steht dort immer NaN.",
        "fr":    "Laisser transparents les points sans résultat, par exemple le matériau sorti du champ de vision, au lieu de les remplir avec la valeur la plus proche dans la région d'intérêt (vue sur l'image de référence). Les fichiers de données y gardent toujours NaN.",
        "es":    "Deja transparentes los puntos sin resultado, por ejemplo el material que salió del campo de visión, en lugar de rellenarlos con el valor más cercano dentro de la región de interés (vista en el fotograma de referencia). Los archivos de datos siempre conservan NaN en ellos.",
    },

    # ===== StrainWindow ========================================================
    "No displacement results.": {
        "zh_CN": "没有位移结果。",
        "zh_TW": "沒有位移結果。",
        "ja":    "変位結果がありません。",
        "ko":    "변위 결과가 없습니다.",
        "de":    "Keine Verschiebungsergebnisse.",
        "fr":    "Aucun résultat de déplacement.",
        "es":    "No hay resultados de desplazamiento.",
    },
    "Frame %1 was not computed.": {
        "zh_CN": "第 %1 帧未计算。",
        "zh_TW": "第 %1 影格未計算。",
        "ja":    "フレーム %1 は計算されていません。",
        "ko":    "프레임 %1은(는) 계산되지 않았습니다.",
        "de":    "Bild %1 wurde nicht berechnet.",
        "fr":    "L'image %1 n'a pas été calculée.",
        "es":    "El fotograma %1 no se ha calculado.",
    },
    "Click Compute Strain to show this field.": {
        "zh_CN": "点击“计算应变”以显示此场。",
        "zh_TW": "點擊「計算應變」以顯示此場。",
        "ja":    "「ひずみを計算」をクリックすると、この場を表示します。",
        "ko":    "'변형률 계산'을 클릭하면 이 필드를 표시합니다.",
        "de":    "Klicken Sie auf „Dehnung berechnen“, um dieses Feld anzuzeigen.",
        "fr":    "Cliquez sur « Calculer la déformation » pour afficher ce champ.",
        "es":    "Haga clic en «Calcular deformación» para mostrar este campo.",
    },
    "No strain for frame %1: click Compute Strain.": {
        "zh_CN": "第 %1 帧没有应变：请点击“计算应变”。",
        "zh_TW": "第 %1 影格沒有應變：請點擊「計算應變」。",
        "ja":    "フレーム %1 のひずみがありません：「ひずみを計算」をクリックしてください。",
        "ko":    "프레임 %1의 변형률이 없습니다: '변형률 계산'을 클릭하세요.",
        "de":    "Keine Dehnung für Bild %1: Klicken Sie auf „Dehnung berechnen“.",
        "fr":    "Aucune déformation pour l'image %1 : cliquez sur « Calculer la déformation ».",
        "es":    "No hay deformación para el fotograma %1: haga clic en «Calcular deformación».",
    },
    "Velocity needs at least two computed frames.": {
        "zh_CN": "速度至少需要两个已计算帧。",
        "zh_TW": "速度至少需要兩個已計算影格。",
        "ja":    "速度には計算済みフレームが 2 つ以上必要です。",
        "ko":    "속도에는 계산된 프레임이 두 개 이상 필요합니다.",
        "de":    "Die Geschwindigkeit benötigt mindestens zwei berechnete Bilder.",
        "fr":    "La vitesse nécessite au moins deux images calculées.",
        "es":    "La velocidad necesita al menos dos fotogramas calculados.",
    },
    "No valid values in frame %1.": {
        "zh_CN": "第 %1 帧没有有效值。",
        "zh_TW": "第 %1 影格沒有有效值。",
        "ja":    "フレーム %1 に有効な値がありません。",
        "ko":    "프레임 %1에 유효한 값이 없습니다.",
        "de":    "Keine gültigen Werte in Bild %1.",
        "fr":    "Aucune valeur valide dans l'image %1.",
        "es":    "No hay valores válidos en el fotograma %1.",
    },
    "Analysis is not available": {
        "zh_CN": "分析功能不可用",
        "zh_TW": "分析功能無法使用",
        "ja":    "解析は利用できません",
        "ko":    "분석을 사용할 수 없습니다",
        "de":    "Analyse ist nicht verfügbar",
        "fr":    "L'analyse n'est pas disponible",
        "es":    "El análisis no está disponible",
    },
    "Analysis is not available (see the application log).": {
        "zh_CN": "分析功能不可用（详见应用程序日志）。",
        "zh_TW": "分析功能無法使用（詳見應用程式日誌）。",
        "ja":    "解析は利用できません（アプリケーションのログを参照してください）。",
        "ko":    "분석을 사용할 수 없습니다(애플리케이션 로그를 확인하세요).",
        "de":    "Analyse ist nicht verfügbar (siehe Anwendungsprotokoll).",
        "fr":    "L'analyse n'est pas disponible (voir le journal de l'application).",
        "es":    "El análisis no está disponible (consulte el registro de la aplicación).",
    },
    "The displacement results changed during the strain computation; its result was discarded.": {
        "zh_CN": "应变计算期间位移结果发生了变化；本次应变结果已丢弃。",
        "zh_TW": "應變計算期間位移結果發生了變化；本次應變結果已捨棄。",
        "ja":    "ひずみ計算中に変位結果が変更されたため、その結果は破棄されました。",
        "ko":    "변형률 계산 중에 변위 결과가 바뀌어 그 결과를 폐기했습니다.",
        "de":    "Die Verschiebungsergebnisse haben sich während der Dehnungsberechnung geändert; ihr Ergebnis wurde verworfen.",
        "fr":    "Les résultats de déplacement ont changé pendant le calcul de déformation ; son résultat a été abandonné.",
        "es":    "Los resultados de desplazamiento cambiaron durante el cálculo de deformación; su resultado se descartó.",
    },
    "Export is not available: %1": {
        "zh_CN": "导出功能不可用：%1",
        "zh_TW": "匯出功能無法使用：%1",
        "ja":    "エクスポートは利用できません：%1",
        "ko":    "내보내기를 사용할 수 없습니다: %1",
        "de":    "Export ist nicht verfügbar: %1",
        "fr":    "L'export n'est pas disponible : %1",
        "es":    "La exportación no está disponible: %1",
    },

    # ===== WorkflowTypePanel (tracking mode, model) ============================
    # "a later reference frame": a reference frame after frame 1, as opposed to
    # frame 1 itself in accumulative mode; the CJK text says so explicitly.
    "Accumulative: every frame is compared directly with frame 1.\n"
    "Preferred whenever the motion stays within the model's reach:\n"
    "no error accumulates from frame to frame.\n\n"
    "Incremental: each frame is compared with a later reference frame\n"
    "and the steps are composed. Needed when the motion from frame 1\n"
    "exceeds the model's reach (large stretch or rotation); every step\n"
    "adds its error.": {
        "zh_CN": "累积式：每帧都直接与第 1 帧比较。\n只要运动不超出模型量程，就优先使用：\n误差不会逐帧累积。\n\n增量式：每帧与第 1 帧之后的某个参考帧比较，\n再将各步结果串接。当相对第 1 帧的运动\n超出模型量程（大拉伸或大转动）时需要使用；每一步\n都会引入自己的误差。",
        "zh_TW": "累積式：每個影格都直接與第 1 影格比較。\n只要運動不超出模型量程，就優先使用：\n誤差不會逐影格累積。\n\n增量式：每個影格與第 1 影格之後的某個參考影格比較，\n再將各步結果串接。當相對第 1 影格的運動\n超出模型量程（大拉伸或大轉動）時需要使用；每一步\n都會引入自己的誤差。",
        "ja":    "累積：すべてのフレームをフレーム 1 と直接比較します。\n動きがモデルの到達範囲内に収まる限り、こちらを推奨します：\nフレーム間で誤差が累積しません。\n\n逐次：各フレームをフレーム 1 より後の参照フレームと比較し、\n各ステップを合成します。フレーム 1 からの動きが\nモデルの到達範囲を超える場合（大きな伸びや回転）に必要です。各ステップで\n誤差が加わります。",
        "ko":    "누적형: 모든 프레임을 프레임 1과 직접 비교합니다.\n움직임이 모델의 도달 범위 안에 있으면 이 방식을 권장합니다:\n프레임 간에 오차가 누적되지 않습니다.\n\n증분형: 각 프레임을 프레임 1 이후의 참조 프레임과 비교하고\n단계들을 합성합니다. 프레임 1로부터의 움직임이\n모델의 도달 범위를 넘을 때(큰 신장이나 회전) 필요하며, 단계마다\n오차가 더해집니다.",
        "de":    "Akkumulativ: Jedes Bild wird direkt mit Bild 1 verglichen.\nZu bevorzugen, solange die Bewegung in der Reichweite des Modells bleibt:\nVon Bild zu Bild sammelt sich kein Fehler an.\n\nInkrementell: Jedes Bild wird mit einem späteren Referenzbild verglichen,\nund die Schritte werden verknüpft. Nötig, wenn die Bewegung ab Bild 1\ndie Reichweite des Modells übersteigt (große Streckung oder Drehung); jeder Schritt\nfügt seinen Fehler hinzu.",
        "fr":    "Cumulatif : chaque image est comparée directement à l'image 1.\nÀ privilégier tant que le mouvement reste dans la portée du modèle :\naucune erreur ne s'accumule d'une image à l'autre.\n\nIncrémental : chaque image est comparée à une image de référence ultérieure\net les étapes sont composées. Nécessaire quand le mouvement depuis l'image 1\ndépasse la portée du modèle (grand étirement ou rotation) ; chaque étape\najoute son erreur.",
        "es":    "Acumulativo: cada fotograma se compara directamente con el fotograma 1.\nPreferible siempre que el movimiento quede dentro del alcance del modelo:\nel error no se acumula de un fotograma a otro.\n\nIncremental: cada fotograma se compara con un fotograma de referencia posterior\ny los pasos se componen. Necesario cuando el movimiento desde el fotograma 1\nsupera el alcance del modelo (gran estiramiento o rotación); cada paso\nañade su error.",
    },
    "RAFTcorr model. A smaller s resolves finer detail; a larger s\n"
    "reaches further (the coarsest correlation level looks 32 s px away).\n\n"
    "s1, s2: small displacements, strain concentrations.\n"
    "s4: balanced.\n"
    "s8: large motion or rotation, or no prior knowledge.": {
        "zh_CN": "RAFTcorr 模型。s 越小，可分辨的细节越精细；s 越大，\n量程越大（最粗相关层级的查找距离为 32 s px）。\n\ns1、s2：小位移、应变集中。\ns4：均衡。\ns8：大运动或大转动，或无先验信息。",
        "zh_TW": "RAFTcorr 模型。s 越小，可分辨的細節越精細；s 越大，\n量程越大（最粗相關層級的查找距離為 32 s px）。\n\ns1、s2：小位移、應變集中。\ns4：均衡。\ns8：大運動或大轉動，或無先驗資訊。",
        "ja":    "RAFTcorr モデル。s が小さいほど細部を分解でき、s が大きいほど\n遠くまで届きます（最も粗い相関レベルは 32 s px 先まで探索）。\n\ns1、s2：小さな変位、ひずみ集中。\ns4：バランス型。\ns8：大きな動きや回転、または事前情報がない場合。",
        "ko":    "RAFTcorr 모델. s가 작을수록 더 미세한 세부를 분해하고, s가 클수록\n더 멀리 도달합니다(가장 거친 상관 단계는 32 s px 떨어진 곳까지 봄).\n\ns1, s2: 작은 변위, 변형률 집중.\ns4: 균형.\ns8: 큰 움직임이나 회전, 또는 사전 정보가 없을 때.",
        "de":    "RAFTcorr-Modell. Ein kleineres s löst feinere Details auf; ein größeres s\nreicht weiter (die gröbste Korrelationsebene schaut 32 s px weit).\n\ns1, s2: kleine Verschiebungen, Dehnungskonzentrationen.\ns4: ausgewogen.\ns8: große Bewegung oder Drehung, oder kein Vorwissen.",
        "fr":    "Modèle RAFTcorr. Un s plus petit résout des détails plus fins ; un s plus grand\nporte plus loin (le niveau de corrélation le plus grossier regarde à 32 s px).\n\ns1, s2 : petits déplacements, concentrations de déformation.\ns4 : équilibré.\ns8 : grand mouvement ou rotation, ou aucune connaissance préalable.",
        "es":    "Modelo RAFTcorr. Un s menor resuelve detalles más finos; un s mayor\nllega más lejos (el nivel de correlación más grueso mira a 32 s px).\n\ns1, s2: desplazamientos pequeños, concentraciones de deformación.\ns4: equilibrado.\ns8: movimiento o rotación grandes, o sin conocimiento previo.",
    },
    "Model": {
        "zh_CN": "模型",
        "zh_TW": "模型",
        "ja":    "モデル",
        "ko":    "모델",
        "de":    "Modell",
        "fr":    "Modèle",
        "es":    "Modelo",
    },
    "Suggest a model": {
        "zh_CN": "推荐模型",
        "zh_TW": "建議模型",
        "ja":    "モデルを提案",
        "ko":    "모델 추천",
        "de":    "Modell vorschlagen",
        "fr":    "Suggérer un modèle",
        "es":    "Sugerir un modelo",
    },
    "Measure the motion of the sequence quickly with s8 and pick the\nfinest model whose reach covers it.": {
        "zh_CN": "先用 s8 快速测量序列的运动，再选出量程足以\n覆盖该运动的最精细模型。",
        "zh_TW": "先以 s8 快速測量序列的運動，再選出量程足以\n涵蓋該運動的最精細模型。",
        "ja":    "s8 でシーケンスの動きをすばやく計測し、その動きを到達範囲で\nカバーできる最も細かいモデルを選びます。",
        "ko":    "s8로 시퀀스의 움직임을 빠르게 측정한 뒤, 도달 범위가 그 움직임을\n포함하는 가장 세밀한 모델을 고릅니다.",
        "de":    "Misst die Bewegung der Sequenz schnell mit s8 und wählt das\nfeinste Modell, dessen Reichweite sie abdeckt.",
        "fr":    "Mesure rapidement le mouvement de la séquence avec s8 et choisit le\nmodèle le plus fin dont la portée le couvre.",
        "es":    "Mide rápidamente el movimiento de la secuencia con s8 y elige el\nmodelo más fino cuyo alcance lo cubre.",
    },
    # Model tooltips: "%1: %2.\nReach ...", %2 being one of the summaries below.
    "finest spatial resolution; small displacements": {
        "zh_CN": "空间分辨率最高；适合小位移",
        "zh_TW": "空間解析度最高；適合小位移",
        "ja":    "空間分解能が最も高い、小さな変位向け",
        "ko":    "가장 높은 공간 분해능; 작은 변위",
        "de":    "feinste räumliche Auflösung; kleine Verschiebungen",
        "fr":    "résolution spatiale la plus fine ; petits déplacements",
        "es":    "resolución espacial más fina; desplazamientos pequeños",
    },
    "fine resolution, several times faster than s1; small to moderate displacements": {
        "zh_CN": "分辨率高，速度是 s1 的数倍；适合小到中等位移",
        "zh_TW": "解析度高，速度是 s1 的數倍；適合小到中等位移",
        "ja":    "分解能が高く s1 の数倍高速、小～中程度の変位向け",
        "ko":    "높은 분해능, s1보다 몇 배 빠름; 작거나 중간 정도의 변위",
        "de":    "feine Auflösung, mehrfach schneller als s1; kleine bis mittlere Verschiebungen",
        "fr":    "résolution fine, plusieurs fois plus rapide que s1 ; déplacements petits à modérés",
        "es":    "resolución fina, varias veces más rápido que s1; desplazamientos pequeños a moderados",
    },
    # "capture range" = the model's reach (glossary: Reach).
    "balanced resolution and capture range": {
        "zh_CN": "分辨率与量程均衡",
        "zh_TW": "解析度與量程均衡",
        "ja":    "分解能と到達範囲のバランス型",
        "ko":    "분해능과 도달 범위의 균형",
        "de":    "ausgewogene Auflösung und Reichweite",
        "fr":    "résolution et portée équilibrées",
        "es":    "resolución y alcance equilibrados",
    },
    "largest capture range, coarsest resolution; large motion and rotation": {
        "zh_CN": "量程最大，分辨率最粗；适合大运动和大转动",
        "zh_TW": "量程最大，解析度最粗；適合大運動與大轉動",
        "ja":    "到達範囲が最大で分解能は最も粗い、大きな動きや回転向け",
        "ko":    "가장 넓은 도달 범위, 가장 거친 분해능; 큰 움직임과 회전",
        "de":    "größte Reichweite, gröbste Auflösung; große Bewegung und Drehung",
        "fr":    "portée la plus grande, résolution la plus grossière ; grands mouvements et rotations",
        "es":    "mayor alcance, resolución más gruesa; movimientos y rotaciones grandes",
    },
    "%1: %2.\nReach (nominal lookup radius 32 s): %3 px.": {
        "zh_CN": "%1：%2。\n量程（名义查找半径 32 s）：%3 px。",
        "zh_TW": "%1：%2。\n量程（名義查找半徑 32 s）：%3 px。",
        "ja":    "%1：%2。\n到達範囲（公称探索半径 32 s）：%3 px。",
        "ko":    "%1: %2.\n도달 범위(공칭 조회 반경 32 s): %3 px.",
        "de":    "%1: %2.\nReichweite (nomineller Suchradius 32 s): %3 px.",
        "fr":    "%1 : %2.\nPortée (rayon de recherche nominal 32 s) : %3 px.",
        "es":    "%1: %2.\nAlcance (radio de búsqueda nominal 32 s): %3 px.",
    },
    "Weights not installed.": {
        "zh_CN": "权重文件未安装。",
        "zh_TW": "權重檔案未安裝。",
        "ja":    "重みファイルがインストールされていません。",
        "ko":    "가중치 파일이 설치되지 않았습니다.",
        "de":    "Gewichte nicht installiert.",
        "fr":    "Poids non installés.",
        "es":    "Pesos no instalados.",
    },

    # ===== Round 2: strings that were hidden from lupdate or untranslated =====
    # --- Field labels (ExportDialog field rows, strain window buttons) ---
    # Same meaning in both contexts: the displacement magnitude |u|. The
    # export dialog's name column is 100 px wide: de keeps the short "Betrag".
    "Magnitude": {
        "zh_CN": "位移大小",
        "zh_TW": "位移大小",
        "ja":    "変位の大きさ",
        "ko":    "변위 크기",
        "de":    "Betrag",
        "fr":    "Norme",
        "es":    "Magnitud",
    },
    "Velocity": {
        "zh_CN": "速度",
        "zh_TW": "速度",
        "ja":    "速度",
        "ko":    "속도",
        "de":    "Geschwindigkeit",
        "fr":    "Vitesse",
        "es":    "Velocidad",
    },
    # --- ExportDialog: column headers, preview tab, units, frame range ---
    "Field": {
        "zh_CN": "场变量",
        "zh_TW": "場變數",
        "ja":    "フィールド",
        "ko":    "필드",
        "de":    "Feld",
        "fr":    "Champ",
        "es":    "Campo",
    },
    "Frame": {
        "zh_CN": "帧",
        "zh_TW": "影格",
        "ja":    "フレーム",
        "ko":    "프레임",
        "de":    "Bild",
        "fr":    "Image",
        "es":    "Fotograma",
    },
    "Auto": {
        "zh_CN": "自动",
        "zh_TW": "自動",
        "ja":    "自動",
        "ko":    "자동",
        "de":    "Auto",
        "fr":    "Auto",
        "es":    "Auto",
    },
    "Colormap": {
        "zh_CN": "颜色映射",
        "zh_TW": "色彩對映",
        "ja":    "カラーマップ",
        "ko":    "색상 맵",
        "de":    "Farbskala",
        "fr":    "Palette",
        "es":    "Mapa de colores",
    },
    "Range": {
        "zh_CN": "范围",
        "zh_TW": "範圍",
        "ja":    "範囲",
        "ko":    "범위",
        "de":    "Bereich",
        "fr":    "Plage",
        "es":    "Rango",
    },
    "Min": {
        "zh_CN": "最小",
        "zh_TW": "最小",
        "ja":    "最小",
        "ko":    "최소",
        "de":    "Min",
        "fr":    "Min",
        "es":    "Mín",
    },
    "Max": {
        "zh_CN": "最大",
        "zh_TW": "最大",
        "ja":    "最大",
        "ko":    "최대",
        "de":    "Max",
        "fr":    "Max",
        "es":    "Máx",
    },
    "Open this tab to render a preview.": {
        "zh_CN": "打开此页以渲染预览。",
        "zh_TW": "開啟此分頁以繪製預覽。",
        "ja":    "このタブを開くとプレビューが描画されます。",
        "ko":    "이 탭을 열면 미리보기가 렌더링됩니다.",
        "de":    "Diesen Reiter öffnen, um eine Vorschau zu rendern.",
        "fr":    "Ouvrez cet onglet pour générer un aperçu.",
        "es":    "Abra esta pestaña para generar una vista previa.",
    },
    "FIELD APPEARANCE": {
        "zh_CN": "场的外观",
        "zh_TW": "場的外觀",
        "ja":    "フィールドの外観",
        "ko":    "필드 모양",
        "de":    "FELDDARSTELLUNG",
        "fr":    "APPARENCE DU CHAMP",
        "es":    "APARIENCIA DEL CAMPO",
    },
    "Apply to all fields": {
        "zh_CN": "应用到所有场",
        "zh_TW": "套用到所有場",
        "ja":    "すべてのフィールドに適用",
        "ko":    "모든 필드에 적용",
        "de":    "Auf alle Felder anwenden",
        "fr":    "Appliquer à tous les champs",
        "es":    "Aplicar a todos los campos",
    },
    "Apply this field's colormap, opacity and auto-range to every enabled field (each field keeps its own min/max).": {
        "zh_CN": "将该场的颜色映射、不透明度和自动范围应用到所有已启用的场（每个场保留各自的最小/最大值）。",
        "zh_TW": "將該場的色彩對映、不透明度和自動範圍套用到所有已啟用的場（每個場保留各自的最小/最大值）。",
        "ja":    "このフィールドのカラーマップ・不透明度・自動範囲を、有効なすべてのフィールドに適用します（各フィールドの最小/最大は保持）。",
        "ko":    "이 필드의 색상 맵, 불투명도, 자동 범위를 활성화된 모든 필드에 적용합니다(각 필드의 최소/최대는 유지).",
        "de":    "Farbskala, Deckkraft und Auto-Bereich dieses Felds auf alle aktivierten Felder anwenden (jedes Feld behält sein eigenes Min/Max).",
        "fr":    "Applique la palette, l'opacité et la plage auto de ce champ à tous les champs activés (chaque champ garde ses propres min/max).",
        "es":    "Aplica el mapa de colores, la opacidad y el rango automático de este campo a todos los campos activados (cada campo conserva su propio mín/máx).",
    },
    "Refresh preview": {
        "zh_CN": "刷新预览",
        "zh_TW": "重新整理預覽",
        "ja":    "プレビューを更新",
        "ko":    "미리보기 새로고침",
        "de":    "Vorschau aktualisieren",
        "fr":    "Actualiser l'aperçu",
        "es":    "Actualizar vista previa",
    },
    # The error text is appended directly: keep the trailing separator.
    "Preview failed: ": {
        "zh_CN": "预览失败：",
        "zh_TW": "預覽失敗：",
        "ja":    "プレビューに失敗しました：",
        "ko":    "미리보기 실패: ",
        "de":    "Vorschau fehlgeschlagen: ",
        "fr":    "Échec de l'aperçu : ",
        "es":    "Error en la vista previa: ",
    },
    "Enable a field on the Images tab to preview.": {
        "zh_CN": "请在“图像”页启用一个场以进行预览。",
        "zh_TW": "請在「影像」頁啟用一個場以進行預覽。",
        "ja":    "プレビューするには「画像」タブでフィールドを有効にしてください。",
        "ko":    "미리보려면 '이미지' 탭에서 필드를 활성화하세요.",
        "de":    "Aktivieren Sie ein Feld im Reiter „Bilder“ für die Vorschau.",
        "fr":    "Activez un champ dans l'onglet « Images » pour l'aperçu.",
        "es":    "Active un campo en la pestaña «Imágenes» para la vista previa.",
    },
    "No data for this field/frame.": {
        "zh_CN": "该场/帧没有数据。",
        "zh_TW": "該場/影格沒有資料。",
        "ja":    "このフィールド/フレームにはデータがありません。",
        "ko":    "이 필드/프레임에 데이터가 없습니다.",
        "de":    "Keine Daten für dieses Feld/Bild.",
        "fr":    "Aucune donnée pour ce champ/cette image.",
        "es":    "No hay datos para este campo/fotograma.",
    },
    "Select:": {
        "zh_CN": "选择：",
        "zh_TW": "選擇：",
        "ja":    "選択：",
        "ko":    "선택:",
        "de":    "Auswählen:",
        "fr":    "Sélectionner :",
        "es":    "Seleccionar:",
    },
    "PHYSICAL UNITS": {
        "zh_CN": "物理单位",
        "zh_TW": "物理單位",
        "ja":    "物理単位",
        "ko":    "물리 단위",
        "de":    "PHYSIKALISCHE EINHEITEN",
        "fr":    "UNITÉS PHYSIQUES",
        "es":    "UNIDADES FÍSICAS",
    },
    "Enable physical units": {
        "zh_CN": "启用物理单位",
        "zh_TW": "啟用物理單位",
        "ja":    "物理単位を有効化",
        "ko":    "물리 단위 활성화",
        "de":    "Physikalische Einheiten aktivieren",
        "fr":    "Activer les unités physiques",
        "es":    "Activar unidades físicas",
    },
    "Scale displacement values by pixel size and show physical units on colorbar labels. "
    "Strain is dimensionless and unaffected.": {
        "zh_CN": "按像素尺寸缩放位移值，并在色条标签上显示物理单位。应变为无量纲量，不受影响。",
        "zh_TW": "按像素尺寸縮放位移值，並在色條標籤上顯示物理單位。應變為無因次量，不受影響。",
        "ja":    "変位値をピクセルサイズでスケールし、カラーバーのラベルに物理単位を表示します。ひずみは無次元量のため影響を受けません。",
        "ko":    "변위 값을 픽셀 크기로 스케일링하고 컬러바 레이블에 물리 단위를 표시합니다. 변형률은 무차원이므로 영향받지 않습니다.",
        "de":    "Verschiebungswerte mit der Pixelgröße skalieren und physikalische Einheiten auf den Farbleistenbeschriftungen anzeigen. Dehnung ist dimensionslos und wird nicht beeinflusst.",
        "fr":    "Met à l'échelle les valeurs de déplacement par la taille de pixel et affiche les unités physiques sur les étiquettes de la barre de couleur. La déformation est sans dimension et n'est pas affectée.",
        "es":    "Escala los valores de desplazamiento por el tamaño de píxel y muestra unidades físicas en las etiquetas de la barra de color. La deformación es adimensional y no se ve afectada.",
    },
    "/ pixel": {
        "zh_CN": "/ 像素",
        "zh_TW": "/ 像素",
        "ja":    "/ ピクセル",
        "ko":    "/ 픽셀",
        "de":    "/ Pixel",
        "fr":    "/ pixel",
        "es":    "/ píxel",
    },
    "Pixel size": {
        "zh_CN": "像素尺寸",
        "zh_TW": "像素尺寸",
        "ja":    "ピクセルサイズ",
        "ko":    "픽셀 크기",
        "de":    "Pixelgröße",
        "fr":    "Taille de pixel",
        "es":    "Tamaño de píxel",
    },
    # Unit suffix after the frame-rate spin box (glossary: FPS stays literal).
    "fps": {lang: "fps" for lang in LANGUAGES},
    "Frame rate": {
        "zh_CN": "帧率",
        "zh_TW": "影格率",
        "ja":    "フレームレート",
        "ko":    "프레임 속도",
        "de":    "Bildrate",
        "fr":    "Fréquence d'images",
        "es":    "Velocidad de fotogramas",
    },
    "FRAME RANGE": {
        "zh_CN": "帧范围",
        "zh_TW": "影格範圍",
        "ja":    "フレーム範囲",
        "ko":    "프레임 범위",
        "de":    "BILDBEREICH",
        "fr":    "PLAGE D'IMAGES",
        "es":    "RANGO DE FOTOGRAMAS",
    },
    "All frames": {
        "zh_CN": "所有帧",
        "zh_TW": "所有影格",
        "ja":    "すべてのフレーム",
        "ko":    "모든 프레임",
        "de":    "Alle Bilder",
        "fr":    "Toutes les images",
        "es":    "Todos los fotogramas",
    },
    "Numbered like the frame slider: frame 1 is the reference image.\n"
    "File names use the same numbers: frame 1 is written as _0001.": {
        "zh_CN": "与帧滑块的编号一致：第 1 帧是参考图像。\n文件名使用相同的编号：第 1 帧写为 _0001。",
        "zh_TW": "與影格滑桿的編號一致：第 1 影格是參考影像。\n檔名使用相同的編號：第 1 影格寫為 _0001。",
        "ja":    "フレームスライダーと同じ番号です。フレーム 1 は参照画像です。\nファイル名も同じ番号を使い、フレーム 1 は _0001 として書き出されます。",
        "ko":    "프레임 슬라이더와 같은 번호입니다. 프레임 1은 참조 이미지입니다.\n파일 이름도 같은 번호를 사용하며, 프레임 1은 _0001로 기록됩니다.",
        "de":    "Nummeriert wie der Bildschieberegler: Bild 1 ist das Referenzbild.\nDateinamen verwenden dieselben Nummern: Bild 1 wird als _0001 geschrieben.",
        "fr":    "Numérotées comme le curseur d'images : l'image 1 est l'image de référence.\nLes noms de fichiers utilisent les mêmes numéros : l'image 1 est écrite _0001.",
        "es":    "Numerados como el control deslizante de fotogramas: el fotograma 1 es la imagen de referencia.\nLos nombres de archivo usan los mismos números: el fotograma 1 se escribe como _0001.",
    },

    # --- ImageMessages ---
    "Cannot open the images": {
        "zh_CN": "无法打开图像",
        "zh_TW": "無法開啟影像",
        "ja":    "画像を開けません",
        "ko":    "이미지를 열 수 없습니다",
        "de":    "Bilder können nicht geöffnet werden",
        "fr":    "Impossible d'ouvrir les images",
        "es":    "No se pueden abrir las imágenes",
    },

    # --- MainWindow: log lines, model suggestion, installed models, About ---
    "Project saved: %1": {
        "zh_CN": "项目已保存：%1",
        "zh_TW": "專案已儲存：%1",
        "ja":    "プロジェクトを保存しました：%1",
        "ko":    "프로젝트 저장됨: %1",
        "de":    "Projekt gespeichert: %1",
        "fr":    "Projet enregistré : %1",
        "es":    "Proyecto guardado: %1",
    },
    "Project opened: %1": {
        "zh_CN": "已打开项目：%1",
        "zh_TW": "已開啟專案：%1",
        "ja":    "プロジェクトを開きました：%1",
        "ko":    "프로젝트 열림: %1",
        "de":    "Projekt geöffnet: %1",
        "fr":    "Projet ouvert : %1",
        "es":    "Proyecto abierto: %1",
    },
    # %1 model key (twice), %2 displacement in px, %3 1-based frame list, %4 reach in px.
    "Suggested: RAFTcorr-%1. Largest displacement of the ROI about %2 px (99th percentile over frames %3); "
    "RAFTcorr-%1 reaches %4 px.": {
        "zh_CN": "推荐：RAFTcorr-%1。ROI 内最大位移约 %2 px（在第 %3 帧上取第 99 百分位数）；RAFTcorr-%1 的量程为 %4 px。",
        "zh_TW": "建議：RAFTcorr-%1。ROI 內最大位移約 %2 px（在第 %3 影格上取第 99 百分位數）；RAFTcorr-%1 的量程為 %4 px。",
        "ja":    "推奨：RAFTcorr-%1。ROI の最大変位は約 %2 px（フレーム %3 での 99 パーセンタイル）、RAFTcorr-%1 の到達範囲は %4 px です。",
        "ko":    "추천: RAFTcorr-%1. ROI의 최대 변위는 약 %2 px(프레임 %3에서의 99번째 백분위수)이며, RAFTcorr-%1의 도달 범위는 %4 px입니다.",
        "de":    "Vorschlag: RAFTcorr-%1. Größte Verschiebung der ROI etwa %2 px (99. Perzentil über die Bilder %3); RAFTcorr-%1 reicht %4 px weit.",
        "fr":    "Suggestion : RAFTcorr-%1. Plus grand déplacement de la ROI d'environ %2 px (99e centile sur les images %3) ; la portée de RAFTcorr-%1 est de %4 px.",
        "es":    "Sugerencia: RAFTcorr-%1. Mayor desplazamiento de la ROI de unos %2 px (percentil 99 sobre los fotogramas %3); RAFTcorr-%1 alcanza %4 px.",
    },
    "This exceeds even the reach of RAFTcorr-s8: use incremental tracking.": {
        "zh_CN": "这甚至超出了 RAFTcorr-s8 的量程：请使用增量追踪。",
        "zh_TW": "這甚至超出了 RAFTcorr-s8 的量程：請使用增量追蹤。",
        "ja":    "これは RAFTcorr-s8 の到達範囲さえ超えています。逐次追跡を使用してください。",
        "ko":    "이는 RAFTcorr-s8의 도달 범위조차 넘습니다: 증분 추적을 사용하세요.",
        "de":    "Das übersteigt sogar die Reichweite von RAFTcorr-s8: Verwenden Sie inkrementelles Tracking.",
        "fr":    "Cela dépasse même la portée de RAFTcorr-s8 : utilisez le suivi incrémental.",
        "es":    "Esto supera incluso el alcance de RAFTcorr-s8: use el seguimiento incremental.",
    },
    # Installed Models dialog: %1 model name, %2 s, %3 reach, %4 weight file or "not installed".
    "%1  (s = %2, reach %3 px): %4": {
        "zh_CN": "%1  （s = %2，量程 %3 px）：%4",
        "zh_TW": "%1  （s = %2，量程 %3 px）：%4",
        "ja":    "%1  （s = %2、到達範囲 %3 px）：%4",
        "ko":    "%1  (s = %2, 도달 범위 %3 px): %4",
        "de":    "%1  (s = %2, Reichweite %3 px): %4",
        "fr":    "%1  (s = %2, portée %3 px) : %4",
        "es":    "%1  (s = %2, alcance %3 px): %4",
    },
    # About box (rich text): the author's name and university stay in English.
    "<b>RAFTcorr %1</b><br>Learned digital image correlation with the RAFTcorr-s1, -s2, -s4 and -s8 models."
    "<br><br>Displacements: reference to deformed, +u right, +v down, in pixels, frame-1 (Lagrangian) "
    "coordinates.<br><br>Interface derived from pyALDIC.<br>Zixiang (Zach) Tong, The University of Texas at "
    "Austin.": {
        "zh_CN": "<b>RAFTcorr %1</b><br>基于学习的数字图像相关，使用 RAFTcorr-s1、-s2、-s4 和 -s8 模型。<br><br>位移：从参考到变形，+u 向右，+v 向下，单位为像素，采用第 1 帧（拉格朗日）坐标。<br><br>界面源自 pyALDIC。<br>Zixiang (Zach) Tong, The University of Texas at Austin.",
        "zh_TW": "<b>RAFTcorr %1</b><br>基於學習的數位影像相關，使用 RAFTcorr-s1、-s2、-s4 與 -s8 模型。<br><br>位移：從參考到變形，+u 向右，+v 向下，單位為像素，採用第 1 影格（拉格朗日）座標。<br><br>介面源自 pyALDIC。<br>Zixiang (Zach) Tong, The University of Texas at Austin.",
        "ja":    "<b>RAFTcorr %1</b><br>RAFTcorr-s1、-s2、-s4、-s8 モデルによる学習型デジタル画像相関。<br><br>変位：参照から変形へ、+u 右、+v 下、ピクセル単位、フレーム 1（ラグランジュ）座標。<br><br>インターフェースは pyALDIC に由来します。<br>Zixiang (Zach) Tong, The University of Texas at Austin.",
        "ko":    "<b>RAFTcorr %1</b><br>RAFTcorr-s1, -s2, -s4, -s8 모델을 사용하는 학습 기반 디지털 이미지 상관법.<br><br>변위: 참조에서 변형으로, +u 오른쪽, +v 아래쪽, 픽셀 단위, 프레임 1(라그랑주) 좌표.<br><br>인터페이스는 pyALDIC에서 파생되었습니다.<br>Zixiang (Zach) Tong, The University of Texas at Austin.",
        "de":    "<b>RAFTcorr %1</b><br>Gelernte digitale Bildkorrelation mit den Modellen RAFTcorr-s1, -s2, -s4 und -s8.<br><br>Verschiebungen: von Referenz zu verformt, +u nach rechts, +v nach unten, in Pixeln, in Koordinaten von Bild 1 (Lagrange).<br><br>Oberfläche abgeleitet von pyALDIC.<br>Zixiang (Zach) Tong, The University of Texas at Austin.",
        "fr":    "<b>RAFTcorr %1</b><br>Corrélation d'images numériques apprise avec les modèles RAFTcorr-s1, -s2, -s4 et -s8.<br><br>Déplacements : de la référence vers l'image déformée, +u vers la droite, +v vers le bas, en pixels, dans les coordonnées de l'image 1 (lagrangiennes).<br><br>Interface dérivée de pyALDIC.<br>Zixiang (Zach) Tong, The University of Texas at Austin.",
        "es":    "<b>RAFTcorr %1</b><br>Correlación digital de imágenes aprendida con los modelos RAFTcorr-s1, -s2, -s4 y -s8.<br><br>Desplazamientos: de referencia a deformada, +u a la derecha, +v hacia abajo, en píxeles, en coordenadas del fotograma 1 (lagrangianas).<br><br>Interfaz derivada de pyALDIC.<br>Zixiang (Zach) Tong, The University of Texas at Austin.",
    },

    # --- PipelineController (run log, progress, error dialogs) ---
    "Stopped.": {
        "zh_CN": "已停止。",
        "zh_TW": "已停止。",
        "ja":    "停止しました。",
        "ko":    "중지되었습니다.",
        "de":    "Gestoppt.",
        "fr":    "Arrêté.",
        "es":    "Detenido.",
    },
    "The GPU ran out of memory. Close other GPU programs, choose a model with a larger s, or lower the memory "
    "fraction (Advanced).": {
        "zh_CN": "GPU 显存不足。请关闭其他占用 GPU 的程序，选择 s 更大的模型，或调低“高级”中的 GPU 显存比例。",
        "zh_TW": "GPU 記憶體不足。請關閉其他使用 GPU 的程式，選擇 s 較大的模型，或調低「進階」中的 GPU 記憶體比例。",
        "ja":    "GPU のメモリが不足しました。GPU を使う他のプログラムを閉じるか、s の大きいモデルを選ぶか、「詳細設定」のメモリの割合を下げてください。",
        "ko":    "GPU 메모리가 부족합니다. GPU를 사용하는 다른 프로그램을 닫거나, s가 더 큰 모델을 선택하거나, '고급 설정'에서 메모리 비율을 낮추세요.",
        "de":    "Der GPU-Speicher reicht nicht aus. Schließen Sie andere GPU-Programme, wählen Sie ein Modell mit größerem s oder verringern Sie den Speicheranteil (Erweitert).",
        "fr":    "La mémoire du GPU est épuisée. Fermez les autres programmes qui utilisent le GPU, choisissez un modèle avec un s plus grand, ou réduisez la fraction de mémoire (Avancé).",
        "es":    "La GPU se quedó sin memoria. Cierre otros programas que usen la GPU, elija un modelo con un s mayor o reduzca la fracción de memoria (Avanzado).",
    },
    "Analysis failed": {
        "zh_CN": "分析失败",
        "zh_TW": "分析失敗",
        "ja":    "解析に失敗しました",
        "ko":    "분석 실패",
        "de":    "Analyse fehlgeschlagen",
        "fr":    "Échec de l'analyse",
        "es":    "El análisis falló",
    },
    # Log line; %1 lists 1-based frame numbers, as everywhere on screen.
    "Reference frames without their own ROI carry the ROI of frame 1 forward: %1": {
        "zh_CN": "没有自己 ROI 的参考帧沿用第 1 帧的 ROI：%1",
        "zh_TW": "沒有自己 ROI 的參考影格沿用第 1 影格的 ROI：%1",
        "ja":    "独自の ROI がない参照フレームはフレーム 1 の ROI を引き継ぎます：%1",
        "ko":    "자체 ROI가 없는 참조 프레임은 프레임 1의 ROI를 이어서 사용합니다: %1",
        "de":    "Referenzbilder ohne eigene ROI übernehmen die ROI von Bild 1: %1",
        "fr":    "Les images de référence sans ROI propre reprennent la ROI de l'image 1 : %1",
        "es":    "Los fotogramas de referencia sin ROI propia arrastran la ROI del fotograma 1: %1",
    },
    "Starting...": {
        "zh_CN": "开始中…",
        "zh_TW": "開始中…",
        "ja":    "開始中…",
        "ko":    "시작 중…",
        "de":    "Wird gestartet…",
        "fr":    "Démarrage…",
        "es":    "Iniciando…",
    },
    "The images changed: the running analysis is cancelled.": {
        "zh_CN": "图像已更改：正在运行的分析已取消。",
        "zh_TW": "影像已變更：正在執行的分析已取消。",
        "ja":    "画像が変更されたため、実行中の解析をキャンセルしました。",
        "ko":    "이미지가 바뀌어 실행 중인 분석을 취소했습니다.",
        "de":    "Die Bilder haben sich geändert: Die laufende Analyse wird abgebrochen.",
        "fr":    "Les images ont changé : l'analyse en cours est annulée.",
        "es":    "Las imágenes cambiaron: se cancela el análisis en curso.",
    },
    "The images changed during the run: its results are discarded.": {
        "zh_CN": "运行期间图像发生了变化：本次结果已丢弃。",
        "zh_TW": "執行期間影像發生了變化：本次結果已捨棄。",
        "ja":    "実行中に画像が変更されたため、その結果は破棄されました。",
        "ko":    "실행 중에 이미지가 바뀌어 그 결과를 폐기했습니다.",
        "de":    "Die Bilder haben sich während des Laufs geändert: Seine Ergebnisse werden verworfen.",
        "fr":    "Les images ont changé pendant l'exécution : ses résultats sont abandonnés.",
        "es":    "Las imágenes cambiaron durante la ejecución: se descartan sus resultados.",
    },
    "No frame was computed; the previous results are kept.": {
        "zh_CN": "没有计算任何帧；保留之前的结果。",
        "zh_TW": "沒有計算任何影格；保留先前的結果。",
        "ja":    "計算されたフレームはありません。以前の結果を保持します。",
        "ko":    "계산된 프레임이 없습니다. 이전 결과를 유지합니다.",
        "de":    "Es wurde kein Bild berechnet; die vorherigen Ergebnisse bleiben erhalten.",
        "fr":    "Aucune image n'a été calculée ; les résultats précédents sont conservés.",
        "es":    "No se calculó ningún fotograma; se conservan los resultados anteriores.",
    },
    "Tracking quality": {
        "zh_CN": "追踪质量",
        "zh_TW": "追蹤品質",
        "ja":    "追跡品質",
        "ko":    "추적 품질",
        "de":    "Tracking-Qualität",
        "fr":    "Qualité du suivi",
        "es":    "Calidad del seguimiento",
    },
    "Part of the sequence may not be tracked correctly.": {
        "zh_CN": "序列中的部分内容可能未被正确追踪。",
        "zh_TW": "序列中的部分內容可能未被正確追蹤。",
        "ja":    "シーケンスの一部が正しく追跡されていない可能性があります。",
        "ko":    "시퀀스의 일부가 올바르게 추적되지 않았을 수 있습니다.",
        "de":    "Ein Teil der Sequenz wurde möglicherweise nicht korrekt verfolgt.",
        "fr":    "Une partie de la séquence n'a peut-être pas été suivie correctement.",
        "es":    "Es posible que parte de la secuencia no se haya seguido correctamente.",
    },
    # 'Residual' is the main window's field button (FieldSelector).
    "Show the 'Residual' field to see where the images do not match after the displacement is applied.": {
        "zh_CN": "显示“残差”场，即可查看施加位移后图像在哪些位置不匹配。",
        "zh_TW": "顯示「殘差」場，即可查看套用位移後影像在哪些位置不相符。",
        "ja":    "「残差」の場を表示すると、変位を適用した後に画像が一致しない場所を確認できます。",
        "ko":    "'잔차' 필드를 표시하면 변위를 적용한 뒤 이미지가 일치하지 않는 곳을 볼 수 있습니다.",
        "de":    "Zeigen Sie das Feld „Residuum“ an, um zu sehen, wo die Bilder nach Anwenden der Verschiebung nicht übereinstimmen.",
        "fr":    "Affichez le champ « Résidu » pour voir où les images ne correspondent pas une fois le déplacement appliqué.",
        "es":    "Muestre el campo «Residuo» para ver dónde no coinciden las imágenes tras aplicar el desplazamiento.",
    },

    # --- StrainController (progress label while computing strain) ---
    "Strain frame %1/%2": {
        "zh_CN": "应变：帧 %1/%2",
        "zh_TW": "應變：影格 %1/%2",
        "ja":    "ひずみ：フレーム %1/%2",
        "ko":    "변형률: 프레임 %1/%2",
        "de":    "Dehnung: Bild %1/%2",
        "fr":    "Déformation : image %1/%2",
        "es":    "Deformación: fotograma %1/%2",
    },
}

# The same English word with a class-specific meaning: (context, source).
CONTEXTUAL: dict[tuple[str, str], dict[str, str]] = {
    # Extra image context around the ROI (tiling), not a blank border.
    ("AdvancedPanel", "Margin"): {
        "zh_CN": "边距",
        "zh_TW": "邊距",
        "ja":    "マージン",
        "ko":    "마진",
        "de":    "Rand",
        "fr":    "Marge",
        "es":    "Margen",
    },
    # Label of the correlation-lookup implementation (Automatic / On demand /
    # Stored volume); the bare glossary word 相关 would read "related".
    ("AdvancedPanel", "Correlation"): {
        "zh_CN": "相关计算",
        "zh_TW": "相關計算",
        "ja":    "相関計算",
        "ko":    "상관 계산",
        "de":    "Korrelation",
        "fr":    "Corrélation",
        "es":    "Correlación",
    },
    # Column header (50 px) over the checkbox that includes a field in the
    # export; elsewhere "Export" is a button and keeps the full verb.
    ("ExportDialog", "Export"): {
        "zh_CN": "导出",
        "zh_TW": "匯出",
        "ja":    "出力",
        "ko":    "내보내기",
        "de":    "Export",
        "fr":    "Exporter",
        "es":    "Exportar",
    },
}

# Calls that pass a disambiguation comment: (source, comment).
DISAMBIGUATED: dict[tuple[str, str], dict[str, str]] = {
    # Export dialog frame range: [From] <spin> [to] <spin>.
    ("From", "Frame range: starting frame"): {
        "zh_CN": "从",
        "zh_TW": "從",
        "ja":    "開始",
        "ko":    "시작",
        "de":    "Von",
        "fr":    "De",
        "es":    "Desde",
    },
    ("to", "Frame range: ending frame"): {
        "zh_CN": "到",
        "zh_TW": "到",
        "ja":    "まで",
        "ko":    "끝",
        "de":    "bis",
        "fr":    "à",
        "es":    "a",
    },
}

# Plural messages: source -> lang -> forms (1 for CJK, 2 for de / fr / es).
NUMERUS_TRANSLATIONS: dict[str, dict[str, tuple[str, ...]]] = {
    # ExportDialog status lines: %n files written, %1 the output folder.
    "Exported %n file(s) → %1": {
        "zh_CN": ("已导出 %n 个文件 → %1",),
        "zh_TW": ("已匯出 %n 個檔案 → %1",),
        "ja":    ("%n 個のファイルをエクスポートしました → %1",),
        "ko":    ("%n 개 파일 내보냄 → %1",),
        "de":    ("%n Datei exportiert → %1", "%n Dateien exportiert → %1"),
        "fr":    ("%n fichier exporté → %1", "%n fichiers exportés → %1"),
        "es":    ("Exportado %n archivo → %1", "Exportados %n archivos → %1"),
    },
    "Exported %n image(s) → %1": {
        "zh_CN": ("已导出 %n 张图像 → %1",),
        "zh_TW": ("已匯出 %n 張影像 → %1",),
        "ja":    ("%n 枚の画像をエクスポートしました → %1",),
        "ko":    ("%n 개 이미지 내보냄 → %1",),
        "de":    ("%n Bild exportiert → %1", "%n Bilder exportiert → %1"),
        "fr":    ("%n image exportée → %1", "%n images exportées → %1"),
        "es":    ("Exportada %n imagen → %1", "Exportadas %n imágenes → %1"),
    },
    "Export cancelled: %n file(s) written before stopping.": {
        "zh_CN": ("导出已取消：停止前已写入 %n 个文件。",),
        "zh_TW": ("匯出已取消：停止前已寫入 %n 個檔案。",),
        "ja":    ("エクスポートをキャンセルしました：停止前に %n 個のファイルを書き込みました。",),
        "ko":    ("내보내기가 취소되었습니다: 중지 전에 파일 %n개를 기록했습니다.",),
        "de":    ("Export abgebrochen: %n Datei vor dem Abbruch geschrieben.",
                  "Export abgebrochen: %n Dateien vor dem Abbruch geschrieben."),
        "fr":    ("Export annulé : %n fichier écrit avant l'arrêt.",
                  "Export annulé : %n fichiers écrits avant l'arrêt."),
        "es":    ("Exportación cancelada: %n archivo escrito antes de detenerse.",
                  "Exportación cancelada: %n archivos escritos antes de detenerse."),
    },
    # ImageMessages log line: %1 x %2 = width x height, %3 an English
    # grey-level description ("8-bit") the code inserts as is.
    "%n image(s), %1 x %2 px, %3": {
        "zh_CN": ("%n 张图像，%1 x %2 px，%3",),
        "zh_TW": ("%n 張影像，%1 x %2 px，%3",),
        "ja":    ("画像 %n 枚、%1 x %2 px、%3",),
        "ko":    ("이미지 %n개, %1 x %2 px, %3",),
        "de":    ("%n Bild, %1 x %2 px, %3", "%n Bilder, %1 x %2 px, %3"),
        "fr":    ("%n image, %1 x %2 px, %3", "%n images, %1 x %2 px, %3"),
        "es":    ("%n imagen, %1 x %2 px, %3", "%n imágenes, %1 x %2 px, %3"),
    },
}


# ---------- Table validation ------------------------------------------------

_PLACEHOLDER = re.compile(r"%(?:\d+|n)")
_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9]*\s*/?>")


def _markers(text: str) -> tuple[list[str], list[str]]:
    """Placeholders and HTML tags of *text*, sorted (order may change)."""
    return sorted(_PLACEHOLDER.findall(text)), sorted(_TAG.findall(text))


def _check_rendering(where: str, source: str, rendering: str) -> None:
    if not rendering.strip():
        raise ValueError(f"{where}: empty translation")
    if _markers(rendering) != _markers(source):
        raise ValueError(
            f"{where}: placeholders/tags {_markers(rendering)} do not match the "
            f"source's {_markers(source)}")


def check_tables() -> None:
    """Refuse a malformed table instead of writing garbage into the catalogs.

    Every entry must cover all LANGUAGES, keep the source's placeholders and
    HTML tags, and be a string (or, for plurals, a tuple with the language's
    number of forms). A plain string filed as plural forms would be indexed
    character by character -- pyALDIC once shipped one-letter translations
    that way.
    """
    plain = [(f"TRANSLATIONS[{s!r}]", s, per) for s, per in TRANSLATIONS.items()]
    plain += [(f"CONTEXTUAL[{k!r}]", k[1], per) for k, per in CONTEXTUAL.items()]
    plain += [(f"DISAMBIGUATED[{k!r}]", k[0], per) for k, per in DISAMBIGUATED.items()]
    for where, source, per_lang in plain:
        missing = set(LANGUAGES) - set(per_lang)
        if missing:
            raise ValueError(f"{where}: no translation for {sorted(missing)}")
        for lang, text in per_lang.items():
            if not isinstance(text, str):
                raise TypeError(f"{where}[{lang!r}] must be a string, got {text!r}")
            _check_rendering(f"{where}[{lang!r}]", source, text)
    for source, per_lang in NUMERUS_TRANSLATIONS.items():
        missing = set(LANGUAGES) - set(per_lang)
        if missing:
            raise ValueError(f"NUMERUS_TRANSLATIONS[{source!r}]: no forms for {sorted(missing)}")
        for lang, forms in per_lang.items():
            where = f"NUMERUS_TRANSLATIONS[{source!r}][{lang!r}]"
            if isinstance(forms, str) or not all(isinstance(f, str) for f in forms):
                raise TypeError(f"{where} must be a tuple of plural forms, got {forms!r}")
            if len(forms) != NUMERUS_FORMS[lang]:
                raise ValueError(f"{where}: {len(forms)} forms, {lang} has {NUMERUS_FORMS[lang]}")
            for form in forms:
                _check_rendering(where, source, form)


# ---------- .ts editing -----------------------------------------------------

CONTEXT_BLOCK = re.compile(r"(<context>\s*<name>(.*?)</name>)(.*?)(</context>)", re.DOTALL)
MESSAGE_BLOCK = re.compile(r"(<message[^>]*>)(.*?)(</message>)", re.DOTALL)
SOURCE_INNER = re.compile(r"<source>(.*?)</source>", re.DOTALL)
COMMENT_INNER = re.compile(r"<comment>(.*?)</comment>", re.DOTALL)
UNFINISHED_PLAIN = re.compile(r'<translation type="unfinished">.*?</translation>', re.DOTALL)
UNFINISHED_NUMERUS = re.compile(
    r'<translation type="unfinished">\s*((?:<numerusform>.*?</numerusform>\s*)+)</translation>', re.DOTALL)
NUMERUS_FORM = re.compile(r"<numerusform>.*?</numerusform>", re.DOTALL)


def _plain_entry(context: str, source: str, comment: str | None) -> dict[str, str] | None:
    entry = CONTEXTUAL.get((context, source))
    if entry is None and comment is not None:
        entry = DISAMBIGUATED.get((source, comment))
    return entry if entry is not None else TRANSLATIONS.get(source)


def _fill_message(context: str, open_tag: str, body: str, lang: str) -> tuple[str, bool]:
    """``(new_body, filled)`` for one ``<message>``; unfinished bodies only."""
    if 'type="unfinished"' not in body:
        return body, False
    src_m = SOURCE_INNER.search(body)
    if src_m is None:
        return body, False
    source = unescape(src_m.group(1))
    comment_m = COMMENT_INNER.search(body)
    comment = unescape(comment_m.group(1)) if comment_m else None

    if 'numerus="yes"' in open_tag:
        forms = NUMERUS_TRANSLATIONS.get(source, {}).get(lang)
        slots = UNFINISHED_NUMERUS.search(body)
        if forms is None or slots is None:
            return body, False
        expected = len(NUMERUS_FORM.findall(slots.group(1)))
        if len(forms) != expected:
            raise ValueError(f"{lang} [{context}] {source!r}: {len(forms)} plural forms, catalog expects {expected}")
        joined = "\n".join(f"            <numerusform>{escape(f)}</numerusform>" for f in forms)
        new = f"<translation>\n{joined}\n        </translation>"
        return body[:slots.start()] + new + body[slots.end():], True

    entry = _plain_entry(context, source, comment)
    rendering = entry.get(lang) if entry is not None else None
    if rendering is None:
        return body, False
    new = f"<translation>{escape(rendering)}</translation>"
    return UNFINISHED_PLAIN.sub(lambda _m: new, body, count=1), True


def fill_text(text: str, lang: str) -> tuple[str, int, list[str]]:
    """Fill *text* (a whole .ts file); return ``(new_text, filled, missing)``.

    *missing* lists the unfinished messages no table covers, as
    ``"[Context] source"``.
    """
    check_tables()
    filled = 0
    missing: list[str] = []

    def per_context(cm: re.Match[str]) -> str:
        nonlocal filled
        head, name, body, tail = cm.group(1), unescape(cm.group(2)), cm.group(3), cm.group(4)

        def per_message(mm: re.Match[str]) -> str:
            nonlocal filled
            new_body, done = _fill_message(name, mm.group(1), mm.group(2), lang)
            if done:
                filled += 1
            elif 'type="unfinished"' in mm.group(2):
                src = SOURCE_INNER.search(mm.group(2))
                missing.append(f"[{name}] {unescape(src.group(1)) if src else '?'}")
            return mm.group(1) + new_body + mm.group(3)

        return head + MESSAGE_BLOCK.sub(per_message, body) + tail

    return CONTEXT_BLOCK.sub(per_context, text), filled, missing


def fill_ts(lang: str, ts_dir: Path = TS_DIR) -> tuple[int, list[str]]:
    """Fill one language's catalog in place; return ``(filled, missing)``."""
    path = ts_dir / f"raftcorr_{lang}.ts"
    # newline="": keep lupdate's LF line endings on Windows as well.
    with open(path, encoding="utf-8", newline="") as fh:
        text = fh.read()
    new_text, filled, missing = fill_text(text, lang)
    if new_text != text:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(new_text)
    return filled, missing


def main() -> int:
    status = 0
    for lang in LANGUAGES:
        filled, missing = fill_ts(lang)
        print(f"  {lang:<6} filled={filled:<4} missing={len(missing)}")
        for item in missing:
            print(f"         no translation: {item}")
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
