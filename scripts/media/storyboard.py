"""On-screen text of the quick-start video, per language.

The Chinese captions use the terms of docs/i18n/glossary.md and the words of the translated interface
they sit under (感兴趣区域, 推荐模型, 量程, 运行 DIC 分析, 位移场, 应变场, 虚拟引伸计, 导出结果).
"""
from __future__ import annotations

from dataclasses import dataclass

from compositor import Caption

TOTAL_STEPS = 8
REPO_URL = "github.com/zachtong/RAFTcorr"
CREDIT = "Zixiang Tong  ·  zachtong@utexas.edu"
GROUP = "Dr. Jin Yang Group  ·  The University of Texas at Austin"


@dataclass(frozen=True)
class Script:
    lang: str                  # LanguageManager code of the interface
    cjk: bool
    title: str
    subtitle: str
    end_tagline: str
    end_install: str
    timelapse: str             # badge of the steps shown faster than real time
    steps: tuple[Caption, ...]


def _steps(rows: list[tuple[str, str]]) -> tuple[Caption, ...]:
    return tuple(Caption(step=f"{i} / {TOTAL_STEPS}", title=t, sub=s) for i, (t, s) in enumerate(rows, start=1))


ZH = Script(
    lang="zh_CN",
    cjk=True,
    title="RAFTcorr",
    subtitle="一分钟上手：从散斑图像到应变场",
    end_tagline="基于学习的数字图像相关：桌面应用、命令行与 Python 库",
    end_install="Windows / macOS 安装包  ·  pip install raftcorr",
    timelapse="延时",
    steps=_steps([
        ("导入图像", "拖入或浏览图像文件夹，第一张图像即为参考图像"),
        ("感兴趣区域", "用矩形框选标距段，再用圆形工具裁掉两个孔"),
        ("选择模型", "「推荐模型」先测量运动，再选出量程够用的最精细模型"),
        ("运行分析", "点击「运行 DIC 分析」；区域太大时自动分块，以适应显存"),
        ("位移场", "切换到位移 V，拖动滑块逐帧查看"),
        ("应变场", "打开应变窗口并计算，这里显示沿加载方向的应变 εyy"),
        ("探针分析", "放置虚拟引伸计，它的应变随帧绘成曲线"),
        ("导出结果", "导出 HDF5 / MAT / NPZ 数据、图片、动画和 HTML 报告"),
    ]),
)

EN = Script(
    lang="en",
    cjk=False,
    title="RAFTcorr",
    subtitle="Quick start: from speckle images to strain in one minute",
    end_tagline="Learned digital image correlation: desktop application, command line and Python library",
    end_install="Windows / macOS installers  ·  pip install raftcorr",
    timelapse="time-lapse",
    steps=_steps([
        ("Import images", "Drop or browse an image folder; the first image is the reference"),
        ("Region of interest", "Box the gauge section, then cut out both holes with the circle tool"),
        ("Model", "Suggest a Model measures the motion and picks the finest model that reaches it"),
        ("Run", "Click Run DIC Analysis; large regions are tiled to fit the GPU memory"),
        ("Displacement", "Switch to V and scrub through the frames"),
        ("Strain", "Open the strain window and compute; here εyy, the strain along the load"),
        ("Probes", "Place a virtual extensometer; its strain plots over the frames"),
        ("Export", "HDF5 / MAT / NPZ data, figures, animations and an HTML report"),
    ]),
)

SCRIPTS = {"zh_CN": ZH, "en": EN}
