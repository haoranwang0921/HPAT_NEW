"""绘图工具：用 Pillow（PIL）直接画论文图表，不依赖 matplotlib。

背景：为了让论文图片完全可复现、可离线生成（不依赖 GUI/字体渲染环境），
本文件直接用 Pillow 在像素层面画图。提供四类图：
- grouped_metric_panels：多指标分组柱状图（每个指标一个小面板）；
- stacked_bar：堆积柱状图（每根柱子按分项堆叠）；
- line_panel：多面板折线图；
- horizontal_grouped_bar：横向分组柱状图。

每个函数都返回生成的图片路径，且输入都是"行字典列表"（与各分析表的
行格式一致），方便把分析结果直接喂给绘图函数。
"""

from __future__ import annotations

import pathlib
from typing import Any


# 统一色板：10 种论文风格的深色系 RGB 颜色，按索引循环使用
PALETTE = [
    (45, 92, 161),    # 深蓝
    (214, 124, 42),   # 橙
    (64, 145, 108),   # 绿
    (143, 86, 153),   # 紫
    (191, 69, 69),    # 红
    (82, 128, 143),   # 青灰
    (226, 169, 58),   # 金黄
    (92, 92, 92),     # 深灰
    (122, 161, 71),   # 草绿
    (95, 111, 176),   # 蓝灰
]


def _pil():
    """延迟导入 Pillow（只在真正画图时才导入，加快模块加载）。

    延迟导入的好处：命令行工具哪怕只是列出可用命令，也不会因为
    本机没装 Pillow 而整体崩溃。

    :return: (Image, ImageDraw, ImageFont) 三个子模块。
    """
    from PIL import Image, ImageDraw, ImageFont  # type: ignore

    return Image, ImageDraw, ImageFont


def _font(size: int = 12):
    """获取指定字号的字体；系统字体不存在时退回 Pillow 默认字体。

    :param size: 字号（像素）。
    :return: 一个可用字体对象。
    """
    _, _, ImageFont = _pil()
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
    except Exception:
        return ImageFont.load_default()  # 兜底：默认位图字体（中英文均可能显示不全）


def _text(draw: Any, xy: tuple[int, int], text: str, fill=(30, 30, 30), size: int = 12, anchor: str | None = None) -> None:
    """在画布上写一行文字（封装字体与颜色参数）。

    :param draw: ImageDraw 画布对象。
    :param xy: 文字左上角坐标。
    :param text: 文字内容。
    :param fill: 文字颜色（RGB）。
    :param size: 字号。
    :param anchor: Pillow 文字锚点（可选）。
    """
    draw.text(xy, text, fill=fill, font=_font(size), anchor=anchor)


def _save(path: pathlib.Path, image: Any) -> pathlib.Path:
    """把 PIL 图片保存到指定路径（自动创建父目录）。

    :param path: 目标图片路径。
    :param image: PIL Image 对象。
    :return: 保存后的路径。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def grouped_metric_panels(
    path: pathlib.Path,
    rows: list[dict[str, Any]],
    *,
    title: str,
    group_key: str,
    series_key: str,
    metric_key: str,
    value_key: str,
    unit_key: str,
) -> pathlib.Path:
    """画"多指标分组柱状图"：最多 3 个指标，每个指标一个小面板。

    例：按模型变体（group）为横轴、数据来源（series）为系列，
    分别展示 能耗/延迟/利用率 三个指标。

    :param path: 保存路径。
    :param rows: 行字典列表。
    :param title: 图标题。
    :param group_key: 分组字段名（如 "model_variant"）。
    :param series_key: 系列字段名（如 "evidence_tier"）。
    :param metric_key: 指标字段名（如 "metric"）。
    :param value_key: 数值字段名（如 "energy_mj"）。
    :param unit_key: 单位字段名（用于坐标轴标签）。
    :return: 保存后的图片路径。
    """
    Image, ImageDraw, _ = _pil()
    image = Image.new("RGB", (1800, 850), "white")  # 整张画布
    draw = ImageDraw.Draw(image)
    _text(draw, (60, 28), title, size=26)
    # 收集去重后的指标、分组、系列（保持出现顺序）
    metrics = []
    for row in rows:
        if row[metric_key] not in metrics:
            metrics.append(row[metric_key])
    groups = []
    for row in rows:
        if row[group_key] not in groups:
            groups.append(row[group_key])
    series = []
    for row in rows:
        if row[series_key] not in series:
            series.append(row[series_key])
    panel_w = 540  # 面板宽
    panel_h = 540  # 面板高
    start_x = 70
    top = 105
    for p, metric in enumerate(metrics[:3]):  # 最多 3 个面板
        x0 = start_x + p * (panel_w + 35)
        y0 = top
        data = [row for row in rows if row[metric_key] == metric]
        max_v = max(float(row[value_key]) for row in data) if data else 1.0  # 用于归一化柱高
        _text(draw, (x0, y0 - 28), f"{metric} ({data[0][unit_key] if data else ''})", size=18)
        # 画坐标轴
        draw.line((x0 + 55, y0, x0 + 55, y0 + panel_h), fill=(80, 80, 80), width=2)
        draw.line((x0 + 55, y0 + panel_h, x0 + panel_w, y0 + panel_h), fill=(80, 80, 80), width=2)
        # 画水平网格线 + 数值标签（5 档）
        for tick in range(5):
            val = max_v * tick / 4.0
            y = y0 + panel_h - int(panel_h * tick / 4.0)
            draw.line((x0 + 50, y, x0 + panel_w, y), fill=(230, 230, 230), width=1)
            _text(draw, (x0 + 4, y - 7), f"{val:.1f}", size=10)
        # 每个分组的位置与柱宽（按系列数均分）
        group_space = (panel_w - 85) / max(len(groups), 1)
        bar_w = max(16, int(group_space / (len(series) + 1)))
        for gi, group in enumerate(groups):
            gx = int(x0 + 70 + gi * group_space)
            _text(draw, (gx + int(group_space / 2) - 26, y0 + panel_h + 12), group.replace("MobileViT-", ""), size=11)
            for si, name in enumerate(series):
                match = [row for row in data if row[group_key] == group and row[series_key] == name]
                if not match:
                    continue
                value = float(match[0][value_key])
                h = int(panel_h * value / max_v) if max_v else 0  # 柱高按最大比例缩放
                bx = gx + si * bar_w
                color = PALETTE[si % len(PALETTE)]
                draw.rectangle((bx, y0 + panel_h - h, bx + bar_w - 2, y0 + panel_h), fill=color)
        # 图例（系列色块）
        for si, name in enumerate(series):
            lx = x0 + 65 + si * 150
            ly = y0 + panel_h + 55
            draw.rectangle((lx, ly, lx + 18, ly + 12), fill=PALETTE[si % len(PALETTE)])
            _text(draw, (lx + 24, ly - 2), name[:20], size=11)
    return _save(path, image)


def stacked_bar(
    path: pathlib.Path,
    rows: list[dict[str, Any]],
    *,
    title: str,
    group_key: str,
    stack_key: str,
    value_key: str,
    ylabel: str,
    max_stacks: int = 10,
) -> pathlib.Path:
    """画堆积柱状图：每根柱子 = 一个分组，柱内按分项（stack）堆叠。

    例：每个模型变体一根柱，柱内堆叠 10 个能耗分项，直观看出哪项
    占比最大。

    :param path: 保存路径。
    :param rows: 行字典列表。
    :param title: 图标题。
    :param group_key: 分组字段名。
    :param stack_key: 堆叠分项字段名。
    :param value_key: 数值字段名。
    :param ylabel: 纵轴标签。
    :param max_stacks: 最多显示的堆叠分项数（防止图例太长）。
    :return: 保存后的图片路径。
    """
    Image, ImageDraw, _ = _pil()
    image = Image.new("RGB", (1500, 900), "white")
    draw = ImageDraw.Draw(image)
    _text(draw, (60, 28), title, size=26)
    groups = []
    stacks = []
    for row in rows:
        if row[group_key] not in groups:
            groups.append(row[group_key])
        if row[stack_key] not in stacks:
            stacks.append(row[stack_key])
    stacks = stacks[:max_stacks]  # 只画前 max_stacks 个分项
    # 每个分组的合计（只累计显示出来的分项）
    totals = {
        group: sum(float(row[value_key]) for row in rows if row[group_key] == group and row[stack_key] in stacks)
        for group in groups
    }
    max_total = max(totals.values()) if totals else 1.0
    x0, y0, w, h = 110, 100, 1040, 610  # 绘图区
    draw.line((x0, y0, x0, y0 + h), fill=(80, 80, 80), width=2)
    draw.line((x0, y0 + h, x0 + w, y0 + h), fill=(80, 80, 80), width=2)
    _text(draw, (22, y0 + 260), ylabel, size=14)
    # 水平网格线
    for tick in range(6):
        val = max_total * tick / 5.0
        y = y0 + h - int(h * tick / 5.0)
        draw.line((x0 - 5, y, x0 + w, y), fill=(230, 230, 230), width=1)
        _text(draw, (x0 - 76, y - 7), f"{val:.1f}", size=11)
    bar_space = w / max(len(groups), 1)
    bar_w = min(170, int(bar_space * 0.55))
    for gi, group in enumerate(groups):
        bx = int(x0 + gi * bar_space + (bar_space - bar_w) / 2)
        base = y0 + h  # 从柱底开始往上堆
        for si, stack in enumerate(stacks):
            value = sum(float(row[value_key]) for row in rows if row[group_key] == group and row[stack_key] == stack)
            seg_h = int(h * value / max_total) if max_total else 0
            draw.rectangle((bx, base - seg_h, bx + bar_w, base), fill=PALETTE[si % len(PALETTE)])
            base -= seg_h  # 上移基准线，形成堆叠
        _text(draw, (bx - 5, y0 + h + 16), str(group).replace("MobileViT-", ""), size=13)
    # 图例
    lx = 1190
    ly = 110
    for si, stack in enumerate(stacks):
        draw.rectangle((lx, ly + si * 32, lx + 20, ly + 14 + si * 32), fill=PALETTE[si % len(PALETTE)])
        _text(draw, (lx + 28, ly - 2 + si * 32), str(stack)[:34], size=12)
    return _save(path, image)


def line_panel(
    path: pathlib.Path,
    rows: list[dict[str, Any]],
    *,
    title: str,
    x_key: str,
    y_key: str,
    series_key: str,
    panel_key: str | None = None,
    ylabel: str = "",
    xlabel: str = "",
) -> pathlib.Path:
    """画多面板折线图：最多 4 个面板（每面板一组数据）。

    例：按 非理想性种类 分面板，x 轴为扫描强度，y 轴为相对误差，
    每个变体一条折线。

    :param path: 保存路径。
    :param rows: 行字典列表。
    :param title: 图标题。
    :param x_key: x 轴字段名（数值）。
    :param y_key: y 轴字段名（数值）。
    :param series_key: 系列字段名（每条折线一个系列）。
    :param panel_key: 面板字段名；None 时只画一个面板。
    :param ylabel: 纵轴标签。
    :param xlabel: 横轴标签。
    :return: 保存后的图片路径。
    """
    Image, ImageDraw, _ = _pil()
    image = Image.new("RGB", (1600, 900), "white")
    draw = ImageDraw.Draw(image)
    _text(draw, (60, 28), title, size=26)
    panels = []
    if panel_key:
        for row in rows:
            if row[panel_key] not in panels:
                panels.append(row[panel_key])
    else:
        panels = ["all"]
    panels = panels[:4]  # 最多 4 个面板
    cols = 2
    panel_w, panel_h = 680, 310
    for pi, panel in enumerate(panels):
        px = 85 + (pi % cols) * 760
        py = 105 + (pi // cols) * 380
        data = [row for row in rows if not panel_key or row[panel_key] == panel]
        xs = sorted({float(row[x_key]) for row in data})
        ys = [float(row[y_key]) for row in data]
        max_y = max(ys) if ys else 1.0
        min_x, max_x = (min(xs), max(xs)) if xs else (0.0, 1.0)
        if max_x == min_x:
            max_x = min_x + 1.0  # 防止 x 范围为零导致除零
        _text(draw, (px, py - 25), str(panel), size=17)
        # 坐标轴
        draw.line((px + 50, py, px + 50, py + panel_h), fill=(80, 80, 80), width=2)
        draw.line((px + 50, py + panel_h, px + panel_w, py + panel_h), fill=(80, 80, 80), width=2)
        _text(draw, (px - 15, py + 120), ylabel, size=12)
        # y 轴刻度
        for tick in range(5):
            y_val = max_y * tick / 4.0
            y = py + panel_h - int(y_val / max_y * (panel_h - 20)) if max_y else py + panel_h
            draw.line((px + 45, y, px + panel_w, y), fill=(235, 235, 235), width=1)
            _text(draw, (px + 4, y - 7), f"{y_val:.2f}", size=10)
        # x 轴刻度
        for tick in range(5):
            x_val = min_x + (max_x - min_x) * tick / 4.0
            x = px + 50 + int((x_val - min_x) / (max_x - min_x) * (panel_w - 70))
            draw.line((x, py + panel_h, x, py + panel_h + 5), fill=(80, 80, 80), width=1)
            _text(draw, (x - 12, py + panel_h + 10), f"{x_val:.1f}", size=10)
        if xlabel:
            _text(draw, (px + int(panel_w * 0.36), py + panel_h + 42), xlabel, size=12)
        series = []
        for row in data:
            if row[series_key] not in series:
                series.append(row[series_key])
        for si, series_name in enumerate(series[:6]):  # 每面板最多 6 条折线
            points = sorted([row for row in data if row[series_key] == series_name], key=lambda r: float(r[x_key]))
            coords = []
            for row in points:
                # 数据点 → 画布坐标
                x = px + 50 + int((float(row[x_key]) - min_x) / (max_x - min_x) * (panel_w - 70))
                y = py + panel_h - int(float(row[y_key]) / max_y * (panel_h - 20)) if max_y else py + panel_h
                coords.append((x, y))
            if len(coords) > 1:
                draw.line(coords, fill=PALETTE[si % len(PALETTE)], width=3)  # 连线
            for x, y in coords:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=PALETTE[si % len(PALETTE)])  # 数据点圆点
        # 图例
        for si, series_name in enumerate(series[:4]):
            lx = px + 60 + si * 145
            ly = py + panel_h + 20
            draw.rectangle((lx, ly, lx + 16, ly + 12), fill=PALETTE[si % len(PALETTE)])
            _text(draw, (lx + 22, ly - 2), str(series_name)[:16], size=10)
    return _save(path, image)


def horizontal_grouped_bar(
    path: pathlib.Path,
    rows: list[dict[str, Any]],
    *,
    title: str,
    group_key: str,
    series_key: str,
    value_key: str,
    xlabel: str,
) -> pathlib.Path:
    """画横向分组柱状图：每组一行，组内系列并排横着画。

    适合值域跨度大的数据（如 MAC 数），横向柱子可容纳更宽的数值标签。

    :param path: 保存路径。
    :param rows: 行字典列表。
    :param title: 图标题。
    :param group_key: 分组字段名（每行一组）。
    :param series_key: 系列字段名。
    :param value_key: 数值字段名。
    :param xlabel: 横轴标签。
    :return: 保存后的图片路径。
    """
    Image, ImageDraw, _ = _pil()
    image = Image.new("RGB", (1500, 900), "white")
    draw = ImageDraw.Draw(image)
    _text(draw, (60, 28), title, size=26)
    groups = []
    series = []
    for row in rows:
        if row[group_key] not in groups:
            groups.append(row[group_key])
        if row[series_key] not in series:
            series.append(row[series_key])
    max_v = max(float(row[value_key]) for row in rows) if rows else 1.0
    x0, y0, w = 330, 110, 900  # 绘图区
    row_h = max(58, int(620 / max(len(groups), 1)))  # 每行高度
    bar_h = max(10, int(row_h / (len(series) + 1)))  # 每根柱高度
    # 坐标轴
    draw.line((x0, y0 - 15, x0, y0 + row_h * len(groups)), fill=(80, 80, 80), width=2)
    draw.line((x0, y0 + row_h * len(groups), x0 + w, y0 + row_h * len(groups)), fill=(80, 80, 80), width=2)
    # 竖网格线
    for tick in range(6):
        x = x0 + int(w * tick / 5.0)
        val = max_v * tick / 5.0
        draw.line((x, y0 - 15, x, y0 + row_h * len(groups)), fill=(232, 232, 232), width=1)
        _text(draw, (x - 14, y0 + row_h * len(groups) + 10), f"{val:.3f}", size=10)
    _text(draw, (x0 + int(w * 0.38), y0 + row_h * len(groups) + 44), xlabel, size=14)
    for gi, group in enumerate(groups):
        gy = y0 + gi * row_h
        _text(draw, (55, gy + int(row_h / 2) - 8), str(group)[:35], size=13)  # 左侧组名
        for si, name in enumerate(series):
            match = [row for row in rows if row[group_key] == group and row[series_key] == name]
            if not match:
                continue
            value = float(match[0][value_key])
            bw = int(w * value / max_v) if max_v else 0  # 柱长按比例缩放
            by = gy + 8 + si * bar_h
            draw.rectangle((x0, by, x0 + bw, by + bar_h - 3), fill=PALETTE[si % len(PALETTE)])
    # 图例
    lx, ly = 1260, 115
    for si, name in enumerate(series):
        draw.rectangle((lx, ly + si * 34, lx + 20, ly + 14 + si * 34), fill=PALETTE[si % len(PALETTE)])
        _text(draw, (lx + 28, ly - 2 + si * 34), str(name)[:24], size=12)
    return _save(path, image)
