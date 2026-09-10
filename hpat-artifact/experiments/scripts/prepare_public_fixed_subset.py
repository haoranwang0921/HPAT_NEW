"""准备公开发布的固定数据子集（subset，即从完整数据集中挑出的一个固定子样本）。

做什么：生成一份"固定不变"的公开图片子集，供论文里的稳健性/敏感性实验
       （扰动检查）使用。子集来源有两种：
       1) public-examples：3 张公开的示例狗图片（PyTorch hub / torchvision 官方示例图）；
       2) imagenette160-valid：Imagenette-160 数据集的官方验证集
          （Imagenette 是 ImageNet 的 10 类子集，本脚本从这里采样出可公开的固定子集）。
       脚本会把每张图的来源、SHA256 哈希（校验文件完整性的指纹）、标签依据等信息
       记入"来源台账"（source ledger），保证子集可追溯、可复现。

数据从哪来：
    public-examples：从 GitHub 等公开 URL 直接下载图片。
    imagenette160-valid：从 fastai 提供的 S3 归档（imagenette2-160.tgz，约 160 像素
        分辨率）下载并解压出验证集。

输出到哪：把 fixed_subset.csv（子集清单）和 fixed_subset_source_ledger.csv（来源台账）
        同时写到 3 处：
        - 运行目录（--output-dir 下的 dataset 目录和 tables 子目录）；
        - 项目根目录 tables/（便于论文附件直接引用）；
        并输出 public_fixed_subset_manifest.json（元信息清单）。

怎么运行（示例）：
    python prepare_public_fixed_subset.py --output-dir run --source imagenette160-valid
    python prepare_public_fixed_subset.py --output-dir run --source public-examples

注意事项：子集是"外部公开数据"；它不是完整的 ImageNet 验证集，也不代表任何
        芯片/设备实测结果，只能支撑"固定子集扰动检查"这类限定性结论。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import tarfile
import urllib.request
from typing import Any

from _common import (
    REPO_ROOT,
    base_manifest,
    ensure_dir,
    project_writes_enabled,
    relative,
    sha256_file,
    write_csv,
    write_json,
)
from hpat_eval.e_local import E_LOCAL_CLAIM_BOUNDARY


# Imagenette-160 官方归档的默认下载地址（fastai 的 S3 存储）
IMAGENETTE160_ARCHIVE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"

# 子集清单 CSV 的列定义（每张图一行，记录路径、标签、来源与边界声明）
PUBLIC_FIXED_SUBSET_FIELDS = [
    "path",
    "label",
    "synset",
    "class_name",
    "source_id",
    "source_url",
    "source_type",
    "source_split",
    "archive_url",
    "archive_sha256",
    "image_sha256",
    "label_basis",
    "external_data_label",
    "claim_boundary",
]

# 来源台账 CSV 的列定义（比子集清单多了下载状态、类型、字节数等下载信息）
PUBLIC_FIXED_SUBSET_SOURCE_FIELDS = [
    "source_id",
    "filename",
    "sha256",
    "label",
    "synset",
    "class_name",
    "source_url",
    "source_type",
    "source_split",
    "download_status",
    "content_type",
    "byte_count",
    "archive_url",
    "archive_sha256",
    "label_basis",
    "external_data_label",
    "claim_boundary",
]

# 公开发布的示例图片清单：只有 3 张（从公共官方来源挑选的狗狗图片），
# label 是 ImageNet-1K 类别编号（如 258 = Samoyed 萨摩耶犬）
PUBLIC_IMAGES = [
    {
        "source_id": "pytorch_hub_dog_samoyed",
        "filename": "pytorch_hub_dog_samoyed.jpg",
        "url": "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
        "label": 258,
        "synset": "",
        "class_name": "Samoyed",
        "label_basis": "PyTorch hub example dog image; visually identifiable Samoyed; ImageNet-1K class index from torchvision category order.",
    },
    {
        "source_id": "torchvision_gallery_dog_pembroke",
        "filename": "torchvision_gallery_dog_pembroke.jpg",
        "url": "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/dog1.jpg",
        "label": 263,
        "synset": "",
        "class_name": "Pembroke",
        "label_basis": "Torchvision gallery dog1 image; visually a corgi/Pembroke-like dog; breed ambiguity is retained in the source ledger.",
    },
    {
        "source_id": "torchvision_gallery_dog_german_shepherd",
        "filename": "torchvision_gallery_dog_german_shepherd.jpg",
        "url": "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/dog2.jpg",
        "label": 235,
        "synset": "",
        "class_name": "German shepherd",
        "label_basis": "Torchvision gallery dog2 image; visually identifiable German shepherd; ImageNet-1K class index from torchvision category order.",
    },
]

# Imagenette 的 10 个类别：synset（WordNet 编号，如 n01440764）→
# 对应的 ImageNet-1K 类别编号 label 与类别名 class_name
IMAGENETTE_LABELS = {
    "n01440764": {"label": 0, "class_name": "tench"},
    "n02102040": {"label": 217, "class_name": "English springer"},
    "n02979186": {"label": 482, "class_name": "cassette player"},
    "n03000684": {"label": 491, "class_name": "chain saw"},
    "n03028079": {"label": 497, "class_name": "church"},
    "n03394916": {"label": 566, "class_name": "French horn"},
    "n03417042": {"label": 569, "class_name": "garbage truck"},
    "n03425413": {"label": 571, "class_name": "gas pump"},
    "n03445777": {"label": 574, "class_name": "golf ball"},
    "n03888257": {"label": 701, "class_name": "parachute"},
}


def _download(url: str, path: pathlib.Path) -> tuple[str, str, int]:
    """下载单个文件（用于 public-examples 的示例图片）。

    参数：url —— 图片的完整下载地址；path —— 保存到本地的文件路径。
    返回：(状态, 内容类型, 字节数) 三元组，供来源台账记录。
    """
    # 带 User-Agent 请求头，避免部分服务器拒绝无标识的脚本请求
    request = urllib.request.Request(url, headers={"User-Agent": "hpat-e-local-fixed-subset/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        content = response.read()  # 一次性读完整个文件
        content_type = response.headers.get("content-type", "")  # 服务器返回的 MIME 类型
    path.write_bytes(content)  # 二进制写入本地
    return "ok", content_type, len(content)


def _download_archive(url: str, path: pathlib.Path) -> tuple[str, str, int]:
    """下载较大的归档文件（用于 Imagenette 的 tgz 压缩包）。

    与 _download 的区别：分块（chunk）流式写入，避免一次性占满内存；
    另外若本地已有非空缓存文件则直接复用（返回状态 cached）。
    参数：url —— 归档下载地址；path —— 本地缓存路径。
    返回：(状态, 内容类型, 字节数) 三元组。
    """
    # 缓存命中：文件已存在且非空，直接复用，不重复下载
    if path.exists() and path.stat().st_size > 0:
        return "cached", "application/gzip", path.stat().st_size
    request = urllib.request.Request(url, headers={"User-Agent": "hpat-e-local-imagenette/1.0"})
    byte_count = 0
    content_type = ""
    with urllib.request.urlopen(request, timeout=120) as response:
        content_type = response.headers.get("content-type", "")
        with path.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)  # 每次读 1 MB
                if not chunk:
                    break  # 读到文件尾
                f.write(chunk)
                byte_count += len(chunk)
    return "ok", content_type, byte_count


def _safe_extract_tgz(archive_path: pathlib.Path, destination: pathlib.Path) -> None:
    """安全解压 tgz 归档，防止"路径穿越"攻击（tar 解压的安全校验）。

    为什么：恶意或损坏的压缩包里可能带有 ../../xxx 之类的成员路径，
    直接解压会把文件写到目标目录之外。这里先逐一检查成员路径，确认
    所有解压目标都落在 destination 内部才真正解压。
    参数：archive_path —— tgz 归档路径；destination —— 解压目标目录。
    返回：无。
    """
    destination_resolved = destination.resolve()  # 解析为绝对规范路径
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            # 安全条件：目标必须在 destination 内（或是 destination 本身）
            if destination_resolved not in target.parents and target != destination_resolved:
                raise ValueError(f"Unsafe tar member path: {member.name}")
        archive.extractall(destination)


def _public_example_rows(output_dir: pathlib.Path, dataset_dir: pathlib.Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[pathlib.Path], dict[str, Any]]:
    """准备"公开示例图片"子集：下载 3 张官方示例图并生成清单与台账。

    参数：output_dir —— 本次运行的输出目录；dataset_dir —— 数据集存放目录。
    返回：(rows, source_rows, outputs, extra) 四元组：
        rows —— 子集清单行（每张图一条）；
        source_rows —— 来源台账行（含下载状态等信息）；
        outputs —— 需要写出的输出文件路径列表；
        extra —— 写进 manifest 的额外元信息。
    """
    rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    # 逐张下载 PUBLIC_IMAGES 里定义的示例图，并计算 SHA256 哈希
    for item in PUBLIC_IMAGES:
        image_path = dataset_dir / item["filename"]
        status, content_type, byte_count = _download(item["url"], image_path)
        image_sha256 = sha256_file(image_path)
        # 给这批外部图片统一打上"来源标签"与"边界声明"
        external_label = "external_public_fixed_subset_image; not ImageNet validation-set evidence"
        # 子集清单与台账共用的公共字段
        common = {
            "label": item["label"],
            "synset": item.get("synset", ""),
            "class_name": item["class_name"],
            "source_url": item["url"],
            "source_type": "public example image",
            "source_split": "not_applicable",
            "archive_url": "",
            "archive_sha256": "",
            "label_basis": item["label_basis"],
            "external_data_label": external_label,
            "claim_boundary": E_LOCAL_CLAIM_BOUNDARY,
        }
        # 子集清单行：面向论文使用，含相对路径与图片哈希
        rows.append(
            {
                "path": item["filename"],
                "source_id": item["source_id"],
                "image_sha256": image_sha256,
                **common,
            }
        )
        # 来源台账行：面向溯源审计，多记录下载状态/类型/字节数
        source_rows.append(
            {
                "source_id": item["source_id"],
                "filename": item["filename"],
                "sha256": image_sha256,
                "download_status": status,
                "content_type": content_type,
                "byte_count": byte_count,
                **common,
            }
        )
    # 输出文件有 6 个：运行目录下 2 个、项目 tables 下 2 个、运行 tables 下 2 个
    tables_dir = ensure_dir(output_dir / "tables")
    outputs = [
        dataset_dir / "fixed_subset.csv",
        dataset_dir / "fixed_subset_source_ledger.csv",
        REPO_ROOT / "tables" / "fixed_subset_public_imagenet_examples.csv",
        REPO_ROOT / "tables" / "fixed_subset_public_imagenet_source_ledger.csv",
        tables_dir / "fixed_subset_public_imagenet_examples.csv",
        tables_dir / "fixed_subset_public_imagenet_source_ledger.csv",
    ]
    extra = {
        "source": "public-examples",
        "image_count": len(rows),
        "label_count": len(rows),
        "class_count": len({row["label"] for row in rows}),
        "external_data_label": "external_public_fixed_subset_image",
        "promotion_note": (
            "This tiny public ImageNet-class subset supports only fixed-subset perturbation checks. "
            "It is not the ImageNet validation set and cannot support broad robustness claims."
        ),
    }
    return rows, source_rows, outputs, extra


def _imagenette_rows(
    output_dir: pathlib.Path,
    dataset_dir: pathlib.Path,
    archive_url: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[pathlib.Path], dict[str, Any]]:
    """准备 Imagenette-160 验证集子集：下载、解压、逐图生成清单与台账。

    做什么（分步）：
      1) 把归档缓存到 downloads 目录（带 SHA256 命名，保证不同 URL 不冲突）；
      2) 计算归档的 SHA256 并安全解压；
      3) 遍历 10 个类别的 val 验证集目录，逐张图片计算哈希并生成行；
      4) 返回子集清单、来源台账、输出文件列表与额外元信息。
    参数：output_dir —— 运行输出目录；dataset_dir —— 数据集存放目录；
        archive_url —— 归档下载地址。
    返回：(rows, source_rows, outputs, extra) 四元组，含义同 _public_example_rows。
    """
    cache_dir = ensure_dir(output_dir / "downloads")
    archive_name = "imagenette2-160.tgz"
    # 若传入了非默认归档地址，则用 URL 哈希生成独立缓存文件名，避免与默认缓存混淆
    if archive_url != IMAGENETTE160_ARCHIVE_URL:
        archive_name = f"imagenette2-160_{hashlib.sha256(archive_url.encode('utf-8')).hexdigest()[:12]}.tgz"
    archive_path = cache_dir / archive_name
    # 下载（或命中缓存）并记录下载信息
    status, content_type, byte_count = _download_archive(archive_url, archive_path)
    archive_sha256 = sha256_file(archive_path)
    # 只在还没有 val 目录时解压，避免重复解压覆盖
    dataset_root = dataset_dir / "imagenette2-160"
    if not (dataset_root / "val").exists():
        _safe_extract_tgz(archive_path, dataset_dir)
    val_dir = dataset_root / "val"
    if not val_dir.exists():
        raise ValueError(f"Imagenette archive did not produce a validation split at {val_dir}")

    rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    external_label = "external_public_imagenette160_validation; not full ImageNet validation; not silicon/device robustness"
    # 遍历 10 个类别目录，目录名即 synset（如 n01440764）
    for synset in sorted(IMAGENETTE_LABELS):
        class_dir = val_dir / synset
        if not class_dir.exists():
            raise ValueError(f"Imagenette validation class directory is missing: {class_dir}")
        label_info = IMAGENETTE_LABELS[synset]
        for image_path in sorted(class_dir.iterdir()):
            # 只保留常见图片格式，跳过其他文件
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            # 用相对路径作为子集里的 path，保证可移植
            rel_path = image_path.relative_to(dataset_dir).as_posix()
            image_sha256 = sha256_file(image_path)
            # source_id 由 synset + 文件名构成，保证每张图全局唯一
            source_id = f"imagenette160_valid_{synset}_{image_path.stem}"
            # 记录标签依据：来自官方验证集划分 + torchvision 类别编号映射
            label_basis = (
                f"Official Imagenette-160 validation split synset {synset}; "
                f"mapped to ImageNet-1K label {label_info['label']} ({label_info['class_name']}) "
                "using torchvision category order."
            )
            common = {
                "label": label_info["label"],
                "synset": synset,
                "class_name": label_info["class_name"],
                "source_url": archive_url,
                "source_type": "fastai Imagenette-160 validation image",
                "source_split": "val",
                "archive_url": archive_url,
                "archive_sha256": archive_sha256,
                "label_basis": label_basis,
                "external_data_label": external_label,
                "claim_boundary": E_LOCAL_CLAIM_BOUNDARY,
            }
            # 子集清单行
            rows.append(
                {
                    "path": rel_path,
                    "source_id": source_id,
                    "image_sha256": image_sha256,
                    **common,
                }
            )
            # 来源台账行（多了下载状态等信息）
            source_rows.append(
                {
                    "source_id": source_id,
                    "filename": rel_path,
                    "sha256": image_sha256,
                    "download_status": status,
                    "content_type": content_type,
                    "byte_count": byte_count,
                    **common,
                }
            )
    if not rows:
        raise ValueError(f"Imagenette validation split produced no supported image files: {val_dir}")

    # 与 _public_example_rows 相同的 6 个输出文件，只是文件名换成 imagenette 版本
    tables_dir = ensure_dir(output_dir / "tables")
    outputs = [
        dataset_dir / "fixed_subset.csv",
        dataset_dir / "fixed_subset_source_ledger.csv",
        REPO_ROOT / "tables" / "fixed_subset_imagenette160_valid.csv",
        REPO_ROOT / "tables" / "fixed_subset_imagenette160_valid_source_ledger.csv",
        tables_dir / "fixed_subset_imagenette160_valid.csv",
        tables_dir / "fixed_subset_imagenette160_valid_source_ledger.csv",
    ]
    extra = {
        "source": "imagenette160-valid",
        "archive_url": archive_url,
        "archive_sha256": archive_sha256,
        "archive_path": archive_path.relative_to(output_dir).as_posix(),
        "archive_download_status": status,
        "archive_content_type": content_type,
        "archive_byte_count": byte_count,
        "image_count": len(rows),
        "label_count": len(rows),
        "class_count": len({row["synset"] for row in rows}),
        "external_data_label": "external_public_imagenette160_validation",
        "promotion_note": (
            "This external-public Imagenette-160 validation subset supports fixed-subset robustness "
            "and sensitivity evidence only. It is not the full ImageNet validation set, not a silicon "
            "measurement, and not a measured edge/mobile deployment baseline."
        ),
    }
    return rows, source_rows, outputs, extra


def run(
    output_dir: pathlib.Path,
    dataset_dir: pathlib.Path,
    source: str = "public-examples",
    archive_url: str = IMAGENETTE160_ARCHIVE_URL,
) -> dict[str, pathlib.Path]:
    """入口函数：按选定的数据源生成固定子集清单、台账与 manifest。

    参数：output_dir —— 本次运行输出目录；dataset_dir —— 数据集存放目录；
        source —— 数据源，'public-examples' 或 'imagenette160-valid'；
        archive_url —— Imagenette 归档地址（仅 source 为 imagenette160-valid 时使用）。
    返回：dict[str, pathlib.Path] —— 各输出文件的路径映射（供调用方打印/引用）。
    """
    ensure_dir(output_dir)
    ensure_dir(dataset_dir)
    # 按数据源分派到对应的生成函数
    if source == "imagenette160-valid":
        rows, source_rows, outputs, manifest_extra = _imagenette_rows(output_dir, dataset_dir, archive_url)
    else:
        rows, source_rows, outputs, manifest_extra = _public_example_rows(output_dir, dataset_dir)

    # 取出 6 个输出路径（索引 0/1 在 dataset 目录，2/3 在项目 tables，4/5 在运行 tables）
    subset_csv = dataset_dir / "fixed_subset.csv"
    source_ledger = dataset_dir / "fixed_subset_source_ledger.csv"
    project_subset_csv = outputs[2]
    project_source_ledger = outputs[3]
    run_subset_csv = outputs[4]
    run_source_ledger = outputs[5]

    # 同一份数据写到 3 处，保证"运行目录"与"项目 tables"都能直接引用
    write_csv(subset_csv, rows, PUBLIC_FIXED_SUBSET_FIELDS)
    write_csv(source_ledger, source_rows, PUBLIC_FIXED_SUBSET_SOURCE_FIELDS)
    write_csv(project_subset_csv, rows, PUBLIC_FIXED_SUBSET_FIELDS)
    write_csv(project_source_ledger, source_rows, PUBLIC_FIXED_SUBSET_SOURCE_FIELDS)
    write_csv(run_subset_csv, rows, PUBLIC_FIXED_SUBSET_FIELDS)
    write_csv(run_source_ledger, source_rows, PUBLIC_FIXED_SUBSET_SOURCE_FIELDS)

    manifest_path = output_dir / "public_fixed_subset_manifest.json"
    # 是否允许写入项目根目录（可能受环境开关控制，见 project_writes_enabled）
    write_project = project_writes_enabled()
    manifest = base_manifest("public_fixed_subset", "external public fixed ImageNet-class subset")
    manifest.update(
        {
            "status": "ok",
            "dataset_dir": relative(dataset_dir),
            # 若不允许写项目根目录，就从 outputs 里剔除索引 2/3 的路径
            "outputs": [
                relative(path)
                for index, path in enumerate(outputs)
                if write_project or index not in {2, 3}
            ],
            "project_write_performed": write_project,
            "claim_boundary": E_LOCAL_CLAIM_BOUNDARY,
            "caffeinate_used": False,
            "caffeinate_reason": "Not used by this preparer; full E-local runs should wrap the caller in caffeinate -dimsu.",
            **manifest_extra,
        }
    )
    write_json(manifest_path, manifest)
    return {
        "subset_csv": subset_csv,
        "source_ledger": source_ledger,
        "project_subset_csv": project_subset_csv,
        "project_source_ledger": project_source_ledger,
        "run_subset_csv": run_subset_csv,
        "run_source_ledger": run_source_ledger,
        "manifest": manifest_path,
    }


def main() -> None:
    """命令行入口：解析参数、调用 run() 并打印输出文件路径（JSON 格式）。

    做什么：把 --output-dir / --dataset-dir / --source / --archive-url 等命令行参数
        传给 run()，完成后用相对路径打印各输出文件，方便上层脚本接住。
    参数：全部来自命令行。
    返回：无。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--dataset-dir",
        default="",
    )
    parser.add_argument("--source", default="imagenette160-valid", choices=["public-examples", "imagenette160-valid"])
    parser.add_argument("--archive-url", default=IMAGENETTE160_ARCHIVE_URL)
    args = parser.parse_args()
    output_dir = pathlib.Path(args.output_dir)
    # 未显式指定 dataset-dir 时，默认放在 output_dir/data 下
    outputs = run(
        output_dir,
        pathlib.Path(args.dataset_dir) if args.dataset_dir else output_dir / "data",
        source=args.source,
        archive_url=args.archive_url,
    )
    # 用相对路径打印结果，便于人工核对与脚本消费
    print(json.dumps({key: relative(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
