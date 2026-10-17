"""知识库 Markdown 格式校验

在提交知识库文件前自检，避免因格式问题导致导入中断或内容丢失。

用法：
    python validate_knowledge.py

退出码：
    0 = 无错误（可能有警告）
    1 = 存在错误，必须修复后才能导入
"""

import re
import sys
from pathlib import Path

from vector_db.knowledge_store import KNOWLEDGE_DIR, load_markdown

HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
PLACEHOLDER = re.compile(r"TODO|待补充|待定|xxx|XXX|XXX+", re.IGNORECASE)

MIN_POINTS = 5
MAX_POINTS = 60

FULL_WIDTH_BRACKETS = "（）"
HALF_WIDTH_BRACKETS = "()"


class FileReport:
    def __init__(self, path: Path):
        self.path = path
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.notes: list[str] = []
        self.point_count = 0
        self.experiment_name = ""

    @property
    def name(self) -> str:
        return self.path.name


def split_front_matter(text: str) -> tuple[dict, str]:
    """拆分 front matter 与正文，返回 (元数据, 正文)"""
    metadata = {"experiment_name": "", "source": ""}

    if not text.startswith("---"):
        return metadata, text

    parts = text.split("---", 2)
    if len(parts) != 3:
        return metadata, text

    for line in parts[1].splitlines():
        key, separator, value = line.partition(":")

        if separator and key.strip() in metadata:
            metadata[key.strip()] = value.strip().strip("'\"")

    return metadata, parts[2]


def scan_points(body: str) -> tuple[int, list[str], list[str], list[str]]:
    """扫描正文，返回 (知识点数, 空知识点, 首个知识点前的正文, 深层级标题)"""
    empty_points: list[str] = []
    deep_headings: list[str] = []
    preamble: list[str] = []
    seen_point = False
    count = 0
    title = ""
    lines: list[str] = []

    def finish() -> None:
        if not title:
            return
        if not "\n".join(lines).strip():
            empty_points.append(title)

    for line in body.splitlines():
        heading = HEADING.match(line)

        if heading:
            level = len(heading.group(1))

            if level == 1:
                finish()
                title = ""
                lines = []
                continue

            if level == 2:
                finish()
                count += 1
                seen_point = True
                title = heading.group(2).strip()
                lines = []
                continue

            if level >= 3:
                deep_headings.append(line.strip())
                lines.append(line)
                continue

        if title:
            lines.append(line)
        elif not seen_point and line.strip():
            preamble.append(line.strip())

    finish()

    return count, empty_points, preamble, deep_headings


def check_brackets(value: str) -> str | None:
    """检测括号全半角混用"""
    has_full = any(ch in value for ch in FULL_WIDTH_BRACKETS)
    has_half = any(ch in value for ch in HALF_WIDTH_BRACKETS)

    if has_full and has_half:
        return "括号全半角混用，会导致检索 0 命中"

    return None


def validate_file(path: Path) -> FileReport:
    report = FileReport(path)

    try:
        raw = path.read_bytes()
    except OSError as e:
        report.errors.append(f"文件无法读取：{e}")
        return report

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        report.errors.append("文件不是 UTF-8 编码，请另存为 UTF-8")
        return report

    metadata, body = split_front_matter(text)
    experiment_name = metadata["experiment_name"]
    report.experiment_name = experiment_name

    if not experiment_name:
        report.errors.append(
            "缺少 experiment_name，导入时会抛出异常并中断整个导入流程"
        )
    else:
        if experiment_name.startswith("请填写"):
            report.errors.append("experiment_name 仍是模板占位文字，未替换")

        bracket_issue = check_brackets(experiment_name)
        if bracket_issue:
            report.warnings.append(f"experiment_name {bracket_issue}")

    if not metadata["source"]:
        report.warnings.append(
            "缺少 source 字段，将回退使用文件名，之后改名会导致旧数据残留"
        )
    elif metadata["source"] != path.name:
        report.warnings.append(
            f"source（{metadata['source']}）与文件名（{path.name}）不一致，易混淆"
        )

    count, empty_points, preamble, deep_headings = scan_points(body)

    for title in empty_points:
        report.warnings.append(f"知识点「{title}」正文为空，导入时会被静默跳过")

    for line in preamble:
        report.warnings.append(
            f"首个知识点之前的内容不会被收录：「{line[:30]}」"
        )

    for line in deep_headings:
        report.warnings.append(
            f"使用了 ### 及更深层级，不会被识别为知识点：{line[:30]}"
        )

    if count and count < MIN_POINTS:
        report.warnings.append(f"仅 {count} 个知识点，建议不少于 {MIN_POINTS} 个")
    elif count > MAX_POINTS:
        report.warnings.append(
            f"共 {count} 个知识点，超过 {MAX_POINTS} 个，建议拆分或合并以控制单块粒度"
        )

    for line in body.splitlines():
        if PLACEHOLDER.search(line):
            report.warnings.append(f"正文含占位内容：{line.strip()[:30]}")
            break

    try:
        documents = load_markdown(path)
        report.point_count = len(documents)
    except ValueError as e:
        report.errors.append(f"解析失败：{e}")
    except Exception as e:
        report.errors.append(f"解析异常：{type(e).__name__}: {e}")

    return report


def main() -> int:
    if not KNOWLEDGE_DIR.exists():
        print(f"知识文档目录不存在：{KNOWLEDGE_DIR}")
        return 1

    files = sorted(KNOWLEDGE_DIR.glob("*.md"))

    if not files:
        print(f"未找到 Markdown 文件：{KNOWLEDGE_DIR}")
        return 1

    reports = [validate_file(path) for path in files]

    for index, report in enumerate(reports, start=1):
        status = "✗" if report.errors else ("⚠" if report.warnings else "✓")

        if report.errors:
            headline = report.errors[0]
        elif report.point_count:
            headline = f"{report.point_count} 个知识点"
        else:
            headline = "0 个知识点"

        print(f"[{index}] {report.name}    {status}  {headline}")

        for message in report.errors:
            print(f"    ✗ {message}")

        for message in report.warnings:
            print(f"    ⚠ {message}")

    name_index: dict[str, list[str]] = {}

    for report in reports:
        if report.errors or not report.experiment_name:
            continue

        name_index.setdefault(report.experiment_name, []).append(report.name)

    duplicates = {k: v for k, v in name_index.items() if len(v) > 1}

    for experiment_name, names in duplicates.items():
        print(f"    ✗ experiment_name「{experiment_name}」在多个文件中重复：{names}")

    error_total = sum(len(r.errors) for r in reports) + sum(
        len(v) for v in duplicates.values()
    )
    warning_total = sum(len(r.warnings) for r in reports)

    print()
    print(f"共 {len(reports)} 个文件，{error_total} 个错误，{warning_total} 个警告。")

    if error_total:
        print("存在错误，请修复后再执行导入。")
        return 1

    print("格式校验通过，可以导入。")
    return 0


if __name__ == "__main__":
    sys.exit(main())