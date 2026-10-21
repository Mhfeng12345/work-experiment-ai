import argparse
from pathlib import Path

from vector_db.knowledge_store import (
    count_knowledge,
    delete_document,
    delete_experiment,
    delete_source,
    get_db,
    get_document,
    import_all,
    import_markdown,
    list_documents,
    list_experiment_keys,
    list_sources,
    search_knowledge,
)


def cmd_import(args):
    if args.file:
        path = Path(args.file)

        if not path.exists():
            raise SystemExit(f"文件不存在：{path}")

        count = import_markdown(path)
        print(f"导入成功：{path.name}，{count} 个知识点")
    else:
        import_all()

    print(f"当前知识点总数：{count_knowledge()}")


def cmd_list(args):
    show_experiments = args.experiments or not args.sources
    show_sources = args.sources or not args.experiments

    if show_experiments:
        keys = list_experiment_keys()
        print(f"实验（{len(keys)}）：")
        for key in keys:
            print(f"  - {key}")

    if show_sources:
        sources = list_sources()
        print(f"来源文件（{len(sources)}）：")
        for source in sources:
            print(f"  - {source}")

    print(f"知识点总数：{count_knowledge()}")


def cmd_search(args):
    try:
        results = search_knowledge(
            question=args.question,
            experiment_key=args.experiment,
            k=args.k,
        )
    except ValueError as e:
        raise SystemExit(str(e))

    if not results:
        print("未检索到相关知识。")
        return

    for index, doc in enumerate(results, start=1):
        print(f"\n--- 结果 {index} ---")
        print("内容：", doc.page_content)
        print("元数据：", doc.metadata)


def cmd_show(args):
    docs = list_documents(
        experiment_key=args.experiment,
        source=args.source,
        limit=args.limit,
    )

    if not docs:
        print("没有匹配的知识点。")
        return

    for index, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        print(f"\n--- {index} | {meta.get('source', '?')} ---")
        print("内容：", doc.page_content)

    print(f"\n共 {len(docs)} 条")


def cmd_delete(args):
    if args.source:
        label = f"来源 {args.source}"
        docs = list_documents(source=args.source)
        do_delete = lambda: delete_source(args.source)
    elif args.experiment:
        label = f"实验 {args.experiment}"
        docs = list_documents(experiment_key=args.experiment)
        do_delete = lambda: delete_experiment(args.experiment)
    else:
        label = f"id {args.id}"
        doc = get_document(args.id)
        docs = [doc] if doc else []
        do_delete = lambda: delete_document(args.id)

    if not docs:
        print(f"未找到匹配记录（{label}）。")
        return

    if not args.yes:
        print(f"将要删除 {label} 的 {len(docs)} 条知识点（预览）：")
        for doc in docs[:10]:
            preview = doc.page_content.replace("\n", " ")[:60]
            print(f"  - {preview}...")
        if len(docs) > 10:
            print(f"  ... 其余 {len(docs) - 10} 条省略")
        print("确认删除请加 --yes")
        return

    removed = do_delete()
    print(f"已删除 {label} 的 {removed} 条知识点。")
    print(f"当前知识点总数：{count_knowledge()}")


def cmd_rebuild(args):
    get_db().reset_collection()
    print("已清空集合，重新导入中……")
    import_all()
    print(f"当前知识点总数：{count_knowledge()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chroma_controller.py",
        description="Chroma 知识库管理工具",
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    p = subparsers.add_parser("import", help="导入/更新知识（默认全量）")
    p.add_argument("--file", help="只导入指定的 Markdown 文件")
    p.set_defaults(func=cmd_import)

    p = subparsers.add_parser("list", help="列出实验简称与来源文件")
    p.add_argument("--experiments", action="store_true", help="只列实验")
    p.add_argument("--sources", action="store_true", help="只列来源文件")
    p.set_defaults(func=cmd_list)

    p = subparsers.add_parser("search", help="语义检索")
    p.add_argument("--experiment", required=True, help="实验简称")
    p.add_argument("--question", required=True, help="检索问题")
    p.add_argument("--k", type=int, default=3, help="返回条数，默认 3")
    p.set_defaults(func=cmd_search)

    p = subparsers.add_parser("show", help="按实验简称/来源查看知识点")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--experiment", help="实验简称")
    group.add_argument("--source", help="来源文件名")
    p.add_argument("--limit", type=int, default=None, help="最多显示条数")
    p.set_defaults(func=cmd_show)

    p = subparsers.add_parser("delete", help="删除知识点（默认只预览）")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", help="按来源文件名删除")
    group.add_argument("--experiment", help="按实验简称删除")
    group.add_argument("--id", help="按知识点 id 删除")
    p.add_argument("--yes", action="store_true", help="确认执行删除")
    p.set_defaults(func=cmd_delete)

    p = subparsers.add_parser("rebuild", help="清空并重新导入全部知识")
    p.set_defaults(func=cmd_rebuild)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
