from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path


DOCS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DOCS_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"
PACKAGE_ROOT = SRC_ROOT / "xtalk"
SERVER_DOCS_ROOT = DOCS_ROOT / "api" / "server"
SAMPLE_APP_ROOT = REPO_ROOT / "examples" / "sample_app"
GITHUB_SOURCE_ROOT = "https://github.com/xcc-zach/xtalk/blob/main"
SERVER_API_MODULES = (
    "xtalk",
    "xtalk.events",
    "xtalk.model_types",
    "xtalk.models",
    "xtalk.serving",
    "xtalk.serving.module_types",
)
LOCALES = ("en", "zh")
NUMPY_SECTION_TRANSLATIONS_ZH = {
    "Attributes": "属性",
    "Examples": "示例",
    "Methods": "方法",
    "Notes": "说明",
    "Other Parameters": "其他参数",
    "Parameters": "参数",
    "Raises": "抛出",
    "Receives": "接收",
    "References": "参考",
    "Returns": "返回",
    "See Also": "另请参阅",
    "Warns": "警告",
    "Warnings": "警告",
    "Yields": "生成",
}


@dataclass
class ObjectDoc:
    name: str
    kind: str
    signature: str = ""
    docstring: str = ""
    source_module: str = ""
    children: list["ObjectDoc"] = field(default_factory=list)
    fields: list["ObjectDoc"] = field(default_factory=list)
    value: str = ""


@dataclass
class ModuleInfo:
    name: str
    path: Path
    docstring: str
    exports: list[str]
    definitions: dict[str, ObjectDoc]
    imports: dict[str, tuple[str, str]]
    star_imports: list[str]


MODULE_CACHE: dict[str, ModuleInfo] = {}
NUMPY_SECTION_NAMES = {
    "Attributes",
    "Examples",
    "Methods",
    "Notes",
    "Other Parameters",
    "Parameters",
    "Raises",
    "Receives",
    "References",
    "Returns",
    "See Also",
    "Warns",
    "Warnings",
    "Yields",
}
NUMPY_FIELD_SECTIONS = {
    "Attributes",
    "Methods",
    "Other Parameters",
    "Parameters",
    "Raises",
    "Receives",
    "Returns",
    "Warns",
    "Yields",
}
NUMPY_LIST_SECTIONS = {"See Also"}
NUMPY_EXAMPLE_SECTIONS = {"Examples"}
NUMPY_SECTION_UNDERLINE_RE = re.compile(r"^-{3,}\s*$")
NUMPY_FIELD_RE = re.compile(r"^(?P<name>.+?)\s*:\s*(?P<annotation>.+)$")


def _doc_path_for_module(module_name: str) -> Path:
    parts = module_name.split(".")
    if module_name == "xtalk":
        return Path("xtalk") / "index.md"
    if _module_path(module_name).name == "__init__.py":
        return Path(*parts) / "index.md"
    return Path(*parts[:-1]) / f"{parts[-1]}.md"


def _localized_doc_path(doc_path: Path, locale: str) -> Path:
    if locale == "en":
        return doc_path
    return doc_path.with_name(f"{doc_path.stem}.{locale}{doc_path.suffix}")


def _github_url_for_path(path: Path) -> str:
    relative_path = path.relative_to(REPO_ROOT).as_posix()
    return f"{GITHUB_SOURCE_ROOT}/{relative_path}"


def _github_url_for_module(module_name: str) -> str | None:
    if not module_name.startswith("xtalk"):
        return None
    return _github_url_for_path(_module_path(module_name))


def _module_path(module_name: str) -> Path:
    relative = Path(*module_name.split("."))
    package_init = SRC_ROOT / relative / "__init__.py"
    if package_init.exists():
        return package_init
    module_file = SRC_ROOT / f"{relative}.py"
    if module_file.exists():
        return module_file
    raise FileNotFoundError(f"Cannot resolve module path for {module_name}")


def _safe_unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""
    return ast.unparse(node)


def _resolve_module_name(current_module: str, import_from: ast.ImportFrom) -> str:
    current_path = _module_path(current_module)
    if current_path.name == "__init__.py":
        package_parts = current_module.split(".")
    else:
        package_parts = current_module.split(".")[:-1]
    if import_from.level:
        trim = max(import_from.level - 1, 0)
        base = package_parts[: len(package_parts) - trim]
        if import_from.module:
            return ".".join(base + import_from.module.split("."))
        return ".".join(base)
    return import_from.module or ""


def _extract_string_list(node: ast.AST | None) -> list[str] | None:
    if node is None:
        return None
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values: list[str] = []
        for element in node.elts:
            if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
                return None
            values.append(element.value)
        return values
    return None


def _extract_lazy_imports(node: ast.AST | None) -> dict[str, tuple[str, str]]:
    """Extract string-keyed lazy import mappings from a module assignment."""
    if not isinstance(node, ast.Dict):
        return {}

    imports: dict[str, tuple[str, str]] = {}
    for key_node, value_node in zip(node.keys, node.values):
        if (
            not isinstance(key_node, ast.Constant)
            or not isinstance(key_node.value, str)
            or not isinstance(value_node, ast.Tuple)
            or len(value_node.elts) != 2
        ):
            continue

        module_node, attr_node = value_node.elts
        if (
            isinstance(module_node, ast.Constant)
            and isinstance(module_node.value, str)
            and isinstance(attr_node, ast.Constant)
            and isinstance(attr_node.value, str)
        ):
            imports[key_node.value] = (module_node.value, attr_node.value)

    return imports


def _format_arg(arg: ast.arg, default: ast.AST | None = None) -> str:
    text = arg.arg
    if arg.annotation is not None:
        text += f": {_safe_unparse(arg.annotation)}"
    if default is not None:
        text += f" = {_safe_unparse(default)}"
    return text


def _format_arguments(args: ast.arguments) -> str:
    parts: list[str] = []
    positional = list(args.posonlyargs) + list(args.args)
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)

    for index, arg in enumerate(args.posonlyargs):
        parts.append(_format_arg(arg, defaults[index]))
    if args.posonlyargs:
        parts.append("/")

    for index, arg in enumerate(args.args, start=len(args.posonlyargs)):
        parts.append(_format_arg(arg, defaults[index]))

    if args.vararg is not None:
        vararg = f"*{args.vararg.arg}"
        if args.vararg.annotation is not None:
            vararg += f": {_safe_unparse(args.vararg.annotation)}"
        parts.append(vararg)
    elif args.kwonlyargs:
        parts.append("*")

    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        parts.append(_format_arg(arg, default))

    if args.kwarg is not None:
        kwarg = f"**{args.kwarg.arg}"
        if args.kwarg.annotation is not None:
            kwarg += f": {_safe_unparse(args.kwarg.annotation)}"
        parts.append(kwarg)

    return ", ".join(parts)


def _format_function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
    signature = f"{prefix}{node.name}({_format_arguments(node.args)})"
    if node.returns is not None:
        signature += f" -> {_safe_unparse(node.returns)}"
    return signature


def _decorator_names(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> list[str]:
    return [_safe_unparse(decorator) for decorator in node.decorator_list]


def _is_public_method(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    if node.name == "__init__":
        return True
    return not node.name.startswith("_")


def _class_field_docs(node: ast.ClassDef) -> list[ObjectDoc]:
    fields: list[ObjectDoc] = []
    for child in node.body:
        if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            if child.target.id.startswith("_"):
                continue
            fields.append(
                ObjectDoc(
                    name=child.target.id,
                    kind="attribute",
                    signature=(
                        f"{child.target.id}: {_safe_unparse(child.annotation)}"
                        if child.annotation is not None
                        else child.target.id
                    ),
                    value=_safe_unparse(child.value),
                )
            )
        elif isinstance(child, ast.Assign):
            for target in child.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    fields.append(
                        ObjectDoc(
                            name=target.id,
                            kind="attribute",
                            signature=target.id,
                            value=_safe_unparse(child.value),
                        )
                    )
    deduped: dict[str, ObjectDoc] = {}
    for field_doc in fields:
        deduped[field_doc.name] = field_doc
    return list(deduped.values())


def _parse_class(node: ast.ClassDef, module_name: str) -> ObjectDoc:
    bases = ", ".join(_safe_unparse(base) for base in node.bases)
    decorators = _decorator_names(node)
    if bases:
        signature = f"class {node.name}({bases})"
    else:
        signature = f"class {node.name}"
    if decorators:
        signature = ", ".join(f"@{decorator}" for decorator in decorators) + "\n" + signature

    methods: list[ObjectDoc] = []
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public_method(child):
            method_kind = "property" if "property" in _decorator_names(child) else "method"
            methods.append(
                ObjectDoc(
                    name=child.name,
                    kind=method_kind,
                    signature=_format_function_signature(child),
                    docstring=ast.get_docstring(child) or "",
                    source_module=module_name,
                )
            )

    return ObjectDoc(
        name=node.name,
        kind="class",
        signature=signature,
        docstring=ast.get_docstring(node) or "",
        source_module=module_name,
        children=methods,
        fields=_class_field_docs(node),
    )


def _parse_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    module_name: str,
) -> ObjectDoc:
    return ObjectDoc(
        name=node.name,
        kind="function",
        signature=_format_function_signature(node),
        docstring=ast.get_docstring(node) or "",
        source_module=module_name,
    )


def _parse_assignment(node: ast.Assign | ast.AnnAssign, module_name: str) -> list[ObjectDoc]:
    docs: list[ObjectDoc] = []
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        docs.append(
            ObjectDoc(
                name=node.target.id,
                kind="attribute",
                signature=(
                    f"{node.target.id}: {_safe_unparse(node.annotation)}"
                    if node.annotation is not None
                    else node.target.id
                ),
                value=_safe_unparse(node.value),
                source_module=module_name,
            )
        )
        return docs

    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                docs.append(
                    ObjectDoc(
                        name=target.id,
                        kind="attribute",
                        signature=target.id,
                        value=_safe_unparse(node.value),
                        source_module=module_name,
                    )
                )
    return docs


def _parse_module(module_name: str) -> ModuleInfo:
    cached = MODULE_CACHE.get(module_name)
    if cached is not None:
        return cached

    path = _module_path(module_name)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    docstring = ast.get_docstring(tree) or ""
    exports: list[str] | None = None
    definitions: dict[str, ObjectDoc] = {}
    imports: dict[str, tuple[str, str]] = {}
    star_imports: list[str] = []
    ordered_names: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    exports = _extract_string_list(node.value)
                    break
                if isinstance(target, ast.Name):
                    imports.update(_extract_lazy_imports(node.value))
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "__all__":
            exports = _extract_string_list(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            imports.update(_extract_lazy_imports(node.value))

        if isinstance(node, ast.ClassDef):
            definitions[node.name] = _parse_class(node, module_name)
            ordered_names.append(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions[node.name] = _parse_function(node, module_name)
            ordered_names.append(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for object_doc in _parse_assignment(node, module_name):
                definitions[object_doc.name] = object_doc
                ordered_names.append(object_doc.name)
        elif isinstance(node, ast.ImportFrom):
            resolved_module = _resolve_module_name(module_name, node)
            for alias in node.names:
                if alias.name == "*":
                    star_imports.append(resolved_module)
                    continue
                local_name = alias.asname or alias.name
                imports[local_name] = (resolved_module, alias.name)
                ordered_names.append(local_name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name.split(".")[0]
                imports[local_name] = (alias.name, "")
                ordered_names.append(local_name)

    if exports is None:
        exports = []
        seen: set[str] = set()
        for name in definitions:
            if name.startswith("_") or name == "__all__":
                continue
            if name not in seen:
                seen.add(name)
                exports.append(name)
        for imported_module in star_imports:
            for exported_name in _exported_names(imported_module):
                if exported_name.startswith("_") or exported_name in seen:
                    continue
                seen.add(exported_name)
                exports.append(exported_name)

    module_info = ModuleInfo(
        name=module_name,
        path=path,
        docstring=docstring,
        exports=exports,
        definitions=definitions,
        imports=imports,
        star_imports=star_imports,
    )
    MODULE_CACHE[module_name] = module_info
    return module_info


def _exported_names(module_name: str) -> list[str]:
    return _parse_module(module_name).exports


def _resolve_object(module_name: str, object_name: str, visited: set[tuple[str, str]] | None = None) -> ObjectDoc | None:
    if visited is None:
        visited = set()
    key = (module_name, object_name)
    if key in visited:
        return None
    visited.add(key)

    if not module_name.startswith("xtalk"):
        return ObjectDoc(
            name=object_name,
            kind="external",
            signature=f"from {module_name} import {object_name}" if object_name else module_name,
            source_module=module_name,
            docstring="External dependency re-exported by this module.",
        )

    module_info = _parse_module(module_name)
    if object_name in module_info.definitions:
        return module_info.definitions[object_name]

    binding = module_info.imports.get(object_name)
    if binding is not None:
        target_module, target_name = binding
        if target_name:
            return _resolve_object(target_module, target_name, visited)
        return ObjectDoc(
            name=object_name,
            kind="module",
            signature=target_module,
            source_module=target_module,
        )

    for imported_module in module_info.star_imports:
        if object_name in _exported_names(imported_module):
            return _resolve_object(imported_module, object_name, visited)

    return None


def _sample_app_modules() -> list[str]:
    modules: set[str] = set()
    for source_path in sorted(SAMPLE_APP_ROOT.glob("*.py")):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "xtalk" or alias.name.startswith("xtalk."):
                        modules.add(alias.name)
            elif (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module
                and (node.module == "xtalk" or node.module.startswith("xtalk."))
            ):
                modules.add(node.module)
    return sorted(modules)


def _model_interface_modules() -> list[str]:
    modules: list[str] = []
    models_root = PACKAGE_ROOT / "models"
    for interface_path in sorted(models_root.glob("*/interfaces.py")):
        relative = interface_path.relative_to(SRC_ROOT).with_suffix("")
        modules.append(".".join(relative.parts))
    return modules


def _server_api_modules() -> list[str]:
    return sorted(
        set(SERVER_API_MODULES)
        | set(_model_interface_modules())
        | set(_sample_app_modules())
    )


def _server_api_doc_entries() -> list[tuple[str, Path]]:
    return [
        (module_name, _doc_path_for_module(module_name))
        for module_name in _server_api_modules()
    ]


def _strip_blank_edges(lines: list[str]) -> list[str]:
    start = 0
    end = len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return [line.rstrip() for line in lines[start:end]]


def _is_numpy_section_heading(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    title = lines[index].strip()
    underline = lines[index + 1].strip()
    return title in NUMPY_SECTION_NAMES and bool(NUMPY_SECTION_UNDERLINE_RE.fullmatch(underline))


def _split_numpy_sections(docstring: str) -> list[tuple[str | None, list[str]]] | None:
    lines = docstring.strip().splitlines()
    sections: list[tuple[str | None, list[str]]] = []
    buffer: list[str] = []
    found_section = False
    index = 0

    while index < len(lines):
        if _is_numpy_section_heading(lines, index):
            found_section = True
            if buffer:
                sections.append((None, _strip_blank_edges(buffer)))
                buffer = []

            title = lines[index].strip()
            index += 2
            section_lines: list[str] = []
            while index < len(lines) and not _is_numpy_section_heading(lines, index):
                section_lines.append(lines[index])
                index += 1
            sections.append((title, _strip_blank_edges(section_lines)))
            continue

        buffer.append(lines[index])
        index += 1

    if buffer:
        sections.append((None, _strip_blank_edges(buffer)))

    return sections if found_section else None


def _normalize_block(lines: list[str]) -> list[str]:
    if not lines:
        return []
    return _strip_blank_edges(textwrap.dedent("\n".join(lines)).splitlines())


def _indent_block(lines: list[str], prefix: str = "  ") -> list[str]:
    return [f"{prefix}{line}" if line else "" for line in lines]


def _parse_numpy_field_entries(lines: list[str]) -> list[tuple[str, list[str]]]:
    entries: list[tuple[str, list[str]]] = []
    index = 0

    while index < len(lines):
        while index < len(lines) and not lines[index].strip():
            index += 1
        if index >= len(lines):
            break

        if lines[index].startswith((" ", "\t")):
            if entries:
                header, description = entries[-1]
                description.append(lines[index])
                entries[-1] = (header, description)
            index += 1
            continue

        header = lines[index].strip()
        index += 1
        description: list[str] = []
        while index < len(lines):
            line = lines[index]
            if line.startswith((" ", "\t")) or not line.strip():
                description.append(line)
                index += 1
                continue
            break
        entries.append((header, description))

    return entries


def _split_numpy_field_header(header: str) -> tuple[str, str | None]:
    match = NUMPY_FIELD_RE.match(header)
    if match is None:
        return header, None
    return match.group("name").strip(), match.group("annotation").strip()


def _render_numpy_field_section(lines: list[str]) -> list[str]:
    rendered: list[str] = []
    for header, description in _parse_numpy_field_entries(lines):
        name, annotation = _split_numpy_field_header(header)
        bullet = f"- `{name}`"
        if annotation:
            bullet += f" (`{annotation}`)"
        rendered.append(bullet)

        normalized_description = _normalize_block(description)
        if normalized_description:
            rendered.extend(_indent_block(normalized_description))
    return rendered


def _render_numpy_list_section(lines: list[str]) -> list[str]:
    rendered: list[str] = []
    items = _parse_numpy_field_entries(lines)
    for header, description in items:
        rendered.append(f"- {header}")
        normalized_description = _normalize_block(description)
        if normalized_description:
            rendered.extend(_indent_block(normalized_description))
    return rendered


def _split_paragraphs(lines: list[str]) -> list[list[str]]:
    paragraphs: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.strip():
            current.append(line.rstrip())
            continue
        if current:
            paragraphs.append(current)
            current = []
    if current:
        paragraphs.append(current)
    return paragraphs


def _render_numpy_examples(lines: list[str]) -> list[str]:
    rendered: list[str] = []
    for paragraph in _split_paragraphs(lines):
        normalized = _normalize_block(paragraph)
        if not normalized:
            continue

        first_line = normalized[0].lstrip()
        if first_line.startswith((">>>", "...")):
            rendered.append("```pycon")
            rendered.extend(normalized)
            rendered.append("```")
        else:
            rendered.extend(normalized)
        rendered.append("")

    return _strip_blank_edges(rendered)


def _localized_numpy_section_title(title: str, locale: str) -> str:
    if locale == "zh":
        return NUMPY_SECTION_TRANSLATIONS_ZH.get(title, title)
    return title


def _render_numpy_sections(
    sections: list[tuple[str | None, list[str]]],
    heading_level: int,
    locale: str,
) -> list[str]:
    rendered: list[str] = []
    for title, lines in sections:
        if not lines:
            continue

        if title is None:
            rendered.extend(lines)
            rendered.append("")
            continue

        rendered.append(
            f"{'#' * heading_level} {_localized_numpy_section_title(title, locale)}"
        )
        rendered.append("")
        if title in NUMPY_FIELD_SECTIONS:
            rendered.extend(_render_numpy_field_section(lines))
        elif title in NUMPY_LIST_SECTIONS:
            rendered.extend(_render_numpy_list_section(lines))
        elif title in NUMPY_EXAMPLE_SECTIONS:
            rendered.extend(_render_numpy_examples(lines))
        else:
            rendered.extend(lines)
        rendered.append("")

    return _strip_blank_edges(rendered)


def _render_docstring(
    docstring: str,
    heading_level: int = 3,
    locale: str = "en",
) -> list[str]:
    if not docstring:
        return []

    numpy_sections = _split_numpy_sections(docstring)
    if numpy_sections is not None:
        return _render_numpy_sections(numpy_sections, heading_level, locale)

    return docstring.strip().splitlines()


def _object_field_title(object_doc: ObjectDoc, locale: str) -> str:
    if locale == "zh":
        if object_doc.kind == "class":
            return "类字段"
        return "字段"
    return f"{object_doc.kind.capitalize()} Fields"


def _object_children_title(object_doc: ObjectDoc, locale: str) -> str:
    if locale == "zh":
        return "方法" if object_doc.kind == "class" else "成员"
    return "Methods" if object_doc.kind == "class" else "Members"


def _render_object(
    object_doc: ObjectDoc,
    current_module: str,
    level: int = 2,
    locale: str = "en",
) -> list[str]:
    heading = "#" * level
    lines = [f"{heading} {object_doc.name}", ""]

    if object_doc.source_module and object_doc.source_module != current_module:
        source_url = _github_url_for_module(object_doc.source_module)
        source_ref = (
            f"[`{object_doc.source_module}`]({source_url})"
            if source_url
            else f"`{object_doc.source_module}`"
        )
        if locale == "zh":
            lines.append(f"_定义于 {source_ref}。_")
        else:
            lines.append(f"_Defined in {source_ref}._")
        lines.append("")

    if object_doc.signature:
        lines.append("```python")
        lines.append(object_doc.signature)
        lines.append("```")
        lines.append("")

    lines.extend(
        _render_docstring(
            object_doc.docstring,
            heading_level=level + 1,
            locale=locale,
        )
    )
    if object_doc.docstring:
        lines.append("")

    if object_doc.value:
        value_label = "值" if locale == "zh" else "Value"
        lines.append(f"**{value_label}:** `{object_doc.value}`")
        lines.append("")

    if object_doc.fields:
        lines.append(f"### {_object_field_title(object_doc, locale)}")
        lines.append("")
        for field_doc in object_doc.fields:
            field_line = f"- `{field_doc.signature}`"
            if field_doc.value:
                field_line += f" = `{field_doc.value}`"
            lines.append(field_line)
        lines.append("")

    if object_doc.children:
        lines.append(f"### {_object_children_title(object_doc, locale)}")
        lines.append("")
        for child in object_doc.children:
            lines.extend(_render_object(child, current_module, level + 2, locale))

    return lines


def _render_module_page(module_name: str, locale: str = "en") -> str:
    module_info = _parse_module(module_name)
    lines = [
        (
            "<!-- 此文件由 generate_server_docs.py 自动生成。 -->"
            if locale == "zh"
            else "<!-- This file is auto-generated by generate_server_docs.py. -->"
        ),
        f"# {module_name}",
        "",
    ]

    lines.extend(_render_docstring(module_info.docstring, heading_level=2, locale=locale))
    if module_info.docstring:
        lines.append("")

    exported_objects = [
        _resolve_object(module_name, exported_name)
        for exported_name in module_info.exports
        if exported_name != "__all__"
    ]

    for object_doc in exported_objects:
        if object_doc is None:
            continue
        lines.extend(_render_object(object_doc, module_name, locale=locale))

    return "\n".join(lines).rstrip() + "\n"


def _render_index(locale: str = "en") -> str:
    doc_entries = _server_api_doc_entries()
    lines = [
        (
            "<!-- 此文件由 generate_server_docs.py 自动生成。 -->"
            if locale == "zh"
            else "<!-- This file is auto-generated by generate_server_docs.py. -->"
        ),
        "# 服务端 API" if locale == "zh" else "# Server API",
        "",
    ]
    lines.extend(
        f"- [`{module_name}`]({_localized_doc_path(doc_path, locale).as_posix()})"
        for module_name, doc_path in doc_entries
    )
    lines.append("")
    return "\n".join(lines)


def _write_if_changed(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def _prune_generated_server_docs() -> None:
    for path in SERVER_DOCS_ROOT.rglob("*.md"):
        path.unlink()

    for directory in sorted(
        [path for path in SERVER_DOCS_ROOT.rglob("*") if path.is_dir()],
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        if directory == SERVER_DOCS_ROOT:
            continue
        try:
            directory.rmdir()
        except OSError:
            pass


def _write_module_docs() -> None:
    for module_name, doc_path in _server_api_doc_entries():
        for locale in LOCALES:
            _write_if_changed(
                SERVER_DOCS_ROOT / _localized_doc_path(doc_path, locale),
                _render_module_page(module_name, locale=locale),
            )


def generate_server_docs() -> None:
    _prune_generated_server_docs()
    _write_module_docs()
    for locale in LOCALES:
        index_path = _localized_doc_path(Path("index.md"), locale)
        _write_if_changed(SERVER_DOCS_ROOT / index_path, _render_index(locale=locale))


if __name__ == "__main__":
    generate_server_docs()
