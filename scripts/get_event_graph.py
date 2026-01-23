# -*- coding: utf-8 -*-
"""
Build Manager <-> Event graph by static AST scanning.

Usage:
  python tools/build_event_graph.py

If graphviz 'dot' is available, PNG will be rendered automatically when --png is provided.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ----------------------------
# Helpers
# ----------------------------


def qname(node: ast.AST) -> Optional[str]:
    """Best-effort qualified name for Name/Attribute chains: a.b.c"""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = qname(node.value)
        if not base:
            return node.attr
        return f"{base}.{node.attr}"
    return None


def simple_name(node: ast.AST) -> Optional[str]:
    """Return last segment of qname() (e.g., a.b.C -> C)."""
    n = qname(node)
    if not n:
        return None
    return n.split(".")[-1]


def is_manager_event_handler_decorator(dec: ast.AST) -> bool:
    """
    Match: @Manager.event_handler(...)
    """
    if not isinstance(dec, ast.Call):
        return False
    fn = dec.func
    if not isinstance(fn, ast.Attribute):
        return False
    if fn.attr != "event_handler":
        return False
    base = qname(fn.value)
    return base == "Manager"


def extract_event_from_event_handler(dec: ast.Call) -> Optional[str]:
    """
    @Manager.event_handler(SomeEvent, ...)
    """
    if not dec.args:
        return None
    return simple_name(dec.args[0])


def is_self_event_bus_publish(call: ast.Call) -> bool:
    """
    Match: self.event_bus.publish(...)
    """
    fn = call.func
    if not isinstance(fn, ast.Attribute):
        return False
    if fn.attr != "publish":
        return False
    base = qname(fn.value)
    return base == "self.event_bus"


def extract_event_from_publish_arg(
    arg0: ast.AST, local_ctor_map: Dict[str, str]
) -> Optional[str]:
    """
    publish(SomeEvent(...)) -> SomeEvent
    publish(evt) where evt = SomeEvent(...) earlier -> SomeEvent
    """
    # Direct constructor call: SomeEvent(...)
    if isinstance(arg0, ast.Call):
        return simple_name(arg0.func)

    # publish(var)
    if isinstance(arg0, ast.Name):
        return local_ctor_map.get(arg0.id)

    return None


# ----------------------------
# Data Model
# ----------------------------


@dataclass
class ManagerInfo:
    manager_name: str
    file_path: str
    handles: Set[str] = field(default_factory=set)  # Events received
    publishes: Set[str] = field(default_factory=set)  # Events published


# ----------------------------
# AST Visitor
# ----------------------------


class FunctionPublishScanner(ast.NodeVisitor):
    """
    Scan a function body to infer what event types are published.

    Heuristic:
      - Track simple assignments: x = SomeEvent(...)
      - Detect self.event_bus.publish(arg0) and map arg0 -> event type
    """

    def __init__(self) -> None:
        self.local_ctor_map: Dict[str, str] = {}  # var -> EventClass
        self.published: Set[str] = set()

    def visit_Assign(self, node: ast.Assign) -> None:
        # x = SomeEvent(...)
        if isinstance(node.value, ast.Call):
            evt = simple_name(node.value.func)
            if evt:
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        self.local_ctor_map[t.id] = evt
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # x: Type = SomeEvent(...)
        if node.value and isinstance(node.value, ast.Call):
            evt = simple_name(node.value.func)
            if evt and isinstance(node.target, ast.Name):
                self.local_ctor_map[node.target.id] = evt
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # self.event_bus.publish(...)
        if is_self_event_bus_publish(node) and node.args:
            evt = extract_event_from_publish_arg(node.args[0], self.local_ctor_map)
            if evt:
                self.published.add(evt)
        self.generic_visit(node)


class ModuleScanner(ast.NodeVisitor):
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.managers: List[ManagerInfo] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Identify Manager subclasses (best-effort): class X(Manager) or class X(..., Manager, ...)
        is_manager_subclass = any(simple_name(b) == "Manager" for b in node.bases)
        if not is_manager_subclass:
            self.generic_visit(node)
            return

        info = ManagerInfo(
            manager_name=node.name,
            file_path=self.file_path,
        )

        # 1) event handlers from decorators
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in item.decorator_list:
                    if is_manager_event_handler_decorator(dec):
                        evt = extract_event_from_event_handler(dec)  # type: ignore[arg-type]
                        if evt:
                            info.handles.add(evt)

        # 2) published events from functions
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fps = FunctionPublishScanner()
                fps.visit(item)
                info.publishes |= fps.published

        self.managers.append(info)
        self.generic_visit(node)


# ----------------------------
# Graph building
# ----------------------------


def scan_python_file(path: Path) -> List[ManagerInfo]:
    try:
        src = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        src = path.read_text(encoding="utf-8", errors="ignore")

    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return []

    scanner = ModuleScanner(str(path))
    scanner.visit(tree)
    return scanner.managers


def scan_root(root: Path) -> List[ManagerInfo]:
    managers: List[ManagerInfo] = []
    for p in root.rglob("*.py"):
        # skip __pycache__ etc
        if any(part.startswith(".") for part in p.parts):
            continue
        if "__pycache__" in p.parts:
            continue
        managers.extend(scan_python_file(p))
    return managers


def build_edges(managers: List[ManagerInfo]) -> List[Tuple[str, str, str]]:
    """
    Return edges: (producer_manager, consumer_manager, event_label)
    Based on shared event types: producer publishes E, consumer handles E.
    """
    producers: Dict[str, Set[str]] = {}
    consumers: Dict[str, Set[str]] = {}

    for m in managers:
        for e in m.publishes:
            producers.setdefault(e, set()).add(m.manager_name)
        for e in m.handles:
            consumers.setdefault(e, set()).add(m.manager_name)

    edges: List[Tuple[str, str, str]] = []
    for e, ps in producers.items():
        cs = consumers.get(e, set())
        for p in ps:
            for c in cs:
                edges.append((p, c, e))
    return edges


def to_dot(managers: List[ManagerInfo], edges: List[Tuple[str, str, str]]) -> str:
    """
    DOT graph:
      - Managers are nodes
      - Edges labeled by Event
      - Also include per-manager tooltip with file path
    """
    # Node ids must be stable and safe; use manager names directly (quoted)
    lines: List[str] = []
    lines.append("digraph EventGraph {")
    lines.append("  rankdir=LR;")
    lines.append('  node [shape=box, style="rounded,filled", fillcolor="#f7f7f7"];')
    lines.append("  edge [fontsize=10];")
    lines.append("")

    # Nodes
    for m in sorted(managers, key=lambda x: x.manager_name):
        label = m.manager_name
        tooltip = m.file_path.replace("\\", "/")
        lines.append(f'  "{m.manager_name}" [label="{label}", tooltip="{tooltip}"];')

    lines.append("")

    # Edges: group same (p,c) with multiple events into one label for readability
    grouped: Dict[Tuple[str, str], List[str]] = {}
    for p, c, e in edges:
        grouped.setdefault((p, c), []).append(e)

    for (p, c), evts in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        evts_sorted = sorted(set(evts))
        # Keep label not too long; if too many, truncate.
        if len(evts_sorted) > 6:
            label = "\\n".join(evts_sorted[:6]) + f"\\n...(+{len(evts_sorted)-6})"
        else:
            label = "\\n".join(evts_sorted)
        lines.append(f'  "{p}" -> "{c}" [label="{label}"];')

    lines.append("}")
    return "\n".join(lines)


# ----------------------------
# CLI
# ----------------------------


def main():

    root = Path("src/xtalk/serving/modules").resolve()
    out = Path("logs/event_graph.dot").resolve()

    managers = scan_root(root)
    edges = build_edges(managers)
    dot = to_dot(managers, edges)
    out.write_text(dot, encoding="utf-8")

    print(f"[ok] scanned managers: {len(managers)}")
    print(f"[ok] inferred edges:    {len(edges)}")
    print(f"[ok] wrote dot:         {out}")


if __name__ == "__main__":
    main()
