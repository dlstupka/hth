"""Shared navigation treatment for long GitHub-flavored Markdown reports."""
from __future__ import annotations

import re


def _slugify_heading(text: str) -> str:
    value = re.sub(r"[`*_]", "", text).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value or "section"


def _navigation_heading(line: str) -> tuple[int, str] | None:
    """Return the visible report level and title for Markdown and details headings."""
    markdown = re.match(r"^(##|###) (.+)$", line)
    if markdown:
        return len(markdown.group(1)), markdown.group(2).strip()

    details_heading = re.fullmatch(
        r"<summary><h([23])>(.+)</h\1></summary>", line.strip()
    )
    if details_heading:
        return int(details_heading.group(1)), details_heading.group(2).strip()

    nested_heading = re.fullmatch(
        r"<summary><strong>(.+)</strong></summary>", line.strip()
    )
    if nested_heading:
        return 4, nested_heading.group(1).strip()

    return None


def add_report_navigation(lines: list[str]) -> list[str]:
    """Add stable navigation and return links to a Markdown report.

    GitHub does not expose headings embedded in ``<summary>`` elements to its
    normal Markdown table of contents. Explicit anchors keep collapsible report
    sections navigable while preserving ``<summary>`` as the first child of its
    corresponding ``<details>`` element.
    """
    headings: list[tuple[int, str, str]] = []
    used: dict[str, int] = {}
    for line in lines:
        heading = _navigation_heading(line)
        if heading is None:
            continue
        level, title = heading
        base = _slugify_heading(title)
        used[base] = used.get(base, 0) + 1
        slug = base if used[base] == 1 else f"{base}-{used[base]}"
        headings.append((level, title, slug))

    if not headings:
        return lines

    navigation = [
        '<a id="table-of-contents"></a>',
        "",
        "<details open>",
        "<summary><strong>Navigation</strong></summary>",
        "",
    ]
    for level, title, slug in headings:
        indent = "  " * max(0, level - 2)
        navigation.append(f"{indent}- [{title}](#{slug})")
    navigation.extend(["", "</details>", ""])

    result: list[str] = []
    inserted_navigation = False
    heading_index = 0
    for line in lines:
        if not inserted_navigation and line.startswith("# "):
            result.append(line)
            result.extend(["", *navigation])
            inserted_navigation = True
            continue

        heading = _navigation_heading(line)
        if heading is not None:
            _, _, slug = headings[heading_index]
            details_open = None
            trailing_blank = False
            if line.lstrip().startswith("<summary>"):
                if result and result[-1] == "":
                    result.pop()
                    trailing_blank = True
                if result and result[-1].strip().startswith("<details"):
                    details_open = result.pop()

            if result and result[-1] != "":
                result.append("")
            if heading_index > 0:
                result.extend(["[↑ Back to Navigation](#table-of-contents)", ""])
            result.append(f'<a id="{slug}"></a>')
            if details_open is not None:
                result.append(details_open)
            result.append(line)
            if trailing_blank:
                result.append("")
            heading_index += 1
        else:
            result.append(line)

    if heading_index:
        result.extend(["", "[↑ Back to Navigation](#table-of-contents)"])
    return result
