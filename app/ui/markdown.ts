/**
 * Small dependency-free Markdown renderer for the whiteboard window.
 *
 * The renderer escapes every raw character first and only reinserts
 * pre-escaped placeholders, so untrusted whiteboard text can never inject
 * HTML. Links are restricted to http(s) targets.
 */

const PLACEHOLDER_PREFIX = "\u0000";
const PLACEHOLDER_SUFFIX = "\u0000";

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function placeholderToken(index: number): string {
  return `${PLACEHOLDER_PREFIX}${index}${PLACEHOLDER_SUFFIX}`;
}

function renderInline(value: string): string {
  const codeSpans: string[] = [];
  let text = value.replace(/`([^`\n]+)`/g, (_match, code: string) => {
    const index = codeSpans.length;
    codeSpans.push(escapeHtml(code));
    return placeholderToken(index);
  });
  text = text.replace(
    /!?\[([^\]]*)\]\((https?:\/\/[^)\s]+)\)/g,
    (_match, label: string, url: string) => {
      const safeUrl = escapeHtml(url);
      const linkLabel = label.length > 0 ? label : url;
      return `<a href="${safeUrl}" target="_blank" rel="noopener noreferrer">${escapeHtml(linkLabel)}</a>`;
    },
  );
  text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  text = text.replace(/(?<!\*)\*([^*\n]+)\*(?!\*)/g, "<em>$1</em>");
  text = text.replace(/~~([^~]+)~~/g, "<del>$1</del>");
  return text.replace(
    placeholderTokenPattern(codeSpans),
    (_match, index: string) => `<code>${codeSpans[Number(index)]}</code>`,
  );
}

function placeholderTokenPattern(codeSpans: unknown[]): RegExp {
  if (codeSpans.length === 0) {
    return /(?!)/;
  }
  return new RegExp(
    `${PLACEHOLDER_PREFIX}(\\d+)${PLACEHOLDER_SUFFIX}`,
    "g",
  );
}

function renderList(lines: string[], ordered: boolean): string {
  const tag = ordered ? "ol" : "ul";
  const items = lines
    .map((line) => {
      const content = ordered
        ? line.replace(/^\s*\d+\.\s+/, "")
        : line.replace(/^\s*[-*+]\s+/, "");
      return `<li>${renderInline(content)}</li>`;
    })
    .join("");
  return `<${tag}>${items}</${tag}>`;
}

function renderQuote(lines: string[]): string {
  const content = lines
    .map((line) => line.replace(/^\s*>\s?/, ""))
    .join("\n");
  return `<blockquote>${renderInline(content)}</blockquote>`;
}

function renderTable(rows: string[][]): string {
  const [headerRow, ...bodyRows] = rows;
  if (headerRow === undefined || bodyRows.length === 0) {
    return "";
  }
  const renderCells = (cells: string[], cellTag: string): string =>
    cells.map((cell) => `<${cellTag}>${renderInline(cell.trim())}</${cellTag}>`).join("");
  const header = `<thead><tr>${renderCells(headerRow, "th")}</tr></thead>`;
  const body = `<tbody>${bodyRows
    .map((cells) => `<tr>${renderCells(cells, "td")}</tr>`)
    .join("")}</tbody>`;
  return `<table>${header}${body}</table>`;
}

function isTableRow(line: string): boolean {
  return line.includes("|");
}

function isTableDelimiter(line: string): boolean {
  return /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/.test(line);
}

function splitTableRow(line: string): string[] {
  return line
    .split("|")
    .slice(1, -1);
}

/**
 * Renders one Markdown document into safe HTML fragments.
 *
 * @param markdown Untrusted Markdown text from the whiteboard.
 * @returns Escaped HTML safe for innerHTML assignment.
 */
export function renderMarkdown(markdown: string): string {
  const fencedBlocks: string[] = [];
  let text = markdown.replace(/\r\n?/g, "\n");
  text = text.replace(/```[^\n]*\n?([\s\S]*?)(?:```|$)/g, (_match, code: string) => {
    const index = fencedBlocks.length;
    fencedBlocks.push(escapeHtml(code.replace(/\n$/, "")));
    return `${placeholderToken(index)}\n`;
  });

  const lines = text.split("\n");
  const blocks: string[] = [];
  let paragraph: string[] = [];
  let index = 0;

  const flushParagraph = (): void => {
    if (paragraph.length === 0) {
      return;
    }
    blocks.push(`<p>${renderInline(paragraph.join("\n"))}</p>`);
    paragraph = [];
  };

  while (index < lines.length) {
    const line = lines[index] ?? "";
    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading !== null) {
      flushParagraph();
      const level = heading[1]?.length ?? 1;
      blocks.push(`<h${level}>${renderInline(heading[2] ?? "")}</h${level}>`);
      index += 1;
      continue;
    }
    if (/^\s*(?:---|\*\*\*|___)\s*$/.test(line)) {
      flushParagraph();
      blocks.push("<hr />");
      index += 1;
      continue;
    }
    if (/^\s*>\s?/.test(line)) {
      flushParagraph();
      const quoteLines: string[] = [];
      while (index < lines.length && /^\s*>\s?/.test(lines[index] ?? "")) {
        quoteLines.push(lines[index] ?? "");
        index += 1;
      }
      blocks.push(renderQuote(quoteLines));
      continue;
    }
    if (/^\s*[-*+]\s+/.test(line)) {
      flushParagraph();
      const listLines: string[] = [];
      while (index < lines.length && /^\s*[-*+]\s+/.test(lines[index] ?? "")) {
        listLines.push(lines[index] ?? "");
        index += 1;
      }
      blocks.push(renderList(listLines, false));
      continue;
    }
    if (/^\s*\d+\.\s+/.test(line)) {
      flushParagraph();
      const listLines: string[] = [];
      while (index < lines.length && /^\s*\d+\.\s+/.test(lines[index] ?? "")) {
        listLines.push(lines[index] ?? "");
        index += 1;
      }
      blocks.push(renderList(listLines, true));
      continue;
    }
    if (
      isTableRow(line) &&
      isTableDelimiter(lines[index + 1] ?? "")
    ) {
      flushParagraph();
      const rows: string[][] = [splitTableRow(line)];
      index += 2;
      while (index < lines.length && isTableRow(lines[index] ?? "")) {
        rows.push(splitTableRow(lines[index] ?? ""));
        index += 1;
      }
      blocks.push(renderTable(rows));
      continue;
    }
    if (line.trim().length === 0) {
      flushParagraph();
      index += 1;
      continue;
    }
    paragraph.push(line);
    index += 1;
  }
  flushParagraph();

  return blocks
    .map((block) =>
      block.replace(
        placeholderTokenPattern(fencedBlocks),
        (_match, blockIndex: string) =>
          `<pre><code>${fencedBlocks[Number(blockIndex)] ?? ""}</code></pre>`,
      ),
    )
    .join("\n");
}
