from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import calendar
import os
import re
from datetime import datetime, timedelta

from langchain.tools import tool, BaseTool
from langchain_chroma import Chroma  # type: ignore


def _resolve_env(*names: str) -> Optional[str]:
    """Return the first non-empty environment variable value among the names."""
    for n in names:
        v = os.getenv(n)
        if v and v.strip():
            return v.strip()
    return None


# ---------------------------------------------------------------------------
# Relative-time resolution for search queries
# ---------------------------------------------------------------------------

_HAS_RELATIVE_TIME = re.compile(
    r"今[天日]|昨[天日]|前天|后天|後天|大前天|大后天|大後天"
    r"|明[天日]|今年|去年|前年|明年|后年|後年|大前年"
    r"|[上下]个?月|本月|[上下]个?周|本周|[上下]个?星期|[上下]个?礼拜|[上下]个?禮拜"
    r"|上上个?月|下下个?月|上上个?周|下下个?周"
    r"|半年前|半年后|半年後|年初|年[底末]|月初|月[底末]"
    r"|\d+\s*[天周年]前|\d+\s*[天周年][后後]"
    r"|\d+\s*个?月前|\d+\s*个?月[后後]"
    r"|\d+\s*个?星期前|\d+\s*个?礼拜前"
    r"|today|yesterday|tomorrow|last\s+week|this\s+week|next\s+week"
    r"|last\s+month|this\s+month|next\s+month"
    r"|last\s+year|this\s+year|next\s+year"
    r"|\d+\s*days?\s*ago|\d+\s*weeks?\s*ago"
    r"|\d+\s*months?\s*ago|\d+\s*years?\s*ago"
    r"|day\s+before\s+yesterday|day\s+after\s+tomorrow",
    re.IGNORECASE,
)


def _add_months(dt: datetime, months: int) -> datetime:
    """Month-level date arithmetic without external dependencies."""
    month = dt.month - 1 + months
    year = dt.year + month // 12
    month = month % 12 + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


def _resolve_relative_time(query: str) -> str:
    """Replace relative time expressions with absolute dates in search queries.

    Covers Chinese and English daily expressions: 今天/昨天/前天/明天/后天,
    今年/去年/前年/明年, 本月/上个月/下个月, 本周/上周/下周, 上周三/下周一,
    N天前/后, N个月前/后, N年前/后, 半年前/后, today/yesterday/tomorrow,
    last/this/next week/month/year, N days/weeks/months/years ago, etc.
    """
    if not _HAS_RELATIVE_TIME.search(query):
        return query

    now = datetime.now()

    def _cn_date(dt: datetime) -> str:
        return dt.strftime("%Y年%m月%d日")

    def _cn_month(dt: datetime) -> str:
        return dt.strftime("%Y年%m月")

    # Chinese weekday mapping: 一=Monday(0) .. 日/天=Sunday(6)
    _CN_WD = {
        "一": 0, "二": 1, "三": 2, "四": 3,
        "五": 4, "六": 5, "日": 6, "天": 6,
    }

    def _last_wd(name: str) -> str:
        wd = _CN_WD.get(name, 0)
        delta = (now.weekday() - wd) % 7 or 7
        return _cn_date(now - timedelta(days=delta))

    def _this_wd(name: str) -> str:
        wd = _CN_WD.get(name, 0)
        delta = (wd - now.weekday()) % 7
        return _cn_date(now + timedelta(days=delta))

    def _next_wd(name: str) -> str:
        wd = _CN_WD.get(name, 0)
        delta = (wd - now.weekday()) % 7
        if delta <= 0:
            delta += 7
        return _cn_date(now + timedelta(days=delta + 7))

    # English weekday mapping
    _EN_WD = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
        "mon": 0, "tue": 1, "wed": 2, "thu": 3,
        "fri": 4, "sat": 5, "sun": 6,
    }

    def _en_last_wd(name: str) -> str:
        wd = _EN_WD.get(name.lower(), 0)
        delta = (now.weekday() - wd) % 7 or 7
        return (now - timedelta(days=delta)).strftime("%Y-%m-%d")

    def _en_this_wd(name: str) -> str:
        wd = _EN_WD.get(name.lower(), 0)
        delta = (wd - now.weekday()) % 7
        return (now + timedelta(days=delta)).strftime("%Y-%m-%d")

    def _en_next_wd(name: str) -> str:
        wd = _EN_WD.get(name.lower(), 0)
        delta = (wd - now.weekday()) % 7
        if delta <= 0:
            delta += 7
        return (now + timedelta(days=delta + 7)).strftime("%Y-%m-%d")

    # Rules: (regex_pattern, replacement_callback)
    # Ordered by specificity — longer / more specific patterns first.
    # Negative lookbehinds prevent false positives like 目前天气 → 目(前天)气.
    rules: List[Tuple[str, Any]] = [
        # ---- Chinese: N + unit + 前/后 ----
        (r"(\d+)\s*年半前", lambda m: f"{now.year - int(m.group(1))}年"),
        (r"(\d+)\s*年半[后後]", lambda m: f"{now.year + int(m.group(1)) + 1}年"),
        (r"(\d+)\s*年前", lambda m: f"{now.year - int(m.group(1))}年"),
        (r"(\d+)\s*年[后後]", lambda m: f"{now.year + int(m.group(1))}年"),
        (r"半年前", lambda m: _cn_month(_add_months(now, -6))),
        (r"半年[后後]", lambda m: _cn_month(_add_months(now, 6))),
        (r"(\d+)\s*个?半?月前",
         lambda m: _cn_month(_add_months(now, -int(m.group(1))))),
        (r"(\d+)\s*个?半?月[后後]",
         lambda m: _cn_month(_add_months(now, int(m.group(1))))),
        (r"(\d+)\s*(?:周|个?星期|个?礼拜|個?禮拜)前",
         lambda m: _cn_date(now - timedelta(weeks=int(m.group(1))))),
        (r"(\d+)\s*(?:周|个?星期|个?礼拜|個?禮拜)[后後]",
         lambda m: _cn_date(now + timedelta(weeks=int(m.group(1))))),
        (r"(\d+)\s*天前",
         lambda m: _cn_date(now - timedelta(days=int(m.group(1))))),
        (r"(\d+)\s*天[后後]",
         lambda m: _cn_date(now + timedelta(days=int(m.group(1))))),

        # ---- Chinese: weekday with 上/这/下 prefix ----
        (r"上(?:个?)?(?:周|星期|礼拜|禮拜)([一二三四五六日天])",
         lambda m: _last_wd(m.group(1))),
        (r"(?:这个?|本)(?:周|星期|礼拜|禮拜)([一二三四五六日天])",
         lambda m: _this_wd(m.group(1))),
        (r"下(?:个?)?(?:周|星期|礼拜|禮拜)([一二三四五六日天])",
         lambda m: _next_wd(m.group(1))),

        # ---- Chinese: fixed day expressions ----
        (r"大前天", lambda m: _cn_date(now - timedelta(days=3))),
        (r"大[后後]天", lambda m: _cn_date(now + timedelta(days=3))),
        (r"(?<!目)(?<!以)(?<!之)(?<!日)前天",
         lambda m: _cn_date(now - timedelta(days=2))),
        (r"(?<!以)(?<!往)(?<!日)[后後]天",
         lambda m: _cn_date(now + timedelta(days=2))),
        (r"昨[天日]", lambda m: _cn_date(now - timedelta(days=1))),
        (r"(?<!说)(?<!证)明[天日]",
         lambda m: _cn_date(now + timedelta(days=1))),
        (r"(?<!如)(?<!至)今[天日]", lambda m: _cn_date(now)),

        # ---- Chinese: fixed year expressions ----
        (r"大前年", lambda m: f"{now.year - 3}年"),
        (r"(?<!目)前年", lambda m: f"{now.year - 2}年"),
        (r"去年", lambda m: f"{now.year - 1}年"),
        (r"今年", lambda m: f"{now.year}年"),
        (r"明年", lambda m: f"{now.year + 1}年"),

        # ---- Chinese: fixed month expressions ----
        (r"上上个?月", lambda m: _cn_month(_add_months(now, -2))),
        (r"上个?月", lambda m: _cn_month(_add_months(now, -1))),
        (r"(?:这个?|本)月", lambda m: _cn_month(now)),
        (r"下下个?月", lambda m: _cn_month(_add_months(now, 2))),
        (r"下个?月", lambda m: _cn_month(_add_months(now, 1))),

        # ---- Chinese: fixed week expressions ----
        (r"上上(?:个?)?(?:周|星期|礼拜|禮拜)",
         lambda m: _cn_date(now - timedelta(weeks=2))),
        (r"上(?:个?)?(?:周|星期|礼拜|禮拜)",
         lambda m: _cn_date(now - timedelta(weeks=1))),
        (r"(?:这个?|本)(?:周|星期|礼拜)",
         lambda m: _cn_date(now)),
        (r"下下(?:个?)?(?:周|星期|礼拜|禮拜)",
         lambda m: _cn_date(now + timedelta(weeks=2))),
        (r"下(?:个?)?(?:周|星期|礼拜|禮拜)",
         lambda m: _cn_date(now + timedelta(weeks=1))),

        # ---- Chinese: period start/end ----
        (r"年初", lambda m: f"{now.year}年1月"),
        (r"年[底末]", lambda m: f"{now.year}年12月"),
        (r"月初", lambda m: _cn_month(now) + "初"),
        (r"月[底末]", lambda m: _cn_month(now) + "底"),

        # ---- English: N + unit + ago ----
        (r"(\d+)\s*years?\s*ago",
         lambda m: str(now.year - int(m.group(1)))),
        (r"(\d+)\s*months?\s*ago",
         lambda m: _add_months(now, -int(m.group(1))).strftime("%Y-%m")),
        (r"(\d+)\s*weeks?\s*ago",
         lambda m: (now - timedelta(weeks=int(m.group(1)))).strftime("%Y-%m-%d")),
        (r"(\d+)\s*days?\s*ago",
         lambda m: (now - timedelta(days=int(m.group(1)))).strftime("%Y-%m-%d")),

        # ---- English: fixed expressions ----
        (r"\bthe\s+day\s+before\s+yesterday\b",
         lambda m: (now - timedelta(days=2)).strftime("%Y-%m-%d")),
        (r"\bthe\s+day\s+after\s+tomorrow\b",
         lambda m: (now + timedelta(days=2)).strftime("%Y-%m-%d")),
        (r"\byesterday\b",
         lambda m: (now - timedelta(days=1)).strftime("%Y-%m-%d")),
        (r"\btomorrow\b",
         lambda m: (now + timedelta(days=1)).strftime("%Y-%m-%d")),
        (r"\btoday\b",
         lambda m: now.strftime("%Y-%m-%d")),

        # ---- English: last/this/next + unit ----
        (r"\blast\s+year\b", lambda m: str(now.year - 1)),
        (r"\bthis\s+year\b", lambda m: str(now.year)),
        (r"\bnext\s+year\b", lambda m: str(now.year + 1)),
        (r"\blast\s+month\b",
         lambda m: _add_months(now, -1).strftime("%Y-%m")),
        (r"\bthis\s+month\b", lambda m: now.strftime("%Y-%m")),
        (r"\bnext\s+month\b",
         lambda m: _add_months(now, 1).strftime("%Y-%m")),
        (r"\blast\s+week\b",
         lambda m: (now - timedelta(weeks=1)).strftime("%Y-%m-%d")),
        (r"\bthis\s+week\b", lambda m: now.strftime("%Y-%m-%d")),
        (r"\bnext\s+week\b",
         lambda m: (now + timedelta(weeks=1)).strftime("%Y-%m-%d")),

        # ---- English: last/this/next + weekday ----
        (r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday"
         r"|mon|tue|wed|thu|fri|sat|sun)\b",
         lambda m: _en_last_wd(m.group(1))),
        (r"\bthis\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday"
         r"|mon|tue|wed|thu|fri|sat|sun)\b",
         lambda m: _en_this_wd(m.group(1))),
        (r"\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday"
         r"|mon|tue|wed|thu|fri|sat|sun)\b",
         lambda m: _en_next_wd(m.group(1))),
    ]

    for pattern, fn in rules:
        query = re.sub(pattern, fn, query, flags=re.IGNORECASE)

    return query


WEB_SEARCH_TOOL = "web_search"


def build_web_search_tool() -> BaseTool:
    """Build a Serper-based web search tool with graceful degradation."""

    # JSON schema for tool arguments
    args_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query string."},
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "default": 5,
                "description": "Max number of results (1-10).",
            },
            "region": {
                "type": "string",
                "description": "Region code, e.g. 'us', 'cn', 'gb'. Optional.",
            },
            "lang": {
                "type": "string",
                "description": "Language code, e.g. 'en', 'zh-CN'. Optional.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    resolved_key = _resolve_env("SERPER_API_KEY", "GOOGLE_SERPER_API_KEY") or ""

    search_description = """Search the web for accurate, up-to-date information.
You MUST call this tool when the user's question involves ANY of the following:
- Weather, news, current events, real-time data (stock prices, scores, exchange rates)
- Specific places, buildings, campuses, addresses, floor numbers, opening hours (e.g. '香港科技大学有几家咖啡厅', '南北小厨在几楼', 'where is the library')
- Restaurants, shops, cafes, businesses: location, menu, price, count, hours
- Specific people, organizations, companies, products, events
- Numbers, statistics, rankings, or comparisons requiring accuracy
- ANY factual question where you are not 100% certain of the answer
GOLDEN RULE: When in doubt, ALWAYS search. Never guess specific facts from memory.
LANGUAGE RULE: The search query MUST use the SAME language as the user's question. If the user speaks Chinese, search in Chinese. If the user speaks English, search in English. Do NOT translate the query to a different language."""

    @tool(WEB_SEARCH_TOOL, args_schema=args_schema)
    def web_search(
        query: str,
        max_results: int = 5,
        region: str | None = None,
        lang: str | None = None,
    ) -> str:
        """Search the web for real-time information and return top results."""
        # Lazy import requests so the module loads even if dependency is missing
        try:
            import requests  # type: ignore
        except Exception:
            return "Search service is unavailable: missing requests package."

        if not resolved_key:
            return "Search service is unavailable: missing SERPER_API_KEY."

        try:
            query = _resolve_relative_time(query)
            headers = {
                "X-API-KEY": resolved_key,
                "Content-Type": "application/json",
            }
            payload: Dict[str, Any] = {"q": query}
            if region:
                payload["location"] = region
            if lang:
                payload["gl"] = region or "us"
                payload["hl"] = lang

            resp = requests.post(
                "https://google.serper.dev/search",
                headers=headers,
                json=payload,
                timeout=12,
            )
            data = resp.json() if resp.ok else {}
            items = (data.get("organic") or [])[
                : max(1, min(10, int(max_results or 5)))
            ]
            if not items:
                return "No relevant results found."
            lines = [
                "[The following are factual results from a search engine. "
                "Cite all times, numbers, and names exactly as shown — "
                "do NOT modify or reinterpret them.]"
            ]
            for i, it in enumerate(items, 1):
                title = (it.get("title") or "").strip()
                snippet = (it.get("snippet") or "").strip()
                link = (it.get("link") or "").strip()
                lines.append(f"{i}. {title} — {snippet}\n{link}")
            return "\n".join(lines)
        except Exception as e:  # fail-safe
            return f"Search error: {e}"

    web_search.description = search_description
    return web_search


TIME_TOOL = "get_time"


def build_time_tool() -> BaseTool:
    """Build a current-time tool with optional timezone, format, and date offset."""

    args_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "IANA timezone, e.g. 'UTC', 'America/Los_Angeles'. Optional.",
            },
            "fmt": {
                "type": "string",
                "description": "strftime format, default '%Y-%m-%d %H:%M:%S'.",
                "default": "%Y-%m-%d %H:%M:%S",
            },
            "offset_days": {
                "type": "integer",
                "description": "Offset in days. E.g. -1 = yesterday, 1 = tomorrow, -3 = 3 days ago, 7 = a week later. Default 0.",
                "default": 0,
            },
            "offset_months": {
                "type": "integer",
                "description": "Offset in months. E.g. -1 = last month, 1 = next month, -6 = half a year ago. Default 0.",
                "default": 0,
            },
            "offset_years": {
                "type": "integer",
                "description": "Offset in years. E.g. -1 = last year, 1 = next year. Default 0.",
                "default": 0,
            },
        },
        "required": [],
        "additionalProperties": False,
    }

    time_description = """Get the current date and time, with optional date offset for relative date questions.
You MUST call this tool when the user asks about:
- Current time (e.g. '现在几点', 'what time is it')
- Current date (e.g. '今天几号', 'what is today's date')
- Day of week (e.g. '今天星期几', 'what day is it')
- Relative dates — use offset parameters instead of calculating yourself:
  * '昨天几号' / 'yesterday' → offset_days=-1
  * '前天' / 'day before yesterday' → offset_days=-2
  * '明天星期几' / 'tomorrow' → offset_days=1
  * '后天' / 'day after tomorrow' → offset_days=2
  * '上个月是几月' / 'last month' → offset_months=-1
  * '下个月' / 'next month' → offset_months=1
  * '去年是哪一年' / 'last year' → offset_years=-1
  * '3天前是几号' / '3 days ago' → offset_days=-3
  * '两个月后' / '2 months later' → offset_months=2
  * '上周三' / 'last Wednesday' → calculate the correct offset_days
- Any question that requires knowing a specific datetime
IMPORTANT: Do NOT calculate dates yourself — always use offset parameters and let this tool compute the exact date."""

    @tool(TIME_TOOL, args_schema=args_schema)
    def get_time(
        timezone: str | None = None,
        fmt: str = "%Y-%m-%d %H:%M:%S",
        offset_days: int = 0,
        offset_months: int = 0,
        offset_years: int = 0,
    ) -> str:
        """Get the current date and time, optionally offset by days/months/years."""
        tzinfo = None
        if timezone:
            try:
                from zoneinfo import ZoneInfo

                tzinfo = ZoneInfo(timezone)
            except Exception:
                tzinfo = None

        now = datetime.now(tzinfo)

        if offset_years or offset_months:
            total_months = now.month - 1 + offset_months + offset_years * 12
            year = now.year + total_months // 12
            month = total_months % 12 + 1
            day = min(now.day, calendar.monthrange(year, month)[1])
            now = now.replace(year=year, month=month, day=day)
        if offset_days:
            now = now + timedelta(days=offset_days)

        try:
            return now.strftime(fmt)
        except Exception:
            return now.strftime("%Y-%m-%d %H:%M:%S")

    get_time.description = time_description
    return get_time


LOCAL_SEARCH_TOOL = "local_search"


def build_local_search_tool(
    db: Chroma,
) -> BaseTool:
    """Build a Chroma-based local vector search tool (read-only).

    Args:
        db: Initialized Chroma instance reused for retrieval.
    """

    args_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Query string."},
            "k": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
                "default": 4,
                "description": "Number of results to return.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    @tool(LOCAL_SEARCH_TOOL, args_schema=args_schema)
    def local_search(query: str, k: int = 4) -> str:
        """
        Search in user uploaded documents. Always try to use this tool if user's query is related to their uploaded docs.
        """
        try:
            if db is None:
                return (
                    "Search service is unavailable: missing Chroma database instance."
                )

            search_kwargs = {"k": int(max(1, min(50, int(k))))}
            retriever = db.as_retriever(
                search_type="similarity", search_kwargs=search_kwargs
            )

            contexts = []
            try:
                if hasattr(retriever, "get_relevant_documents"):
                    contexts = retriever.get_relevant_documents(query)
                elif hasattr(retriever, "invoke"):
                    contexts = retriever.invoke(query)
                elif callable(retriever):
                    contexts = retriever(query)
                else:
                    return "Retriever does not expose a supported retrieval method."
            except Exception:
                try:
                    contexts = retriever.invoke(query)
                except Exception as e:
                    return f"Error during database retrieval: {e}"

            texts = [getattr(r, "page_content", str(r)) for r in contexts]
            return "\n\n".join(texts) if texts else "No relevant information found."
        except Exception as e:
            return f"Error during database retrieval: {e}"

    return local_search
