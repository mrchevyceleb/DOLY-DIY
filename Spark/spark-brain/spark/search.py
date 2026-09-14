"""Web search tool — no API key required.

Primary: `ddgs` package (pip). Fallback: DuckDuckGo Lite HTML scrape
via urllib, so the tool still works if the package is missing.
"""
import html
import re
import sys
import urllib.parse
import urllib.request


def _log(msg):
    print(f"[search] {msg}", file=sys.stderr)


def _ddgs(query, max_results):
    from ddgs import DDGS
    out = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            out.append({"title": r.get("title", ""),
                        "snippet": r.get("body", "") or r.get("snippet", ""),
                        "url": r.get("href", "") or r.get("url", "")})
    return out


def _html_fallback(query, max_results):
    """DuckDuckGo HTML endpoint via POST (no JS, no key)."""
    data = urllib.parse.urlencode({"q": query}).encode()
    req = urllib.request.Request("https://html.duckduckgo.com/html/", data=data,
                                 headers={"User-Agent": "Mozilla/5.0 (X11; Linux armv7l) "
                                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                                          "Chrome/120.0 Safari/537.36"})
    page = urllib.request.urlopen(req, timeout=8).read().decode("utf-8", errors="replace")
    links = re.findall(r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', page, re.S)
    snips = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', page, re.S)
    results = []
    for i, (href, title) in enumerate(links[:max_results]):
        snippet = html.unescape(re.sub(r"<[^>]+>", "", snips[i])) if i < len(snips) else ""
        # unwrap ddg redirect (uddg=...)
        m = re.search(r"uddg=([^&]+)", href)
        if m:
            href = urllib.parse.unquote(m.group(1))
        results.append({"title": html.unescape(re.sub(r"<[^>]+>", "", title)).strip(),
                        "snippet": " ".join(snippet.split()), "url": href})
    return results


def web_search(query, max_results=4):
    """Return [{title, snippet, url}], possibly empty. Never raises."""
    query = (query or "").strip()
    if not query:
        return []
    for fn in (_ddgs, _html_fallback):
        try:
            results = fn(query, max_results)
            if results:
                return results
        except Exception as e:
            _log(f"{fn.__name__} failed: {e}")
    return []


def context_block(query, results):
    """Render results as a compact context block for the LLM."""
    lines = [f"Web search results for '{query}' (just retrieved):"]
    for i, r in enumerate(results, 1):
        t = " ".join((r["title"] or "").split())[:80]
        s = " ".join((r["snippet"] or "").split())[:240]
        lines.append(f"{i}. {t} — {s} [{r['url'][:100]}]")
    return "\n".join(lines)
