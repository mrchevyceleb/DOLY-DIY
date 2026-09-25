"""Web tools — no API key required.

Search: `ddgs` package (pip). Fallback: DuckDuckGo Lite HTML scrape
via urllib, so the tool still works if the package is missing.
Read: stdlib HTML-to-text page fetcher for following a result link.

The brain triggers these itself by replying with 'SEARCH: <query>' or
'READ: <url>' as its entire first spoken line (see parse_tool_call).
"""
import html
import ipaddress
import re
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request

_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux armv7l) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"}


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
                                 headers=dict(_UA))
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
        lines.append(f"{i}. {t} — {s} [{(r['url'] or '')[:300]}]")
    return "\n".join(lines)


def urls_in_context(context):
    """Every http(s) URL mentioned in a tool-context block."""
    return set(re.findall(r"https?://[^\s\]\)\"'>]+", context or ""))


# --------------------------------------------------------------- page fetch
_BLOCK_RE = re.compile(r"(?is)<(script|style|head|nav|footer|aside|svg|form|noscript)"
                       r"[^>]*>.*?</\1>")
_COMMENT_RE = re.compile(r"(?s)<!--.*?-->")
_TAG_RE = re.compile(r"<[^>]+>")
_TITLE_RE = re.compile(r"(?is)<title[^>]*>(.*?)</title>")
_REDIRECTS = (301, 302, 303, 307, 308)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Auto-following would skip the public-host revalidation below."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _public_http_target(url):
    """True only for http(s) URLs whose host resolves to public space.

    Blocks SSRF style reads of loopback, LAN, link-local, and reserved
    ranges (the brain runs on Matt's network; a prompted or injected
    READ must never point it at internal services).
    """
    try:
        parsed = urllib.parse.urlsplit(url)
    except ValueError:
        return False
    if parsed.scheme.lower() not in ("http", "https"):
        return False
    if parsed.username or parsed.password:          # no credentials relay
        return False
    try:
        host = (parsed.hostname or "").strip("[]").lower()
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    except ValueError:                               # malformed port / IPv6 literal
        return False
    if not host:
        return False
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError, ValueError):
        return False
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False
        if not (ip.is_global and not ip.is_reserved):
            return False
    return True


def read_page(url, max_chars=3500, timeout_s=6):
    """Fetch an http(s) page and return {'title', 'text'} or None.

    Stdlib only: public-host gate on every redirect hop, content-type
    gate, drops script/style/nav blocks and tags, keeps readable text,
    capped at max_chars so a page never floods her small context.
    """
    url = (url or "").strip()
    opener = urllib.request.build_opener(_NoRedirect)
    raw = None
    for _ in range(4):  # initial fetch + up to 3 redirect hops
        if not _public_http_target(url):
            return None
        req = urllib.request.Request(url, headers=dict(_UA))
        try:
            resp = opener.open(req, timeout=timeout_s)
        except urllib.error.HTTPError as e:
            if e.code in _REDIRECTS and e.headers.get("Location"):
                url = urllib.parse.urljoin(url, e.headers["Location"])
                continue
            return None
        with resp:
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if not any(t in ctype for t in ("html", "text/plain", "xml", "json")):
                return None
            raw = resp.read(2_000_000).decode("utf-8", errors="replace")
        break
    if raw is None:
        return None
    m = _TITLE_RE.search(raw)
    title = ""
    if m:
        title = " ".join(html.unescape(_TAG_RE.sub(" ", m.group(1))).split())[:120]
    text = _COMMENT_RE.sub(" ", _BLOCK_RE.sub(" ", raw))
    text = " ".join(html.unescape(_TAG_RE.sub(" ", text)).split())
    if not text:
        return None
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + " …"
    return {"title": title, "text": text}


def page_block(url, page):
    """Render a fetched page as a compact context block for the LLM."""
    title = (page or {}).get("title") or url
    return f"Fetched page '{title}' ({url}):\n{(page or {}).get('text', '')}"


# ------------------------------------------------------------- brain trigger
_TOOL_CALL_RE = re.compile(r"^\s*(search|read)\s*:\s*(.+?)\s*$", re.IGNORECASE)


def parse_tool_call(sentence):
    """Detect the brain demanding a web tool in its first spoken line.

    'SEARCH: <query>' -> ("search", query); 'READ: <url>' -> ("read", url).
    None for ordinary speech, so a normal reply never misroutes into a tool.
    READ keeps its URL byte-exact apart from one sentence-final period —
    legal URL characters like '!' are never stripped.
    """
    m = _TOOL_CALL_RE.match(sentence or "")
    if not m:
        return None
    kind = m.group(1).lower()
    arg = m.group(2).strip().strip("\"'")
    if not arg:
        return None
    if kind == "read":
        arg = re.sub(r"[.…]$", "", arg.strip())
        return ("read", arg) if re.match(r"^https?://", arg, re.I) else None
    return "search", arg.rstrip(" .!?…")
