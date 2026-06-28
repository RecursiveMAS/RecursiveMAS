"""Tavily web search with multi-key rotation, distinguishing rate-limits from credit exhaustion.

Tavily returns HTTP 429 for two different conditions:
  - rate limit: transient; carries a `retry-after` header / a rate-limit message ->
    back off and keep the key.
  - usage / credit limit: permanent for that key; message mentions usage/credit/plan/quota ->
    mark the key exhausted and rotate to the next.
401 -> invalid/dead key -> exhaust. Error body: {"detail":{"error":"..."}}.

Keys are tried round-robin so load and credits spread evenly; exhausted keys are shared
across processes via an flock'd file. When every key is usage-exhausted, a sentinel file is
written and TAVILY_ALL_KEYS_EXHAUSTED is returned (so a credit wall is never silently turned
into empty results). An all-rate-limited pass backs off and retries without exhausting keys.
"""
import os
import json
import time
import random
import urllib.request
import urllib.error

try:
    import fcntl
except Exception:  # pragma: no cover
    fcntl = None

TAVILY_URL = "https://api.tavily.com/search"
EXHAUSTED_MARKER = "[TAVILY_ALL_KEYS_EXHAUSTED]"
_USAGE_HINTS = ("usage", "credit", "plan", "quota", "monthly", "exceeded your", "limit reached")
_RATE_HINTS = ("rate limit", "too many", "per minute", "per second", "rpm", "slow down")


def load_keys(path):
    if not path or not os.path.isfile(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            k = line.strip()
            if k and not k.startswith("#"):
                out.append(k)
    return out


def _load_exhausted(path):
    if not path or not os.path.isfile(path):
        return set()
    try:
        with open(path) as f:
            return {x.strip() for x in f if x.strip()}
    except OSError:
        return set()


def _mark_exhausted(path, key):
    if not path:
        return
    try:
        with open(path, "a+") as f:
            if fcntl is not None:
                fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            cur = {x.strip() for x in f if x.strip()}
            if key not in cur:
                f.write(key + "\n")
                f.flush()
            if fcntl is not None:
                fcntl.flock(f, fcntl.LOCK_UN)
    except OSError:
        pass


def _one_search(key, query, depth, max_results, timeout=30):
    """Return (status_code, payload_or_detail, retry_after_seconds_or_None)."""
    body = json.dumps({
        "api_key": key,
        "query": query,
        "search_depth": depth,
        "max_results": max_results,
    }).encode("utf-8")
    req = urllib.request.Request(TAVILY_URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return 200, json.loads(r.read()), None
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            detail = ""
        ra = None
        try:
            rah = e.headers.get("retry-after") or e.headers.get("Retry-After")
            ra = float(rah) if rah is not None else None
        except Exception:
            ra = None
        return e.code, detail, ra
    except Exception as e:
        return -1, f"{type(e).__name__}: {e}", None


def _classify_429(detail, retry_after):
    """Return 'usage' (permanent, exhaust key) or 'rate' (transient, back off)."""
    d = (detail or "").lower()
    if any(h in d for h in _USAGE_HINTS):
        return "usage"
    if retry_after is not None or any(h in d for h in _RATE_HINTS):
        return "rate"
    # ambiguous 429 -> treat as transient rate-limit (NEVER false-exhaust a key)
    return "rate"


def _format(query, resp, max_results, max_chars):
    lines = [f"Query: {query}"]
    ans = resp.get("answer")
    if ans:
        lines += ["Answer:", str(ans).strip()]
    results = resp.get("results") or []
    if results:
        lines.append("Results:")
    for i, item in enumerate(results[:max_results], 1):
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        content = " ".join(str(item.get("content") or "").split())
        if len(content) > 320:
            content = content[:320].rstrip() + " ..."
        lines.append(f"[{i}] {title}")
        if url:
            lines.append(f"URL: {url}")
        if content:
            lines.append(content)
    if not ans and not results:
        lines.append("[no search results]")
    out = "\n".join(lines)
    return out[:max_chars]


def tavily_search(query, keys_path, exhausted_path=None, sentinel_path=None,
                  depth="advanced", max_results=4, max_chars=6000,
                  max_passes=4):
    """Round-robin search; rotate on usage-exhaustion (429-usage)/401, back off on rate-limit (429-rate)."""
    keys = load_keys(keys_path)
    if not keys:
        return "[search unavailable] no Tavily keys configured"
    # random start offset spreads load evenly across keys
    start = random.randrange(len(keys))
    order = [keys[(start + i) % len(keys)] for i in range(len(keys))]

    last_rate_detail = None
    for attempt in range(max_passes):
        exhausted = _load_exhausted(exhausted_path)
        live = [k for k in order if k not in exhausted]
        if not live:
            break  # every key usage-exhausted
        rate_limited_all = True
        max_retry_after = 0.0
        for key in live:
            code, payload, retry_after = _one_search(key, query, depth, max_results)
            if code == 200:
                return _format(query, payload, max_results, max_chars)
            if code == 401:
                _mark_exhausted(exhausted_path, key)
                continue
            if code == 429:
                kind = _classify_429(payload, retry_after)
                if kind == "usage":
                    _mark_exhausted(exhausted_path, key)
                    continue
                # rate-limited: skip this key this pass, try the next (do NOT exhaust)
                last_rate_detail = payload
                if retry_after:
                    max_retry_after = max(max_retry_after, retry_after)
                continue
            # 400/403/5xx/network: transient or request issue -> try next key, keep all keys
            rate_limited_all = False
            last_rate_detail = f"http={code} {str(payload)[:120]}"
            continue
        # finished a pass without a 200. If everything was rate-limited, back off and retry.
        if attempt < max_passes - 1:
            time.sleep(min(max_retry_after or (2.0 * (attempt + 1)), 15.0))
    # could not get a result
    exhausted = _load_exhausted(exhausted_path)
    if all(k in exhausted for k in keys):
        if sentinel_path:
            try:
                with open(sentinel_path, "a") as f:
                    f.write(f"ALL_KEYS_EXHAUSTED t={time.strftime('%Y-%m-%d %H:%M:%S')} q={str(query)[:80]}\n")
            except OSError:
                pass
        return f"{EXHAUSTED_MARKER} all {len(keys)} Tavily keys out of credits"
    # transient: keys still live but rate-limited/erroring this round
    return f"[search temporarily unavailable] {str(last_rate_detail)[:160]}"
