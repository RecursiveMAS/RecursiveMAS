"""LLM-as-judge for free-form QA (search) datasets.

Scores a model prediction against the gold answer with a large LLM judge over an
OpenAI-compatible chat-completions API: a SYSTEM_PROMPT + VERIFICATION_PROMPT, fields
(question / gold / parsed prediction / raw output[:4000]), JSON {analysis, true_false}
output, temperature 0.0, max_tokens 256.

Configuration is read ONLY from the environment (no provider/account details are
baked into the repo). All three are required; a missing one fails loud:
  API_KEY        bearer token for the judge endpoint
  API_BASE_URL   OpenAI-compatible base or full chat-completions URL
  API_MODEL      judge model id
"""
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

MAX_RAW_CHARS = 4000
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0

# ---- judge prompt ----
SYSTEM_PROMPT = (
    "You are an impartial judge for question answering evaluation. "
    "You must decide whether the model's final answer should be counted as correct. "
    "Be strict about factual equivalence, but ignore harmless formatting differences."
)
VERIFICATION_PROMPT = """Given a question, a gold correct answer, and a model prediction, determine whether the prediction should be counted as correct.

Question:
{question}

Gold Correct Answer:
{gold_answer}

Model Parsed Final Answer:
{pred_answer}

Model Raw Output:
{raw_output}

Instructions:
- Judge semantic correctness, not exact string identity.
- Prefer the parsed final answer. Use the raw output only if the parsed answer is missing, incomplete, or ambiguous.
- Allow small formatting differences, capitalization differences, punctuation differences, and common aliases.
- Allow equivalent date formats when they denote the same date or same year-level answer if the gold only specifies a year.
- Allow answer spans that are minimally longer but still unambiguously identify the same entity.

Return ONLY one JSON object with exactly two fields:
{{
  "analysis": "<brief explanation>",
  "true_false": true
}}
"""


def require_config() -> Dict[str, str]:
    """Return {api_key, url, model} from the environment, or raise loudly if unset."""
    api_key = os.environ.get("API_KEY", "").strip()
    base_url = os.environ.get("API_BASE_URL", "").strip()
    model = os.environ.get("API_MODEL", "").strip()
    missing = [name for name, val in (("API_KEY", api_key), ("API_BASE_URL", base_url), ("API_MODEL", model)) if not val]
    if missing:
        raise RuntimeError(
            "LLM judge is required for this dataset but is not configured. "
            "Set the environment variable(s): " + ", ".join(missing) + "."
        )
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    return {"api_key": api_key, "url": url, "model": model}


def build_prompt(sample: Dict[str, object]) -> str:
    raw_output = str(sample.get("raw_output") or "")
    if MAX_RAW_CHARS > 0 and len(raw_output) > MAX_RAW_CHARS:
        raw_output = raw_output[:MAX_RAW_CHARS]
    return VERIFICATION_PROMPT.format(
        question=str(sample.get("question") or ""),
        gold_answer=str(sample.get("gold_answer") or ""),
        pred_answer=str(sample.get("pred_answer") or ""),
        raw_output=raw_output,
    )


def parse_judge_json(text: str) -> Optional[bool]:
    """Return the true_false verdict, or None if it cannot be parsed."""
    if not text:
        return None
    cands = list(re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE))
    s = text.find("{")
    e = text.rfind("}")
    if s != -1 and e != -1 and e > s:
        cands.append(text[s:e + 1])
    for c in cands:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict) and "true_false" in obj:
                v = obj["true_false"]
                if isinstance(v, bool):
                    return v
                return str(v).strip().lower() in {"true", "1", "yes"}
        except Exception:
            continue
    m = re.search(r'"true_false"\s*:\s*(true|false)', text, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower() == "true"
    return None


def _judge_one(sample: Dict[str, object], cfg: Dict[str, str], retries: int = 5) -> bool:
    """Return the judge verdict for one sample. Transient HTTP errors are retried, then
    raised (fail loud). An unparseable verdict is treated as incorrect (conservative)."""
    body = json.dumps({
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(sample)},
        ],
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
    }).encode("utf-8")
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                cfg["url"], data=body,
                headers={"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"},
            )
            resp = json.loads(urllib.request.urlopen(req, timeout=90).read())
            verdict = parse_judge_json(resp["choices"][0]["message"]["content"])
            return bool(verdict)  # unparseable -> None -> False (conservative)
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2.0 * (attempt + 1))
                continue
            raise RuntimeError(f"LLM judge request failed (HTTP {e.code}).") from e
        except Exception as e:  # network / timeout / decode
            last_err = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"LLM judge request failed after {retries} retries: {last_err}")


def judge_samples(samples: List[Dict[str, object]], max_workers: int = 10) -> List[bool]:
    """Judge every sample concurrently; returns a list of booleans aligned to samples."""
    cfg = require_config()
    verdicts: List[bool] = [False] * len(samples)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_judge_one, s, cfg): i for i, s in enumerate(samples)}
        for fut in as_completed(futs):
            verdicts[futs[fut]] = fut.result()
    return verdicts
