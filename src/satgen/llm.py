"""Model clients. Retries live in call_with_retry so every provider shares one backoff policy."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import threading
import time
from typing import Callable, Protocol

log = logging.getLogger("satgen")

GEN_MODEL = os.getenv("SATGEN_GEN_MODEL", "gemini-2.5-flash-lite")
REVIEW_MODEL = os.getenv("SATGEN_REVIEW_MODEL", "claude-sonnet-5-5")
TIMEOUT_S = float(os.getenv("SATGEN_TIMEOUT_S", "60"))


class LLM(Protocol):
    model: str

    def complete(
        self, prompt: str, system: str = "", temperature: float = 0.7, json_mode: bool = False
    ) -> tuple[str, int, int]: ...


def call_with_retry(
    client: LLM,
    prompt: str,
    *,
    system: str = "",
    temperature: float = 0.7,
    json_mode: bool = False,
    max_retries: int = 4,
    base_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[str, int, int]:
    """Call the model, retrying failures with exponential backoff and full jitter.

    Rate limits and timeouts are routine at batch scale (about 12,600 calls for
    1,800 questions), so one failed call must never end the run.
    """
    attempt = 0
    while True:
        try:
            return client.complete(prompt, system=system, temperature=temperature, json_mode=json_mode)
        except Exception as exc:  # SDKs raise many types; treat all as retryable
            if attempt >= max_retries:
                raise
            delay = random.uniform(0, base_delay * 2**attempt)
            log.warning("%s call failed (%s); retry %d in %.1fs", client.model, exc, attempt + 1, delay)
            sleep(delay)
            attempt += 1


class GeminiClient:
    def __init__(self, model: str = GEN_MODEL):
        from google import genai  # pip install google-genai
        from google.genai import types

        self.model = model
        self._types = types
        self._client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options=types.HttpOptions(timeout=int(TIMEOUT_S * 1000)),
        )

    def complete(self, prompt, system="", temperature=0.7, json_mode=False):
        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                system_instruction=system or None,
                temperature=temperature,
                response_mime_type="application/json" if json_mode else None,
            ),
        )
        usage = resp.usage_metadata
        return (
            resp.text or "",
            usage.prompt_token_count or 0,
            (usage.candidates_token_count or 0) + (getattr(usage, "thoughts_token_count", 0) or 0),
        )


class ClaudeClient:
    def __init__(self, model: str = REVIEW_MODEL):
        import anthropic  # pip install anthropic

        self.model = model
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=TIMEOUT_S, max_retries=0)

    def complete(self, prompt, system="", temperature=0.0, json_mode=False):
        if json_mode:
            system = (system + "\n\n" if system else "") + "Reply with a single JSON object and nothing else."
        kw = {} if _fixed_sampling(self.model) else {"temperature": temperature}
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            **kw,
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        return text, msg.usage.input_tokens, msg.usage.output_tokens


def _fixed_sampling(model: str) -> bool:
    """Claude 5-family models reject non-default temperature/top_p with a 400, so leave them unset.

    Solver diversity on those models comes only from independent calls.
    """
    return re.match(r"claude-(opus|sonnet|haiku)-5", model) is not None


def make_client(model: str) -> LLM:
    if model.startswith("gemini"):
        return GeminiClient(model)
    if model.startswith("claude"):
        return ClaudeClient(model)
    raise ValueError(f"no client for model {model!r}; expected a gemini-* or claude-* id")


# Offline simulation

_WORDS = [
    (
        "A coastal survey recorded steady tide patterns over three decades.",
        "robust",
        "strong",
        ["fragile", "loud", "sudden"],
    ),
    (
        "A city council debated a new transit plan for the river district.",
        "tenuous",
        "weak",
        ["lasting", "hostile", "complete"],
    ),
    (
        "A botanist catalogued unusual ferns in a mountain valley.",
        "prolific",
        "abundant",
        ["scarce", "slow", "ancient"],
    ),
    (
        "A violinist rehearsed a piece written by a little-known composer.",
        "austere",
        "plain",
        ["lavish", "rapid", "playful"],
    ),
    (
        "An archivist restored letters exchanged between two astronomers.",
        "candid",
        "frank",
        ["guarded", "formal", "brief"],
    ),
    (
        "A research team monitored soil moisture across several farms.",
        "erratic",
        "irregular",
        ["steady", "gradual", "modest"],
    ),
    ("A novelist revisited the harbor town where she grew up.", "wistful", "longing", ["cheerful", "angry", "hurried"]),
    (
        "An engineer tested a bridge model under repeated loads.",
        "resilient",
        "durable",
        ["brittle", "costly", "narrow"],
    ),
    ("A museum curator arranged an exhibit on early printing.", "meticulous", "careful", ["careless", "bold", "dated"]),
    (
        "A linguist compared verb forms in two neighboring dialects.",
        "subtle",
        "delicate",
        ["obvious", "harsh", "common"],
    ),
    (
        "A pastry chef adjusted a recipe for high-altitude baking.",
        "pragmatic",
        "practical",
        ["idealistic", "lazy", "lavish"],
    ),
    (
        "A historian examined trade records from a medieval port.",
        "sporadic",
        "occasional",
        ["constant", "ample", "hidden"],
    ),
]
_WORD_INDEX = {w: (subj, w, right, wrong) for subj, w, right, wrong in _WORDS}
_EQ = re.compile(r"(\d+)x \+ (\d+) = (\d+)")


def _stable_rng(*parts: object) -> random.Random:
    digest = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest()
    return random.Random(int.from_bytes(digest, "big"))


class SimulatedClient:
    """Deterministic stand-in that lets the whole pipeline run with no API keys.

    It plays all three roles. As generator it draws from a small bank and
    deliberately plants a wrong answer key in about `bad_key_rate` of items,
    so the gate and the reviewer have real errors to catch. As solver it
    picks the correct choice with probability `solver_accuracy`. As reviewer
    it knows the truth and corrects planted keys. Replies depend only on the
    prompt text, so results are identical with any level of concurrency.

    The numbers it produces demonstrate the mechanics; they say nothing about
    how real models behave. Real measurements come from `satgen eval`.
    """

    def __init__(self, model: str = "simulated", solver_accuracy: float = 0.8, bad_key_rate: float = 0.15):
        self.model = model
        self._acc = solver_accuracy
        self._bad = bad_key_rate
        self._seen: dict[str, int] = {}
        self._lock = threading.Lock()

    def _nth(self, prompt: str) -> int:
        with self._lock:
            self._seen[prompt] = self._seen.get(prompt, 0) + 1
            return self._seen[prompt]

    def complete(self, prompt, system="", temperature=0.7, json_mode=False):
        t_in = max(1, len(prompt) // 4)
        rng = _stable_rng(self.model, prompt, self._nth(prompt))
        if "Return JSON with keys: qtype" in prompt:
            return self._generate(prompt), t_in, 120
        if "Return JSON with keys:\n  solved_answer" in prompt:
            return self._review(prompt, rng), t_in, 60
        return self._solve(prompt, rng), t_in, 3

    def _generate(self, prompt: str) -> str:
        i = int(re.search(r"Variation: (\d+)", prompt).group(1))
        planted = _stable_rng("bad", i).random() < self._bad
        if "spr_" in prompt.split("type:")[1].split("\n")[0]:
            a, x, b = i % 5 + 2, i % 9 + 1, i % 7 + 1
            key = x + 1 if planted else x
            return json.dumps(
                {
                    "qtype": "spr_linear_equations",
                    "format": "spr",
                    "passage": "",
                    "stem": f"If {a}x + {b} = {a * x + b}, what is the value of x?",
                    "choices": {},
                    "answer": str(key),
                    "explanation": f"Subtract {b} from both sides, then divide by {a}: x = {key}.",
                }
            )
        idx = i if i % 7 else i - 1  # every 7th item repeats the previous one to exercise dedupe
        subj, word, right, wrong = _WORDS[idx % len(_WORDS)]
        choices = dict(zip("ABCD", [right, *wrong], strict=True))
        key = "BCD"[i % 3] if planted else "A"
        return json.dumps(
            {
                "qtype": "words_in_context",
                "format": "mcq",
                "passage": f"{subj} Reviewers described the outcome as {word}, a judgment that shaped the next round of work.",
                "stem": f'As used in the text, "{word}" most nearly means',
                "choices": choices,
                "answer": key,
                "explanation": f'The context points to "{choices[key]}".',
            }
        )

    @staticmethod
    def _truth(prompt: str) -> tuple[str, dict[str, str]]:
        """Return the correct answer for the question embedded in the prompt."""
        eq = _EQ.search(prompt)
        if eq:
            a, b, c = map(int, eq.groups())
            return str((c - b) // a), {}
        word = re.search(r'"(\w+)" most nearly means', prompt).group(1)
        right = _WORD_INDEX[word][2]
        choices = dict(re.findall(r"^([A-D])\) (.+)$", prompt, re.MULTILINE))
        letter = next(k for k, v in choices.items() if v == right)
        return letter, choices

    def _solve(self, prompt: str, rng: random.Random) -> str:
        truth, choices = self._truth(prompt)
        if rng.random() < self._acc:
            return truth
        if choices:
            return rng.choice([k for k in choices if k != truth])
        return str(int(truth) + rng.choice([-2, -1, 1, 2]))

    def _review(self, prompt: str, rng: random.Random) -> str:
        truth, choices = self._truth(prompt)
        shown_key = re.search(r"Answer key: (\S+)", prompt)
        key = shown_key.group(1) if shown_key else truth  # a key-hidden review never sees the key
        verdict = {"solved_answer": truth, "key_correct": key == truth, "unambiguous": True, "explanation": None}
        if not shown_key:
            verdict["explanation"] = f"The answer is {truth}."
        if key != truth:
            shown = choices.get(truth, truth)
            verdict["explanation"] = (
                f'The correct answer is {truth} ("{shown}").' if choices else f"Solving the equation gives x = {truth}."
            )
            verdict["notes"] = "answer key was wrong"
        elif rng.random() < 0.05:
            verdict.update(unambiguous=False, notes="wording is ambiguous")
        else:
            verdict["notes"] = "ok"
        return json.dumps(verdict)
