"""JevFieldVerifier: check LLM-extracted fields against their source text with Jev.

Extraction is a generation task, so the agent's chat model does it. Verification is a decision task,
so Jev does it. For every extracted field the agent must cite the page it came from and quote the
evidence verbatim. Code first checks the quote really is on that page. Then one Jev request judges
every field at once with two propositions each:

* ``<field>_supported``: the value is what the cited page states.
* ``<field>_right_item``: the value answers the field's description, not a neighbouring figure
  (prior year, a segment, a different line item, a different date).

Thresholds turn the probabilities into ``accept`` / ``reject`` / ``review`` per field.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .jev import JevClient, JevError, Noul


@dataclass(frozen=True)
class ExtractedField:
    name: str
    description: str  # what the field means, e.g. "Total net sales for fiscal 2025, USD millions"
    value: Any
    page: int  # 1-based page the value came from
    quote: str  # verbatim evidence from that page


@dataclass(frozen=True)
class FieldVerdict:
    field: ExtractedField
    outcome: str  # "accept" | "reject" | "review"
    reason: str
    supported: float | None = None
    right_item: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("’", "'").replace("—", "-")).strip().lower()


def quote_on_page(quote: str, page_text: str) -> bool:
    """Whitespace- and quote-insensitive containment check."""
    return bool(quote.strip()) and _normalize(quote) in _normalize(page_text)


class JevFieldVerifier:
    def __init__(
        self,
        jev: JevClient,
        *,
        accept_at: float = 0.9,
        reject_at: float = 0.1,
        max_page_chars: int = 12000,
    ):
        if not 0.0 <= reject_at < accept_at <= 1.0:
            raise ValueError("need 0 <= reject_at < accept_at <= 1")
        self._jev = jev
        self._accept_at = accept_at
        self._reject_at = reject_at
        self._max_page_chars = max_page_chars
        self.last_cost: float | None = None

    async def verify(self, fields: list[ExtractedField], pages: Mapping[int, str]) -> list[FieldVerdict]:
        """Verify ``fields`` against ``pages`` (page number -> text). Returns one verdict per field, in order."""
        verdicts: dict[str, FieldVerdict] = {}
        to_check: list[ExtractedField] = []

        for f in fields:
            page_text = pages.get(f.page)
            if page_text is None:
                verdicts[f.name] = FieldVerdict(f, "reject", f"cited page {f.page} is not in the document")
            elif not quote_on_page(f.quote, page_text):
                verdicts[f.name] = FieldVerdict(f, "reject", f"quote not found on page {f.page}")
            else:
                to_check.append(f)

        if to_check:
            verdicts.update(await self._ask_jev(to_check, pages))
        return [verdicts[f.name] for f in fields]

    async def _ask_jev(self, fields: list[ExtractedField], pages: Mapping[int, str]) -> dict[str, FieldVerdict]:
        state = {
            "pages": {str(f.page): self._page_window(pages[f.page], f.quote) for f in fields},
            "fields": {
                f.name: {"description": f.description, "value": f.value, "page": str(f.page), "quote": f.quote}
                for f in fields
            },
        }
        # Phrase each check as a complete proposition with the description and value spelled out.
        # Measured on the Amazon 10-K: this form scores correct fields 0.98-0.99 and a prior-year
        # figure 0.02, whereas asking about `fields.X.value` by reference scored 0.84-0.98 and 0.19.
        questions: dict[str, Noul] = {}
        for f in fields:
            page = f"`pages.{f.page}`"
            what = f.description.rstrip(".")
            questions[f"{f.name}_supported"] = Noul(
                instructions=f"According to {page}, the {what} is {f.value}. Differences only in formatting "
                f"(units, thousands separators, date format, capitalization) still count."
            )
            questions[f"{f.name}_right_item"] = Noul(
                instructions=f"On {page}, {f.value} is the figure for the {what}, and not a look-alike on the same "
                f"page such as a different year, period, segment, line item, total, or as-of date."
            )

        try:
            decision = await self._jev.decide(state, questions)
        except JevError as error:
            return {f.name: FieldVerdict(f, "review", f"Jev unavailable ({error})") for f in fields}

        self.last_cost = decision.usage.cost
        out: dict[str, FieldVerdict] = {}
        for f in fields:
            supported = decision.noul(f"{f.name}_supported")
            right_item = decision.noul(f"{f.name}_right_item")
            out[f.name] = FieldVerdict(f, *self._judge(supported, right_item), supported=supported, right_item=right_item)
        return out

    def _page_window(self, page_text: str, quote: str) -> str:
        """Whole page when it fits the budget; otherwise a window around the quote so it is never cut off."""
        if len(page_text) <= self._max_page_chars:
            return page_text
        # locate the quote tolerating whitespace differences, then centre the window on it
        pattern = r"\s+".join(re.escape(tok) for tok in quote.split())
        match = re.search(pattern, page_text, re.IGNORECASE)
        centre = match.start() if match else 0
        start = max(0, min(centre - self._max_page_chars // 2, len(page_text) - self._max_page_chars))
        return page_text[start : start + self._max_page_chars]

    def _judge(self, supported: float, right_item: float) -> tuple[str, str]:
        checks = {"supported": supported, "right_item": right_item}
        failed = {k: v for k, v in checks.items() if v <= self._reject_at}
        if failed:
            return "reject", "failed " + ", ".join(f"{k}={v:.2f}" for k, v in failed.items())
        if all(v >= self._accept_at for v in checks.values()):
            return "accept", "supported by cited page"
        unsure = {k: v for k, v in checks.items() if v < self._accept_at}
        return "review", "uncertain " + ", ".join(f"{k}={v:.2f}" for k, v in unsure.items())
