import asyncio

from strands_jev import ExtractedField, JevFieldVerifier, quote_on_page

PAGES = {
    12: "AMAZON.COM, INC. (Exact name of registrant)\nDelaware   91-1646860\n(State or other jurisdiction of incorporation)",
    48: "Total net sales  574,785  637,959  716,924\nNet income $ 30,425 $ 59,248 $ 77,670",
}

SALES = ExtractedField(
    name="total_net_sales",
    description="Total net sales for fiscal 2025, USD millions",
    value=716924,
    page=48,
    quote="Total net sales 574,785 637,959 716,924",
)
STATE = ExtractedField(
    name="state_of_incorporation",
    description="State of incorporation",
    value="Delaware",
    page=12,
    quote="Delaware 91-1646860",
)


def nouls(**probs):
    return {k: {"type": "noul", "noul": v} for k, v in probs.items()}


def run(verifier, fields, pages=PAGES):
    return asyncio.run(verifier.verify(fields, pages))


def test_quote_matching_ignores_whitespace_and_case():
    assert quote_on_page("total net sales  574,785", PAGES[48])
    assert not quote_on_page("Total net sales 999", PAGES[48])
    assert not quote_on_page("   ", PAGES[48])


def test_one_request_verifies_all_fields_and_accepts_clear_ones(jev, fake):
    fake.answers = nouls(
        total_net_sales_supported=0.98,
        total_net_sales_right_item=0.95,
        state_of_incorporation_supported=0.99,
        state_of_incorporation_right_item=0.97,
    )
    verdicts = run(JevFieldVerifier(jev), [SALES, STATE])

    assert [v.outcome for v in verdicts] == ["accept", "accept"]
    assert verdicts[0].supported == 0.98 and verdicts[0].right_item == 0.95
    assert len(fake.requests) == 1
    sent = fake.requests[0]
    assert set(sent["state"]["pages"]) == {"48", "12"}
    assert sent["state"]["fields"]["total_net_sales"]["value"] == 716924
    assert set(sent["questions"]) == {
        "total_net_sales_supported",
        "total_net_sales_right_item",
        "state_of_incorporation_supported",
        "state_of_incorporation_right_item",
    }
    # questions are complete propositions with the description and value inlined
    supported = sent["questions"]["total_net_sales_supported"]["instructions"]
    assert "`pages.48`" in supported and "716924" in supported and "Total net sales for fiscal 2025" in supported
    right_item = sent["questions"]["total_net_sales_right_item"]["instructions"]
    assert "716924" in right_item and "different year" in right_item


def test_long_page_is_windowed_around_the_quote(jev, fake):
    fake.answers = nouls(total_net_sales_supported=0.99, total_net_sales_right_item=0.99)
    long_page = ("filler text " * 2000) + PAGES[48] + (" more filler" * 2000)
    verifier = JevFieldVerifier(jev, max_page_chars=3000)
    (verdict,) = run(verifier, [SALES], {48: long_page})
    assert verdict.outcome == "accept"
    sent_page = fake.requests[0]["state"]["pages"]["48"]
    assert len(sent_page) <= 3000
    assert "716,924" in sent_page


def test_wrong_year_figure_is_rejected(jev, fake):
    prior_year = ExtractedField("total_net_sales", SALES.description, 637959, 48, SALES.quote)
    fake.answers = nouls(total_net_sales_supported=0.9, total_net_sales_right_item=0.04)
    (verdict,) = run(JevFieldVerifier(jev), [prior_year])
    assert verdict.outcome == "reject"
    assert "right_item=0.04" in verdict.reason


def test_middle_probability_goes_to_review(jev, fake):
    fake.answers = nouls(total_net_sales_supported=0.6, total_net_sales_right_item=0.95)
    (verdict,) = run(JevFieldVerifier(jev), [SALES])
    assert verdict.outcome == "review"
    assert "supported=0.60" in verdict.reason


def test_quote_not_on_page_is_rejected_without_calling_jev(jev, fake):
    bogus = ExtractedField("ceo", "CEO name", "Andrew R. Jassy", 12, "Andrew R. Jassy President and CEO")
    (verdict,) = run(JevFieldVerifier(jev), [bogus])
    assert verdict.outcome == "reject"
    assert "quote not found" in verdict.reason
    assert fake.requests == []


def test_unknown_page_is_rejected(jev, fake):
    off = ExtractedField("x", "x", 1, 999, "anything")
    (verdict,) = run(JevFieldVerifier(jev), [off])
    assert verdict.outcome == "reject"
    assert "999" in verdict.reason


def test_jev_failure_sends_everything_to_review(jev, fake):
    fake.status_code = 502
    verdicts = run(JevFieldVerifier(jev), [SALES, STATE])
    assert [v.outcome for v in verdicts] == ["review", "review"]
    assert "502" in verdicts[0].reason


def test_mixed_precheck_and_jev_keeps_input_order(jev, fake):
    bogus = ExtractedField("ceo", "CEO", "Nobody", 12, "not on the page")
    fake.answers = nouls(total_net_sales_supported=0.99, total_net_sales_right_item=0.99)
    verdicts = run(JevFieldVerifier(jev), [bogus, SALES])
    assert [v.field.name for v in verdicts] == ["ceo", "total_net_sales"]
    assert [v.outcome for v in verdicts] == ["reject", "accept"]
    assert set(fake.requests[0]["questions"]) == {"total_net_sales_supported", "total_net_sales_right_item"}
