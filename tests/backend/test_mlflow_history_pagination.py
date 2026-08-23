"""Run history must follow MLflow's pagination, not read one page.

The defect this pins: `client.search_runs(max_results=200)` looks like "give
me up to 200 runs", but `max_results` is a PAGE SIZE and a Databricks-hosted
tracking server picks its own. Against the live workspace it returned 1 run
on page one and 2 on page two for a three-run experiment, so reading a single
page silently hid every run but the most recent — and because the Results
page derives its dataset dropdown from that list, the whole of history
collapsed to whichever dataset ran last.

It is invisible on a file-backed store, which answers any of these queries in
one page. That is why it only ever showed up in cloud mode.

`_find_run` is paged for the same reason and the stakes are higher there: a
miss reads as "this run has no data" on the Results page rather than as a
missing table row.
"""

from __future__ import annotations

from app.orchestration.mlflow_history import _MAX_SEARCH_PAGES, MLflowHistoryStore


class _Page(list):
    """A PagedList stand-in: a list that also carries a continuation token."""

    def __init__(self, items, token=None):
        super().__init__(items)
        self.token = token


class _FakeClient:
    """Serves prepared pages and records how it was called."""

    def __init__(self, pages):
        self._pages = list(pages)
        self.calls: list[dict] = []

    def search_runs(self, experiment_ids, filter_string="", max_results=None, order_by=None, page_token=None):
        self.calls.append(
            {
                "experiment_ids": experiment_ids,
                "filter_string": filter_string,
                "max_results": max_results,
                "order_by": order_by,
                "page_token": page_token,
            }
        )
        if not self._pages:
            return _Page([])
        return self._pages.pop(0)


def _search(client, **kwargs):
    kwargs.setdefault("limit", 200)
    return MLflowHistoryStore._search_runs(client, "exp-1", **kwargs)


# --- the bug: pages after the first were dropped -----------------------


def test_runs_from_every_page_are_collected():
    """The exact shape the live workspace returned: 1 then 2."""
    client = _FakeClient([_Page(["a"], token="t1"), _Page(["b", "c"], token=None)])

    assert _search(client) == ["a", "b", "c"]


def test_a_single_page_still_works():
    client = _FakeClient([_Page(["a", "b"], token=None)])

    assert _search(client) == ["a", "b"]
    assert len(client.calls) == 1


def test_server_order_is_preserved_across_pages():
    """Pages are concatenated in arrival order — 'newest first' must survive
    paging, or the dropdown reorders itself."""
    client = _FakeClient([_Page([3, 2], token="t"), _Page([1], token=None)])

    assert _search(client) == [3, 2, 1]


def test_the_continuation_token_is_sent_back():
    client = _FakeClient([_Page(["a"], token="t1"), _Page(["b"], token=None)])
    _search(client)

    assert client.calls[0]["page_token"] is None
    assert client.calls[1]["page_token"] == "t1"


# --- bounds ------------------------------------------------------------


def test_the_limit_is_honoured_across_pages():
    client = _FakeClient([_Page([1, 2], token="t"), _Page([3, 4], token="t"), _Page([5], token=None)])

    assert _search(client, limit=3) == [1, 2, 3]


def test_paging_stops_once_the_limit_is_reached():
    """No page is fetched that cannot contribute — this is what stops a
    200-run limit from becoming 200 round trips on a chatty server."""
    client = _FakeClient([_Page([1, 2], token="t"), _Page([3], token=None)])
    _search(client, limit=2)

    assert len(client.calls) == 1


def test_each_page_asks_only_for_what_is_still_needed():
    client = _FakeClient([_Page([1], token="t"), _Page([2], token=None)])
    _search(client, limit=5)

    assert client.calls[0]["max_results"] == 5
    assert client.calls[1]["max_results"] == 4


def test_a_server_that_always_returns_a_token_cannot_loop_forever():
    """Bounded as well as token-driven — an endless token would otherwise
    hang the request thread, and there is only one worker."""

    class _Endless:
        def __init__(self):
            self.calls = 0

        def search_runs(self, *args, **kwargs):
            self.calls += 1
            return _Page([], token="always")

    client = _Endless()
    assert MLflowHistoryStore._search_runs(client, "exp-1", limit=200) == []
    assert client.calls == _MAX_SEARCH_PAGES


# --- backward compatibility -------------------------------------------


def test_a_plain_list_response_ends_the_loop():
    """An SDK or stub that returns a bare list has no `.token`; behaviour
    must collapse to exactly the old single-call path."""
    client = _FakeClient([["a", "b"]])

    assert _search(client) == ["a", "b"]
    assert len(client.calls) == 1


def test_no_runs_at_all_is_an_empty_list_not_an_error():
    assert _search(_FakeClient([_Page([], token=None)])) == []


# --- _find_run's filtered lookup ---------------------------------------


def test_a_filtered_lookup_passes_the_filter_on_every_page():
    client = _FakeClient([_Page([], token="t1"), _Page(["match"], token=None)])
    found = _search(client, limit=1, filter_string="tags.run_id = 'r-1'")

    assert found == ["match"]
    assert [c["filter_string"] for c in client.calls] == [
        "tags.run_id = 'r-1'",
        "tags.run_id = 'r-1'",
    ]


def test_an_empty_first_page_with_a_token_is_not_a_missing_run():
    """The `_find_run` failure mode: giving up on page one reports a run
    that exists as having no data."""
    client = _FakeClient([_Page([], token="t1"), _Page([], token="t2"), _Page(["match"], token=None)])

    assert _search(client, limit=1) == ["match"]


def test_a_genuinely_absent_run_still_reports_absent():
    client = _FakeClient([_Page([], token="t1"), _Page([], token=None)])

    assert _search(client, limit=1) == []
