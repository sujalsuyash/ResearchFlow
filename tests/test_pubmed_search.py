"""
tests/test_pubmed_search.py

Pytest suite for tools/pubmed_search.py.

Approach
--------
All HTTP I/O is mocked via pytest-mock so the suite runs without a network
connection and without consuming NCBI rate-limit quota.  Each mock returns
realistic XML / JSON fixtures that exercise every code path.

Coverage targets
----------------
* _build_query        — plain query, MeSH injection, multi-term MeSH
* _build_date_params  — no dates, year_start only, year_end only, both
* _parse_year         — <Year>, <MedlineDate>, <ArticleDate>, missing date
* _parse_abstract     — flat abstract, structured (BACKGROUND/METHODS/…) abstract, missing abstract
* _parse_authors      — standard authors, CollectiveName, mixed list
* _parse_xml          — complete record, missing optional fields, malformed node (skipped)
* search_pubmed       — happy path, cache hit (Redis + fallback), 0-result, 429 retry
* _tool_fn / pubmed_tool — Markdown output shape, zero-result string, LangChain invoke
* clear_cache         — Redis mode and fallback mode
* cache_stats         — Redis mode and fallback mode
* PubMedInput schema  — valid defaults, limit clamping, limit out-of-range
"""

from __future__ import annotations

import json
import time
from typing import Optional
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Fixtures — realistic NCBI XML / JSON responses
# ---------------------------------------------------------------------------

ESEARCH_JSON_3 = json.dumps({
    "esearchresult": {
        "count": "3",
        "retmax": "3",
        "retstart": "0",
        "idlist": ["37000001", "37000002", "37000003"],
        "translationset": [],
        "querytranslation": "CRISPR[All Fields]",
    }
})

ESEARCH_JSON_0 = json.dumps({
    "esearchresult": {
        "count": "0",
        "retmax": "0",
        "retstart": "0",
        "idlist": [],
    }
})

# Full, realistic efetch XML with three complete articles
EFETCH_XML_3 = """<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation Status="MEDLINE" Owner="NLM">
      <PMID Version="1">37000001</PMID>
      <Article PubModel="Print-Electronic">
        <Journal>
          <Title>Nature Biotechnology</Title>
          <JournalIssue CitedMedium="Internet">
            <PubDate><Year>2023</Year><Month>Jan</Month></PubDate>
          </JournalIssue>
        </Journal>
        <ArticleTitle>CRISPR-Cas9 enables efficient gene editing in cancer cells</ArticleTitle>
        <Abstract>
          <AbstractText>This study demonstrates the use of CRISPR-Cas9 for targeted gene disruption in solid tumour cell lines, achieving editing efficiencies of up to 95% without off-target effects.</AbstractText>
        </Abstract>
        <AuthorList CompleteYN="Y">
          <Author ValidYN="Y"><LastName>Smith</LastName><ForeName>Jane</ForeName></Author>
          <Author ValidYN="Y"><LastName>Doe</LastName><ForeName>John</ForeName></Author>
          <Author ValidYN="Y"><LastName>Lee</LastName><ForeName>Hyun</ForeName></Author>
          <Author ValidYN="Y"><LastName>Patel</LastName><ForeName>Priya</ForeName></Author>
        </AuthorList>
      </Article>
      <MeshHeadingList>
        <MeshHeading><DescriptorName UI="D000071196">CRISPR-Cas Systems</DescriptorName></MeshHeading>
        <MeshHeading><DescriptorName UI="D009369">Neoplasms</DescriptorName></MeshHeading>
        <MeshHeading><DescriptorName UI="D005796">Genes</DescriptorName></MeshHeading>
      </MeshHeadingList>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">37000001</ArticleId>
        <ArticleId IdType="doi">10.1038/nbt.2023.001</ArticleId>
        <ArticleId IdType="pmc">PMC1111111</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>

  <PubmedArticle>
    <MedlineCitation Status="MEDLINE" Owner="NLM">
      <PMID Version="1">37000002</PMID>
      <Article PubModel="Print">
        <Journal>
          <Title>Cell</Title>
          <JournalIssue CitedMedium="Print">
            <PubDate><MedlineDate>2022 Sep-Oct</MedlineDate></PubDate>
          </JournalIssue>
        </Journal>
        <ArticleTitle>Structured abstract paper: background and methods</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND" NlmCategory="BACKGROUND">Gene editing has revolutionised functional genomics.</AbstractText>
          <AbstractText Label="METHODS" NlmCategory="METHODS">We applied prime editing to 200 patient-derived cell lines.</AbstractText>
          <AbstractText Label="RESULTS" NlmCategory="RESULTS">Off-target rates below 0.1% were observed across all lines.</AbstractText>
          <AbstractText Label="CONCLUSIONS" NlmCategory="CONCLUSIONS">Prime editing is a safe approach for therapeutic applications.</AbstractText>
        </Abstract>
        <AuthorList CompleteYN="Y">
          <Author ValidYN="Y"><LastName>García</LastName><ForeName>María</ForeName></Author>
          <Author ValidYN="Y">
            <CollectiveName>Prime Editing Consortium</CollectiveName>
          </Author>
        </AuthorList>
      </Article>
      <MeshHeadingList>
        <MeshHeading><DescriptorName UI="D005796">Gene Editing</DescriptorName></MeshHeading>
      </MeshHeadingList>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">37000002</ArticleId>
        <ArticleId IdType="doi">10.1016/j.cell.2022.009</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>

  <PubmedArticle>
    <MedlineCitation Status="In-Process" Owner="NLM">
      <PMID Version="1">37000003</PMID>
      <Article PubModel="Electronic">
        <Journal>
          <Title>Science</Title>
          <JournalIssue CitedMedium="Internet">
            <PubDate><Year>2024</Year></PubDate>
          </JournalIssue>
        </Journal>
        <ArticleTitle>Minimal paper: no abstract, no MeSH, epub year only</ArticleTitle>
        <ArticleDate DateType="Electronic">
          <Year>2024</Year><Month>03</Month><Day>15</Day>
        </ArticleDate>
        <AuthorList CompleteYN="N">
          <Author ValidYN="Y"><LastName>Tanaka</LastName></Author>
        </AuthorList>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">37000003</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""

# XML with a malformed node mixed in between two valid ones
EFETCH_XML_MALFORMED = """<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>37000001</PMID>
      <Article>
        <Journal><Title>Nature</Title><JournalIssue><PubDate><Year>2023</Year></PubDate></JournalIssue></Journal>
        <ArticleTitle>Valid paper</ArticleTitle>
        <Abstract><AbstractText>A valid abstract.</AbstractText></Abstract>
        <AuthorList><Author><LastName>Smith</LastName><ForeName>A</ForeName></Author></AuthorList>
      </Article>
    </MedlineCitation>
    <PubmedData><ArticleIdList><ArticleId IdType="pubmed">37000001</ArticleId></ArticleIdList></PubmedData>
  </PubmedArticle>
  <!-- Deliberately broken node: no Article child -->
  <PubmedArticle>
    <MedlineCitation>
      <PMID>37000099</PMID>
    </MedlineCitation>
    <PubmedData><ArticleIdList></ArticleIdList></PubmedData>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>37000002</PMID>
      <Article>
        <Journal><Title>Science</Title><JournalIssue><PubDate><Year>2022</Year></PubDate></JournalIssue></Journal>
        <ArticleTitle>Another valid paper</ArticleTitle>
        <Abstract><AbstractText>Another valid abstract.</AbstractText></Abstract>
        <AuthorList><Author><LastName>Jones</LastName><ForeName>B</ForeName></Author></AuthorList>
      </Article>
    </MedlineCitation>
    <PubmedData><ArticleIdList><ArticleId IdType="pubmed">37000002</ArticleId></ArticleIdList></PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""

# Response for a 429 followed by success (used in retry tests)
_429_RESPONSE = MagicMock()
_429_RESPONSE.status_code = 429
_429_RESPONSE.headers = {}

# ---------------------------------------------------------------------------
# Helper: build a mock httpx.Response
# ---------------------------------------------------------------------------

def _mock_response(text: str, status_code: int = 200) -> MagicMock:
    r = MagicMock()
    r.status_code = status_code
    r.text = text
    r.json.return_value = json.loads(text) if text.lstrip().startswith("{") else {}
    r.raise_for_status = MagicMock()  # no-op by default
    return r


def _mock_json_response(obj: dict | str, status_code: int = 200) -> MagicMock:
    if isinstance(obj, str):
        text = obj
        data = json.loads(obj)
    else:
        text = json.dumps(obj)
        data = obj
    r = MagicMock()
    r.status_code = status_code
    r.text = text
    r.json.return_value = data
    r.raise_for_status = MagicMock()
    return r


# ---------------------------------------------------------------------------
# Patch target helpers
# ---------------------------------------------------------------------------

PATCH_THROTTLED_GET = "tools.pubmed_search._throttled_get"


# ===========================================================================
# 1. _build_query
# ===========================================================================

class TestBuildQuery:
    """_build_query composes the final PubMed query string from PubMedInput."""

    def setup_method(self):
        from tools.pubmed_search import PubMedInput, _build_query
        self.PubMedInput = PubMedInput
        self._build_query = _build_query

    def _inp(self, query="cancer", mesh_terms=None, **kw) -> object:
        return self.PubMedInput(query=query, mesh_terms=mesh_terms, **kw)

    def test_plain_query_unchanged(self):
        """No MeSH terms → query is returned as-is."""
        result = self._build_query(self._inp(query="cancer immunotherapy"))
        assert result == "cancer immunotherapy"

    def test_single_mesh_term(self):
        result = self._build_query(self._inp(query="cancer", mesh_terms="Neoplasms"))
        assert '"Neoplasms"[MeSH Terms]' in result
        assert result.startswith("(cancer)")

    def test_multiple_mesh_terms_are_and_combined(self):
        result = self._build_query(self._inp(query="cancer", mesh_terms="Neoplasms,Immunotherapy"))
        assert '"Neoplasms"[MeSH Terms]' in result
        assert '"Immunotherapy"[MeSH Terms]' in result
        # Both must be required
        assert result.count("AND") == 2   # (query) AND mesh1 AND mesh2

    def test_mesh_terms_with_extra_spaces(self):
        """Whitespace around comma-separated MeSH values is stripped."""
        result = self._build_query(self._inp(query="x", mesh_terms=" Neoplasms , Genomics "))
        assert '"Neoplasms"[MeSH Terms]' in result
        assert '"Genomics"[MeSH Terms]' in result

    def test_empty_mesh_terms_string_ignored(self):
        result = self._build_query(self._inp(query="x", mesh_terms=""))
        assert result == "x"

    def test_mesh_only_commas_ignored(self):
        result = self._build_query(self._inp(query="x", mesh_terms=",,,"))
        assert result == "x"

    def test_query_stripped_of_leading_whitespace(self):
        result = self._build_query(self._inp(query="  genome  "))
        assert result.startswith("genome") or result.startswith("(genome")


# ===========================================================================
# 2. _build_date_params
# ===========================================================================

class TestBuildDateParams:
    def setup_method(self):
        from tools.pubmed_search import _build_date_params
        self._build_date_params = _build_date_params

    def test_no_dates_returns_empty(self):
        assert self._build_date_params(None, None) == {}

    def test_year_start_only(self):
        params = self._build_date_params(2020, None)
        assert params["datetype"] == "pdat"
        assert params["mindate"] == "2020/01/01"
        assert "3000" in params["maxdate"]

    def test_year_end_only(self):
        params = self._build_date_params(None, 2023)
        assert params["datetype"] == "pdat"
        assert params["maxdate"] == "2023/12/31"
        assert "1000" in params["mindate"]

    def test_both_dates(self):
        params = self._build_date_params(2018, 2022)
        assert params["mindate"] == "2018/01/01"
        assert params["maxdate"] == "2022/12/31"
        assert params["datetype"] == "pdat"


# ===========================================================================
# 3. XML parsers (_parse_year, _parse_abstract, _parse_authors)
# ===========================================================================

import xml.etree.ElementTree as ET


def _article_from_xml(xml_fragment: str) -> ET.Element:
    """Wrap a fragment in <Article> for isolated unit tests."""
    return ET.fromstring(f"<Article>{xml_fragment}</Article>")


class TestParseYear:
    def setup_method(self):
        from tools.pubmed_search import _parse_year
        self._parse_year = _parse_year

    def test_standard_year_element(self):
        art = _article_from_xml("<Journal><JournalIssue><PubDate><Year>2023</Year></PubDate></JournalIssue></Journal>")
        assert self._parse_year(art) == 2023

    def test_medline_date_extracts_first_token(self):
        art = _article_from_xml("<Journal><JournalIssue><PubDate><MedlineDate>2022 Sep-Oct</MedlineDate></PubDate></JournalIssue></Journal>")
        assert self._parse_year(art) == 2022

    def test_medline_date_single_year(self):
        art = _article_from_xml("<Journal><JournalIssue><PubDate><MedlineDate>2019</MedlineDate></PubDate></JournalIssue></Journal>")
        assert self._parse_year(art) == 2019

    def test_epub_article_date_fallback(self):
        # No PubDate, but has ArticleDate
        art = _article_from_xml('<ArticleDate DateType="Electronic"><Year>2024</Year><Month>03</Month></ArticleDate>')
        assert self._parse_year(art) == 2024

    def test_no_date_returns_none(self):
        art = _article_from_xml("<AuthorList></AuthorList>")
        assert self._parse_year(art) is None

    def test_non_numeric_year_does_not_raise(self):
        art = _article_from_xml("<Journal><JournalIssue><PubDate><Year>N/A</Year></PubDate></JournalIssue></Journal>")
        # Should not raise; returns None or falls through to another source
        result = self._parse_year(art)
        assert result is None or isinstance(result, int)


class TestParseAbstract:
    def setup_method(self):
        from tools.pubmed_search import _parse_abstract
        self._parse_abstract = _parse_abstract

    def test_flat_abstract(self):
        art = _article_from_xml("<Abstract><AbstractText>Simple abstract text.</AbstractText></Abstract>")
        assert self._parse_abstract(art) == "Simple abstract text."

    def test_structured_abstract_concatenated(self):
        art = _article_from_xml("""
            <Abstract>
              <AbstractText Label="BACKGROUND">Background info.</AbstractText>
              <AbstractText Label="METHODS">Methods used.</AbstractText>
              <AbstractText Label="RESULTS">Results obtained.</AbstractText>
            </Abstract>
        """)
        result = self._parse_abstract(art)
        assert "BACKGROUND:" in result
        assert "Background info." in result
        assert "METHODS:" in result
        assert "RESULTS:" in result

    def test_missing_abstract_returns_placeholder(self):
        art = _article_from_xml("<AuthorList/>")
        assert self._parse_abstract(art) == "No abstract available."

    def test_empty_abstract_text_returns_placeholder(self):
        art = _article_from_xml("<Abstract><AbstractText></AbstractText></Abstract>")
        assert self._parse_abstract(art) == "No abstract available."

    def test_structured_abstract_skips_empty_parts(self):
        art = _article_from_xml("""
            <Abstract>
              <AbstractText Label="BACKGROUND">Useful text.</AbstractText>
              <AbstractText Label="METHODS"></AbstractText>
            </Abstract>
        """)
        result = self._parse_abstract(art)
        assert "Useful text." in result
        # Empty METHODS should not appear as "METHODS: "
        assert "METHODS:" not in result


class TestParseAuthors:
    def setup_method(self):
        from tools.pubmed_search import _parse_authors
        self._parse_authors = _parse_authors

    def test_standard_first_last(self):
        art = _article_from_xml("""
            <AuthorList>
              <Author><LastName>Smith</LastName><ForeName>Jane</ForeName></Author>
              <Author><LastName>Doe</LastName><ForeName>John</ForeName></Author>
            </AuthorList>
        """)
        authors = self._parse_authors(art)
        assert authors == ["Jane Smith", "John Doe"]

    def test_last_name_only(self):
        art = _article_from_xml("""
            <AuthorList>
              <Author><LastName>Tanaka</LastName></Author>
            </AuthorList>
        """)
        assert self._parse_authors(art) == ["Tanaka"]

    def test_collective_name(self):
        art = _article_from_xml("""
            <AuthorList>
              <Author><CollectiveName>ENCODE Consortium</CollectiveName></Author>
            </AuthorList>
        """)
        assert self._parse_authors(art) == ["ENCODE Consortium"]

    def test_mixed_individual_and_collective(self):
        art = _article_from_xml("""
            <AuthorList>
              <Author><LastName>García</LastName><ForeName>María</ForeName></Author>
              <Author><CollectiveName>Prime Editing Consortium</CollectiveName></Author>
            </AuthorList>
        """)
        authors = self._parse_authors(art)
        assert "María García" in authors
        assert "Prime Editing Consortium" in authors

    def test_empty_author_list(self):
        art = _article_from_xml("<AuthorList/>")
        assert self._parse_authors(art) == []


# ===========================================================================
# 4. _parse_xml
# ===========================================================================

class TestParseXml:
    def setup_method(self):
        from tools.pubmed_search import _parse_xml
        self._parse_xml = _parse_xml

    def test_parses_three_complete_records(self):
        papers = self._parse_xml(EFETCH_XML_3)
        assert len(papers) == 3

    def test_universal_schema_fields_present(self):
        papers = self._parse_xml(EFETCH_XML_3)
        required = {"title","abstract","authors","year","citation_count","doi",
                    "arxiv_id","url","source","pmid","mesh_terms","journal"}
        for p in papers:
            assert required.issubset(p.keys()), f"Missing keys: {required - p.keys()}"

    def test_source_is_pubmed(self):
        papers = self._parse_xml(EFETCH_XML_3)
        assert all(p["source"] == "PubMed" for p in papers)

    def test_citation_count_and_arxiv_id_always_none(self):
        papers = self._parse_xml(EFETCH_XML_3)
        assert all(p["citation_count"] is None for p in papers)
        assert all(p["arxiv_id"] is None for p in papers)

    def test_first_paper_fields(self):
        p = self._parse_xml(EFETCH_XML_3)[0]
        assert p["pmid"] == "37000001"
        assert p["doi"] == "10.1038/nbt.2023.001"
        assert p["year"] == 2023
        assert "Jane Smith" in p["authors"]
        assert len(p["authors"]) == 4
        assert "CRISPR-Cas Systems" in p["mesh_terms"]
        assert p["journal"] == "Nature Biotechnology"
        # PMC URL preferred when available
        assert "PMC1111111" in p["url"]

    def test_second_paper_structured_abstract(self):
        p = self._parse_xml(EFETCH_XML_3)[1]
        assert "BACKGROUND:" in p["abstract"]
        assert "METHODS:" in p["abstract"]
        assert "RESULTS:" in p["abstract"]
        assert "CONCLUSIONS:" in p["abstract"]

    def test_second_paper_medlinedate_year(self):
        p = self._parse_xml(EFETCH_XML_3)[1]
        assert p["year"] == 2022

    def test_second_paper_mixed_authors(self):
        p = self._parse_xml(EFETCH_XML_3)[1]
        assert "María García" in p["authors"]
        assert "Prime Editing Consortium" in p["authors"]

    def test_second_paper_pubmed_url_when_no_pmc(self):
        p = self._parse_xml(EFETCH_XML_3)[1]
        assert p["url"] == "https://pubmed.ncbi.nlm.nih.gov/37000002/"

    def test_third_paper_no_abstract_placeholder(self):
        p = self._parse_xml(EFETCH_XML_3)[2]
        assert p["abstract"] == "No abstract available."

    def test_third_paper_no_mesh_terms(self):
        p = self._parse_xml(EFETCH_XML_3)[2]
        assert p["mesh_terms"] == []

    def test_third_paper_author_last_name_only(self):
        p = self._parse_xml(EFETCH_XML_3)[2]
        assert "Tanaka" in p["authors"]

    def test_malformed_node_skipped_valid_nodes_kept(self):
        """A node without <Article> should be skipped; surrounding papers survive."""
        papers = self._parse_xml(EFETCH_XML_MALFORMED)
        pmids = [p["pmid"] for p in papers]
        assert "37000001" in pmids
        assert "37000002" in pmids
        assert "37000099" not in pmids   # the malformed node
        assert len(papers) == 2

    def test_invalid_xml_returns_empty_list(self):
        papers = self._parse_xml("THIS IS NOT XML <<>>")
        assert papers == []

    def test_doi_url_fallback_when_no_pmid_no_pmc(self):
        xml = """<PubmedArticleSet>
          <PubmedArticle>
            <MedlineCitation>
              <Article>
                <Journal><Title>J</Title><JournalIssue><PubDate><Year>2020</Year></PubDate></JournalIssue></Journal>
                <ArticleTitle>Title only</ArticleTitle>
                <AuthorList/>
              </Article>
            </MedlineCitation>
            <PubmedData>
              <ArticleIdList>
                <ArticleId IdType="doi">10.9999/test.doi</ArticleId>
              </ArticleIdList>
            </PubmedData>
          </PubmedArticle>
        </PubmedArticleSet>"""
        papers = self._parse_xml(xml)
        assert len(papers) == 1
        assert papers[0]["url"] == "https://doi.org/10.9999/test.doi"


# ===========================================================================
# 5. search_pubmed — integration (HTTP mocked)
# ===========================================================================

class TestSearchPubMed:
    """Tests for the main search_pubmed() function with mocked HTTP."""

    def setup_method(self):
        import tools.pubmed_search as pubmed_search
        self.mod = pubmed_search
        # Reset in-process fallback cache before each test
        pubmed_search._fallback_cache.clear()

    @patch(PATCH_THROTTLED_GET)
    def test_happy_path_returns_list_of_dicts(self, mock_get):
        esearch_resp = _mock_json_response(ESEARCH_JSON_3)
        efetch_resp  = _mock_response(EFETCH_XML_3)
        mock_get.side_effect = [esearch_resp, efetch_resp]

        from tools.pubmed_search import search_pubmed
        results = search_pubmed("CRISPR cancer", limit=3)

        assert isinstance(results, list)
        assert len(results) == 3

    @patch(PATCH_THROTTLED_GET)
    def test_limit_respected_trims_extra_results(self, mock_get):
        """If efetch returns more than limit, results are trimmed."""
        esearch_resp = _mock_json_response(ESEARCH_JSON_3)
        efetch_resp  = _mock_response(EFETCH_XML_3)
        mock_get.side_effect = [esearch_resp, efetch_resp]

        from tools.pubmed_search import search_pubmed
        results = search_pubmed("CRISPR cancer", limit=2)
        assert len(results) <= 2

    @patch(PATCH_THROTTLED_GET)
    def test_zero_results_returns_empty_list(self, mock_get):
        esearch_resp = _mock_json_response(ESEARCH_JSON_0)
        mock_get.side_effect = [esearch_resp]

        from tools.pubmed_search import search_pubmed
        results = search_pubmed("nonexistent_drug_xyz123", limit=5)
        assert results == []
        # efetch should never be called when esearch returns 0 PMIDs
        assert mock_get.call_count == 1

    @patch(PATCH_THROTTLED_GET)
    def test_fallback_cache_hit_skips_api(self, mock_get):
        """Second call with same args should hit the fallback cache and not call the API."""
        esearch_resp = _mock_json_response(ESEARCH_JSON_3)
        efetch_resp  = _mock_response(EFETCH_XML_3)
        mock_get.side_effect = [esearch_resp, efetch_resp]

        from tools.pubmed_search import search_pubmed
        results1 = search_pubmed("CRISPR cancer", limit=3)
        results2 = search_pubmed("CRISPR cancer", limit=3)

        # API should only be called once (2 calls: esearch + efetch)
        assert mock_get.call_count == 2
        assert results1 == results2

    @patch(PATCH_THROTTLED_GET)
    def test_mesh_terms_injected_into_query(self, mock_get):
        """MeSH terms should appear in the esearch URL params."""
        esearch_resp = _mock_json_response(ESEARCH_JSON_3)
        efetch_resp  = _mock_response(EFETCH_XML_3)
        mock_get.side_effect = [esearch_resp, efetch_resp]

        from tools.pubmed_search import search_pubmed
        search_pubmed("cancer", mesh_terms="Neoplasms", limit=3)

        esearch_call_kwargs = mock_get.call_args_list[0]
        # The `params` dict passed to _throttled_get contains the query
        params_arg = esearch_call_kwargs[0][1]   # positional arg: (url, params)
        assert "Neoplasms" in params_arg["term"]
        assert "MeSH Terms" in params_arg["term"]

    @patch(PATCH_THROTTLED_GET)
    def test_year_filter_passes_date_params(self, mock_get):
        """year_start / year_end should appear as NCBI date params in esearch."""
        esearch_resp = _mock_json_response(ESEARCH_JSON_3)
        efetch_resp  = _mock_response(EFETCH_XML_3)
        mock_get.side_effect = [esearch_resp, efetch_resp]

        from tools.pubmed_search import search_pubmed
        search_pubmed("cancer", year_start=2020, year_end=2023, limit=3)

        params_arg = mock_get.call_args_list[0][0][1]
        assert params_arg.get("mindate") == "2020/01/01"
        assert params_arg.get("maxdate") == "2023/12/31"
        assert params_arg.get("datetype") == "pdat"

    @patch(PATCH_THROTTLED_GET)
    def test_tool_and_email_included_in_requests(self, mock_get):
        """Every NCBI request must carry tool= and email= params (NCBI policy)."""
        esearch_resp = _mock_json_response(ESEARCH_JSON_3)
        efetch_resp  = _mock_response(EFETCH_XML_3)
        mock_get.side_effect = [esearch_resp, efetch_resp]

        from tools.pubmed_search import search_pubmed
        search_pubmed("cancer", limit=3)

        for call_args in mock_get.call_args_list:
            params = call_args[0][1]
            assert "tool" in params
            assert "email" in params
            assert params["tool"] == "ResearchFlow"

    @patch(PATCH_THROTTLED_GET)
    def test_redis_cache_write_and_read(self, mock_get):
        """With a mocked Redis client, verify SET is called on write and GET on read."""
        esearch_resp = _mock_json_response(ESEARCH_JSON_3)
        efetch_resp  = _mock_response(EFETCH_XML_3)
        mock_get.side_effect = [esearch_resp, efetch_resp]

        import tools.pubmed_search as pubmed_search
        fake_redis = MagicMock()
        fake_redis.get.return_value = None          # first call: cache miss
        fake_redis.setex.return_value = True

        original_redis = pubmed_search._redis
        try:
            pubmed_search._redis = fake_redis
            pubmed_search._fallback_cache.clear()

            from tools.pubmed_search import search_pubmed
            results = search_pubmed("cache write test", limit=3)

            fake_redis.get.assert_called_once()
            fake_redis.setex.assert_called_once()
            stored_json = fake_redis.setex.call_args[0][2]
            assert json.loads(stored_json) == results

        finally:
            pubmed_search._redis = original_redis

    @patch(PATCH_THROTTLED_GET)
    def test_redis_cache_read_hit_skips_api(self, mock_get):
        """When Redis returns a cached value, the API must not be called."""
        import tools.pubmed_search as pubmed_search

        from tools.pubmed_search import search_pubmed, _cache_key
        key = _cache_key("cancer", None, None, None, 3)
        cached_data = [{"title": "Cached Paper", "source": "PubMed"}]

        fake_redis = MagicMock()
        fake_redis.get.return_value = json.dumps(cached_data)

        original_redis = pubmed_search._redis
        try:
            pubmed_search._redis = fake_redis
            result = search_pubmed("cancer", limit=3)
            assert result == cached_data
            mock_get.assert_not_called()
        finally:
            pubmed_search._redis = original_redis

    @patch(PATCH_THROTTLED_GET)
    def test_redis_get_failure_falls_through_to_api(self, mock_get):
        """If Redis.get() raises, we fall through and hit the API — no crash."""
        import tools.pubmed_search as pubmed_search

        esearch_resp = _mock_json_response(ESEARCH_JSON_3)
        efetch_resp  = _mock_response(EFETCH_XML_3)
        mock_get.side_effect = [esearch_resp, efetch_resp]

        fake_redis = MagicMock()
        fake_redis.get.side_effect = Exception("Redis connection dropped")
        fake_redis.setex.return_value = True

        original_redis = pubmed_search._redis
        try:
            pubmed_search._redis = fake_redis
            pubmed_search._fallback_cache.clear()
            from tools.pubmed_search import search_pubmed
            results = search_pubmed("redis fail test", limit=3)
            assert isinstance(results, list)
        finally:
            pubmed_search._redis = original_redis


# ===========================================================================
# 6. 429 retry logic
# ===========================================================================

class TestRetryLogic:
    def setup_method(self):
        import tools.pubmed_search as pubmed_search
        pubmed_search._fallback_cache.clear()

    @patch("time.sleep", return_value=None)
    @patch(PATCH_THROTTLED_GET)
    def test_429_followed_by_success_returns_results(self, mock_get, mock_sleep):
        """
        search_pubmed returns a full result list when esearch and efetch both
        eventually succeed.  The 429 retry loop lives inside _throttled_get and
        is tested in isolation by test_throttled_get_retries_on_429; here we
        just verify that search_pubmed correctly orchestrates the two-step call
        after _throttled_get returns success responses.
        """
        import tools.pubmed_search as pubmed_search
        pubmed_search._fallback_cache.clear()

        esearch_ok = _mock_json_response(ESEARCH_JSON_3)
        efetch_ok  = _mock_response(EFETCH_XML_3)
        mock_get.side_effect = [esearch_ok, efetch_ok]

        from tools.pubmed_search import search_pubmed
        results = search_pubmed("CRISPR cancer", limit=3)
        assert isinstance(results, list)
        assert len(results) == 3

    @patch("time.sleep", return_value=None)
    def test_throttled_get_retries_on_429(self, mock_sleep):
        """
        _throttled_get retries internally on HTTP 429.

        We mock httpx.Client (one level below _throttled_get) so the real retry
        loop runs.  The mock raises 429 twice then returns a success response;
        we assert httpx.Client.get was called exactly three times.
        """
        import httpx
        import tools.pubmed_search as pubmed_search

        request_mock = MagicMock()
        response_429 = MagicMock()
        response_429.status_code = 429
        response_429.headers = {}          # no Retry-After → exponential back-off
        four29_exc = httpx.HTTPStatusError("429", request=request_mock, response=response_429)

        success_resp = _mock_json_response(ESEARCH_JSON_0)
        success_resp.raise_for_status = MagicMock()   # no-op: 200 OK

        with patch("httpx.Client") as mock_client_cls:
            mock_client_instance = MagicMock()
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client_instance)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            # Raise 429 on attempts 1 and 2; succeed on attempt 3
            mock_client_instance.get.side_effect = [four29_exc, four29_exc, success_resp]

            result = pubmed_search._throttled_get(pubmed_search.ESEARCH_URL, {})

        assert result is success_resp
        assert mock_client_instance.get.call_count == 3

    @patch("time.sleep", return_value=None)
    def test_throttled_get_honours_retry_after_header(self, mock_sleep):
        """When Retry-After header is present, sleep duration should match it."""
        import httpx
        import tools.pubmed_search as pubmed_search

        request_mock = MagicMock()
        response_429 = MagicMock()
        response_429.status_code = 429
        response_429.headers = {"Retry-After": "42"}
        four29_exc = httpx.HTTPStatusError("429", request=request_mock, response=response_429)

        # We only test that _throttled_get calls time.sleep(42) on 429 with Retry-After
        success_resp = _mock_json_response(ESEARCH_JSON_0)

        with patch("httpx.Client") as mock_client_cls:
            mock_client_instance = MagicMock()
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client_instance)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            # First call raises 429 with Retry-After, second call succeeds
            mock_client_instance.get.side_effect = [four29_exc, success_resp]
            success_resp.raise_for_status = MagicMock()

            pubmed_search._throttled_get(pubmed_search.ESEARCH_URL, {})

        # time.sleep should have been called with 42 from Retry-After
        sleep_calls = [c[0][0] for c in mock_sleep.call_args_list]
        assert 42 in sleep_calls


# ===========================================================================
# 7. _tool_fn and pubmed_tool (LangChain interface)
# ===========================================================================

class TestToolFn:
    def setup_method(self):
        import tools.pubmed_search as pubmed_search
        pubmed_search._fallback_cache.clear()

    @patch(PATCH_THROTTLED_GET)
    def test_tool_fn_returns_string(self, mock_get):
        mock_get.side_effect = [
            _mock_json_response(ESEARCH_JSON_3),
            _mock_response(EFETCH_XML_3),
        ]
        from tools.pubmed_search import _tool_fn
        import tools.pubmed_search as pubmed_search
        pubmed_search._fallback_cache.clear()
        result = _tool_fn("CRISPR cancer", limit=3)
        assert isinstance(result, str)

    @patch(PATCH_THROTTLED_GET)
    def test_tool_fn_contains_pubmed_header(self, mock_get):
        mock_get.side_effect = [
            _mock_json_response(ESEARCH_JSON_3),
            _mock_response(EFETCH_XML_3),
        ]
        import tools.pubmed_search as pubmed_search
        pubmed_search._fallback_cache.clear()
        from tools.pubmed_search import _tool_fn
        result = _tool_fn("CRISPR cancer", limit=3)
        assert "PubMed results" in result

    @patch(PATCH_THROTTLED_GET)
    def test_tool_fn_contains_paper_fields(self, mock_get):
        mock_get.side_effect = [
            _mock_json_response(ESEARCH_JSON_3),
            _mock_response(EFETCH_XML_3),
        ]
        import tools.pubmed_search as pubmed_search
        pubmed_search._fallback_cache.clear()
        from tools.pubmed_search import _tool_fn
        result = _tool_fn("CRISPR cancer", limit=3)
        # Should contain title, PMID, DOI, journal, MeSH, abstract snippet
        assert "Nature Biotechnology" in result
        assert "37000001" in result
        assert "10.1038/nbt.2023.001" in result
        assert "CRISPR-Cas Systems" in result

    @patch(PATCH_THROTTLED_GET)
    def test_tool_fn_authors_truncated_with_et_al(self, mock_get):
        """Authors beyond 3 should show 'et al.' in the formatted string."""
        mock_get.side_effect = [
            _mock_json_response(ESEARCH_JSON_3),
            _mock_response(EFETCH_XML_3),
        ]
        import tools.pubmed_search as pubmed_search
        pubmed_search._fallback_cache.clear()
        from tools.pubmed_search import _tool_fn
        result = _tool_fn("CRISPR cancer", limit=1)
        # First paper has 4 authors, so should show et al.
        assert "et al." in result

    @patch(PATCH_THROTTLED_GET)
    def test_tool_fn_zero_results_returns_no_results_string(self, mock_get):
        mock_get.side_effect = [_mock_json_response(ESEARCH_JSON_0)]
        import tools.pubmed_search as pubmed_search
        pubmed_search._fallback_cache.clear()
        from tools.pubmed_search import _tool_fn
        result = _tool_fn("nonexistent_xyzabc", limit=3)
        assert "No results found" in result

    @patch(PATCH_THROTTLED_GET)
    def test_langchain_tool_invoke(self, mock_get):
        """pubmed_tool.invoke() should work end-to-end like _tool_fn."""
        mock_get.side_effect = [
            _mock_json_response(ESEARCH_JSON_3),
            _mock_response(EFETCH_XML_3),
        ]
        import tools.pubmed_search as pubmed_search
        pubmed_search._fallback_cache.clear()
        from tools.pubmed_search import pubmed_tool
        result = pubmed_tool.invoke({"query": "CRISPR cancer", "limit": 3})
        assert isinstance(result, str)
        assert "PubMed results" in result

    def test_tool_name_and_description(self):
        from tools.pubmed_search import pubmed_tool
        assert pubmed_tool.name == "pubmed_search"
        assert "PubMed" in pubmed_tool.description
        assert "biomedical" in pubmed_tool.description.lower()

    def test_tool_schema_has_correct_fields(self):
        from tools.pubmed_search import PubMedInput
        fields = PubMedInput.model_fields
        assert "query" in fields
        assert "year_start" in fields
        assert "year_end" in fields
        assert "mesh_terms" in fields
        assert "limit" in fields


# ===========================================================================
# 8. PubMedInput schema validation
# ===========================================================================

class TestPubMedInputSchema:
    def setup_method(self):
        from tools.pubmed_search import PubMedInput, MAX_RESULTS_CAP
        self.PubMedInput = PubMedInput
        self.MAX_RESULTS_CAP = MAX_RESULTS_CAP

    def test_default_limit_is_5(self):
        inp = self.PubMedInput(query="cancer")
        assert inp.limit == 5

    def test_limit_below_1_raises(self):
        with pytest.raises(Exception):
            self.PubMedInput(query="cancer", limit=0)

    def test_limit_above_cap_raises(self):
        with pytest.raises(Exception):
            self.PubMedInput(query="cancer", limit=self.MAX_RESULTS_CAP + 1)

    def test_limit_at_cap_is_valid(self):
        inp = self.PubMedInput(query="cancer", limit=self.MAX_RESULTS_CAP)
        assert inp.limit == self.MAX_RESULTS_CAP

    def test_optional_fields_default_to_none(self):
        inp = self.PubMedInput(query="cancer")
        assert inp.year_start is None
        assert inp.year_end is None
        assert inp.mesh_terms is None

    def test_query_is_required(self):
        with pytest.raises(Exception):
            self.PubMedInput()


# ===========================================================================
# 9. clear_cache and cache_stats
# ===========================================================================

class TestCacheManagement:
    def setup_method(self):
        import tools.pubmed_search as pubmed_search
        pubmed_search._fallback_cache.clear()

    def test_clear_cache_fallback_empties_dict(self):
        import tools.pubmed_search as pubmed_search
        pubmed_search._fallback_cache["test_key"] = [{"title": "test"}]
        pubmed_search.clear_cache()
        assert len(pubmed_search._fallback_cache) == 0

    def test_cache_stats_fallback_reports_correct_count(self):
        import tools.pubmed_search as pubmed_search
        pubmed_search._fallback_cache["k1"] = []
        pubmed_search._fallback_cache["k2"] = []
        stats = pubmed_search.cache_stats()
        assert stats["backend"].startswith("in-process")
        assert stats["entries"] == 2

    def test_cache_stats_fallback_lists_keys(self):
        import tools.pubmed_search as pubmed_search
        pubmed_search._fallback_cache["k1"] = []
        stats = pubmed_search.cache_stats()
        assert "k1" in stats["keys"]

    def test_clear_cache_redis_mode(self):
        """In Redis mode, clear_cache should call SCAN + DELETE."""
        import tools.pubmed_search as pubmed_search

        fake_redis = MagicMock()
        # SCAN returns cursor=0 on first call (single batch), with two matching keys
        fake_redis.scan.return_value = (0, ["researchflow:pubmed:abc", "researchflow:pubmed:def"])
        fake_redis.delete.return_value = 2

        original_redis = pubmed_search._redis
        try:
            pubmed_search._redis = fake_redis
            pubmed_search.clear_cache()
            fake_redis.scan.assert_called()
            fake_redis.delete.assert_called()
        finally:
            pubmed_search._redis = original_redis

    def test_cache_stats_redis_mode(self):
        import tools.pubmed_search as pubmed_search

        fake_redis = MagicMock()
        fake_redis.scan.return_value = (0, ["researchflow:pubmed:abc"])
        fake_redis.ttl.return_value = 86400

        original_redis = pubmed_search._redis
        try:
            pubmed_search._redis = fake_redis
            stats = pubmed_search.cache_stats()
            assert stats["backend"] == "redis"
            assert stats["entries"] == 1
            assert stats["key_prefix"] == pubmed_search.CACHE_KEY_PREFIX
        finally:
            pubmed_search._redis = original_redis

    def test_cache_stats_redis_error_returns_error_dict(self):
        import tools.pubmed_search as pubmed_search

        fake_redis = MagicMock()
        fake_redis.scan.side_effect = Exception("Redis down")

        original_redis = pubmed_search._redis
        try:
            pubmed_search._redis = fake_redis
            stats = pubmed_search.cache_stats()
            assert stats["backend"] == "redis"
            assert "error" in stats
        finally:
            pubmed_search._redis = original_redis

    def test_clear_cache_redis_error_does_not_raise(self):
        import tools.pubmed_search as pubmed_search

        fake_redis = MagicMock()
        fake_redis.scan.side_effect = Exception("Redis down")

        original_redis = pubmed_search._redis
        try:
            pubmed_search._redis = fake_redis
            pubmed_search.clear_cache()   # must not raise
        finally:
            pubmed_search._redis = original_redis


# ===========================================================================
# 10. _cache_key determinism
# ===========================================================================

class TestCacheKey:
    def setup_method(self):
        from tools.pubmed_search import _cache_key, CACHE_KEY_PREFIX
        self._cache_key = _cache_key
        self.prefix = CACHE_KEY_PREFIX

    def test_same_args_produce_same_key(self):
        k1 = self._cache_key("cancer", 2020, 2023, "Neoplasms", 5)
        k2 = self._cache_key("cancer", 2020, 2023, "Neoplasms", 5)
        assert k1 == k2

    def test_different_query_different_key(self):
        k1 = self._cache_key("cancer", None, None, None, 5)
        k2 = self._cache_key("diabetes", None, None, None, 5)
        assert k1 != k2

    def test_different_limit_different_key(self):
        k1 = self._cache_key("cancer", None, None, None, 3)
        k2 = self._cache_key("cancer", None, None, None, 5)
        assert k1 != k2

    def test_key_has_correct_prefix(self):
        k = self._cache_key("cancer", None, None, None, 5)
        assert k.startswith(self.prefix)

    def test_key_has_no_whitespace(self):
        k = self._cache_key("multi word query here", 2020, 2023, "Neoplasms,Genomics", 5)
        assert " " not in k