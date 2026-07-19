"""Unit tests for the pure helpers in review_core (no API key needed).

Run with:  python -m pytest test_review_core.py   (or)   python -m unittest
"""

import unittest

import review_core as rc


class TestJson(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(rc.parse_json('{"a": 1}'), {"a": 1})

    def test_fenced(self):
        self.assertEqual(rc.parse_json('```json\n{"a": 1}\n```'), {"a": 1})

    def test_embedded(self):
        self.assertEqual(rc.parse_json('here you go: {"a": 1} thanks'), {"a": 1})

    def test_garbage(self):
        self.assertEqual(rc.parse_json("not json at all"), {})

    def test_none(self):
        self.assertEqual(rc.parse_json(None), {})


class TestNormalize(unittest.TestCase):
    def test_decision(self):
        self.assertEqual(rc.normalize_decision("INCLUDE"), "include")
        self.assertEqual(rc.normalize_decision("should exclude"), "exclude")
        self.assertEqual(rc.normalize_decision("dunno"), "unclear")
        self.assertEqual(rc.normalize_decision(None), "unclear")

    def test_confidence(self):
        self.assertEqual(rc.normalize_confidence(0.5), 0.5)
        self.assertEqual(rc.normalize_confidence(85), 0.85)   # percentage
        self.assertEqual(rc.normalize_confidence("bad"), 0.0)
        self.assertEqual(rc.normalize_confidence(150), 1.0)   # 150% -> clamped to 1.0


class TestRis(unittest.TestCase):
    SAMPLE = (
        "TY  - JOUR\n"
        "TI  - A trial of widget therapy\n"
        "AU  - Smith, J\n"
        "AU  - Doe, A\n"
        "PY  - 2021\n"
        "AB  - Background: widgets.\n"
        "AB  - Methods: RCT.\n"
        "ER  - \n"
        "TY  - JOUR\n"
        "T1  - Second paper\n"
        "PY  - 2019\n"
        "ER  - \n"
    )

    def test_count(self):
        recs = rc.parse_ris(self.SAMPLE, source="f.ris")
        self.assertEqual(len(recs), 2)

    def test_fields(self):
        recs = rc.parse_ris(self.SAMPLE)
        self.assertEqual(recs[0].title, "A trial of widget therapy")
        self.assertEqual(recs[0].year, "2021")
        self.assertIn("Smith, J", recs[0].authors)
        self.assertIn("Doe, A", recs[0].authors)
        self.assertIn("widgets", recs[0].abstract)
        self.assertEqual(recs[1].title, "Second paper")


class TestCsv(unittest.TestCase):
    def test_headers(self):
        csv_text = "Title,Abstract,Year,Authors\nMy Study,Some abstract,2020,Roe R\n"
        recs = rc.parse_citation_csv(csv_text, source="x.csv")
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].title, "My Study")
        self.assertEqual(recs[0].year, "2020")
        self.assertEqual(recs[0].abstract, "Some abstract")


class TestSchema(unittest.TestCase):
    def test_parse(self):
        fields = rc.parse_schema(
            "sample size | number | Total N\n"
            "study_design | enum(RCT, cohort) |\n"
            "# a comment\n"
            "notes | text |\n"
        )
        self.assertEqual(len(fields), 3)
        self.assertEqual(fields[0].name, "sample_size")  # spaces -> underscore
        self.assertEqual(fields[0].type, "number")
        self.assertEqual(fields[1].type, "enum")
        self.assertEqual(fields[1].values, ["RCT", "cohort"])

    def test_prompt_block(self):
        fields = rc.parse_schema("x | enum(a, b) |")
        block = rc.schema_to_prompt_block(fields)
        self.assertIn("one of: a, b", block)


class TestNeedsReview(unittest.TestCase):
    def test_low_confidence(self):
        res = {"decision": "include", "confidence": 0.4}
        self.assertTrue(rc.needs_review(res, 0.7))

    def test_exclude_flagged(self):
        res = {"decision": "exclude", "confidence": 0.99}
        self.assertTrue(rc.needs_review(res, 0.7, review_excludes=True))

    def test_confident_include_ok(self):
        res = {"decision": "include", "confidence": 0.99}
        self.assertFalse(rc.needs_review(res, 0.7))


class TestNormalizeExtraction(unittest.TestCase):
    def test_shape(self):
        fields = rc.parse_schema("age | number |\nsex | text |")
        parsed = {"fields": {"age": {"value": 42, "source_quote": "aged 42"}, "sex": "F"}}
        out = rc.normalize_extraction(parsed, fields)
        self.assertEqual(out["age"]["value"], "42")
        self.assertEqual(out["age"]["source_quote"], "aged 42")
        self.assertEqual(out["sex"]["value"], "F")


if __name__ == "__main__":
    unittest.main()
