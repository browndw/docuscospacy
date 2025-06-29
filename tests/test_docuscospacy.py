# This code is a test suite for the docuscospacy library,
# which provides functions for analyzing text corpora using spaCy.

import unittest
import polars as pl
import spacy
import docuscospacy as ds
from docuscospacy.validation import ModelValidationError


class TestCorpusAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the spaCy model once for all tests
        try:
            cls.nlp = spacy.load("en_docusco_spacy")
        except Exception:
            cls.nlp = None  # Skip tests if model is not available

        # Create a simple corpus DataFrame
        cls.corpus = pl.DataFrame(
            {
                "doc_id": ["doc1", "doc2"],
                "text": ["This is a test.", "Another test document."],
            }
        )

    def setUp(self):
        if self.nlp is None:
            self.skipTest("spaCy model 'en_docusco_spacy' not available")

    def test_docuscope_parse_with_invalid_model(self):
        # Test with an invalid model
        invalid_nlp = spacy.blank("en")
        with self.assertRaises(ModelValidationError):
            ds.docuscope_parse(self.corpus, invalid_nlp)

    def test_docuscope_parse_skips_none_text(self):
        # Corpus with one None text value
        corpus_with_none = pl.DataFrame(
            {"doc_id": ["doc1", "doc2"], "text": [None, "Another test document."]}
        )
        df = ds.docuscope_parse(corpus_with_none, self.nlp)
        # Only doc2 should be present in the output
        self.assertTrue((df["doc_id"] == "doc2").all())

    def test_docuscope_parse(self):
        df = ds.docuscope_parse(self.corpus, self.nlp)
        self.assertIsInstance(df, pl.DataFrame)
        self.assertIn("token", df.columns)
        self.assertIn("pos_tag", df.columns)
        self.assertIn("ds_tag", df.columns)

    def test_frequency_table(self):
        tokens_table = ds.docuscope_parse(self.corpus, self.nlp)
        freq_df = ds.frequency_table(tokens_table, count_by="pos")
        self.assertIsInstance(freq_df, pl.DataFrame)
        self.assertIn("Token", freq_df.columns)
        self.assertIn("AF", freq_df.columns)
        self.assertIn("RF", freq_df.columns)

    def test_tags_table(self):
        tokens_table = ds.docuscope_parse(self.corpus, self.nlp)
        tags_df = ds.tags_table(tokens_table, count_by="pos")
        self.assertIsInstance(tags_df, pl.DataFrame)
        self.assertIn("Tag", tags_df.columns)
        self.assertIn("AF", tags_df.columns)
        self.assertIn("RF", tags_df.columns)

    def test_dispersions_table(self):
        tokens_table = ds.docuscope_parse(self.corpus, self.nlp)
        disp_df = ds.dispersions_table(tokens_table, count_by="pos")
        self.assertIsInstance(disp_df, pl.DataFrame)
        self.assertIn("Token", disp_df.columns)
        self.assertIn("Carrolls_D2", disp_df.columns)

    def test_tags_dtm(self):
        tokens_table = ds.docuscope_parse(self.corpus, self.nlp)
        dtm_df = ds.tags_dtm(tokens_table, count_by="pos")
        self.assertIsInstance(dtm_df, pl.DataFrame)
        self.assertIn("doc_id", dtm_df.columns)
        self.assertIn("NN1", dtm_df.columns)
        self.assertIn("DD1", dtm_df.columns)
        self.assertIn("AT1", dtm_df.columns)
        self.assertIn("VBZ", dtm_df.columns)

    def test_ngrams(self):
        tokens_table = ds.docuscope_parse(self.corpus, self.nlp)
        # Use a small span and low min_frequency for test coverage
        ngram_df = ds.ngrams(
            tokens_table, span=2, min_frequency=0, count_by="pos"
        )  # noqa: E501
        self.assertIsInstance(ngram_df, pl.DataFrame)
        # Check that expected columns exist
        self.assertIn("Token_1", ngram_df.columns)
        self.assertIn("Token_2", ngram_df.columns)
        self.assertIn("AF", ngram_df.columns)
        self.assertIn("RF", ngram_df.columns)

    def test_clusters_by_token(self):
        tokens_table = ds.docuscope_parse(self.corpus, self.nlp)
        # Use a token from your test corpus, e.g., "test"
        result = ds.clusters_by_token(
            tokens_table, node_word="test", span=2, count_by="pos"
        )  # noqa: E501
        # Should always return a DataFrame, even if empty
        self.assertIsInstance(result, pl.DataFrame)
        # If not empty, check for expected columns
        if result.height > 0:
            self.assertIn("Token_1", result.columns)
            self.assertIn("Token_2", result.columns)
            self.assertIn("AF", result.columns)
            self.assertIn("RF", result.columns)

    def test_clusters_by_tag(self):
        tokens_table = ds.docuscope_parse(self.corpus, self.nlp)
        # Use a tag from your test corpus, e.g., "NN1" (adjust as needed)
        result = ds.clusters_by_tag(
            tokens_table, tag="NN1", span=2, count_by="pos"
        )  # noqa: E501
        self.assertIsInstance(result, pl.DataFrame)
        # If not empty, check for expected columns
        if result.height > 0:
            self.assertIn("Token_1", result.columns)
            self.assertIn("Token_2", result.columns)
            self.assertIn("AF", result.columns)
            self.assertIn("RF", result.columns)

    def test_kwic_center_node(self):
        tokens_table = ds.docuscope_parse(self.corpus, self.nlp)
        kwic_df = ds.kwic_center_node(tokens_table, node_word="test")
        self.assertIsInstance(kwic_df, pl.DataFrame)
        # If not empty, check for expected columns
        if kwic_df.height > 0:
            self.assertIn("Doc ID", kwic_df.columns)
            self.assertIn("Pre-Node", kwic_df.columns)
            self.assertIn("Node", kwic_df.columns)
            self.assertIn("Post-Node", kwic_df.columns)

    def test_coll_table(self):
        tokens_table = ds.docuscope_parse(self.corpus, self.nlp)
        coll_df = ds.coll_table(
            tokens_table, node_word="test", statistic="npmi", count_by="pos"
        )  # noqa: E501
        self.assertIsInstance(coll_df, pl.DataFrame)
        # If not empty, check for expected columns
        if coll_df.height > 0:
            self.assertIn("Token", coll_df.columns)
            self.assertIn("Tag", coll_df.columns)
            self.assertIn("Freq Span", coll_df.columns)
            self.assertIn("Freq Total", coll_df.columns)
            self.assertIn("MI", coll_df.columns)

    def test_keyness_table(self):
        # Target corpus (already in setUpClass as self.corpus)
        # Reference corpus: different texts
        reference_corpus = pl.DataFrame(
            {
                "doc_id": ["ref1", "ref2"],
                "text": ["Reference document one.", "Reference document two."],
            }
        )
        # Parse both corpora
        target_tokens = ds.docuscope_parse(self.corpus, self.nlp)
        reference_tokens = ds.docuscope_parse(reference_corpus, self.nlp)
        # Generate frequency tables
        target_freq = ds.frequency_table(target_tokens, count_by="pos")
        reference_freq = ds.frequency_table(reference_tokens, count_by="pos")
        # Run keyness_table
        keyness_df = ds.keyness_table(target_freq, reference_freq)
        self.assertIsInstance(keyness_df, pl.DataFrame)
        # If not empty, check for expected columns
        if keyness_df.height > 0:
            self.assertIn("Token", keyness_df.columns)
            self.assertIn("Tag", keyness_df.columns)
            self.assertIn("LL", keyness_df.columns)
            self.assertIn("LR", keyness_df.columns)
            self.assertIn("PV", keyness_df.columns)
            self.assertIn("RF", keyness_df.columns)
            self.assertIn("RF_Ref", keyness_df.columns)
            self.assertIn("AF", keyness_df.columns)
            self.assertIn("AF_Ref", keyness_df.columns)
            self.assertIn("Range", keyness_df.columns)
            self.assertIn("Range_Ref", keyness_df.columns)

    def test_dtm_weight(self):
        tokens_table = ds.docuscope_parse(self.corpus, self.nlp)
        dtm = ds.tags_dtm(tokens_table, count_by="pos")[0]  # get pos DTM
        weighted = ds.dtm_weight(dtm, scheme="prop")
        self.assertIsInstance(weighted, pl.DataFrame)
        self.assertIn("doc_id", weighted.columns)
        # Should have same columns as dtm
        self.assertEqual(set(weighted.columns), set(dtm.columns))

    def test_dtm_simplify(self):
        tokens_table = ds.docuscope_parse(self.corpus, self.nlp)
        dtm = ds.tags_dtm(tokens_table, count_by="pos")[0]
        simple_dtm = ds.dtm_simplify(dtm)
        self.assertIsInstance(simple_dtm, pl.DataFrame)
        self.assertIn("doc_id", simple_dtm.columns)
        # Should have fewer columns than original dtm (aggregated tags)
        self.assertLess(len(simple_dtm.columns), len(dtm.columns))

    def test_freq_simplify(self):
        tokens_table = ds.docuscope_parse(self.corpus, self.nlp)
        freq = ds.frequency_table(tokens_table, count_by="pos")
        simple_freq = ds.freq_simplify(freq)
        self.assertIsInstance(simple_freq, pl.DataFrame)
        self.assertIn("Tag", simple_freq.columns)
        # Check that all tags are from the simplified set (no raw POS prefixes)
        simplified_prefixes = [
            "NounCommon",
            "VerbLex",
            "Adjective",
            "Adverb",
            "Pronoun",
            "Preposition",
            "Conjunction",
            "NounOther",
            "VerbBe",
            "VerbOther",
            "Other",  # noqa: E501
        ]
        for tag in simple_freq["Tag"].to_list():
            self.assertTrue(
                any(tag.startswith(prefix) for prefix in simplified_prefixes)
            )  # noqa: E501

    def test_direct_import_micusp_mini(self):
        import polars as pl
        from docuscospacy.data import micusp_mini

        self.assertIsInstance(micusp_mini, pl.DataFrame)
        self.assertIn("doc_id", micusp_mini.columns)
        self.assertIn("text", micusp_mini.columns)


if __name__ == "__main__":
    unittest.main()
