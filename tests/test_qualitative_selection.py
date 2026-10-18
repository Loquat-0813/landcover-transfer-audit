import unittest

from p1eval.qualitative_selection import select_oem_qualitative_identifiers


class QualitativeSelectionTest(unittest.TestCase):
    def test_selects_distinct_maximal_examples(self):
        names = ("built", "herbaceous_vegetation", "cropland")
        b0 = {
            "a": [[0, 0, 0], [0, 10, 90], [0, 0, 100]],
            "b": [[0, 0, 0], [0, 50, 50], [0, 0, 100]],
            "c": [[0, 0, 0], [0, 50, 50], [0, 0, 100]],
        }
        b1 = {
            "a": [[0, 0, 0], [0, 90, 10], [0, 0, 100]],
            "b": [[0, 0, 0], [0, 50, 50], [0, 90, 10]],
            "c": [[0, 0, 0], [0, 55, 45], [0, 0, 100]],
        }
        result = select_oem_qualitative_identifiers(b0, b1, names, min_reference_pixels=1)
        self.assertEqual(result["selected"]["rangeland_recovery"]["identifier"], "a")
        self.assertEqual(result["selected"]["cropland_tradeoff"]["identifier"], "b")
        self.assertEqual(result["selection_status"], "post_hoc_illustrative_only")

    def test_rejects_unpaired_inputs(self):
        with self.assertRaises(ValueError):
            select_oem_qualitative_identifiers({"a": [[1, 0], [0, 1]]}, {}, ("herbaceous_vegetation", "cropland"))
