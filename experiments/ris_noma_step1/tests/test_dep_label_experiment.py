"""The extra margin applies to the validation DEP bound after ordinary checks."""
import unittest

import numpy as np

try:
    from ris_noma.dep_label_experiment import choose_labels
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    choose_labels = None


@unittest.skipIf(choose_labels is None, "Install the step 6b environment")
class TestDEPLabelSelection(unittest.TestCase):
    def test_slack_filters_one_pool_before_ranking_by_label_bob_rate(self):
        config = {"requirements": {"qos_public": [.1, .1], "outage_bob_max": .1},
                  "bounds": {"mb": [.05, .49], "s": [.05, .95]}}
        proposals, assessments = [], {}
        for name, rate, lower, outage in [
            ("fast", 4.4, .903, .05), ("buffered", 4.2, .912, .05),
            ("wider", 4.1, .921, .05), ("bad_outage", 5., .99, .2),
        ]:
            proposals.append({"proposal_id": name, "seed": 1, "xi": .9, "gamma_target": .92, "label_feasible": True,
                              "label_metrics": {"dep_min": .925, "bob_mean_rate_bit_s_hz": rate,
                                                "allocation_sic_order": {"mb": .2, "public": [.35, .45], "s": .5}}})
            assessments[name] = {
                "metrics": {"dep_min": .907, "outage_ceiling_margins_positive": True,
                            "constraints": {"checks": {"layout_bob_mean_gain_rule": True}}},
                "screen": {"public_rate_lower_sic_order": [.5, .5], "bob_outage_upper": outage, "dep_uniform_lower": lower},
            }
        labels = choose_labels(config, np.array([1, 0]), proposals, assessments, .9, [0., .005, .01, .02, .03])
        self.assertEqual([row["proposal_id"] if row else None for row in labels],
                         ["fast", "buffered", "buffered", "wider", None])
        np.testing.assert_allclose(labels[1]["allocation"], [.2, .45, .35, .5])
