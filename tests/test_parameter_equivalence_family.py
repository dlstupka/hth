import unittest

from hth.regression.parameter_space import (
    EQUIVALENCE_FAMILY_SENTINEL,
    parameter_set_equivalence_family_id,
    parameter_set_equivalence_family_payload,
    parameter_set_equivalence_family_size,
    parameter_set_id,
)
from hth.regression.parameter_provenance import attach_identity, grid_definition


class ParameterEquivalenceFamilyTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "profiles": {"baseline": {"live": 1, "dead": 0}},
            "parameters": {"live": {"type": "int", "values": [1, 2]}},
            "zombie_parameters": {"dead": {"type": "int", "values": [0, 1], "pinned_value": 0}},
            "equivalence_parameters": ["dead"],
        }

    def test_exact_ids_differ_but_family_is_invariant_across_enrolled_dimension(self):
        a = {"live": 2, "dead": 0}
        b = {"live": 2, "dead": 1}
        self.assertNotEqual(parameter_set_id(a), parameter_set_id(b))
        self.assertEqual(parameter_set_equivalence_family_id(a, self.config), parameter_set_equivalence_family_id(b, self.config))
        self.assertEqual(parameter_set_equivalence_family_size(self.config), 2)

    def test_sentinel_is_hash_payload_only(self):
        executable = {"live": 2, "dead": 1}
        payload = parameter_set_equivalence_family_payload(executable, self.config)
        self.assertEqual(payload["dead"], EQUIVALENCE_FAMILY_SENTINEL)
        self.assertEqual(executable["dead"], 1)

    def test_live_grid_includes_baseline_pinned_zombie_dimension(self):
        grid = grid_definition(self.config)
        self.assertEqual(grid["cartesian_count"], 2)
        self.assertEqual(grid["values"]["dead"], [0])
        result = {"parameters": {"live": 2, "dead": 0}}
        attach_identity(result, "test", self.config)
        self.assertIsNotNone(result["parameter_grid_ordinal"])
        self.assertEqual(result["parameter_set_equivalence_family_size"], 2)

    def test_zombie_inclusive_grid_restores_retained_domain(self):
        grid = grid_definition(self.config, include_zombies=True)
        self.assertEqual(grid["cartesian_count"], 4)
        self.assertEqual(grid["values"]["dead"], [0, 1])


if __name__ == "__main__":
    unittest.main()
