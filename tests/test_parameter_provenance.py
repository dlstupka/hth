import json
import tempfile
import unittest
from pathlib import Path

from hth.regression.parameter_provenance import (
    attach_identity,
    build_provenance,
    grid_ordinal,
    parameter_identity_sha256,
    parameters_from_ordinal,
    provenance_from_legacy_parameters,
    resolve_parameter_set,
)
from hth.regression.parameter_space import parameter_set_id


CONFIG = {
    "detector": "example",
    "parameters": {
        "a": {"values": [1, 2]},
        "b": {"values": [0.1, 0.2, 0.3]},
    },
    "profiles": {"baseline": {"a": 1, "b": 0.1}},
}


class ParameterProvenanceTests(unittest.TestCase):
    def test_legacy_id_is_preserved_but_full_identity_is_namespaced(self):
        parameters = {"a": 2, "b": 0.3}
        result = {"parameter_set_id": parameter_set_id(parameters), "parameters": parameters}
        attach_identity(result, "example", CONFIG)
        self.assertEqual(result["parameter_set_id"], parameter_set_id(parameters))
        self.assertEqual(len(result["parameter_identity_sha256"]), 64)
        other = parameter_identity_sha256("other", parameters, schema_version="1")
        self.assertNotEqual(result["parameter_identity_sha256"], other)

    def test_grid_ordinal_round_trip(self):
        params = {"a": 2, "b": 0.2}
        ordinal = grid_ordinal(CONFIG, params)
        provenance = build_provenance("example", CONFIG, [], strategy="exhaustive", complete_cartesian=True)
        self.assertEqual(parameters_from_ordinal(provenance, ordinal), params)

    def test_grid_registry_does_not_store_every_cartesian_definition(self):
        results = []
        for a in [1, 2]:
            for b in [0.1, 0.2, 0.3]:
                results.append({"parameter_set_id": parameter_set_id({"a": a, "b": b}), "parameters": {"a": a, "b": b}})
        provenance = build_provenance("example", CONFIG, results, strategy="exhaustive", complete_cartesian=True)
        self.assertEqual(provenance["grid"]["cartesian_count"], 6)
        self.assertEqual(provenance["explicit_parameter_sets"], {})
        self.assertEqual(provenance["coverage"]["evaluated_grid_parameter_sets"], 6)

    def test_non_grid_result_is_stored_explicitly(self):
        result = {"parameter_set_id": parameter_set_id({"a": 7, "b": 0.2}), "parameters": {"a": 7, "b": 0.2}}
        provenance = build_provenance("example", CONFIG, [result], strategy="binary-refine", complete_cartesian=False)
        self.assertEqual(len(provenance["explicit_parameter_sets"]), 1)
        resolved = resolve_parameter_set(provenance, result["parameter_set_id"])
        self.assertEqual(resolved["parameters"], {"a": 7, "b": 0.2})

    def test_legacy_twelve_character_id_resolves_from_grid(self):
        params = {"a": 2, "b": 0.3}
        provenance = build_provenance("example", CONFIG, [], strategy="legacy", complete_cartesian=False)
        resolved = resolve_parameter_set(provenance, parameter_set_id(params))
        self.assertEqual(resolved["parameters"], params)
        self.assertEqual(resolved["source"], "grid")

    def test_old_parameters_json_can_be_used_as_legacy_resolver(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "parameters.json"
            path.write_text(json.dumps({"detector": "example", "strategy": "exhaustive", "configuration": CONFIG}), encoding="utf-8")
            provenance = provenance_from_legacy_parameters(path)
            params = {"a": 1, "b": 0.2}
            resolved = resolve_parameter_set(provenance, parameter_set_id(params))
            self.assertEqual(resolved["parameters"], params)

    def test_python_json_type_difference_remains_identifiable(self):
        i = parameter_identity_sha256("example", {"x": 1}, schema_version="1")
        f = parameter_identity_sha256("example", {"x": 1.0}, schema_version="1")
        self.assertNotEqual(i, f)


if __name__ == "__main__":
    unittest.main()
