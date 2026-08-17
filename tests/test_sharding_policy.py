import contextlib
import io
import unittest
from pathlib import Path
from hth.regression.runner import parse_args
ROOT=Path(__file__).resolve().parents[1]
class ShardingPolicyTests(unittest.TestCase):
    def test_regression_field_location_and_default(self):
        s=(ROOT/'.github/workflows/regress-detector.yml').read_text(); self.assertLess(s.index('      manual_execution_shape:'),s.index('      sharding:')); self.assertLess(s.index('      sharding:'),s.index('      runner:')); self.assertIn('default: auto',s[s.index('      sharding:'):s.index('      runner:')]); self.assertNotIn('      shards:',s)
    def test_optimizer_field_location_and_default(self):
        s=(ROOT/'.github/workflows/execution-optimizer.yml').read_text(); self.assertLess(s.index('      early_stop:'),s.index('      sharding:')); self.assertLess(s.index('      sharding:'),s.index('      resume:')); self.assertIn('default: "1"',s[s.index('      sharding:'):s.index('      resume:')])
    def test_driver_contract(self):
        s=(ROOT/'tools/run-detector-regressions.sh').read_text(); self.assertIn('sharding_policy="${SHARDING:-auto}"',s); self.assertIn('planned_shards=$((shard_pipeline_count * sharding_policy))',s)
    def test_critical_can_shard(self):
        a=parse_args(['--detector-config','d.json','--golden-set','g.json','--image-root','i','--output','o','--strategy','critical','--shard-count','2']); self.assertEqual(a.shard_count,2)
    def test_binary_refine_cannot_shard(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                parse_args(['--detector-config','d.json','--golden-set','g.json','--image-root','i','--output','o','--strategy','binary-refine','--shard-count','2'])
        self.assertIn('binary-refine cannot be sharded because its search path is sequential', stderr.getvalue())
    def test_dispatch_and_resume_propagation(self):
        r=(ROOT/'hth/regression_dispatch.py').read_text(); o=(ROOT/'hth/optimizer_dispatch.py').read_text(); w=(ROOT/'.github/workflows/execution-optimizer.yml').read_text(); q=(ROOT/'hth/optimizer_resume.py').read_text(); self.assertIn('"sharding": args.sharding',r); self.assertIn('"sharding": args.sharding',o); self.assertIn('--sharding "${{ inputs.sharding }}"',w); self.assertIn('"sharding": sharding',q)
if __name__=='__main__': unittest.main()
