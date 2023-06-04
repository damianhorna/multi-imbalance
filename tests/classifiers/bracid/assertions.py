def assert_almost_equal(ctx, bracid_closest_rule_per_example, initial_correct_rules):
    # Make sure confusion matrix, closest rule per example are correct at the beginning
    for example_id in bracid_closest_rule_per_example:
        rule_id, dist = bracid_closest_rule_per_example[example_id]
        with ctx.subTest(f"{example_id}: rule_id"):
            ctx.assertEqual(rule_id, initial_correct_rules[example_id].rule_id)
        with ctx.subTest(f"{example_id}: dist"):
            ctx.assertAlmostEqual(dist, initial_correct_rules[example_id].dist, delta=0.001)
