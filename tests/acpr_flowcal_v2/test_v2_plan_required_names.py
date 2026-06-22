from pathlib import Path


REQUIRED = ['test_v2_action_subspace_train_only.py', 'test_v2_atlas_schema.py', 'test_v2_axis_control_adapter.py', 'test_v2_axis_direction_flow.py', 'test_v2_best_selector.py', 'test_v2_camera_compensated_motion.py', 'test_v2_checkpoint_migration.py', 'test_v2_config_binding.py', 'test_v2_contextual_reason_target.py', 'test_v2_control_normalization.py', 'test_v2_course_circular_residual.py', 'test_v2_dynamic_descriptor.py', 'test_v2_equal_mass_random.py', 'test_v2_firewall_forward_invariance.py', 'test_v2_firewall_gradient_ratios.py', 'test_v2_flow_grammar.py', 'test_v2_foreground_supervisor.py', 'test_v2_formal_import_graph.py', 'test_v2_hardpair_contradiction.py', 'test_v2_implementation_manifest.py', 'test_v2_intervention_recompute.py', 'test_v2_lane_flow_field.py', 'test_v2_lateral_direction_sign.py', 'test_v2_local_transport.py', 'test_v2_named_outputs.py', 'test_v2_no_cache_contract.py', 'test_v2_optimizer_groups.py', 'test_v2_prefix_future_no_leak.py', 'test_v2_preflight_json_contract.py', 'test_v2_pu_unknown_schedule.py', 'test_v2_reason_memory_54.py', 'test_v2_recovery_reaudit_contract.py', 'test_v2_renderer_schema.py', 'test_v2_representation_pcgrad.py', 'test_v2_review_pass_binding.py', 'test_v2_scaled_gradient.py', 'test_v2_scheduler_step_resume.py', 'test_v2_stage_controller.py', 'test_v2_temporal_necessity.py', 'test_v2_transport_warp.py', 'test_v2_transported_predicate_tracker.py']


def test_all_plan_named_v2_tests_exist():
    root = Path('tests/acpr_flowcal_v2')
    missing = [name for name in REQUIRED if not (root / name).exists()]
    assert not missing
