import torch

from fate_x.acpr_dynflow_swin.dynamic_predicate_field import DynamicPredicateFieldBuilder
from fate_x.acpr_dynflow_swin.predicate_transfer import PredicateQueryTransfer


def test_oia_transfer_changes_runtime_predicate_queries(tmp_path):
    source = torch.randn(32, 384)
    checkpoint = tmp_path / "oia.pth"
    torch.save({"model": {"predicate_head": {"predicate_queries": source}}}, checkpoint)

    transfer = PredicateQueryTransfer(dim=256, name_features=torch.randn(32, 768))
    transfer.load_oia_query(checkpoint, "model.predicate_head.predicate_queries")
    queries, report = transfer()
    field = DynamicPredicateFieldBuilder(dim=256)
    grid = torch.randn(2, 3, 4, 4, 256)
    output = field(grid, base_queries=queries)

    assert report["loaded"] is True
    assert report["source_shape"] == [32, 384]
    assert output.query_states.shape == (2, 3, 32, 256)
    assert torch.allclose(output.query_states[:, 0], queries.unsqueeze(0))


def test_oia_name_source_and_residual_each_affect_queries():
    transfer = PredicateQueryTransfer(dim=16, name_features=torch.randn(32, 24))
    baseline, _ = transfer()
    with torch.no_grad():
        transfer.name_features.add_(0.25)
    changed_name, _ = transfer()
    with torch.no_grad():
        transfer.oia_query = torch.randn(32, 12)
        transfer._materialize_oia_mapper(12)
    changed_oia, _ = transfer()
    with torch.no_grad():
        transfer.domain_residual.add_(0.1)
    changed_residual, _ = transfer()

    assert not torch.allclose(baseline, changed_name)
    assert not torch.allclose(changed_name, changed_oia)
    assert not torch.allclose(changed_oia, changed_residual)
