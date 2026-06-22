from fate_x.acpr_dynflow.predicate_transfer import PredicateQueryInitializer

def test_query_transfer_report_and_grad():
    q, report = PredicateQueryInitializer()()
    assert q.shape[0] == 32
    assert 'residual_norm' in report

