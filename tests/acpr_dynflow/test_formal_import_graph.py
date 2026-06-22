def test_formal_import_graph_excludes_legacy():
    text=open('fate_x/acpr_dynflow/model.py', encoding='utf-8').read()+open('fate_x/engine/train_acpr_dynflow.py', encoding='utf-8').read()
    for forbidden in ['ACPRFlowModel','ACPRFlowCalV2Model','TokenPMTAdapter','LogSinkhornTransport','TemporalEvidenceMemory']:
        assert forbidden not in text

