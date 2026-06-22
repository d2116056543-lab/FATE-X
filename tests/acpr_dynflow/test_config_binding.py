from fate_x.acpr_dynflow.config import load_dynflow_config

def test_config_binding_manifest():
    cfg=load_dynflow_config('configs/acpr_dynflow_v1_bddx_32f_224.yaml')
    assert cfg.get('data','frames') == 32
    assert 'model.predicates.assignment' in cfg.consumer_manifest

