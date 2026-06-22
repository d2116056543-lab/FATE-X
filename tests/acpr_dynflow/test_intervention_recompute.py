from ._helper import make_output
from fate_x.acpr_dynflow.interventions import run_intervention

def test_intervention_changes_forward():
    model,batch,out=make_output()
    off=run_intervention(model,batch,'all_flow_off')
    assert off.diagnostics['intervention'] == 'all_flow_off'

