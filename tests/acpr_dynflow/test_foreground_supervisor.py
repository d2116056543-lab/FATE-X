def test_supervisor_no_detach_tokens():
    text=open('fate_x/engine/supervise_acpr_dynflow_foreground.py', encoding='utf-8').read()
    for bad in ['Start-Process','Start-Job','nohup','DETACHED_PROCESS']:
        assert bad not in text

