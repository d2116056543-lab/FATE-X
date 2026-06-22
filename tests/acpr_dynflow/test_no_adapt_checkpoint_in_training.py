def test_no_adapt_checkpoint_symbol_in_formal_trainer():
    text=open('fate_x/engine/train_acpr_dynflow.py', encoding='utf-8').read()
    assert 'adapt_checkpoint' not in text.lower()
    assert 'model.bin' not in text.lower()

