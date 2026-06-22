from pathlib import Path
path = Path('fate_x/acpr_flow/model.py')
text = path.read_text(encoding='utf-8')
old = '''        temporal_code = time * freq
        return base + 0.02 * temporal_code
'''
new = '''        state_modulation = 1.0 + 0.25 * torch.tanh(base)
        temporal_code = time * freq * state_modulation
        return base + 0.05 * temporal_code
'''
if old not in text:
    raise SystemExit('temporal code return block not found')
path.write_text(text.replace(old, new), encoding='utf-8')

test = Path('tests/test_acpr_flow_control_temporal_path.py')
text = test.read_text(encoding='utf-8')
text = text.replace('    state = torch.zeros(4, 16)\n', '    torch.manual_seed(7)\n    state = torch.randn(4, 16)\n')
test.write_text(text, encoding='utf-8')
print('fixed state-conditioned temporal code')
