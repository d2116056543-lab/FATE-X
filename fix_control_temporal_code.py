from pathlib import Path
path = Path('fate_x/acpr_flow/model.py')
text = path.read_text(encoding='utf-8')
old = '''        time = torch.linspace(0.0, 1.0, steps, device=base.device, dtype=base.dtype).view(1, steps, 1)
        freq = torch.arange(1, base.shape[-1] + 1, device=base.device, dtype=base.dtype).view(1, 1, -1)
        # Fixed temporal code keeps the continuous-control path from producing
        # identical per-frame predictions before reason-memory adaptation.
        temporal_code = torch.sin(time * freq * math.pi)
        return base + 0.02 * temporal_code
'''
new = '''        time = torch.linspace(-1.0, 1.0, steps, device=base.device, dtype=base.dtype).view(1, steps, 1)
        freq = torch.linspace(0.5, 1.5, base.shape[-1], device=base.device, dtype=base.dtype).view(1, 1, -1)
        # Fixed temporal code keeps the continuous-control path from producing
        # identical per-frame predictions before reason-memory adaptation.
        temporal_code = time * freq
        return base + 0.02 * temporal_code
'''
if old not in text:
    raise SystemExit('temporal code block not found')
path.write_text(text.replace(old, new), encoding='utf-8')
print('fixed temporal code')
