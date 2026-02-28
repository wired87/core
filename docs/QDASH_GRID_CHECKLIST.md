# QDash Frontend: Grid Binary Frame Integration Checklist

Use this checklist when integrating the grid live stream into the qdash frontend.

## WebSocket Binary Handling

- [ ] On WebSocket `message` event, distinguish binary from text:
  - `event.data instanceof ArrayBuffer` or `event.data instanceof Blob`
  - If binary, parse as grid frame; if text, parse as JSON
- [ ] Parse binary frame header (little-endian):
  - `step = new DataView(buffer).getUint32(0, true)`
  - `n = new DataView(buffer).getUint32(4, true)`
  - `data = new Float32Array(buffer, 8, n)`
- [ ] Use double buffer: store incoming frame, swap on each `requestAnimationFrame` to avoid tearing

## Rendering

- [ ] Derive positions from `data` using `env_cfg.dims` and `amount_nodes` (see `grid/visualizer.py` for position logic)
- [ ] Draw scatter plot from positions
- [ ] Handle step 0 only when using core.guard + grid subprocess flow; multi-step requires jax/gnn path

## References

- [GRID_STREAM_PROTOCOL.md](GRID_STREAM_PROTOCOL.md) – binary frame format
- `grid/visualizer.py` – position derivation and plot layout
