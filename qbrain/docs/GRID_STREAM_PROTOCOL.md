# Grid Live Stream Protocol

Binary WebSocket protocol for streaming grid data from backend to frontend at 60 FPS.

## Data Flow

- **Step 0**: Pushed from `core.guard.Guard.main()` after `converter()` produces components. This happens in-process before the grid subprocess runs.
- **Grid subprocess** (`py -m jax_test.grid --cfg ...`): Decodes DB, builds model, writes `model_out.json` and `model_out_data.npz`. It does **not** run a simulation loop and does **not** stream frames. Only step 0 is streamed from core.
- **Multi-step streaming**: For per-time-step streaming, the simulation must run in-process (e.g. via `jax_test/gnn`) or through a streaming-capable runner; the grid subprocess path does not support it.

## Enabling

Set `GRID_STREAM_ENABLED=true` in the backend environment. When enabled, the Relay creates a `GridStreamer` on WebSocket connect and pushes frames during START_SIM and (when wired) during jax simulation steps.

## Binary Frame Format

Each WebSocket binary message (opcode 0x2) has the following layout (little-endian):

| Offset | Size | Field  | Description              |
|--------|------|--------|--------------------------|
| 0      | 4    | `step` | uint32, time step index  |
| 4      | 4    | `n`    | uint32, number of floats |
| 8      | n*4  | `data` | float32 array (raw)      |

**Total**: 8 + n*4 bytes.

## Frontend Handling

1. On `binary` message (not `text`): parse header with `DataView`:
   - `step = view.getUint32(0, true)`
   - `n = view.getUint32(4, true)`
   - `data = new Float32Array(buffer, 8, n)`

2. Store in double buffer; swap on each frame.

3. Render with `requestAnimationFrame`: draw scatter from `data` using positions derived from `env_cfg.dims` and `amount_nodes` (see `jax_test/grid/visualizer.py` for position logic).

## Message Type

Frames are sent as raw binary. No JSON envelope. The frontend distinguishes grid frames from text messages by checking `event.data instanceof ArrayBuffer` (or `Blob`).
