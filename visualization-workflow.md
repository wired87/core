# Visualization Workflow (LIVE_DATA)

Cinematic real-time simulation rendering with minimal latency and high stability.

---

## Architecture

```
Frontend (qdash)
    │
    ▼
Backend (qbrain WS Relay + Global Registry)
    ▲
    │
GPU Container (jax_test)

Transport: WebSocket (JSON for LIVE_DATA; binary frames optional for grid stream)
```

---

## Frame Protocol

**LIVE_DATA (JSON):**

- `type`: `"LIVE_DATA"`
- `auth`: `{ "user_id", "env_id" }`
- `data`: `dict[str, any]` — keys from param/field schema, values are arrays (list of numbers) or scalars. No JSON arrays at top level; structure is key → value.

**Binary (optional, grid stream):**

- Header: type, env_id, frame_id, timestamp, shape, dtype (as per existing GridStreamer when enabled).
- No JSON arrays in the binary path.

---

## Component Responsibilities

### Backend (qbrain)

- **Global registry** (`qbrain/utils/ws_registry.py`): `env_id` / `user_id:env_id` → `{ gpu_ws, clients: set, created_at, last_frame_id }`.
- **Frame relay**: On receive of `LIVE_DATA` from a GPU connection, look up channel by stored `user_id`/`env_id` and resend the payload to all frontend clients in that channel.
- **Backpressure**: Drop or throttle frames if send queue grows (optional; can be extended).
- **Heartbeat**: Optional heartbeat every 10s for GPU and clients (extend as needed).
- **Cleanup**: On disconnect, remove the connection from the registry (frontend or GPU) and run `cleanup_empty()`.
- **Optional**: Batch persistence of frames (future).

### GPU (jax_test)

- **VIS_FPS** env: Throttle how often LIVE_DATA is sent (e.g. every N steps).
- **Frame payload**: Use `build_live_data_payload(cfg, time_db_flat)` in `jax_test/grid/live_payload.py` to convert `time_db[0]` (current state) to `dict[keys, shaped_param]` via param_ctlr and keys list.
- **Guard**: Holds WebSocket client instance; at preprocessing (start of `run()`), if `WS_ENDP` is set, connect and send first message `type: "gpu"` with `user_id` and `env_id` (from env vars), then store the WS and wire a `_send_live` callback to GNN.
- **Simulate loop** (`jax_test/gnn/gnn.py`): After `save_t_step()`, if `_send_live` is set, get current state (e.g. `time_construct[0]`), call payload builder, send over the Guard’s WebSocket.

### Frontend (qdash)

- **Single WebSocket** at `/run/`; dedicated handling for message type `LIVE_DATA`.
- **latestFrameRef**: Updated in the WebSocket callback (no render inside callback); holds `{ data, env_id }`.
- **requestAnimationFrame loop**: A live matplot-style view reads from `latestFrameRef` and draws; rendering is decoupled from the socket for smooth animation.
- **WebGL / Canvas**: LiveView, OscilloscopeView, and LiveMatplotView (LIVE_DATA dict) in the control engine section; double-buffer / rAF as needed.
- **Selected env**: Show LIVE_DATA for the selected env (filter by `env_id` when present).

---

## Performance Rules

- Never send full DB state every iteration.
- Never poll DB for live frames.
- Never render inside the WebSocket callback; use a ref and rAF (or similar) to drive updates.
- Use binary transport where applicable (grid stream already supports it).
- Enforce FPS throttle (VIS_FPS) on the GPU side.

---

## Stability

- Heartbeat every 10s (to be added or extended).
- Drop stale frames when backpressure is high (optional).
- Remove dead clients on disconnect and run registry cleanup.
- Clean up empty channels in the global registry.

---

## File Reference

| Area     | Files |
|----------|--------|
| qdash    | `qdash/src/qdash/websocket.js` (LIVE_DATA handler, latestFrameRef), `qdash/src/qdash/components/LiveMatplotView.js`, `landing_page.js` |
| qbrain   | `qbrain/utils/ws_registry.py`, `qbrain/relay_station.py` (registry, GPU registration, LIVE_DATA relay) |
| jax_test | `qbrain/jax_test/guard.py` (WS_ENDP, _live_ws, _send_live), `qbrain/jax_test/grid/live_payload.py`, `qbrain/jax_test/gnn/gnn.py` (simulate loop) |

---

## Future TODO

The following are **not** implemented and are explicitly marked as future work:

- [ ] Delta compression mode for LIVE_DATA
- [ ] Snapshot replay system
- [ ] Adaptive FPS control (client or server driven)
- [ ] Frame interpolation smoothing
- [ ] Multi-backend horizontal scaling (e.g. Redis pub/sub for relay)
- [ ] GPU memory pooling
- [ ] Replay export to video
