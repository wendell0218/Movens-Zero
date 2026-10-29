# Operations

## Configuration and Upgrades

Launchers load `.sim2real.env` from the repository root. The example files in
`config/` list the current environment variables. `MOVENS_HOST` is the server LAN
address; `MOVENS_PORT` defaults to 8014. `MOVENS_DEPLOY_PYTHON` selects client Python;
`MOVENS_SERVER_PYTHON` selects server Python. `WBC_ROOT` locates the installed
controller. `WBC_PLANNER` can override its planner asset path.

For an existing installation, regenerate the local env file from the current
example and reapply that host's paths, network settings, and reviewed operating
limits. Previous environment variable names and internal script names are no
longer accepted. Keep `server.sh` and `client.sh` as the public entry points.

The protocol identifier is `movens-root40-v1`. Stop the client and server before
upgrading, replace the entire deployment directory on both hosts, update local
configuration, verify checksums, and restart the server first. Do not mix old and
new client/server revisions. Roll back both directories and their corresponding
local env files together if an update fails.

## Health and Network

```bash
curl http://127.0.0.1:8014/health
```

Expect `status=ok`, `backend=checkpoint`, `checkpoint_verified=true`,
`model_action_dim=40`, `action_dim=36`, and the digest of the selected checkpoint. Pin it explicitly with
`MOVENS_CHECKPOINT_SHA256` when a fixed model identity is required. `busy=true` means another client owns the model.
Only one inference client is supported at a time.

The service has no authentication or TLS. Use a trusted LAN or an authenticated
tunnel, not a publicly exposed port. The G1 must reach the workstation's bind
address. Local controller packets use queued, non-conflated ZMQ on port 5556.

## Model and Timing

The model predicts 40D root40 features; the wire carries 36D absolute poses:
29 IsaacLab-order joints, root xyz, and quaternion wxyz. At 50 Hz, inference emits
30 frames and advances 20, retaining a 10-frame RTC prefix. The prefix is rebased
using the latest observation and heading anchor. Checkpoint keys and normalization
must match the bundled configuration; changing labels does not convert a 39D model.

Text and speech keep the same conditioning input within a task. Speech uses the
complete recording; music uses a moving 0.6-second window. Video uses 8 fps,
0.6-second windows, and 50176-pixel frames. Video decoding is cached for the most recent frame cursor; uncached decoding
is included in the actual inference deadline. Uploaded media is SHA-256 checked; at most eight media items are cached
until server restart.

Publication deadlines follow a fixed monotonic clock, avoiding accumulated drift
from a single delayed send. Lateness above 180 ms stops the segment instead of
fast-forwarding it. Stale-state, joint-bound, first-frame jump, quaternion, and
mechanical-velocity checks remain active. The shared first-frame jump default is
0.35 rad; do not relax it to hide hardware or motion errors.

## Logs and Limitations

- Logs normally go to `/tmp/movens_real_logs`. Bridge logs and generated CSVs can
  be overwritten when preparing another task; archive them before rerunning.
- Rejected motion snapshots contain the inputs and error needed for diagnosis.
- The tested controller needs a 46-frame lookahead, exceeding the first 30-frame
  model chunk. Initial waiting/playback lag remains; the deadline correction does
  not remove it or justify changing the trained encoder horizon.
- A previous short replay showed a possible final-packet/EOF race. Full final-frame
  delivery remains an acceptance item.
- Preflight, unit tests and dry runs do not establish physical tracking quality.
  Validate the controller independently before live model-driven motion.
- Shared temporary state/flag paths support one robot session, not concurrent
  deployments on the same machine.

## Closed-loop Simulation

Use [sim2sim](../sim2sim/README.md) for MuJoCo plus SONIC. It runs physics on a
separate wall-clock loop and explicitly selects `--state-feedback live`, so the
complete observed root pose and joints reach the model. The real robot bridge
retains its prior `reference_heading` convention by default: measured joints and
height with reference root XY/orientation. These are separate feedback protocols.
