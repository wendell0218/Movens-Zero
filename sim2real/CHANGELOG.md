# Deployment Changes

## 2026-10-26

- Rename the public deployment directory to `sim2real`.
- Use the public root40 model loader and automatically detect S/M/L/XL dimensions.
- Make checkpoint identity pinning explicit and retain root40/RTC contract checks.
- Add optional complete live-state feedback and per-chunk latency records for sim2sim.
- Keep the historical deployment source and controller activation limits.

## 2026-10-12

- Standardize Movens module names, environment variables and terminal messages.
- Separate quick-start instructions, dependencies and operational details.
- Remove development comments and historical source-path metadata.
- Keep model parameters, numerical inference, controller limits and activation keys unchanged.
- Require a paired client/server update for the renamed protocol and configuration.
