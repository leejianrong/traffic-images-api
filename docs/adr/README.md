# Architecture decision records

Short records of the decisions that shape this codebase: what the situation was,
what we chose, and what we now have to live with. They exist because each of
these looks like an arbitrary detail from the outside, and each one has already
been "cleaned up" at least once in someone's head before they read the reason.

One file per decision, numbered in the order they were taken, never renumbered.
A superseded ADR stays in place with its status changed and a pointer forward.

| # | Decision | Status |
|---|---|---|
| [0001](0001-onnxruntime-not-ultralytics.md) | ONNX Runtime for inference, not the `ultralytics` package | Accepted |
| [0002](0002-raw-ansi-raster-not-rich.md) | The raster is printed raw, not through Rich | Accepted |
| [0003](0003-two-test-layers.md) | Two test layers: fast/offline by default, integration opt-in | Accepted |

## Writing a new one

Copy the shape of an existing file: title, status, date, then **context →
decision → consequences**. Keep it to one screen. Record the decision and its
price, not a tutorial — if it needs a diagram it is probably documentation, not
an ADR.
