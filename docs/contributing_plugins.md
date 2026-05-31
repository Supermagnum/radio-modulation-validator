# Contributing custom-mode plugins

Custom-mode plugins validate composite or novel modulations that the standard
family/order CNN cannot classify. They run when a sidecar uses
`expected_family: "custom"` and `expected_order` matches a registered plugin
`mode_id`.

## Sidecar format

```json
{
  "source": "gr-sleipnir",
  "block_name": "SleipnirTxHier",
  "expected_family": "custom",
  "expected_order": "sleipnir_8qpsk",
  "sample_rate_hz": 48000,
  "center_freq_hz": 0,
  "snr_db": null,
  "notes": "8-carrier QPSK, 900 baud per carrier, 1300 Hz spacing"
}
```

`rmv validate` routes to the plugin and skips the CNN. The JSON output includes a
`custom_mode` object with metrics and `pass_overall`.

## Adding a new plugin

1. Copy `src/rmv/plugins/sleipnir_8qpsk.py` as a template.
2. Subclass `CustomModeValidator` in `src/rmv/plugins/base.py`.
3. Set a unique `mode_id` and implement `validate()` and `describe()`.
4. Register the plugin in `src/rmv/plugins/registry.py` inside
   `_register_builtin_plugins()`.
5. Add tests in `tests/test_plugins.py` (synthetic IQ is preferred; no large files).
6. Add an example sidecar under `iq_samples/` if you have a real capture.
7. Open a pull request with plugin description and pass criteria.

## CLI

```bash
rmv plugins list
rmv plugins describe sleipnir_8qpsk
rmv validate iq_samples/gr-sleipnir/tx_output.iq
```

## Built-in plugins

| mode_id | Description |
|---------|-------------|
| `sleipnir_8qpsk` | Eight parallel QPSK carriers in one composite IQ stream (1300 Hz spacing, 900 baud) |

See `rmv plugins describe sleipnir_8qpsk` for measurement and pass criteria details.
