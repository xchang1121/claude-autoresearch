# ascendc

Format: a CANNBot-style direct-invoke AscendC project. `--kernel` points at
an `ascendc_op/` directory; its parent directory must contain `kernel.py`.

Supported example:

```text
add_custom/
  reference.py
  kernel.py
  ascendc_op/
    CMakeLists.txt
    op_kernel/
    op_extension/
```

Run from the CA repo root after setting `defaults.dsl: ascendc`,
`defaults.backend: ascend`, and `defaults.framework: torch`:

```text
/autoresearch --ref ar_examples/ascendc/add_custom/reference.py --kernel ar_examples/ascendc/add_custom/ascendc_op --op-name add_custom --devices 0
```

The AscendC adapter copies the whole `ascendc_op/` project, rebuilds it, and
imports sibling `kernel.py` as the `ModelNew` entry.
