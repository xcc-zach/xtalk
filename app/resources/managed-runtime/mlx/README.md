# MLX runtime resources

`prepare_managed_runtime.py` stages the generated `mlx-swift_Cmlx.bundle`
beside this directory and maps it to the app bundle's resource root, where MLX
looks it up. Model weights are not bundled; the managed-model installer
downloads pinned snapshots on first use.
