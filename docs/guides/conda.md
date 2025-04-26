# Using a Conda Environment YAML

This guide explains how to create and use a Conda environment from a YAML file.

## 1. Create a New Environment from a YAML File

If you have an `environment.yaml` file, you can recreate the environment by running:

```bash
conda env create -f environment.yaml
```

This will create a new environment with the same dependencies specified in the YAML.

## 2. Activate the Environment

After creating the environment, activate it:

```bash
conda activate carbon
```

## 3. Update an Existing Environment

If you already have an environment and want to update it based on a YAML file, you can run:

```bash
conda env update -f environment.yaml --prune
```

- `--prune` removes dependencies that are no longer required.

## 4. Exporting Your Own Environment (Optional)

If you want to create your own `environment.yaml` from an existing Conda environment:

```bash
conda env export > environment.yaml
```

Or for a specific environment:

```bash
conda env export -n your_env_name > environment.yaml
```
