# Installation and Training Guide

This guide provides step-by-step instructions for installing and running a training job using the `alignment-handbook` repository. Follow the steps below to set up your environment and start the training process.

## Step 1: Clone the Repository
First, clone the repository from GitHub to your local machine.
```bash
git clone https://github.com/huggingface/alignment-handbook.git
```

## Step 2: Navigate to the Repository Directory
Change your directory to the cloned repository.
```bash
cd ./alignment-handbook/
```

## Step 3: Install the Package
Use `pip` to install the package.
```bash
python -m pip install .
```

## Step 4: Install `flash-attn`
Install the `flash-attn` package with the `--no-build-isolation` flag.
```bash
python -m pip install flash-attn --no-build-isolation
```

## Step 5: Launch the Training Job
Set the log level for `accelerate` to `info` and launch the training job using the provided configuration files.
```bash
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft.py recipes/zephyr-7b-beta/sft/config_full.yaml
```

### Note:
If the training job fails due to missing attributes in `SFTConfig`, follow the steps below to fix the issue.

## Handling Missing Attributes in `SFTConfig`
1. Open the `sft_trainer.py` file in edit mode.
2. Locate the attribute causing the error.
3. If the attribute is accessed using the `.` notation, change it to use `getattr` with a default value of `None`.

### Example:
Change this:
```python
args.dataset_batch_size
```
To this:
```python
getattr(args, "dataset_batch_size", None)
```

## Conclusion
By following these steps, you should be able to set up the environment and run the training job successfully. If you encounter any issues, ensure that you have followed each step carefully and made the necessary adjustments for handling missing attributes in `SFTConfig`.

For more information and detailed documentation, refer to the repository's [README file](https://github.com/huggingface/alignment-handbook).