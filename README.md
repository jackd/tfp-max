# tfp-max

Package containing tensorflow probability implementation of distributions resulting from taking the max of other distributions.

## Installation

```bash
# install tensorflow somehow
git clone https://github.com/jackd/tfp-max
pip install tfp-max
```

## Pre-commit

This package uses [pre-commit](https://pre-commit.com/) to ensure commits meet minimum criteria. To Install, use

```bash
pip install pre-commit
pre-commit install
```

This will ensure git hooks are run before each commit. While it is not advised to do so, you can skip these hooks with

```bash
git commit --no-verify -m "commit message"
```
