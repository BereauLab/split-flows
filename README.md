# Split-Flows: Measure Transport and Information Loss Across Molecular Resolutions

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
![UV Install Check](https://github.com/hummerichsander/split-flows/actions/workflows/uv-install-check.yml/badge.svg)
![Tests](https://github.com/hummerichsander/split-flows/actions/workflows/pytest.yml/badge.svg)
[![codecov](https://codecov.io/gh/hummerichsander/split-flows/branch/public/graph/badge.svg)](https://codecov.io/gh/hummerichsander/split-flows)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2511.01464-b31b1b.svg)](https://arxiv.org/abs/2511.01464)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

</div>

## Overview

Split-flows provide a probabilistic bridge between molecular resolutions, enabling conditional backmapping and direct measurement of the configuration-dependent (local) information loss.

<div align="center">
    <img src="figures/Figure_1.png" alt="Figure 1" style="max-width: 500px; width: 100%;">
</div>

## Installation

Clone the repository and navigate to the project directory:

```bash
git clone git@github.com:hummerichsander/split-flows.git
cd split-flows
```

To install the project dependencies use uv (if you have not installed uv yet, check out the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/)) and run:

```bash
uv sync
```
