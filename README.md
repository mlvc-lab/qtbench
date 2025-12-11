<p align="center">
<img src="assets/eng_logo.png" alt="ENC Logo" width="450">
</p>

<h2><p align="center">Model Compression Toolbox for LLMs and Diffusion Models</p></h2>

<p align="center">
    <a href="https://github.com/mit-han-lab/deepcompressor/blob/master/LICENSE">
        <img alt="Apache License" src="https://img.shields.io/github/license/mit-han-lab/deepcompressor">
    </a>
</p>

## News

- [2025/12] 🔥 Start of this model compression toolbox! Let's make great contributions together!

## Key Features

***DeepCompressor*** is an open source model compression toolbox for large language models and diffusion models based on PyTorch. DeepCompressor currently supports fake quantization with any integer and floating-point data type within 8 bits, e.g., INT8, INT4 and FP4_E2M1. Here are examples that implement the following algorithms.

- [Post-training quantization for large language models](/examples/llm/):
  - Weight-only Quantization
    - [AWQ (W4A16)](/examples/llm/configs/awq.yaml)
    - [GPTQ (W4A16)](/examples/llm/configs/gptq.yaml)
  - Weight-Activation Quantization
    - [SmoothQuant (W8A8)](/examples/llm/configs/smoothquant-static.yaml)
  - Weight-Activation and KV-Cache Quantization
    - [QoQ (W4A8KV4)](/examples/llm/)
- [Post-training quantization for diffusion models](/examples/diffusion/):
  - Weight-Activation Quantization
    - [SVDQuant (W4A4)](/examples/diffusion/configs/svdquant/)
    - [QuaRTZ (W4A4)](/examples/diffusion/configs/quartz/)

DeepCompressor also contains examples that integrate with other inference libraries.

- [Deploy weight-only quantized LLMs with TinyChat](/examples/llm/)
- [Deploy quantized LLMs with QServe]((/examples/llm/))
- [Deploy quantized diffusion models with Nunchaku](/examples/diffusion/)

## Installation

### Install from Source

1. Clone this repository and navigate to deepcompressor folder

```
git clone https://github.com/mlvc-lab/qtbench
cd qtbench
```

2. Install Package

```
conda env create -f environment.yml
poetry install
```

## Acknowledgments

Quantization Bench is forked from DeepCompressor library from MIT Han Lab.
Many thanks to the original authors for their great work!

