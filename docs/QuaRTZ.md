# QuaRTZ: Post-Training Quantization via Residual Truncation and Zero Suppression

QuaRTZ is an activation only quantization method that leverages residual truncation using zero suppression to achieve high accuracy with low-bit quantization.
It is designed to be hardware-friendly and efficient, making it suitable for deployment on resource-constrained devices, especially in edge computing scenarios.

## Formulation

Let $X \in \mathbb{R}^{N \times D}$ be the input activation tensor to be quantized where $N$ is the batch size and $D$ is the feature dimension.
Our aim is to create compact and useful $k$-bit integer representations of $X$.
To achieve informative bit representation, QuaRTZ employs a two-step quantization process: initial RTN quantization followed by residual truncation with zero suppression.

First, we use $k'$-bit RTN quantization to obtain a fine-grained, higher-precision representation of the input activation.
Like any RTN(Round-To-Nearest) based quantization method, QuaRTZ first quantizes the input activation $X$ to a $k'$-bit representation.
Normally, $k'=8$.

```math
Q_x(X) = \text{clamp} \left( \text{round} \left( \frac{X}{s_x} \right), -127, 127 \right)
```

where $s_x$ is the scale factor.

Here, we can use different approaches; whether we use absolute max value or use min-max range.

For absolute max setting,

```math
s_x = \frac{\text{amax}(|X|)}{2^{k-1}}
```

.

For min-max range setting,

```math
s_x = \frac{\text{amax}(X) - \text{amin}(X)}{2^{k} -1}
```

.

At second stage, we further compress the quantized representation by truncating less significant bits and applying zero suppression.
Intuition behind this is that in many practical scenarios, activation values majorly cluster around zero which makes most significant bits redundant(being $0$).
If we can identify and remove these redundant bits, we can achieve higher compression without losing significant information.
For outlier values, failing to preserve their magnitude can lead to significant accuracy degradation. To prevent this, we preserve the most significant bits for outlier values and truncate the less significant bits.

**Example**

| S | M6 | M5 | M4 | M3 | M2 | M1 | M0 | SINT8 Range | Flag ($F$) |
|  :-: | :-: | :-: | :-: | :-: | :-: | :-: |:-: | :-: | :-: |
| 0 | 0 | 0 | 0 | 0 | X | X | X | +/- 0 ~ 7 | 0 (0b000) |
| 0 | 0 | 0 | 0 | 1 | X | X | - | +/- 8 ~ 15 | 1 (0b001) |
| 0 | 0 | 0 | 1 | X | X | - | - | +/- 16 ~ 31 | 2 (0b010) |
| 0 | 0 | 1 | X | X | - | - | - | +/- 32 ~ 63 | 3 (0b011) |
| 0 | 1 | X | X | - | - | - | - | +/- 64 ~ 127 | 4 (0b100) |

## Practical Implementation

Such representation can be highly effective to reduce bit-width, but it adds an 3-bit overhead per activation value.
This makes naive implementation equivalent to 7-bit quantization.

To reduce this memory overhead, we group multiple activation values and share the flag bits among them.
This is,

  1. group $G$ activation values.
  2. apply absolute value and save the sign bits separately.
  3. perform bitwise OR operation on the magnitude bits to determine the maximum flag value for the group.
  4. Apply same quantization scheme using the shared flag value for the entire group.

The caveat is, if there are single outlier value in the group and all other values are small, the entire group will use higher bit-width representation.
In this case, most of the small values will be truncated and induce significant information loss.
However, statistically, such cases are very rare (< 0.01%) and the overall quantization error remains similar to 8-bit quantization.
