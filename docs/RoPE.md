# RoPE Frequencies: How Are They Calculated?

RoPE (Rotary Positional Embedding) rotates pairs of dimensions in the Query (Q) and Key (K) vectors.

For example, if:

```text
head_dim = 8
```

A Q vector:

```text
[2, 4, 6, 8, 1, 3, 5, 7]
```

is divided into pairs:

```text
Pair 0 → [2, 4]
Pair 1 → [6, 8]
Pair 2 → [1, 3]
Pair 3 → [5, 7]
```

Each pair gets a different frequency.

---

## 1. RoPE Frequency Formula

The frequency for each pair is:

$$
frequency_i = \frac{1}{base^{2i/d}}
$$

Where:

- $i$ = pair index: `0, 1, 2, ...`
- $d$ = head dimension
- $base$ = RoPE base, traditionally `10000`

For a token at a particular position, the rotation angle is:

$$
angle_i = position \times frequency_i
$$

The flow is:

```text
Pair Index
    ↓
Calculate Frequency
    ↓
frequency = 1 / base^(2i/d)
    ↓
Token Position × Frequency
    ↓
Rotation Angle
    ↓
Rotate Dimension Pair
```

---

## 2. Example: head_dim = 8

We have:

$$
d = 8
$$

Therefore:

$$
8 / 2 = 4
$$

dimension pairs.

```text
Pair 0 → dimensions [0, 1]
Pair 1 → dimensions [2, 3]
Pair 2 → dimensions [4, 5]
Pair 3 → dimensions [6, 7]
```

Assume:

$$
base = 10000
$$

---

### Pair 0

Pair index:

$$
i = 0
$$

Frequency:

$$
frequency_0 =
\frac{1}{10000^{2(0)/8}}
$$

$$
=
\frac{1}{10000^0}
$$

$$
= 1
$$

```text
Pair 0 → frequency = 1
```

---

### Pair 1

Pair index:

$$
i = 1
$$

Frequency:

$$
frequency_1 =
\frac{1}{10000^{2(1)/8}}
$$

$$
=
\frac{1}{10000^{0.25}}
$$

$$
= 0.1
$$

```text
Pair 1 → frequency = 0.1
```

---

### Pair 2

Pair index:

$$
i = 2
$$

Frequency:

$$
frequency_2 =
\frac{1}{10000^{2(2)/8}}
$$

$$
=
\frac{1}{10000^{0.5}}
$$

$$
= 0.01
$$

```text
Pair 2 → frequency = 0.01
```

---

### Pair 3

Pair index:

$$
i = 3
$$

Frequency:

$$
frequency_3 =
\frac{1}{10000^{2(3)/8}}
$$

$$
=
\frac{1}{10000^{0.75}}
$$

$$
= 0.001
$$

```text
Pair 3 → frequency = 0.001
```

---

## 3. Frequency Summary

| Pair | Dimensions | Frequency |
|---|---|---:|
| Pair 0 | `[0, 1]` | `1` |
| Pair 1 | `[2, 3]` | `0.1` |
| Pair 2 | `[4, 5]` | `0.01` |
| Pair 3 | `[6, 7]` | `0.001` |

These frequencies are used for **every token**.

What changes from token to token is the position.

---

## 4. Frequency Is Not the Rotation Angle

This distinction is important.

Frequency tells us how quickly a pair rotates as token position changes.

The actual rotation angle is:

$$
angle_i = position \times frequency_i
$$

For example:

```text
Token position = 5
```

### Pair 0

$$
angle_0 = 5 \times 1 = 5
$$

### Pair 1

$$
angle_1 = 5 \times 0.1 = 0.5
$$

### Pair 2

$$
angle_2 = 5 \times 0.01 = 0.05
$$

### Pair 3

$$
angle_3 = 5 \times 0.001 = 0.005
$$

Therefore:

```text
Pair 0 → rotate by 5 radians
Pair 1 → rotate by 0.5 radians
Pair 2 → rotate by 0.05 radians
Pair 3 → rotate by 0.005 radians
```

---

## 5. What Does Frequency Mean?

Frequency controls how quickly a dimension pair's rotation changes across token positions.

For:

```text
frequency = 1
```

the angle changes quickly:

```text
Position:  0   1   2   3   4   5
Angle:     0   1   2   3   4   5
```

For:

```text
frequency = 0.1
```

the angle changes more slowly:

```text
Position:  0    1    2    3    4    5
Angle:     0   0.1  0.2  0.3  0.4  0.5
```

For:

```text
frequency = 0.01
```

the angle changes even more slowly:

```text
Position:  0     1     2     3
Angle:     0    .01   .02   .03
```

So:

```text
High Frequency
    ↓
Angle changes quickly
    ↓
Captures finer positional differences


Low Frequency
    ↓
Angle changes slowly
    ↓
Captures broader positional differences
```

---

## 6. Does RoPE Always Reduce Frequency by 1/10?

No.

The pattern:

```text
1
↓
0.1
↓
0.01
↓
0.001
```

only happened because we used:

$$
base = 10000
$$

and:

$$
d = 8
$$

The reduction between consecutive frequencies depends on:

$$
base^{2/d}
$$

Each next frequency is:

$$
frequency_{i+1}
=
\frac{frequency_i}{base^{2/d}}
$$

Therefore:

```text
base
  ↓
Controls the overall frequency range

head_dim (d)
  ↓
Controls how frequencies are spaced
```

---

## 7. Why Does head_dim = 8 Give 1, 0.1, 0.01?

For:

$$
d = 8
$$

the exponent increases by:

$$
\frac{2}{d}
=
\frac{2}{8}
=
0.25
$$

The exponents are:

```text
Pair 0 → 0
Pair 1 → 0.25
Pair 2 → 0.50
Pair 3 → 0.75
```

Since:

$$
10000^{0.25} = 10
$$

each next frequency is divided by `10`.

Therefore:

```text
1
↓ ÷ 10
0.1
↓ ÷ 10
0.01
↓ ÷ 10
0.001
```

This pattern is specific to:

```text
base = 10000
head_dim = 8
```

---

## 8. Example: head_dim = 64

Suppose:

$$
d = 64
$$

The exponent increment becomes:

$$
\frac{2}{64}
=
0.03125
$$

The frequencies decrease more gradually.

The first few frequencies are approximately:

| Pair | Frequency |
|---|---:|
| Pair 0 | `1.0000` |
| Pair 1 | `0.7499` |
| Pair 2 | `0.5623` |
| Pair 3 | `0.4217` |

So:

```text
1
↓
0.75
↓
0.56
↓
0.42
↓
...
```

It does not become:

```text
1 → 0.1 → 0.01
```

The number of dimensions affects the spacing between frequencies.

---

## 9. Complete Mental Model

Suppose:

```text
head_dim = 8
```

Then:

```text
Q Vector

[2, 4, 6, 8, 1, 3, 5, 7]

        ↓

Split into pairs

[2,4] [6,8] [1,3] [5,7]

        ↓

Pair indices

  0     1     2     3

        ↓

Calculate frequencies

1     0.1    0.01   0.001

        ↓

Token Position × Frequency

        ↓

Rotation angle for each pair

        ↓

Rotate every pair

        ↓

Position-aware Q
```

The same process is applied to K:

```text
K
↓
Split into pairs
↓
Calculate frequencies
↓
Position × Frequency
↓
Rotation angles
↓
Rotate pairs
↓
Position-aware K
```

---

## Key Takeaway

RoPE divides Q and K vectors into dimension pairs.

Each pair gets a frequency:

$$
frequency_i = \frac{1}{base^{2i/d}}
$$

For each token:

$$
angle_i = position \times frequency_i
$$

Each pair is then rotated using its own angle.

Different frequencies allow different dimension pairs to represent positional changes at different scales.