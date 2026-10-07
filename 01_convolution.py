"""
CS5720 Neural Network and Deep Learning - Home Assignment 3
Q1: 2-D convolution from scratch

Slide a 3x3 filter over the given 5x5 input with stride 1 and padding 0.
The filter is not flipped. Each output is the dot product of the filter
and the window underneath it, which is the convolution used in CNNs.
"""

import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Values from the assignment sheet.
INPUT_MATRIX = np.array(
    [
        [1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0],
    ],
    dtype=np.int64,
)

FILTER = np.array(
    [
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
    ],
    dtype=np.int64,
)

STRIDE = 1
PADDING = 0


def conv2d(image, kernel, stride, padding):
    """2-D convolution by a sliding dot product. No library conv is used.

    padding is the number of zeros added on each side. stride is the step
    between windows. The kernel is used as given, with no flip.
    """
    if padding < 0:
        raise ValueError("padding must be >= 0")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    if padding > 0:
        image = np.pad(image, pad_width=padding, mode="constant", constant_values=0)

    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape
    if image_h < kernel_h or image_w < kernel_w:
        raise ValueError("kernel does not fit in the input")

    # Same size formula as Part I: floor((N - F + 2P) / S) + 1.
    # Padding was already applied, so P is not added again here.
    out_h = (image_h - kernel_h) // stride + 1
    out_w = (image_w - kernel_w) // stride + 1
    output = np.zeros((out_h, out_w), dtype=np.int64)
    flat_kernel = kernel.ravel()

    for row in range(out_h):
        for col in range(out_w):
            top = row * stride
            left = col * stride
            window = image[top : top + kernel_h, left : left + kernel_w]
            output[row, col] = np.dot(window.ravel(), flat_kernel)

    return output


def format_matrix(matrix):
    lines = []
    for row in matrix:
        lines.append("  " + "  ".join(f"{int(value):2d}" for value in row))
    return "\n".join(lines)


def main():
    output = conv2d(INPUT_MATRIX, FILTER, stride=STRIDE, padding=PADDING)
    # Shown only to make the stride explanation concrete. The required map is stride 1.
    stride_two = conv2d(INPUT_MATRIX, FILTER, stride=2, padding=0)

    lines = [
        "Q1: Convolution from scratch",
        "",
        "Input",
        format_matrix(INPUT_MATRIX),
        "",
        "Filter",
        format_matrix(FILTER),
        "",
        f"Stride = {STRIDE}, Padding = {PADDING}",
        "",
        "Output feature map",
        format_matrix(output),
        "",
        f"Output shape: {output.shape}",
        "",
        "Effect of changing the stride from 1 to 2",
        (
            "The filter steps two pixels instead of one. With N = 5, F = 3, "
            "and P = 0, the output size is floor((5 - 3) / 2) + 1 = 2, so the "
            "map is 2 x 2 instead of 3 x 3. Every other window is skipped, so "
            "the map is smaller and the spatial sampling is coarser. The four "
            "values below are the windows whose top-left corners are (0, 0), "
            "(0, 2), (2, 0), and (2, 2)."
        ),
        "",
        "Stride = 2, Padding = 0 (illustration only)",
        "Output feature map",
        format_matrix(stride_two),
        "",
        f"Output shape: {stride_two.shape}",
        "",
    ]
    text = "\n".join(lines)
    print(text)

    out_path = os.path.join(OUTPUT_DIR, "q1_feature_map.txt")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
