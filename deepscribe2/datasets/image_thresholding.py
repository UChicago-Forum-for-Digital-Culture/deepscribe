from typing import Iterable, Optional, Tuple

import torch
from torch.nn import functional as f

# copypasted from https://github.com/nopperl/torch-image-binarization/tree/main


def histogram(
    image: torch.Tensor, bins=256, range: Optional[Iterable[float]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculates the pixel value histogram for a grayscale image.
    While `torch.histogram` does not work with CUDA or `torch.compile`,
    this implementation does.

    Parameters:
        image: grayscale input image of shape (B, H, W) and float dtype.
        nbins: number of bins used to calculate the image histogram.
        range: value range of the bins of form (min, max).

    Returns:
        (counts, bin_edges): Two tensors of shape (nbins,)
    """
    if not range:
        range_min, range_max = image.min(), image.max()
    elif len(range) == 2:
        range_min, range_max = range[0], range[1]
    else:
        raise ValueError("range needs to be iterable of form: (min, max).")
    counts = torch.empty(bins, device=image.device, dtype=image.dtype)
    torch.histc(image, bins, min=range_min, max=range_max, out=counts)
    bin_edges = torch.linspace(
        range_min, range_max, bins, device=counts.device, dtype=counts.dtype
    )
    return counts, bin_edges


def threshold_otsu(image: torch.Tensor, nbins=256) -> torch.Tensor:
    """Return threshold value based on Otsu's method.

    Parameters:
        image: grayscale input image of shape (B, H, W) and float dtype.
        nbins: number of bins used to calculate the image histogram.

    Returns:
        threshold: A threashold in $[0,1]$ which can be used to binarize
        the grayscale image.

    References:
       [1]: https://en.wikipedia.org/wiki/Otsu's_Method
       [2]: https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_otsu

    Examples:
    >>> threshold = threshold_otsu(image)
    >>> binary = image <= threshold
    """
    counts, bin_edges = histogram(image, nbins, range=(0, 1))

    # class probabilities for all possible thresholds
    weight1 = torch.cumsum(counts, dim=0)
    weight2 = torch.cumsum(counts.flip(dims=(0,)), dim=0).flip(dims=(0,))
    # class means for all possible thresholds
    mean1 = torch.cumsum(counts * bin_edges, dim=0) / weight1
    mean2 = (
        torch.cumsum((counts * bin_edges).flip(dims=(0,)), dim=0)
        / weight2.flip(dims=(0,))
    ).flip(dims=(0,))

    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = torch.argmax(variance12)
    threshold = idx / nbins

    return threshold


def su(img: torch.Tensor, w=3, n_min=3) -> torch.Tensor:
    """Binarizes an image using the Su algorithm.

    Arguments:
        img: grayscale input image of shape (B, H, W) and float dtype
        w: window size, recommended to set w=n_min
        n_min: min high contrast pixels, recommended to set w=n_min

    Returns:
        image of shape (B, H, W)

    References:
        [1]: https://doi.org/10.1145/1815330.1815351
    """
    eps = 1e-10
    batch_size, height, width = img.shape

    # construct contrast image
    windows = f.unfold(f.pad(img, pad=[w // 2] * 4, mode="replicate"), kernel_size=w)
    local_max = torch.max(windows, dim=0).values
    local_min = torch.min(windows, dim=0).values
    contrast = (local_max - local_min) / (local_max + local_min + eps)

    # find high-contrast pixels
    threshold = threshold_otsu(contrast)
    hi_contrast = torch.where(
        contrast < threshold,
        torch.tensor(0, dtype=img.dtype),
        torch.tensor(1, dtype=img.dtype),
    )
    del contrast
    hi_contrast_windows = f.unfold(
        f.pad(
            hi_contrast.view(height, width).unsqueeze(0),
            pad=[w // 2] * 4,
            mode="replicate",
        ),
        kernel_size=w,
    )

    # classify pixels
    hi_contrast_count = hi_contrast_windows.sum(axis=0)

    e_sum = torch.sum(
        windows * hi_contrast_windows, axis=0
    )  # matrix multiplication in axes 2 and 3
    e_mean = (
        e_sum / hi_contrast_count
    )  # produces nan if hi_contrast_count == 0, but since only pixels with hi_contrast_count >= n_min are considered, these values are ignored anyway
    e_mean = torch.where(torch.isnan(e_mean), 0, e_mean)
    e_std = torch.square((windows - e_mean) * hi_contrast_windows).mean(axis=0)
    del windows, hi_contrast_windows
    e_std = torch.sqrt(e_std)
    e_std = torch.where(torch.isnan(e_std), 0, e_std)
    result = torch.zeros_like(img)
    result[
        (hi_contrast_count.view(height, width) >= n_min)
        & (img <= e_mean.view(height, width) + e_std.view(height, width) / 2)
    ] = 1

    return result
