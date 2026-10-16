import torch
from torchvision import transforms


class ImagePreprocessor:
    """
    Preprocessing di una singola immagine del Google Cartoon Set.

    Input:
        PIL.Image RGB, originariamente 500x500.

    Output:
        torch.Tensor di shape [3, image_size, image_size].

    Il preprocessing esegue:
        1. resize;
        2. conversione a tensor;
        3. normalizzazione opzionale.

    La normalizzazione rimane configurabile perché la scelta
    definitiva non è ancora stata presa.
    """

    def __init__(self, image_size=32, normalization="zero_one"):

        self.image_size = image_size
        self.normalization = normalization

        if image_size <= 0:
            raise ValueError(
                "image_size deve essere maggiore di 0."
            )

        if normalization not in ("zero_one", "minus_one_one"):
            raise ValueError(
                "normalization deve essere "
                "'zero_one' oppure 'minus_one_one'."
            )

        transform_list = [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor()
        ]

        if normalization == "minus_one_one":
            transform_list.append(
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            )

        self.transform = transforms.Compose(transform_list)

    def __call__(self, image):
        """
        Applica il preprocessing all'immagine.

        Args:
            image: PIL.Image.

        Returns:
            Tensor float32 di shape [3, image_size, image_size].
        """

        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self.transform(image)

        return tensor