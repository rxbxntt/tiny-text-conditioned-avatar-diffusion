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

    3. conversione dei valori nell'intervallo [0, 1].
    """

    def __init__(self, image_size=32):
        self.image_size = image_size

        if image_size <= 0:
            raise ValueError(
                "image_size deve essere maggiore di 0."
            )

        self.transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor()
        ])

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