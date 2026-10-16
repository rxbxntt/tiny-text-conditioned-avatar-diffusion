from pathlib import Path
import csv

from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset


EXPECTED_ATTRIBUTES = {
    "eye_angle",
    "eye_lashes",
    "eye_lid",
    "chin_length",
    "eyebrow_weight",
    "eyebrow_shape",
    "eyebrow_thickness",
    "face_shape",
    "facial_hair",
    "hair",
    "eye_color",
    "face_color",
    "hair_color",
    "glasses",
    "glasses_color",
    "eye_slant",
    "eyebrow_width",
    "eye_eyebrow_distance",
}


class CartoonSetDataset(Dataset):
    """
    Dataset loader per Google Cartoon Set 10k.

    Ogni sample è composto da:
        - una immagine PNG
        - un file CSV con i relativi attributi

    Il loader NON esegue il preprocessing 32x32.
    """

    def __init__(self, root_dir, expected_samples=10_000):
        self.root_dir = Path(root_dir)
        self.expected_samples = expected_samples

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory non trovata: {self.root_dir}"
            )

        if not self.root_dir.is_dir():
            raise NotADirectoryError(
                f"Il percorso non è una directory: {self.root_dir}"
            )

        self.samples = self._build_index()

        if len(self.samples) != self.expected_samples:
            raise ValueError(
                f"Numero di sample inatteso: "
                f"trovati {len(self.samples)}, "
                f"attesi {self.expected_samples}"
            )

    def _build_index(self):
        """
        Costruisce l'indice:

            image_path <-> metadata_path

        L'associazione viene fatta tramite il nome del file.
        """

        image_files = sorted(self.root_dir.glob("*.png"))

        samples = []

        for image_path in image_files:
            metadata_path = image_path.with_suffix(".csv")

            if not metadata_path.exists():
                raise FileNotFoundError(
                    f"Metadata mancante per {image_path.name}: "
                    f"{metadata_path.name}"
                )

            samples.append(
                {
                    "image": image_path,
                    "metadata": metadata_path,
                }
            )

        return samples

    @staticmethod
    def _load_metadata(metadata_path):
        """
        Legge il CSV associato a una immagine.

        Esempio reale:

            "eye_angle", 2, 3

        Il primo valore numerico rappresenta il valore
        dell'attributo per quella specifica immagine.
        """

        metadata = {}

        with metadata_path.open(
            mode="r",
            encoding="utf-8",
            newline=""
        ) as file:

            reader = csv.reader(file)

            for row in reader:

                if len(row) < 2:
                    raise ValueError(
                        f"Riga metadata non valida in "
                        f"{metadata_path.name}: {row}"
                    )

                attribute = row[0].strip()

                try:
                    value = int(row[1])
                except ValueError as exc:
                    raise ValueError(
                        f"Valore non numerico per '{attribute}' "
                        f"in {metadata_path.name}"
                    ) from exc

                metadata[attribute] = value

        # Controlliamo che gli attributi siano quelli attesi.
        attributes = set(metadata.keys())

        if attributes != EXPECTED_ATTRIBUTES:
            missing = EXPECTED_ATTRIBUTES - attributes
            unexpected = attributes - EXPECTED_ATTRIBUTES

            raise ValueError(
                f"Metadata non validi in {metadata_path.name}. "
                f"Mancanti: {missing}. "
                f"Inaspettati: {unexpected}."
            )

        return metadata

    @staticmethod
    def _load_image(image_path):
        """
        Apre e valida una immagine.

        Il resize a 32x32 NON viene fatto qui.
        """

        try:
            with Image.open(image_path) as image:

                # Verifica che il file immagine sia integro.
                image.verify()

            # Dopo verify() bisogna riaprire l'immagine.
            with Image.open(image_path) as image:

                if image.mode not in ("RGB", "RGBA"):
                    raise ValueError(
                        f"Formato immagine non supportato: "
                        f"{image_path.name} "
                        f"(mode={image.mode})"
                    )

                return image.convert("RGB").copy()

        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(
                f"Immagine corrotta o non leggibile: "
                f"{image_path}"
            ) from exc

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        image = self._load_image(sample["image"])

        metadata = self._load_metadata(sample["metadata"])

        return {
            "image": image,
            "metadata": metadata,
        }