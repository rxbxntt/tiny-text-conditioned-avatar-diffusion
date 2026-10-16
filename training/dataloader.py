from torch.utils.data import DataLoader


def collate_cartoon_samples(batch):
    """
    Keeps images and metadata in their original form.

    The dataset returns samples like:
        {
            "image": PIL.Image,
            "metadata": dict
        }

    We do not convert them to tensors here because the
    image preprocessing is handled by the training step.

    Args:
        batch:
            List of samples returned by CartoonSetDataset.

    Returns:
        The same list of samples.
    """
    return batch


def create_dataloader(
    dataset,
    batch_size,
    shuffle=True,
    num_workers=0,
):
    """
    Creates the DataLoader for an already initialized dataset.

    Args:
        dataset:
            CartoonSetDataset instance.
        batch_size:
            Number of samples per batch.
        shuffle:
            Whether to shuffle the dataset.
        num_workers:
            Number of DataLoader worker processes.

    Returns:
        PyTorch DataLoader.
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_cartoon_samples,
    )

    return dataloader