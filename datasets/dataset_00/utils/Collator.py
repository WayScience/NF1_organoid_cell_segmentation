import pathlib

from torch.utils.data import default_collate


def collator(batch):
    """
    Ensures that pathlib paths can be collated by the dataloader, and
    ensures that other metadata consist of lists instead of pytorch tensors.
    """
    collated = {}
    skip_collate_subkeys = {
        "Metadata_Input_Slices",
        "Metadata_Target_Slices",
        "Metadata_Crop_Coordinates",
        "Metadata_Dataset_ID",
    }

    def tensor_to_list(v):
        if hasattr(v, "tolist"):
            return v.tolist()
        return v

    for key in batch[0]:
        values = [d[key] for d in batch]

        if isinstance(values[0], pathlib.Path):
            collated[key] = values

        elif key == "metadata":
            metadata_collated = {}
            for meta_key in values[0]:
                meta_values = [v[meta_key] for v in values]

                if meta_key in skip_collate_subkeys:
                    if isinstance(meta_values[0], dict):
                        metadata_collated[meta_key] = {
                            k: tensor_to_list(v)
                            for k, v in default_collate(meta_values).items()
                        }
                    else:
                        metadata_collated[meta_key] = [
                            tensor_to_list(v) for v in meta_values
                        ]
                else:
                    metadata_collated[meta_key] = default_collate(meta_values)

            collated["metadata"] = metadata_collated

        else:
            collated[key] = default_collate(values)

    return collated
