from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader, has_file_allowed_extension
from typing import Callable, Optional, Tuple, Any, List, Dict, Union, cast
import os

IMG_EXTENSIONS = [".jpg", ".png", ".jpeg"]


def make_dataset_empty_okay(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)

    # if class_to_idx is None:
    #     _, class_to_idx = find_classes(directory)
    # elif not class_to_idx:
    #     raise ValueError(
    #         "'class_to_index' must have at least one entry to collect any samples."
    #     )

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError(
            "Both extensions and is_valid_file cannot be None or not None at the same time"
        )

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    # empty_classes = set(class_to_idx.keys()) - available_classes
    # if empty_classes:
    #     msg = (
    #         f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
    #     )
    #     if extensions is not None:
    #         msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
    #     raise FileNotFoundError(msg)

    return instances


# WIP!!!
# modified to align classes that may be missing.
class HotspotDatasetFolder(DatasetFolder):
    def __init__(
        self,
        root: str,
        max_category_id: int,
        loader: Callable[[str], Any] = default_loader,
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self.max_category_id = max_category_id
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform,
            target_transform,
            is_valid_file,
        )

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_dataset_empty_okay(
            directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file
        )

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        category_ids = sorted(
            entry.name for entry in os.scandir(directory) if entry.is_dir()
        )
        if not category_ids:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {str(i): i for i in range(self.max_category_id)}
        return category_ids, class_to_idx
