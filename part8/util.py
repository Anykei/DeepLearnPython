import os, shutil, pathlib

original_dir = pathlib.Path("./cat_vs_dog/training_set/training_set")
# original_dir = pathlib.Path("./cat_vs_dog/training_set/training_set")
new_base_dir = pathlib.Path("cat_vs_dog_small")


def make_subset(subset_name, start_index, end_index):
    for category in ("cat", "dog"):
        dir = new_base_dir / subset_name / category
        os.makedirs(dir)
        fnames = [f"{category}.{i}.jpg" for i in range(start_index, end_index)]
        for fname in fnames:
            shutil.copyfile(src=original_dir / category / fname, dst=dir / fname)


make_subset("train", start_index=1, end_index=1000)
make_subset("validation", start_index=1000, end_index=1500)
make_subset("test", start_index=1500, end_index=2500)
