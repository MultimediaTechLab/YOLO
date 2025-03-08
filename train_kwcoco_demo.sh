#!/bin/bash
__doc__="
YOLO Training Tutorial with KWCOCO DemoData
===========================================

This demonstrates an end-to-end YOLO pipeline on toydata generated with kwcoco.
"

# Define where we will store results
BUNDLE_DPATH=$HOME/demo-yolo-kwcoco-train
mkdir -p "$BUNDLE_DPATH"

echo "
Generate Toy Data
-----------------

Now that we know where the data and our intermediate files will go, lets
generate the data we will use to train and evaluate with.

The kwcoco package comes with a commandline utility called 'kwcoco toydata' to
accomplish this.
"

# Define the names of the kwcoco files to generate
TRAIN_FPATH=$BUNDLE_DPATH/vidshapes_rgb_train/data.kwcoco.json
VALI_FPATH=$BUNDLE_DPATH/vidshapes_rgb_vali/data.kwcoco.json
TEST_FPATH=$BUNDLE_DPATH/vidshapes_rgb_test/data.kwcoco.json

# Generate toy datasets using the "kwcoco toydata" tool
kwcoco toydata vidshapes32-frames10 --dst "$TRAIN_FPATH"
kwcoco toydata vidshapes4-frames10 --dst "$VALI_FPATH"
kwcoco toydata vidshapes2-frames6 --dst "$TEST_FPATH"

# Ensure legacy COCO structure for now
kwcoco conform "$TRAIN_FPATH" --inplace --legacy=True
kwcoco conform "$VALI_FPATH" --inplace --legacy=True
kwcoco conform "$TEST_FPATH" --inplace --legacy=True


echo "
Create the YOLO Configuration
-----------------------------

Constructing the YOLO configuration is not entirely kwcoco aware
so we need to set
"
# In the current version we need to write configs to the repo itself.
# Its a bit gross, but this should be somewhat robust.
# Find where the yolo repo is installed (we need to be careful that this is the
# our fork of the WongKinYiu variant
REPO_DPATH=$(python -c "import yolo, pathlib; print(pathlib.Path(yolo.__file__).parent.parent)")
MODULE_DPATH=$(python -c "import yolo, pathlib; print(pathlib.Path(yolo.__file__).parent)")
CONFIG_DPATH=$(python -c "import yolo.config, pathlib; print(pathlib.Path(yolo.config.__file__).parent / 'dataset')")
echo "REPO_DPATH = $REPO_DPATH"
echo "MODULE_DPATH = $MODULE_DPATH"
echo "CONFIG_DPATH = $CONFIG_DPATH"

DATASET_CONFIG_FPATH=$CONFIG_DPATH/kwcoco-demo.yaml

# Hack to construct the class part of the YAML
CLASS_YAML=$(python -c "if 1:
    import kwcoco
    train_fpath = kwcoco.CocoDataset('$TRAIN_FPATH')
    categories = train_fpath.categories().objs
    # It would be nice to have better class introspection, but in the meantime
    # do the same sorting as yolo.tools.data_conversion.discretize_categories
    categories = sorted(categories, key=lambda cat: cat['id'])
    class_num = len(categories)
    class_list = [c['name'] for c in categories]
    print(f'class_num: {class_num}')
    print(f'class_list: {class_list}')
")


CONFIG_YAML="
path: $BUNDLE_DPATH
train: $TRAIN_FPATH
validation: $VALI_FPATH

$CLASS_YAML
"

echo "$CONFIG_YAML" > "$DATASET_CONFIG_FPATH"


# This might only work in development mode, otherwise we will get site packages
# That still might be fine, but we do want to fix this to run anywhere.
cd "$REPO_DPATH"
LOG_BATCH_VIZ_TO_DISK=1 python -m yolo.lazy \
    task=train \
    dataset=kwcoco-demo \
    use_wandb=False \
    out_path="$BUNDLE_DPATH"/training \
    name=kwcoco-demo \
    cpu_num=0 \
    device=0 \
    accelerator=auto \
    task.data.batch_size=2 \
    "image_size=[224,224]" \
    task.optimizer.args.lr=0.0003

LOG_BATCH_VIZ_TO_DISK=1 python -m yolo.lazy \
    task=train \
    dataset=kwcoco-demo \
    use_wandb=False \
    out_path="$BUNDLE_DPATH"/training \
    name=kwcoco-demo \
    cpu_num=0 \
    device=0 \
    accelerator=auto \
    task.data.batch_size=2 \
    "image_size=[224,224]" \
    task.optimizer.args.lr=0.0003


### TODO: show how to validate

# Grab a checkpoint
CKPT_FPATH=$(python -c "if 1:
    import pathlib
    ckpt_dpath = pathlib.Path('$BUNDLE_DPATH') / 'training/train/kwcoco-demo/checkpoints'
    checkpoints = sorted(ckpt_dpath.glob('*'))
    print(checkpoints[-1])
")
echo "CKPT_FPATH = $CKPT_FPATH"


#DISABLE_RICH_HANDLER=1
LOG_BATCH_VIZ_TO_DISK=1 python -m yolo.lazy \
    task=validation \
    dataset=kwcoco-demo \
    use_wandb=False \
    out_path="$BUNDLE_DPATH"/training \
    name=kwcoco-demo \
    cpu_num=0 \
    device=0 \
    weight="'$CKPT_FPATH'" \
    accelerator=auto \
    "task.data.batch_size=2" \
    "image_size=[224,224]"


### TODO: show how to run inference
