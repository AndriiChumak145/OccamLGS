# DATASET_NAME="teatime"
# OUTPUT_DIR="/home/dani/concept-scenesplat/OccamLGS/outputs"
DATASET_NAME="ycbv_000052"
OUTPUT_DIR="/home/dani/concept-scenesplat/OccamLGS/outputs/"

# cd ~/workspace/occamlgs

# python train.py -s /home/dani/concept-scenesplat/data/$DATASET_NAME -m $OUTPUT_DIR/$DATASET_NAME --iterations 30000
# python train.py \
#   -s /home/dani/concept-scenesplat/concept-pose/data/ycbv/test/000052 \
#   -m $OUTPUT_DIR/$DATASET_NAME \
#   --iterations 30000 \
#   --images rgb \
#   --mask_rgb \
#   --mask_source mask_visib \
#   --mask_instance_idx 2

python train.py \
  -s /home/dani/concept-scenesplat/data/$DATASET_NAME \
  -m $OUTPUT_DIR/$DATASET_NAME \
  --iterations 30000 \
  --images rgb

# python render.py -m $OUTPUT_DIR/$DATASET_NAME --iteration 30000

# python gaussian_feature_extractor.py -m $OUTPUT_DIR/$DATASET_NAME --iteration 30000 --eval --feature_level 1 \
  # --feature_dim 1536 --max_feature_views 25
# python feature_map_renderer.py -m $OUTPUT_DIR/$DATASET_NAME \
#   --iteration 30000 --eval --feature_level 1 \
#   --feature_dim 1536 --max_feature_views 3

# python feature_map_renderer.py -m $OUTPUT_DIR/$DATASET_NAME \
#   --iteration 30000 --eval --feature_level 1 --max_feature_views 3