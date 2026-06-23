#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

DATASET_NAME="ycbv_000052"
OUTPUT_DIR="/home/dani/concept-scenesplat/OccamLGS/outputs"
MAX_FEAT_VIEWS=(
    25
    900
)

CONFIGS=(
    "configs/siglip_so400m.yaml"
    "configs/siglip_giant.yaml"
    "configs/siglip_large.yaml"
    "configs/siglip_base.yaml"
)

for MAX_FEAT_VIEWS in "${MAX_FEAT_VIEWS[@]}"; do

    for CONFIG_PATH in "${CONFIGS[@]}"; do
        
        # Extract filename without extension (e.g., "siglip_so400m")
        CONFIG_NAME=$(basename "$CONFIG_PATH" .yaml)
        
        echo "=========================================================="
        echo " Running Sweep: ${MAX_FEAT_VIEWS} views | Model: $CONFIG_NAME"
        echo "=========================================================="

        python gaussian_feature_extractor.py \
            -m "$OUTPUT_DIR/$DATASET_NAME" \
            --iteration 30000 \
            --eval \
            --feature_level 1 \
            --max_feature_views "$MAX_FEAT_VIEWS" \
            --online_siglip \
            --config "$CONFIG_PATH" \
            --upsample_method "naf"

        python feature_map_renderer.py \
            -m "$OUTPUT_DIR/$DATASET_NAME" \
            --iteration 30000 \
            --eval \
            --feature_level 1 \
            --skip_test \
            --max_feature_views 1 \
            --online_siglip \
            --config "$CONFIG_PATH" \
            --upsample_method "naf"

        echo "Generating Grad-CAM overlay for ${MAX_FEAT_VIEWS}v_${CONFIG_NAME}..."
        cd ..
        python other/late_GradCAM.py \
            --config "OccamLGS/$CONFIG_PATH" \
            --prefix "${MAX_FEAT_VIEWS}v_${CONFIG_NAME}"
        cd OccamLGS
            
    done
done

echo "Sweep complete!"