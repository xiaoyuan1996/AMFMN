cd ..
CARD=7

CUDA_VISIBLE_DEVICES=$CARD python train.py --path_opt option/RSITMD_AMFMN.yaml

CUDA_VISIBLE_DEVICES=$CARD python test.py --path_opt option/RSITMD_AMFMN.yaml
