cd ..
CARD=5

CUDA_VISIBLE_DEVICES=$CARD python train.py --path_opt option/RSICD_AMFMN.yaml

CUDA_VISIBLE_DEVICES=$CARD python test.py --path_opt option/RSICD_AMFMN.yaml
