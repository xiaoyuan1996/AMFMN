## AMFMN
### Welcome :+1:_<big>`Fork and Star`</big>_:+1:, then we'll let you know when we update

```bash
#### News:
#### 2021.5.22: ---->AMFMN is expected to be released after the RSITMD open to access.<----
#### 2021.7.29: ---->The code of AMFMN is expected to be released before September<----
#### 2021.8.03: ---->The code of AMFMN has been open to access<----
```



```bash
Installation

We recommended the following dependencies:
Python 3
PyTorch > 0.3
Numpy
h5py
nltk
yaml
```

```bash
File Structure:
-- checkpoint    # savepath of ckpt and logs

-- data          # soorted anns of four datesets
    -- rsicd_precomp
        -- train_caps.txt     # train anns
        -- train_filename.txt # corresponding imgs
        -- test_caps.txt      # test anns
        -- test_filename.txt  # corresponding imgs
        -- images             # rsicd images here
    -- rsitmd_precomp
        ...
    -- sydney_precomp
        ...
    -- ucm_precomp
        ...

-- exec         # .sh file

-- layers        # models define

-- logs          # tensorboard save file

-- option        # different config for different datasets and models

-- Rct           # calc Lct, which is not published this time

-- util          # some script for data processing

-- vocab         # vocabs for different datasets

-- seq2vec       # some files about seq2vec
    -- bi_skip.npz
    -- bi_skip.npz.pkl
    -- btable.npy
    -- dictionary.txt
    -- uni_skip.npz
    -- uni_skip.npz.pkl
    -- utable.npy

-- data.py       # load data
-- engine.py     # details about train and val
-- test.py       # test k-fold answers
-- test_single.py    # test one model
-- train.py      # main file
-- utils.py      # some tools
-- vocab.py      # generate vocab

Note:
1. In order to facilitate reproduction, we have provided processed annotations.
2. We prepare some used file::
  (1)[seq2vec (Password:NIST)](https://pan.baidu.com/s/1jz61ZYs8NZflhU_Mm4PbaQ)
  (2)[RSICD images (Password:NIST)](https://pan.baidu.com/s/1lH5m047P9m2IvoZMPsoDsQ)
  (3)[UCM images (Password:NIST)](https://pan.baidu.com/s/1K7vfmx2TfMFyViqTyJQSgg)
  (4)[SYDNEY images (Password:NIST)](https://pan.baidu.com/s/1C-P674uYh7-XKUDpcJL2aA)
```



```bash
Run: (We take the dataset RSITMD as an example)
Step1:
    Put the images of different datasets in ./data/{dataset}_precomp/images/

    --data
        --rsitmd_precomp
        -- train_caps.txt     # train anns
        -- train_filename.txt # corresponding imgs
        -- test_caps.txt      # test anns
        -- test_filename.txt  # corresponding imgs
        -- images             # images here
            --img1.jpg
            --img2.jpg
            ...

Step2:
    Modify the corresponding yaml in ./options.

    Regard RSITMD_AMFMN.yaml as opt, which you need to change is:
        opt['dataset']['data_path']  # change to precomp path
        opt['dataset']['image_path']  # change to image path
        opt['model']['seq2vec']['dir_st'] # some files about seq2vec

Step3:
    Bash the ./sh in ./exec.
    Note the GPU define in specific .sh file.

    cd exec
    bash run_amfmn_rsitmd.sh

Note: We use k-fold verity to do a fair compare. Other details please see the code itself.
```

## Citation
If you feel this code helpful or use this code or dataset, please cite it as
```
Z. Yuan et al., "Exploring a Fine-Grained Multiscale Method for Cross-Modal Remote Sensing Image Retrieval," in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2021.3078451.
```
