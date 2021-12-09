# -----------------------------------------------------------
# "Exploring a Fine-Grained Multiscale Method for Cross-Modal Remote Sensing Image Retrieval"
# Yuan, Zhiqiang and Zhang, Wenkai and Fu, Kun and Li, Xuan and Deng, Chubo and Wang, Hongqi and Sun, Xian
# IEEE Transactions on Geoscience and Remote Sensing 2021
# Writen by YuanZhiqiang, 2021.  Our code is depended on MTFN
# ------------------------------------------------------------
import nltk
import argparse
import numpy as np
from tqdm import tqdm

from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction,corpus_bleu
from nltk.corpus import stopwords
stop_words = [k for k in set(stopwords.words('english'))] + [',', '.']
from nltk.stem import PorterStemmer
ps = PorterStemmer()

# Load from txt
def load_from_txt(filename, encoding="utf-8"):
    f = open(filename,'r' ,encoding=encoding)
    contexts = f.readlines()
    return contexts

def calc_sim(sent_visual, sent_caption):
    def sent_preprocess(sent):
        # tokenize
        sent = word_tokenize(sent)

        # PorterStemmer
        sent = [ps.stem(word.lower()) for word in sent]

        # stop words and PorterStemmer
        sent_remove_stop_words = [word for word in sent if word not in stop_words]
        return sent, sent_remove_stop_words

    # preprocess
    visual_words = [list(sent_preprocess(sent)) for sent in sent_visual]

    sent_visual = [vw[0] for vw in visual_words]
    sent_visual_remove_stop_words = [vw[1] for vw in visual_words]

    sent_caption, sent_caption_remove_stop_words = sent_preprocess(sent_caption)

    # score1: bleu --> semantic
    if sent_caption in sent_visual:
        score1 = 1
    else:
        score1 = sentence_bleu(
            sent_visual,
            sent_caption,
            weights = (0.25, 0.25, 0.25, 0.25),
         )

    # score2: entity --> physical meaning
    tmp_score = [len(set(v_sent) & set(sent_caption_remove_stop_words))/len(v_sent) for v_sent in sent_visual_remove_stop_words if len(v_sent) != 0 ]
    score2 = np.mean(tmp_score)

    return np.mean([score1, score2])

def main_process(val_txt_path, image_text_rate, output_dir):

    # load from txt
    txt_pool = load_from_txt(val_txt_path)

    # calc image and caption size
    image_size = int(len(txt_pool) // image_text_rate)
    caption_size = len(txt_pool)

    # define sims
    sim_t2t_matrix = np.zeros((image_size, caption_size))

    # calc sims
    for i in tqdm(range(image_size)):
        sent_visual = txt_pool[i*image_text_rate : (i+1)*image_text_rate] # extract sent_visual
        for j in range(caption_size):
            sent_caption = txt_pool[j]
            s = calc_sim(sent_visual, sent_caption)
            sim_t2t_matrix[i, j] = s

    # save
    np.save(output_dir + val_txt_path.split("/")[-1].split(".")[0] + ".npy",
            sim_t2t_matrix,
            allow_pickle=True)
    print("save successful")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_txt_path', default='../data/coco_precomp/val_caps.txt',
                        help='Path to val caps')
    parser.add_argument('--image_text_rate', default=5,
                        help='int(text // image)')
    parser.add_argument('--output_dir', default='../data/coco_precomp/',
                        help='Output directory.')
    opt = parser.parse_args()
    print(opt)

    # process
    main_process(
        val_txt_path=opt.val_txt_path,
        image_text_rate= opt.image_text_rate,
        output_dir= opt.output_dir
    )

    # sent_visual = [
    #     "There are some buildings with white and grey roofs .",
    #     "Some buildings with white and grey roofs .",
    #     "There are some buildings with cars parked beside them .",
    #     "There are some buildings with cars parked beside them .",
    #     "Buildings and cars ."
    # ]
    # sent_caption = "Buildings and cars ."
    # sim = calc_sim(sent_visual, sent_caption)
    # print(sim)
