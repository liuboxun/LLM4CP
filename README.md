# LLM4CP
B. Liu, X. Liu, S. Gao, X. Cheng and L. Yang, "LLM4CP: Adapting Large Language Models for Channel Prediction," in Journal of Communications and Information Networks, vol. 9, no. 2, pp. 113-125, June 2024, doi: 10.23919/JCIN.2024.10582829. [[paper]](https://ieeexplore.ieee.org/document/10582829)
<br>

## Dependencies and Installation
- Python 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/))
- Pytorch 2.0.0
- NVIDIA GPU + CUDA
- Python packages: `pip install -r requirements.txt`


## Dataset Preparation
The datasets used in this paper can be downloaded in the following links.  
[[Training Dataset]](https://pan.baidu.com/s/19DtLPftHomCb6_1V2lREtw?pwd=3gbv)
[[Testing Dataset]](https://pan.baidu.com/s/10KzmwC1jncozOGNZ02Hlaw?pwd=sxfd)

 


## Get Started
Training and testing codes are in the current folder. 

-   The code for training is in `train.py`, while the code for test is in `test_tdd_full.py` and `test_fdd_full.py`. we also provide our pretrained model in [[Weights]](https://pan.baidu.com/s/1lysOqCyw44SGDQrH33Os5Q?pwd=nmqw).
    
-   For full shot training, you need to set the file_path in the main function to match your training dataset. For example, if you want to try a full-shot experiment in a TDD scenario, you need to modify the `train_TDD_r_path` and `train_TDD_t_path` in `train.py` to the locations of your downloaded `H_U_his_train.mat` and `H_U_pre_train.mat`, respectively. Then, you can run `train.py`.
-   For few shot training, you need to set the file_path in the main function to match your training dataset. Then, you can set `is_few=1` when creating the training set in `train.py` like this: `train_set = Dataset_Pro(train_TDD_r_path, train_TDD_t_path, is_few=1)` and run `train.py`.

-   For testing, you also need to set the file_path in the main function to match your testing dataset. Then, you can run `test_tdd_full.py` to obtain the results in Figure 7 of the paper, and you can run `test_fdd_full.py` to obtain the results in Figure 8 of the paper. You can also try loading the data under `Testing Dataset/Umi` to test the models' zero-shot performance.

## Citation
If you find this repo helpful, please cite our paper.
```latex
@article{liu2024llm4cp,
  title={LLM4CP: Adapting Large Language Models for Channel Prediction},
  author={Liu, Boxun and Liu, Xuanyu and Gao, Shijian and Cheng, Xiang and Yang, Liuqing},
  journal={arXiv preprint arXiv:2406.14440},
  year={2024}
```