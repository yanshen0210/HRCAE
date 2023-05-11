# HRCAE: A new unsupervised anomaly detection method for machine tools under noises
* Core codes for the paper ["Hybrid robust convolutional autoencoder for unsupervised anomaly detection of machine tools under noises"](https://www.sciencedirect.com/science/article/pii/S0736584522001259)
* Created by Shen Yan, Haidong Shao, Yiming Xiao, Bin Liu, Jiafu Wan.
* Journal: Robotics and Computer-Integrated Manufacturing
## Our operating environment
* Python 3.8
* pytorch  1.10.1
* and other necessary libs

## Guide 
* This repository provides a concise framework for unsupervised anomaly detection for machine tools under noises. It includes a demo dataset; the pre-processing process for the data and the model proposed in the paper. We have also integrated 2 baseline methods for comparison.
* You just need to run `start_procedure.py`. You can also adjust the structure and parameters of the model to suit your needs.

## Pakages
* `data` contians a demo dataset
* `datasets` contians the pre-processing process and the type of added noise for the data
* `models` contians the proposed model and 2 base models
* `utils` contians train&val&test processes

## Citation
If you use our work as a comparison model, please cite:
```
@paper{HRCAE,
  title = {Hybrid robust convolutional autoencoder for unsupervised anomaly detection of machine tools under noises},
  author = {Shen Yan, Haidong Shao, Yiming Xiao, Bin Liu, Jiafu Wan},
  journal = {Robotics and Computer-Integrated Manufacturing},
  volume = {79},
  pages = {102441},
  year = {2023},
  doi = {https://doi.org/10.1016/j.rcim.2022.102441},
  url = {https://www.sciencedirect.com/science/article/pii/S0736584522001259},
}
```
If our work is useful to you, please star it, it is the greatest encouragement to our open source work, thank you very much!

## Contact
- yanshen0210@gmail.com
