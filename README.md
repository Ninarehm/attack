# Exacerbating Algorithmic Bias through Fairness Attacks
This repository contains code for the introduced attacks in [Exacerbating Algorithmic Bias through Fairness Attacks](https://arxiv.org/pdf/2012.08723.pdf) paper. If you find it useful please cite: 
```
@article{mehrabi2020exacerbating,
  title={Exacerbating Algorithmic Bias through Fairness Attacks},
  author={Mehrabi, Ninareh and Naveed, Muhammad and Morstatter, Fred and Galstyan, Aram},
  journal={arXiv preprint arXiv:2012.08723},
  year={2020}
}
```

This code builds upon the code developed by Pang Wei Koh and Percy Liang in 2017. We have left their LICENSE.md file to give due credit to these researchers, and to document that their license allows us to build upon their work. Please also give them credit by citing: 

 ```
@article{koh2018stronger,
  title={Stronger data poisoning attacks break data sanitization defenses},
  author={Koh, Pang Wei and Steinhardt, Jacob and Liang, Percy},
  journal={arXiv preprint arXiv:1811.00741},
  year={2018}
}
 ```
 ```
@inproceedings{koh2017understanding,
  title={Understanding black-box predictions via influence functions},
  author={Koh, Pang Wei and Liang, Percy},
  booktitle={Proceedings of the 34th International Conference on Machine Learning-Volume 70},
  pages={1885--1894},
  year={2017},
  organization={JMLR. org}
}
 ```

If you find the influence attack on fairness useful you may also cite:
 ```
@article{zafar2015learning,
  title={Learning fair classifiers},
  author={Zafar, Muhammad Bilal and Valera, Isabel and Rodriguez, Manuel Gomez and Gummadi, Krishna P},
  journal={stat},
  volume={1050},
  pages={29},
  year={2015}
}
 ```
The citations of the datasets are as follows:
	For German and Drug consumption datasets cite:
 ```
@misc{Dua:2019 ,
author = "Dua, Dheeru and Graff, Casey",
year = "2017",
title = "{UCI} Machine Learning Repository",
url = "http://archive.ics.uci.edu/ml",
institution = "University of California, Irvine, School of Information and Computer Sciences" }
 ```
 For the COMPAS dataset cite: 	
 ```
@article{larson2016compas,
  title={Compas analysis},
  author={Larson, J and Mattu, S and Kirchner, L and Angwin, J},
  journal={GitHub, available at: https://github. com/propublica/compas-analysis[Google Scholar]},
  year={2016}
}
 ```
# Tested Environments 
The code was tested in different environments. The following requirements would work:  
Python 3.6  
1.11.0 < Tensorflow <= 1.12.3  
0.20.3 <= scikit-learn <= 0.23.1  
cvxpy 0.4.11  
CVXcanon <= 0.1.1  
scipy 1.1.0  
0.23 <= Pandas <= 1.1.4  
Matplotlib <= 3.3.3  
seaborn <= 0.11.0  
IPython <= 7.16.1  


# Running Instructions
The dataset can be replaced by the dataset of your choice.

To run the influence attack on fairness (IAF):
```bash
python run_gradient_em_attack.py --em_iter 0 --total_grad_iter 10000 --dataset german --use_slab --epsilon 0.1 --method IAF --sensitive_feature_idx 36 --sensitive_attr_filename german_group_label.npz
```

To run the random anchoring attack (RAA):
```bash
python run_gradient_em_attack.py --em_iter 0 --total_grad_iter 10000 --dataset german --use_slab --epsilon 0.1 --method RAA --sensitive_feature_idx 36 --sensitive_attr_filename german_group_label.npz
```

To run the non-random anchoring attack (NRAA):
```bash
python run_gradient_em_attack.py --em_iter 0 --total_grad_iter 10000 --dataset german --use_slab --epsilon 0.1 --method NRAA --sensitive_feature_idx 36 --sensitive_attr_filename german_group_label.npz
```

# Contact  
If you have questions you can contact ninarehm at usc.edu
