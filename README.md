Contextual Encoding for Translation Quality Estimation
===
Implemented by [Junjie Hu](http://www.cs.cmu.edu/~junjieh/)

Contact: junjieh@cs.cmu.edu

If you use the codes in this repo, please cite our [WMT18 paper](http://aclweb.org/anthology/W18-6462).

	@InProceedings{hu-EtAl:2018:WMT,
	  author    = {Hu, Junjie  and  Chang, Wei-Cheng  and  Wu, Yuexin  and  Neubig, Graham},
	  title     = {Contextual Encoding for Translation Quality Estimation},
	  booktitle = {Proceedings of the Third Conference on Machine Translation: Shared Task Papers},
	  month     = {October},
	  year      = {2018},
	  address   = {Belgium, Brussels},
	  publisher = {Association for Computational Linguistics},
	  pages     = {788--793},
	  url       = {http://www.aclweb.org/anthology/W18-6462}
	  }
	


Install 
==
    sklearn 
    pytorch 0.4.1
    scipy
    scipy.stats


Download Data
==
	cd data/WMT2018QE/
	bash download_data.sh

Train Model
==
    bash scripts/WMT2018QE/de-en/train-word-ClBasic-final.sh [GPU ID]
