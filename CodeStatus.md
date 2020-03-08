`Status`
* ❔  = the code has not been finished.
* ✔️ = the code is finshed, but can be improved.
* ✅ = the optimisation is finished and the code is ready to release.

The released code should have at least one of the following features. 

`Feature`
* Accurate
* Efficient
* Compact

Each module will be updated by the latest state-of-the-art method. Here, we prefer the methods that are training reproducible.

|   Module           | Main Contributor   | Status  | Training Data    | Testing Data    | Method | Feature|
|--------------------|:------------------:|:-------:|:----------------:|:---------------:|:------:|:------:|
|1. Open Source Code and Data: Overview|Jiankang Deng||||||
|2. Fundamentals of Deep Neural Networks|Aston Zhang||||||
|3. Face Detection|Yang Liu|✔️|WiderFace Train|WiderFace Test|[HAMBox](https://arxiv.org/abs/1912.09231)| Accurate|
|4. Face Alignment|Bin Xiao|✔️|WFLW Train|WFLW Test|[HRNet](https://arxiv.org/abs/1904.04514)|Accurate|
|5. Face Parsing|Jinpeng Lin||Helen Train|Helen Test|[RoI Tanh-Warping](https://arxiv.org/pdf/1906.01342.pdf)|Accurate|
|6. 3D Face Landmarking and Morphing|Jiangzhu Guo|✔️|300W-LP|AFLW2000-3D|[3DDFA](https://arxiv.org/abs/1804.01005)||
|7. Face Similarity Metric Learning|Hao Liu||MS1M|LFW|||
|8. Face Feature Learning|Jiankang Deng|✔️|Celeb500K|IJBC and MegaFace|[ArcFace](https://arxiv.org/abs/1801.07698)|Accurate|
|9. Pose Invariant Face Recognition|Jian Zhao||MS1M|CFP|[DLN](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhao_Towards_Pose_Invariant_CVPR_2018_paper.pdf)||
|10. Age Invariant Face Recognition|Hao Wang||CAF|CACD-VS|[DAL](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhao_Towards_Pose_Invariant_CVPR_2018_paper.pdf)||
|11. 3D Face Recognition|Donghyun Kim|❔|Bosphorus|Bosphorus|[Deep 3D](https://arxiv.org/pdf/1703.10714.pdf)||
|12. Heterogeneous Face Recognition|H. Ahmed|❔|I2BVSD|I2BVSD|[Multi-spectral](https://link.springer.com/chapter/10.1007/978-3-319-68195-5_108)||
|13. Face Spoofing Detection|Shifeng Zhang||CASIA-SURF|CASIA-SURF|[Multi-modal Anti-spoofing](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_A_Dataset_and_Benchmark_for_Large-Scale_Multi-Modal_Face_Anti-Spoofing_CVPR_2019_paper.pdf)||
|14. Adversarial Face Perturbations|Gaurav Goswami||multiDB|MEDS|[Adversarial Perturbations](https://link.springer.com/article/10.1007/s11263-019-01160-w#Sec13)||
|15. Face Expression Recognition|Zhiwen Zhao||EmotioNet Train|EmotioNet Test|[Attention](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zhiwen_Shao_Deep_Adaptive_Attention_ECCV_2018_paper.pdf)||
|16. Face Attribute Recognition|Rasmus Rothe||IMDB-WIKI Train|IMDB-WIKI Test|[DEX](https://data.vision.ee.ethz.ch/cvl/publications/papers/articles/eth_biwi_01299.pdf)||
|17. Multi-task Learning for Face Analysis|Rajeev Ranjan||AFLW|multiDB|[HyperFace](https://arxiv.org/pdf/1603.01249.pdf)||
|18. Face Recognition in Video|Jiaolong Yang||MS1M|YTF|[NAN](https://arxiv.org/pdf/1603.05474.pdf)||
|19. Face Super-resolution|Jing Yang|❔|WiderFace Train|WiderFace Test|[Learn Degradation](http://openaccess.thecvf.com/content_ECCV_2018/papers/Adrian_Bulat_To_learn_image_ECCV_2018_paper.pdf)||
|20. Face Data Augmentation|Iacopo Masi||COW|IJB-A|[Face-Specific Augmentation](https://talhassner.github.io/home/projects/augmented_faces/Masietal_IJCV2019.pdf)||
|21. Face Animation|Justus Thies|❔|||[Face2Face](https://ieeexplore.ieee.org/document/7780631)||
|22. Explainable face recognition|Xiaoming Liu|❔|||[Interpretable Face](https://arxiv.org/pdf/1805.00611.pdf)||
|23. Bias in Face Recognition|Xiaoming Liu|❔|||[De-biasing](https://arxiv.org/pdf/1911.08080.pdf)||
|24. Face Recognition System|Jiankang Deng|✔️|Celeb500K|FRVT|[ArcFace](https://arxiv.org/abs/1801.07698)|Accurate|
