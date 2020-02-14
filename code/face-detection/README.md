## Face Detection by S³FD: Single Shot Scale-invariant Face Detector

Shifeng Zhang et al

### Requirement
* pytorch 
* opencv 
* numpy 
* easydict

### Prepare data 
1. download WIDER face dataset
2. modify data/config.py 
3. ``` python prepare_wider_data.py ```


### Train
``` 
python train.py --batch_size 4 --dataset face
``` 

### Evalution
according to yourself dataset path,modify data/config.py 
1. Evaluate on AFW.
```
python afw_test.py
```
2. Evaluate on FDDB 
```
python fddb_test.py
```
3. Evaluate on PASCAL  face 
``` 
python pascal_test.py
```
4. test on WIDER FACE 
```
python wider_test.py
```
### Demo 

```
python demo.py
```

### Result
1. AFW PASCAL FDDB
<div align="center">
<img src="https://github.com/yxlijun/S3FD.pytorch/blob/master/img/AFW.png" height="200px" alt="afw" >
<img src="https://github.com/yxlijun/S3FD.pytorch/blob/master/img/pascal.png" height="200px" alt="pascal" >
<img src="https://github.com/yxlijun/S3FD.pytorch/blob/master/img/FDDB.png" height="200px" alt="fddb" >     
</div>

	AFW AP=99.81 paper=99.85 
	PASCAL AP=98.77 paper=98.49
	FDDB AP=0.975 paper=0.983
	WIDER FACE:
	Easy AP=0.925 paper = 0.927
	Medium AP=0.925 paper = 0.924
	Hard AP=0.854 paper = 0.852

2. demo
<div align="center">
<img src="https://github.com/yxlijun/S3FD.pytorch/blob/master/tmp/test2.jpg" height="400px" alt="afw" >
</div>


### References
If you find this work or code is helpful in your research, please cite:
````
@InProceedings{Zhang2017ICCV,
author = {Zhang, Shifeng and Zhu, Xiangyu and Lei, Zhen and Shi, Hailin and Wang, Xiaobo and Li, Stan Z.},
title = {S3FD: Single Shot Scale-Invariant Face Detector},
booktitle = {ICCV},
year = {2017}
}
````