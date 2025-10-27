# DCPM-ADMET
ADMET predict
Code and trained model for our paper **DCPM-ADMET: Fusion of Dual-channel Pre-trained Model and Molecular Fingerprints to enhance Drug ADMET Properties Prediction**
![ADMET ](https://github.com/zhangzhangleilei/DCPM-ADMET/blob/main/admet.jpg)
<br>

## Installation
DCPM-ADMET can be downloaded by following the commands below.
```bash
git clone zhangzhangleilei/DCPM-ADMET.git
cd DCPM-ADMET

or

wget https://github.com/zhangzhangleilei/DCPM-ADMET.git
cd DCPM-ADMET
```
<br>

## Data
Pre-training [data](https://drive.google.com/drive/folders/1f4EJeGH-_pI642Axghp47LmfoCNfi91u) <br>
MoleculeNet [data](https://drive.google.com/drive/folders/1OUOD88vmmotz0yFa-BmKZBOSmF4F-k4e) <br>

<br>

## Code
We have provided the pre-training [code](https://github.com/zhangzhangleilei/DCPM-ADMET/blob/main/pretrain/train.py)
```bash
python train.py --"testdata" [test_data.csv](https://github.com/zhangzhangleilei/DCPM-ADMET/blob/main/pretrain/test_data.csv) --"testpkl" [test_molfeats.pkl](https://github.com/zhangzhangleilei/DCPM-ADMET/blob/main/pretrain/test_molfeats.pkl) --"data_url" [pretrain.csv](https://drive.google.com/drive/folders/1f4EJeGH-_pI642Axghp47LmfoCNfi91u) --"save_url" ./
```
<br>

## Web
We have developed a web server for the above process to facilitate its usage.
```bash
http://admet.bioai-global.com/
```
<br>

## Contact
If you have any problems with downloading or using the model, please contact zhangleilei0327@163.com. We will reply in a timely manner upon seeing your message.
