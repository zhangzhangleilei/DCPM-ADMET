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
Pre-training [data](https://drive.google.com/drive/folders/1OB6pemOHuCrgLjsDDu5z0_VoFxMqPjp4)
MoleculeNet [data](https://drive.google.com/drive/folders/1OUOD88vmmotz0yFa-BmKZBOSmF4F-k4e)
Fintune classification [data](https://drive.google.com/drive/folders/1aQyotm79I5GT2sEGI_OgQdeQDDPC1NiT)
Fintune regression [data](https://drive.google.com/drive/folders/1-7i7YSSs4QTrc5k7yf980Yk85rA8UPGn)
<br>

## Code
We have provided the pre-training [code](https://github.com/zhangzhangleilei/DCPM-ADMET/blob/main/pretrain/pretrain.py)

```bash
python pretrain.py --"data_url" --"device_num" --"save_url" --"check_point_path"
```

You can use the fine-tuning [code](https://github.com/zhangzhangleilei/DCPM-ADMET/blob/main/fintune/finetune.py) to obtain a custom model that matches your task
```bash
python fintune.py --"fintune data path" --"model path" --"save model path" --"checkpoint"
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
