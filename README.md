# 基于PaddleX的人脸、戴口罩和吸烟目标检测

## 一、安装PaddleX


```python
!pip install paddlex==2.0.0
```

## 二、解压数据集


```python
!unzip -oq /home/aistudio/data/data128757/VOC_MASK_smoke.zip 
```

## 三、生成数据集文件

PaddleX支持VOC格式数据，
## **安装Paddlex**
	首先安装1.83以上版本的Paddlex（1.83一下版本的已经不支持此划分命令）


```python
#如果下一句代码报错，请先运行此代码
#!pip install paddlex==2.0rc -q
```


```python
!paddlex --split_dataset --format VOC --dataset_dir VOC_MASK_smoke --val_value 0.2 --test_value 0.1
```

## 四、数据预处理
这里使用了图像混合、随机像素变换、随机膨胀、随即裁剪、随机水平翻转等数据增强方法。


```python
from paddlex import transforms as T

train_transforms = T.Compose([
    T.MixupImage(mixup_epoch=250), T.RandomDistort(),
    T.RandomExpand(im_padding_value=[123.675, 116.28, 103.53]), T.RandomCrop(),
    T.RandomHorizontalFlip(), T.BatchRandomResize(
        target_sizes=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608],
        interp='RANDOM'), T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transforms = T.Compose([
    T.Resize(
        608, interp='CUBIC'), T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```


```python

train_dataset = pdx.datasets.VOCDetection(
    data_dir='VOC_MASK_smoke',
    file_list='VOC_MASK_smoke/train_list.txt',
    label_list='VOC_MASK_smoke/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.VOCDetection(
    data_dir='VOC_MASK_smoke',
    file_list='VOC_MASK_smoke/val_list.txt',
    label_list='VOC_MASK_smoke/labels.txt',
    transforms=eval_transforms)
```

## 五、训练模型
这里定义了一个YOLOv3，使用DarkNet53作为主干网络（该参数训练50epoch差不多就收敛了）；


```python
import paddlex as pdx
num_classes = len(train_dataset.labels)
model = pdx.det.YOLOv3(num_classes=num_classes, backbone='DarkNet53')
model.train(
    num_epochs=270,
    train_dataset=train_dataset,
    train_batch_size=8,
    eval_dataset=eval_dataset,
    learning_rate=0.000125,
    lr_decay_epochs=[216, 243],
    warmup_steps=1000,
    warmup_start_lr=0.0,
    save_interval_epochs=5,
    save_dir='output/yolov3_darknet53',
    use_vdl=True)
```

## 六、模型检测结果及可视化


```python
import paddlex as pdx
eval_details_file = 'output/yolov3_darknet53/best_model/eval_details.json'
pdx.det.draw_pr_curve(eval_details_file, save_dir='./output/yolov3_darknet53')
```

# 目标检测/实例分割准确率-召回率可视化

![](https://ai-studio-static-online.cdn.bcebos.com/50e7d2e4958c4cb4b83f1dc9e1e41058bb4ec03a872c4598b81d7357a886ace0)



```python
#预测模型
import paddlex as pdx
model = pdx.load_model('output/yolov3_darknet53/best_model')
image_name = 'VOC_MASK_smoke/JPEGImages/1_Handshaking_Handshaking_1_164.jpg'
result = model.predict(image_name)
pdx.det.visualize(image_name, result, threshold=0.5, save_dir='./output/yolov3_darknet53')
```

    2022-02-20 15:29:23 [INFO]	Model[YOLOv3] loaded.
    2022-02-20 15:29:23 [INFO]	The visualized result is saved at ./output/yolov3_darknet53/visualize_1_Handshaking_Handshaking_1_164.jpg



![](https://ai-studio-static-online.cdn.bcebos.com/9dfd2df10171457ca16e96ccbb5618968fc4643ac55140fe89e75586cedbf646)

