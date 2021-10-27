# readme for dataset
> 为了方便暂时用中文描述，之后改成英文

- `basic_dataset.py`：此文件定义了我们自己采集的ithor-syn数据集。该数据集由一个个物体组成，每个物体目录下有一个`imgs`目录和一个`masks`目录，分别包含rgb image和binary mask。使用`BasicDataset`需要传入`img_dir`和`mask_dir`。在我们的实验中`resize`一般设置为(256, 256)。`BasicDataset_nshot`是`BasicDataset`的子集，只包含n个training sample，用于做n-shot。

- `davis_dataset.py`：DAVIS数据集