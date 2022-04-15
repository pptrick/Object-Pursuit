# readme for `coeffnet`
> 为了方便暂时用中文描述，之后改成英文

## hypernet
定义hypernet最上层结构的文件为`hypernet.py`，其中`class Hypernet(nn.Module)`为hypernet的整体结构，包含若干个conv-hyper block。

定义hypernet block结构的文件为`hypernet_block.py`，其中`HypernetFCBlock`为fully-connected版本的block，现在已经不使用。`HypernetConvBlock`为现在使用的convolution版本。

## segmentation network(deeplab)
segmentation“网络”的定义位于`deeplab_block`中，事实上我们的segmentation网络实现上只是一些函数，同时接受training data以及weights作为输入；因此`deeplab_block`是将deeplab结构转换成了函数形式。其中`resnet.py`就是deeplab的backbone（resnet18）

## overall structure
由于在pursuit以及其他应用中还需要优化z或叠加系数coeff，并使用函数版本的deeplab进行segmentation，我在`coeffnet.py`和`coeffnet_simple.py`中还定义了一些上层网络。这两个文件中都定义了`Singlenet`和`Coeffnet`两个网络，其中`Singlenet`包含可优化的z参数，`Coeffnet`包含可优化的叠加系数coeff，二者都是将图片作为输入，输出predicted mask。

`coeffnet.py`和`coeffnet_simple.py`两个文件的不同在于，`coeffnet_simple.py`中定义的网络的参数仅包括z/coeff，在forward时，需要额外定义hypernet和backbone并作为`forward`函数参数传入；而`coeffnet.py`中定义的网络的参数不仅包括z/coeff，还包括hypernet以及可能有的backbone，相当于将所有要用到的网络打包到了一个大网络中。

网络预训练模型载入（以下针对`coeffnet.py`文件）：
- 载入pretrained z: 对于`Singlenet`，可能需要使用pretrained z进行初始化，这时候可以使用`load_z`函数，load的对象可以是singlenet的参数文件(.pth)或者z记录文件(.json)
- 载入pretrained bases: 对于`Coeffnet`，可能需要载入预训练的bases，用于叠加；这时候直接将bases的目录（一个包含很多.json文件的目录，其中的.son文件称为z记录文件）传入`Coeffnet`的`base_dir`参数。
- 载入pretrained hypernet: 调用`init_hypernet`函数即可，`hypernet_path`为pretrained model的路径(.pth文件)，`freeze`为是否在载入后固定其参数，使得其在训练过程中不变。
- 载入pretrained backbone: 需要注意，若`use_backbone`为False，则网络不定义backbone，backbone的参数由hypernet生成，这时候不能进行初始化backbone的操作。若`use_backbone`为True，网络会定义一个backbone，这时调用`init_backbone`即可完成载入。

注意尽可能不要使用`torch.load`直接载入，因为网络结构不尽相同，以上载入都做了兼容处理。

另外，用于multi-objects joint training的`Multinet`位于`coeffnet.py`文件中。