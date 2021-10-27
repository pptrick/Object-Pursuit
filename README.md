# Object-Pursuit
Object Pursuit: An Object Space for Visual Understanding

## file structure

- `BaseAnalyze`: Test how pursuit bases distribute in space; Visualize the combination coefficients.
- `DataCollector`: Collecting synthetic data on ithor platform.
- `Pursuit`: The main part of our project.
  - `object_pursuit`: the main part of object pursuit algorithm. 
    - `pursuit.py`: The algorithm framework
    - `train.py`: The training units for pursuit. Contains both redundancy check and train new representations. 
    - `data_selector.py`: Control the object flow, create a continual learning setting.
    - `rm_redundency.py`: Remove redundancy in bases.
  - `dataset`: constructing torch dataset. `basic_dataset.py` is for ithor synthetic data, `davis_dataset.py` and `kitti_dataset.py` is for DAVIS and KITTI dataset. (both real data)；Other files are some utils for data processing.
  - `model`: Networks and models in pursuit. The Implementation of the hypernetwork and the total network architecture are in `/coeffnet`; `deeplabv3` and `unet` are just test models to test segmentation backbones.
    - `/coeffnet/hypernet.py`: implement hypernetwork architecture
    - `/coeffnet/hypernet_conv.py`: implement convolution hyper-block
    - `/coeffnet/deeplab_block`: implement deeplab that takes network weights as input.
  - `loss`: the loss criterions. we have implement dice loss, IOU loss.
    - `memory_loss.py`: The implementation of forgetting preventing term. Prevent the network from forgetting.
  - `evaluation`: 
    - `seen_obj.py`: test for whether an object is seen/unseen.
    - `eval_net.py`: The evaluation unit.
  - `main.py`: to run the object pursuit algorithm
  - `train_multi.py`: joint training on multiple bases. Also view as pre-train process.
  - `eval_multi.py`: evaluate joint training
  - `eval_pursuit`：evaluate pursuit process