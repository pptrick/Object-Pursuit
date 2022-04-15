# readme
This directory (`pretrain`) contains code for joint pretraining. Different from 'Object Pursuit', which learns objects sequentially, 'Joint Pretraining' learns all objects at once, updating the hypernet and bases simultaneously. Specifically, during training, each base corresponds to an object, and a mini-batch only contains data of one object. When training on a data batch of object $O_A$, the hypernet and the representation of $O_A$ (also called 'the $O_A$ base') are updated. The representation $z$ is a vector and is the input of the hypernet.

Here are the specific descriptions of the modules in `pretrain`:

- `_main.py` : the top-level file of 'joint pretrain', parsing arguments and running joint training.
- `_dataset.py` : the definition of `MultiJointDataset,` which is a dataset that combines all object data together. Also, it defines dataloaders for different object datasets (e.g., iThor, DAVIS, Youtube-VOS).
- `_model.py` : the definition of `Multinet`, which contains a hypernet and $n$ representation vector $z$ ($n$ stands for the object number).
- `_train.py` : a script for joint training.
- `_eval.py` : a script for evaluation of joint training. Specifically, evaluate each object separately using its representation and the current hypernet.

You may also find a script `joint_pretrain.py` outside this directory which is the entry script of this part.
