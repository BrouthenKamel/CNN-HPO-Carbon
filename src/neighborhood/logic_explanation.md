# Neighborhood definition for A CNN model

Before we start with the neighborhood defintion, we have to first define what are the hyperparmeters "dimensions" that will construct the different seach space directions.

## 1. Hyperparameters list

This list contains the different model hyperparameters to be explored when executing the optimization algorithm "in our case, simulated annealing", each paired with a corresponding value range, hyperparameters and their corresponding range are inspired from [paper 1 and paper 2](#4-references)

### 1.1 Convolutional Hyperparameters

- Number of kernels (kc) : [32, 64, 96, 100, 128, 160, 192, 224, 256]
- Size of kernels (ks) [3, 5, 7]
- Activation function (f) : [relu, leaky relu, elu, tanh, linear]
- Stride (s) : fixed (1)
- Padding (p) : fixed (SAME)

### 1.2 Pooling Hyperparameters

- dropout rate (dp): [0.3, 0.4, 0.5]
- Size of kernels (ksp) [2, 3]
- type (pt) : [MAX, AVG]
- Stride (sp) : fixed (2)

### 1.3 Fully connected layers Hyperparameters

- Nb neurons per layer (uf) : [16, 32, 64, 128, 256, 512]
- dropout rate (df): [0.1, 0.2, 0.3, 0.4, 0.5]
- activation function (af): [relu, leaky relu, elu, tanh, linear]

### 1.4 Training Hyperparameters

- Batch size : [64, 128, 256]
- Learning rate : [0.0001, 0.001, 0.01, 0.0002, 0.0005, 0.0008, 0.002, 0.004, 0.005, 0.008]
- Epochs
- Optimizer
- Loss Function


## 3. Neighborhood definition

Given a current hyperparameter configuration, a neighbour is generated as follows :

```
# withing parameters it sepecifies whether it wants to adjust Conv, fully connected, training or ALL hyperparameters

# if conv will be taken into consideration in the neighborhood logic
For every convolutional block CB :
    generate p ~ U[0,1]
    if p < 0.5 :
        Select one hyperparameter randomly either Conv, BatchNorm or Pool
            if we are in conv modify one hyperparmeter selected randomly
            if for batchNorm, 50 50 chance for false to switch to true or vise versa
            if for Pool, select randomly one hyperparameter and change its value

# if the fully connected is also considered in the neighboorhood logic
For every fully connected block FCB :
    generate p ~ U[0,1]
    if p < 0.5 :
        Select one hyperparameter randomly hp
        Modify hp in its value

# if training is included as well
    generate p ~ U[0,1]
    if p < 0.5 :
        select one hyperpaprameter randomly 
        and modify the hp


the modfication process of an hp value is as follows
let hp_current_value = 5
let hp_possible_values = [1,2,3,4,5,6,7]
then future_hp_value = either(4 or 6) # in other words resp. previous or next in the list

if there are extreme cases like hp_current = 2 and hp_possible_values = [1,2] and future_hp_value = next in the list, then the value is kept the same and not changes will be done
```

## 4. References

**Paper 1 :** Multi-objective simulated annealing for hyper-parameter optimization in convolutional neural networks, by **_Ayla Gülcü_** and **_Zeki Kuş_**

**Paper 2 :** SA-CNN: Application to text categorization issues
using simulated annealing-based convolutional
neural network optimization, by **_Zihao GUO_** and **_Yueying CAO_**

