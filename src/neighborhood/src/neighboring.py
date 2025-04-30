import torch
import torch.nn as nn
from typing import Union
import random
from typing import Literal
from enum import Enum


from src.schema.model import ModelArchitecture
from src.schema.block import CNNBlock, MLPBlock
from src.schema.layer import ConvLayer, PoolingLayer, PoolingType, DropoutLayer, LinearLayer, ActivationLayer, ActivationType, PaddingType, AdaptivePoolingLayer
from src.schema.training import TrainingParams, OptimizerType

from pydantic import BaseModel

# this dictionary contains the possible values in the search space, we can figure out a better way to store them

search_space = {
    "epochs": [10, 100, 200],
    "batch_size": [32, 64, 128, 256],
    "learning_rate": [0.0001, 0.001, 0.01, 0.0002, 0.0005, 0.0008, 0.002, 0.004, 0.005, 0.008],
    "momentum": [None],
    "weight_decay": [None],
    "filters": [32, 64, 96, 100, 128, 160, 192, 224, 256],
    "kernel_size": [2, 3, 4, 5, 6, 7],
    "stride": [0, 1, 2, 3],
    "padding": [0, 1, 2, 3],
    "output_size": [3],
    "rate": [0.1, 0.2, 0.3, 0.4, 0.5],
    "neurons": [16, 32, 64, 128, 256, 512]
}

# this is a carefully designed recursive function that gets an object as an input and it will recursively apply the neighborhood algorithm 
# inorder to modify parts of that object's values according to the neighborhood algorithm explained in the document
# uncomment the printing messages to check the values modified
# the method is robust to adding new features to layer.py or block.py or any other class defining file

def modify_value(object, block_modification_ratio = 0.5,search_space = search_space, perturbation_intensity = 1, perturbation_nature = "Local"):
    # perturbation_intensity [1, 2, ...] : if a given block will be modified, this parameter will decide how many hyperparameters to change, by default it is 1 hyperparameter per block
    # perturbation_nature ["Local","Random"] : if a hyperparamter will be modified, this parameter will decide whether to tweak it to a relatively close value "Local", or select randomly a new value off the hyperparameter's search space "Random"


    # if we are dealing with a list, such as a list of CNNBlocks then we will call the recursive function on each element
    if (isinstance(object, list)):
        for element in object:
            modify_value(element)
    # if we are dealing with a BaseModel type, we will modify its content with a 50 50 chance
    elif (isinstance(object,BaseModel)):
        p = random.uniform(0, 1)
        if p<block_modification_ratio:
            component_keys = type(object).model_fields.keys()
            modifiable_keys = list(component_keys)
            # we have to keep only the modifiable components of the object
            for component_key in component_keys : 
                if(getattr(object,component_key)== None):
                    modifiable_keys.remove(component_key)
            
            for iter in range(perturbation_intensity):
                # then we will select at random one of the objects to be modified
                if len(modifiable_keys)!=0:
                    component_key_to_be_modified = random.choice(modifiable_keys)
                    new_object = getattr(object,component_key_to_be_modified)
                    # if this componenet happens to also be a list or a BaseModel, we recursively modify
                    if ( isinstance(new_object,list) or isinstance(new_object,BaseModel)):
                        modify_value(new_object)
                    # if not, this means we are dealing with floats, strings or enums, we will modify them directly
                    else :
                        # Case one : we have ENUMs so we will modify them according to the ENUM values set in classes such as ActivationType
                        if (issubclass(type(new_object),Enum)):
                            possible_values = list(type(new_object))
                            new_value = random.choice(possible_values)
                            setattr(object,component_key_to_be_modified,new_value)
                            print('have modified ', component_key_to_be_modified, ' to ', new_value)
                        # case two : we have values that are supposed to be a in a specefic range, in this case we move up or down the ladder with the value
                        else:
                            value_set = search_space[component_key_to_be_modified]
                            old_value = new_object
                            if old_value in value_set and perturbation_nature=="Local":
                                old_index = value_set.index(old_value)
                                new_offset = random.choice([-1,1])
                                if old_index+new_offset >= 0 and old_index+new_offset < len(value_set) :
                                    new_value = value_set[old_index+new_offset]
                                    setattr(object,component_key_to_be_modified,new_value)
                                    print('have modified ', component_key_to_be_modified, ' to ', new_value)
                            else :
                                new_value = random.choice(value_set)
                                setattr(object,component_key_to_be_modified,new_value)
                                print('have modified ', component_key_to_be_modified, ' to ', new_value)






# Now let's move on to an example by applying it on AlexNet


# AlexNet architecture representation
AlexNetArchitecture = ModelArchitecture(
    cnn_blocks=[
        CNNBlock(
            conv_layer=ConvLayer(filters=64, kernel_size=5, stride=1, padding=2),
            activation_layer=ActivationLayer(type=ActivationType.RELU.value),
            pooling_layer=PoolingLayer(type=PoolingType.MAX.value, kernel_size=2, stride=2, padding=0),
        ),
        CNNBlock(
            conv_layer=ConvLayer(filters=192, kernel_size=5, stride=1, padding=2),
            activation_layer=ActivationLayer(type=ActivationType.RELU.value),
            pooling_layer=PoolingLayer(type=PoolingType.MAX.value, kernel_size=2, stride=2, padding=0),
        ),
        CNNBlock(
            conv_layer=ConvLayer(filters=384, kernel_size=3, stride=1, padding=1),
            activation_layer=ActivationLayer(type=ActivationType.RELU.value),
        ),
        CNNBlock(
            conv_layer=ConvLayer(filters=256, kernel_size=3, stride=1, padding=1),
            activation_layer=ActivationLayer(type=ActivationType.RELU.value),
        ),
        CNNBlock(
            conv_layer=ConvLayer(filters=256, kernel_size=3, stride=1, padding=1),
            activation_layer=ActivationLayer(type=ActivationType.RELU.value),
            pooling_layer=PoolingLayer(type=PoolingType.MAX.value, kernel_size=2, stride=2, padding=0)
        ),
    ],
    adaptive_pooling_layer=AdaptivePoolingLayer(
        type=PoolingType.AVG.value,
        output_size=3
    ),
    mlp_blocks=[
        MLPBlock(
            dropout_layer=DropoutLayer(rate=0.5),
            linear_layer=LinearLayer(neurons=4096),
            activation_layer=ActivationLayer(type=ActivationType.RELU.value),
        ),
        MLPBlock(
            dropout_layer=DropoutLayer(rate=0.5),
            linear_layer=LinearLayer(neurons=(4096)),
            activation_layer=ActivationLayer(type=ActivationType.RELU.value),
        )
    ],
    training_params=TrainingParams(
        epochs=10,
        batch_size=32,
        learning_rate=0.001,
        optimizer=OptimizerType.SGD.value,
        momentum=None,
        weight_decay=None
    )
)


# set the blocks that you will modify, if say you want to modify cnn_blocks only just mention cnn_blocks in the list below
to_modify = ['cnn_blocks', 'adaptive_pooling_layer','mlp_blocks','training_params']

for bloc in to_modify:
    element = getattr(AlexNetArchitecture,bloc)
    modify_value(element,block_modification_ratio=0.5,perturbation_intensity=3,perturbation_nature="Random")



# AlexNetArchitecutre will be then modified, you can check by adding print(AlexNetArchitecture), in addition to uncommenting
# the prints written in the modify_value() method to keep track of what exactly has been modified