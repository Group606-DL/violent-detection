from .i3d_model import Inception_Inflated3d, Inception_Inflated3d_Top, add_i3d_top
from keras import Model, optimizers

def layers_freeze(keras_model: Model) -> Model:
    for layer in keras_model.layers:
        layer.trainable = False

    return keras_model


def get_visual_model(input_shape, num_classes, type_model='rgb'):
    # build model for optical flow data
    # and load pretrained weights (trained on imagenet and kinetics dataset)
    weights_name = f'{type_model}_imagenet_and_kinetics'
    i3d_model_without_top = Inception_Inflated3d(include_top=False,
                                                 weights=weights_name,
                                                 input_shape=input_shape,
                                                 classes=num_classes)
    i3d_model_without_top = layers_freeze(i3d_model_without_top)
    i3d_model_with_top = add_i3d_top(i3d_model_without_top, num_classes, dropout_prob=0.5)

    optimizer = optimizers.Adam(learning_rate=0.01)
    i3d_model_with_top.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return i3d_model_with_top
