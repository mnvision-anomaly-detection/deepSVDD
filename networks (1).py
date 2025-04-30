from tensorflow import keras


__all__ = ['mnist_lenet', 'cifar_lenet']


def mnist_lenet(H=32):
    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(8, (5, 5), padding='same', use_bias=False, input_shape=(28, 28, 1)))
    model.add(keras.layers.LeakyReLU(1e-2))
    model.add(keras.layers.BatchNormalization(epsilon=1e-4, trainable=False))
    model.add(keras.layers.MaxPool2D())

    model.add(keras.layers.Conv2D(4, (5, 5), padding='same', use_bias=False))
    model.add(keras.layers.LeakyReLU(1e-2))
    model.add(keras.layers.BatchNormalization(epsilon=1e-4, trainable=False))
    model.add(keras.layers.MaxPool2D())

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(H, use_bias=False))

    return model


def cifar_lenet(H=128):
    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(32, (5, 5), strides=(3, 3), padding='same', use_bias=False, input_shape=(32, 32, 3)))
    model.add(keras.layers.LeakyReLU(1e-2))
    model.add(keras.layers.BatchNormalization(epsilon=1e-4, trainable=False))

    model.add(keras.layers.Conv2D(64, (5, 5), strides=(3, 3), padding='same', use_bias=False))
    model.add(keras.layers.LeakyReLU(1e-2))
    model.add(keras.layers.BatchNormalization(epsilon=1e-4, trainable=False))

    model.add(keras.layers.Conv2D(128, (5, 5), strides=(3, 3), padding='same', use_bias=False))
    model.add(keras.layers.LeakyReLU(1e-2))
    model.add(keras.layers.BatchNormalization(epsilon=1e-4, trainable=False))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(H, use_bias=False))

    return model

# def custom_lenet(H=128):
#     model = keras.models.Sequential()

#     model.add(keras.layers.Conv2D(32, (3, 3), strides=1, padding='same', input_shape=(64, 64, 3)))
#     model.add(keras.layers.LeakyReLU(0.1))
#     model.add(keras.layers.BatchNormalization())

#     model.add(keras.layers.Conv2D(64, (3, 3), strides=2, padding='same'))
#     model.add(keras.layers.LeakyReLU(0.1))
#     model.add(keras.layers.BatchNormalization())

#     model.add(keras.layers.Conv2D(128, (3, 3), strides=2, padding='same'))
#     model.add(keras.layers.LeakyReLU(0.1))
#     model.add(keras.layers.BatchNormalization())

#     model.add(keras.layers.Conv2D(256, (3, 3), strides=2, padding='same'))
#     model.add(keras.layers.LeakyReLU(0.1))
#     model.add(keras.layers.BatchNormalization())

#     model.add(keras.layers.GlobalAveragePooling2D())
#     model.add(keras.layers.Dense(H, use_bias=False))

#     return model

# def custom_lenet(H=128):
#     model = keras.models.Sequential()

#     # 첫 번째 Conv2D 레이어
#     model.add(keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same', use_bias=False, input_shape=(64, 64, 3)))
#     model.add(keras.layers.LeakyReLU(1e-2))
#     model.add(keras.layers.BatchNormalization(epsilon=1e-4, trainable=False))

#     # 두 번째 Conv2D 레이어
#     model.add(keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
#     model.add(keras.layers.LeakyReLU(1e-2))
#     model.add(keras.layers.BatchNormalization(epsilon=1e-4, trainable=False))

#     # 세 번째 Conv2D 레이어
#     model.add(keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
#     model.add(keras.layers.LeakyReLU(1e-2))
#     model.add(keras.layers.BatchNormalization(epsilon=1e-4, trainable=False))

#     # 네 번째 Conv2D 레이어 추가
#     model.add(keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same', use_bias=False))
#     model.add(keras.layers.LeakyReLU(1e-2))
#     model.add(keras.layers.BatchNormalization(epsilon=1e-4, trainable=False))

#     # 다섯 번째 Conv2D 레이어 추가 (기본적으로 크기를 더 작은 필터로)
#     model.add(keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same', use_bias=False))
#     model.add(keras.layers.LeakyReLU(1e-2))
#     model.add(keras.layers.BatchNormalization(epsilon=1e-4, trainable=False))

#     model.add(keras.layers.Flatten())  # 평탄화
#     model.add(keras.layers.Dense(H, use_bias=False))  # Fully connected layer

#     return model

def custom_lenet(H=128):
    model = keras.models.Sequential()

    # Conv 레이어 부분 (위와 동일)
    model.add(keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same', use_bias=False, input_shape=(128, 128, 3)))
    model.add(keras.layers.LeakyReLU(1e-2))
    model.add(keras.layers.BatchNormalization(epsilon=1e-4, trainable=False))
    model.add(keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(keras.layers.LeakyReLU(1e-2))
    model.add(keras.layers.BatchNormalization(epsilon=1e-4, trainable=False))
    model.add(keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(keras.layers.LeakyReLU(1e-2))
    model.add(keras.layers.BatchNormalization(epsilon=1e-4, trainable=False))

    # Dense 레이어 추가 (이미지 특징을 학습하는 Dense 레이어)
    model.add(keras.layers.Flatten())  # Conv2D 레이어의 출력은 Flatten해야 Dense로 연결 가능
    model.add(keras.layers.Dense(512, activation='relu'))  # Dense 레이어
    model.add(keras.layers.Dropout(0.5))  # Dropout 레이어 (과적합 방지)
    model.add(keras.layers.Dense(256, activation='relu'))  # Dense 레이어
    model.add(keras.layers.Dropout(0.5))  # Dropout 레이어 (과적합 방지)

    model.add(keras.layers.Dense(H, use_bias=False))  # 마지막 Dense 레이어

    return model




