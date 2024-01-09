import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Reshape, Flatten, Input

def build_generator(input_shape):
    model = Sequential()
    model.add(Dense(128, input_shape=input_shape, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(output_shape, activation='sigmoid'))
    return model


input_shape = (100,)
output_shape = 64

generator = build_generator(input_shape)

def build_discriminator(input_shape):
    model = Sequential()
    model.add(Dense(256, input_shape=input_shape, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

discriminator = build_discriminator((output_shape,))
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

discriminator.trainable = True

gan_input = Input(shape=(100,))
x = generator(gan_input)
gan_output = discriminator(x)

gan = Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer='adam')
gan.summary()

def train_gan(epochs, batch_size):
    for epoch in range(epochs):

        noise = np.random.normal(0, 1, size=(batch_size, 100))

        generated_data = generator.predict(noise)

        real_data = np.random.random((batch_size, 64)) #secure key

        d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))

        d_loss_fake = discriminator.train_on_batch(generated_data, np.zeros((batch_size, 1)))

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train the generator
        noise = np.random.normal(0, 1, size=(batch_size, 100))
        valid_labels = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, valid_labels)

        # Print progress
        print(f"{epoch}/{epochs} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")


train_gan(epochs=11, batch_size=32)

noise = np.random.normal(0, 1, size=(1, 100))

discriminator.save("discriminator_model.h5")



video_path = "sample12.mp4"  # Replace with the path to your video file
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_path = "output_video_with_watermark.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    generated_watermark = generator.predict(noise)
    generated_watermark = generated_watermark.reshape((64,))  # Adjust the size accordingly

    generated_watermark = generated_watermark.reshape((64, 1))

    resized_watermark = cv2.resize(generated_watermark, (width, height), interpolation=cv2.INTER_LINEAR)

    if resized_watermark.shape[:2] == frame.shape[:2]:

        if len(resized_watermark.shape) == 2:
            resized_watermark = cv2.cvtColor(resized_watermark, cv2.COLOR_GRAY2BGR)

        frame = frame.astype(np.float32)
        resized_watermark = resized_watermark.astype(np.float32)

        alpha = 0.5
        blended_frame = cv2.addWeighted(frame, 1 - alpha, resized_watermark, alpha, 0)

        blended_frame = np.clip(blended_frame, 0, 255).astype(np.uint8)

        out.write(blended_frame)

        cv2.imshow('Video with Watermark', blended_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Size mismatch: Frame and watermark sizes do not match.")

cap.release()
out.release()
cv2.destroyAllWindows()



# Evaluating veracity

video_path = "sample12.mp4"
cap = cv2.VideoCapture(video_path)


fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

target_size = (64, 64)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, target_size)

    normalized_frame = resized_frame / 255.0

    input_frame = np.expand_dims(normalized_frame, axis=0)

    flattened_input = input_frame.flatten()

    flattened_input = flattened_input[:64]

    frame_probability = discriminator.predict(np.expand_dims(flattened_input, axis=0))

    print(f"Probability of frame: {frame_probability[0, 0]}")

    cv2.imshow('Watermarked Frame', resized_frame.astype(np.uint8))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()