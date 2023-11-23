# import equinox as eqx
# import jax
# import jax.random as random
# import jax.numpy as jnp
# import tensorflow as tf
# import tensorflow_datasets as tfds
# from jax.scipy.special import logsumexp

# import optax
# import time


# batch_size = 128
# n_targets = 10
# data_dir = '/tmp/tfds'


# def relu(x):
#     return jnp.maximum(0, x)


# def predict(model, image):
#     return model(image)


# def one_hot(x, k, dtype=jnp.float32):
#     """Create a one-hot encoding of x of size k."""
#     return jnp.array(x[:, None] == jnp.arange(k), dtype)


# def accuracy(model, images, targets):
#     target_class = jnp.argmax(targets, axis=1)
#     predicted_class = jnp.argmax(batched_predict(model, images), axis=1)
#     return jnp.mean(predicted_class == target_class)


# def cross_entropy_loss(logits, labels):
#     return -jnp.mean(jnp.sum(labels * logits, axis=-1))


# def loss_fn(model, images, labels):
#     preds = batched_predict(model, images)
#     return cross_entropy_loss(preds, labels)


# # Make a batched version of the `predict` function
# batched_predict = jax.vmap(predict, in_axes=(None, 0))

# def load_mnist():
#     mnist_data, info = tfds.load(name="mnist", batch_size=-1, data_dir=data_dir, with_info=True)
#     mnist_data = tfds.as_numpy(mnist_data)
#     train_data, test_data = mnist_data['train'], mnist_data['test']
#     num_labels = info.features['label'].num_classes
#     h, w, c = info.features['image'].shape
#     num_pixels = h * w * c

#     # Full train set
#     train_images, train_labels = train_data['image'], train_data['label']
#     train_images = jnp.reshape(train_images, (len(train_images), num_pixels))
#     train_labels = one_hot(train_labels, num_labels)

#     # Full test set
#     test_images, test_labels = test_data['image'], test_data['label']
#     test_images = jnp.reshape(test_images, (len(test_images), num_pixels))
#     test_labels = one_hot(test_labels, num_labels)
#     return train_images, train_labels, test_images, test_labels, num_labels, num_pixels


# class NN(eqx.Module):
#     layers: list

#     def __init__(self, layer_sizes, key):
#         keys = jax.random.split(key, len(layer_sizes) - 1)
#         self.layers = []
#         for in_size, out_size, key in zip(layer_sizes[:-1], layer_sizes[1:], keys):
#             self.layers.append(eqx.nn.Linear(in_size, out_size, key=key))

#     def __call__(self, x):
#         for layer in self.layers[:-1]:
#             x = jax.nn.relu(layer(x))
#         return self.layers[-1](x)

# def get_train_batches():
#     # as_supervised=True gives us the (image, label) as a tuple instead of a dict
#     ds = tfds.load(name='mnist', split='train', as_supervised=True, data_dir=data_dir)
#     # You can build up an arbitrary tf.data input pipeline
#     ds = ds.batch(batch_size).prefetch(1)
#     # tfds.dataset_as_numpy converts the tf.data.Dataset into an iterable of NumPy arrays
#     return tfds.as_numpy(ds)

# def run_adam():
#     train_images, train_labels, test_images, test_labels, num_labels, num_pixels = load_mnist()

#     key = random.PRNGKey(0)
#     layer_sizes = [num_pixels, 512, 512, num_labels]

#     model = NN(layer_sizes, key)

#     step_size = 0.01
#     optimizer = optax.adam(step_size)
#     opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

#     # Update function
#     @jax.jit
#     def update(model, x, y, opt_state):
#         loss_value, grads = jax.value_and_grad(loss_fn)(model, x, y)
#         updates, opt_state = optimizer.update(grads, opt_state)
#         model = eqx.apply_updates(model, updates)
#         return loss_value, model, opt_state

#     num_epochs = 100
#     for epoch in range(num_epochs):
#         start_time = time.time()
#         for x, y in get_train_batches():
#             x = jnp.reshape(x, (len(x), num_pixels))
#             y = one_hot(y, num_labels)
#             _, model, opt_state = update(model, x, y, opt_state)
#         epoch_time = time.time() - start_time

#         train_acc = accuracy(model, train_images, train_labels)
#         test_acc = accuracy(model, test_images, test_labels)
#         print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
#         print("Training set accuracy {}".format(train_acc))
#         print("Test set accuracy {}".format(test_acc))



# if __name__ == "__main__":
#     run_adam()