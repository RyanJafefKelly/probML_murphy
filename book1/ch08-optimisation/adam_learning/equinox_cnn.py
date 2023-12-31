# -*- coding: utf-8 -*-
"""mnist.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vMB6GrKLJUiRjjVZon4THNmu34DZtTbI

# Convolutional Neural Network on MNIST

This is an introductory example, intended for those who are new to both JAX and Equinox. This example builds a CNN to classify MNIST, and demonstrates:

- How to create a custom neural network using Equinox;
- When and why to use the `eqx.filter_{...}` functions;
- What your neural network looks like "under the hood" (like a PyTree).

The JAX ecosystem is build around a number of libraries, that each do a single thing. So in addition to Equinox (for model building), this example also uses [Optax](https://github.com/deepmind/optax) to train the network, and [jaxtyping](https://github.com/google/jaxtyping) to provide type annotations.

This example is available as a Jupyter notebook [here](https://github.com/patrick-kidger/equinox/blob/main/examples/mnist.ipynb).

!!! FAQ "What's the difference between JAX and Equinox?"

    [JAX](https://github.com/google/jax) is the underlying library for numerical routines: it provides JIT compilation, autodifferentiation, and operations like matrix multiplication etc. However it deliberately does *not* provide anything to do with any particular use case, like neural networks -- these are delegated to downstream libraries.

    Equinox is one such library. It provides neural network operations, plus many more advanced features. Go back and take a look at the [All of Equinox](../all-of-equinox.md) page once you've finished this example!
"""

import jax
import jax.numpy as jnp
import optax  # https://github.com/deepmind/optax
import torch  # https://pytorch.org
import torchvision  # https://pytorch.org
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping

import equinox as eqx

# Hyperparameters

BATCH_SIZE = 64
LEARNING_RATE = 3e-4
STEPS = 300
PRINT_EVERY = 30
SEED = 5678


def run():
    key = jax.random.PRNGKey(SEED)

    """## The dataset

    We load the MNIST dataset using PyTorch.

    JAX deliberately does not provide any built-in datasets or dataloaders! This is because there are already some well-curated datasets and dataloaders available elsewhere -- so it is common to use JAX alongside another library to provide these.

    - If you like PyTorch, then see [here](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) for a guide to its `DataSet` and `DataLoader` classes.
    - If you like TensorFlow, then see [here](https://www.tensorflow.org/guide/data) for a guide to its `tf.data` pipeline.
    - If you like NumPy -- which is a good choice for small in-memory datasets -- then see [here](../train_rnn/) for an example.
    """

    normalise_data = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_dataset = torchvision.datasets.MNIST(
        "MNIST",
        train=True,
        download=True,
        transform=normalise_data,
    )
    test_dataset = torchvision.datasets.MNIST(
        "MNIST",
        train=False,
        download=True,
        transform=normalise_data,
    )
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    # Checking our data a bit (by now, everyone knows what the MNIST dataset looks like)
    dummy_x, dummy_y = next(iter(trainloader))
    dummy_x = dummy_x.numpy()
    dummy_y = dummy_y.numpy()
    print(dummy_x.shape)  # 64x1x28x28
    print(dummy_y.shape)  # 64
    print(dummy_y)

    """We can see that our input has the shape `(64, 1, 28, 28)`. 64 is the batch size, 1 is the number of input channels (MNIST is greyscale) and 28x28 are the height and width of the image in pixels. The label is of shape `(64,)`, and each value is a number from 0 to 9.

    ## The model

    Our convolutional neural network (CNN) will store a list of all its operations. There is no explicit requirement to do it that way, it's simply convenient for this example.

    These operations can be any JAX operation. Some of these will be Equinox's built in layers (e.g. convolutions), and some of them will be functions from JAX itself (e.g. `jax.nn.relu` as an activation function).
    """

    class CNN(eqx.Module):
        layers: list

        def __init__(self, key):
            key1, key2, key3, key4 = jax.random.split(key, 4)
            # Standard CNN setup: convolutional layer, followed by flattening,
            # with a small MLP on top.
            self.layers = [
                eqx.nn.Conv2d(1, 3, kernel_size=4, key=key1),
                eqx.nn.MaxPool2d(kernel_size=2, stride=1),
                jax.nn.relu,
                jnp.ravel,
                eqx.nn.Linear(1728, 512, key=key2),
                jax.nn.sigmoid,
                eqx.nn.Linear(512, 64, key=key3),
                jax.nn.relu,
                eqx.nn.Linear(64, 10, key=key4),
                jax.nn.log_softmax,
            ]

        def __call__(self, x: Float[Array, "1 28 28"]) -> Float[Array, "10"]:
            for layer in self.layers:
                x = layer(x)
            return x


    key, subkey = jax.random.split(key, 2)
    model = CNN(subkey)

    """As with everything in Equinox, our model is a PyTree. That is to say, just a nested collection of objects. Some of these object are JAX arrays; for example `model.layers[0].weight` is the kernel of our convolution. And some of these objects are essentially arbitrary Python objects; for example `model.layers[-1]` is `jax.nn.log_softmax`, which is just a Python function like any other.

    Equinox provides a nice `__repr__` for its modules, so we can just print out what our PyTree looks like:
    """

    print(model)

    """Given some data, we can perform inference on our model.

    (**Note** that here we are using JAX operation outside of a JIT'd region. This is very slow! You shouldn't write it like this except when exploring things in a notebook.)
    """

    def loss(
        model: CNN, x: Float[Array, "batch 1 28 28"], y: Int[Array, " batch"]
    ) -> Float[Array, ""]:
        # Our input has the shape (BATCH_SIZE, 1, 28, 28), but our model operations on
        # a single input input image of shape (1, 28, 28).
        #
        # Therefore, we have to use jax.vmap, which in this case maps our model over the
        # leading (batch) axis.
        pred_y = jax.vmap(model)(x)
        return cross_entropy(y, pred_y)


    def cross_entropy(
        y: Int[Array, " batch"], pred_y: Float[Array, "batch 10"]
    ) -> Float[Array, ""]:
        # y are the true targets, and should be integers 0-9.
        # pred_y are the log-softmax'd predictions.
        pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
        return -jnp.mean(pred_y)


    # Example loss
    loss_value = loss(model, dummy_x, dummy_y)
    print(loss_value.shape)  # scalar loss
    # Example inference
    output = jax.vmap(model)(dummy_x)
    print(output.shape)  # batch of predictions

    """### Filtering

    In the next cells we can see an example of when we should use the filter methods provided by Equinox. For instance, the following code generates an error:
    """

    # This is an error!
    # jax.value_and_grad(loss)(model, dummy_x, dummy_y)

    """When we write `jax.value_and_grad(loss)(model, ...)`, we are asking JAX to differentiate the function `loss` with respect to its first argument `model`. (To compute the gradients on its parameters.)

    However, `model` includes several things that aren't parameters! Look back up at the PyTree print-out from earlier, and we see lines like e.g. `<wrapped function relu>` -- this isn't a parameter and isn't even an array.

    We need to split our model into the bit we want to differentiate (its parameters), and the bit we don't (everything else). If we want to, then we can do this manually:
    """

    # This will work!
    params, static = eqx.partition(model, eqx.is_array)


    def loss2(params, static, x, y):
        model = eqx.combine(params, static)
        return loss(model, x, y)


    loss_value, grads = jax.value_and_grad(loss2)(params, static, dummy_x, dummy_y)
    print(loss_value)

    """It's quite common that all arrays represent parameters, so that "the bit we want to differentiate" really just means "all arrays". As such, Equinox provides a convenient wrapper `eqx.filter_value_and_grad`, which does the above partitioning-and-combining for us: it automatically splits things into arrays and non-arrays, and then differentiates with respect to all arrays in the first argument:"""

    # This will work too!
    value, grads = eqx.filter_value_and_grad(loss)(model, dummy_x, dummy_y)
    print(value)

    """The Equinox `eqx.filter_{...}` functions are essentially the same as the corresponding JAX functions, and they're just smart enough to handle non-arrays without raising an error. So if you're unsure, you can simply always use the Equinox filter functions.

    ## Evaluation

    As with most machine learning tasks, we need some methods to evaluate our model on some testdata. For this we create the following functions.

    Notice that we used `eqx.filter_jit` instead of `jax.jit` since as usual our model contains non-arrays (e.g. those `relu` activation functions), and those aren't arrays that can be differentiated.
    """

    loss = eqx.filter_jit(loss)  # JIT our loss function from earlier!


    @eqx.filter_jit
    def compute_accuracy(
        model: CNN, x: Float[Array, "batch 1 28 28"], y: Int[Array, " batch"]
    ) -> Float[Array, ""]:
        """This function takes as input the current model
        and computes the average accuracy on a batch.
        """
        pred_y = jax.vmap(model)(x)
        pred_y = jnp.argmax(pred_y, axis=1)
        return jnp.mean(y == pred_y)

    def evaluate(model: CNN, testloader: torch.utils.data.DataLoader):
        """This function evaluates the model on the test dataset,
        computing both the average loss and the average accuracy.
        """
        avg_loss = 0
        avg_acc = 0
        for x, y in testloader:
            x = x.numpy()
            y = y.numpy()
            # Note that all the JAX operations happen inside `loss` and `compute_accuracy`,
            # and both have JIT wrappers, so this is fast.
            avg_loss += loss(model, x, y)
            avg_acc += compute_accuracy(model, x, y)
        return avg_loss / len(testloader), avg_acc / len(testloader)

    evaluate(model, testloader)

    """## Training

    Now it's time to train our model using Optax!
    """

    optim = optax.adamw(LEARNING_RATE)

    def train(
        model: CNN,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        optim: optax.GradientTransformation,
        steps: int,
        print_every: int,
    ) -> CNN:
        # Just like earlier: It only makes sense to train the arrays in our model,
        # so filter out everything else.
        opt_state = optim.init(eqx.filter(model, eqx.is_array))

        # Always wrap everything -- computing gradients, running the optimiser, updating
        # the model -- into a single JIT region. This ensures things run as fast as
        # possible.
        @eqx.filter_jit
        def make_step(
            model: CNN,
            opt_state: PyTree,
            x: Float[Array, "batch 1 28 28"],
            y: Int[Array, " batch"],
        ):
            loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
            updates, opt_state = optim.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss_value

        # Loop over our training dataset as many times as we need.
        def infinite_trainloader():
            while True:
                yield from trainloader

        for step, (x, y) in zip(range(steps), infinite_trainloader()):
            # PyTorch dataloaders give PyTorch tensors by default,
            # so convert them to NumPy arrays.
            x = x.numpy()
            y = y.numpy()
            model, opt_state, train_loss = make_step(model, opt_state, x, y)
            if (step % print_every) == 0 or (step == steps - 1):
                test_loss, test_accuracy = evaluate(model, testloader)
                print(
                    f"{step=}, train_loss={train_loss.item()}, "
                    f"test_loss={test_loss.item()}, test_accuracy={test_accuracy.item()}"
                )
        return model

    model = train(model, trainloader, testloader, optim, STEPS, PRINT_EVERY)

"""This is actually a pretty bad final accuracy, as MNIST is so easy. Try tweaking this example to make it better!

!!! Tip "Next steps"

    Hopefully this example has given you a taste of how models are built using JAX and Equinox. For next steps, take a look at the [JAX documentation](https://jax.readthedocs.io/) for more information on JAX, the [All of Equinox page](../all-of-equinox.md) for a summary of everything Equinox can do, or [training an RNN](../train_rnn) for another example.
"""

if __name__ == "__main__":
    run()