
## Loss

When the network has 3 output nodes indicating Cat, Dog, Bird
And the input picture is of Cat
And the prediction is `[0.6, 0,3, 0.1]` for each of them respectively.
Then it means that network is sure it's only 60% Cat.
So the loss is `-log(0.4)=0.398`,
Where `0.4` is the sum of all other predictions

We want Loss to be LOW(0)


## Accuracy

When the network has 3 output nodes indicating Cat, Dog, Bird
And we have 5 batches of data
And the output should be `[Cat, Dog, Cat, Cat, Bird]`
And the prediction really is `[Cat, Cat, Bird, Bird, Bird]`
Then it means in 3 cases the network predicted the output incorrectly
And in 2 cases it was correct
So the accuracy is `2/5`

We want Accuracy to be HIGH(1).

## Overfitting

If a test performance differs from training by more than 10%,
it means that the model is overfit.

One option to prevent overfitting is to change the model’s size. If a model is not
learning at all, one solution might be to try a larger model. If your model is learning, but there’s a
divergence between the training and testing data, it could mean that you should try a smaller
model. One general rule to follow when selecting initial model hyperparameters is to find the
smallest model possible that still learns.

Initially, you can
very quickly (usually within minutes) try different settings (e.g., layer sizes) to see if the models
are learning something. If they are, train the models fully — or at least significantly longer — and
compare results to pick the best set of hyperparameters. Another possibility is to create a list of
different hyperparameter sets and train the model in a loop using each of those sets at a time to
pick the best set at the end. The reasoning here is that the fewer neurons you have, the less chance
you have that the model is memorizing the data.

The process of trying different model settings is called **hyperparameter searching**
