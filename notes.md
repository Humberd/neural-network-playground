
## Loss

When the network has 3 output nodes indicating Cat, Dog, Bird
And the input picture is of Cat
And the prediction is `[0.6, 0,3, 0.1]` for each of them respectively.
Then it means that network is sure it's only 60% Cat.
So the loss is `-log(0.4)=0.398`,
Where `0.4` is the sum of all other predictions

## Accuracy

When the network has 3 output nodes indicating Cat, Dog, Bird
And we have 5 batches of data
And the output should be `[Cat, Dog, Cat, Cat, Bird]`
And the prediction really is `[Cat, Cat, Bird, Bird, Bird]`
Then it means in 3 cases the network predicted the output incorrectly
And in 2 cases it was correct
So the accuracy is `2/5`

