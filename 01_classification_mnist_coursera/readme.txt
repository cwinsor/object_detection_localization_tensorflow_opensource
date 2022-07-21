This is the coursera basic TensorFlow classifier - https://www.coursera.org/projects/tensorflow-beginner-basic-image-classification

The dataset is MNIST
Input comes as image patch (28x28 integer).
Output comes as a digit in the range [0..9]
It is already split into [training, test], [x, y]

We want a classifier (10 classes).

Gold...
print(set(a_list)) to quick-see the range of values of (say) the output class

Procedure:

Output data:
convert to one-hot.  tensorflow.keras.utils tocategorical for this

Input data:
Un-patch the image into array of 784. This is not a CNN. Use np.reshape for this.
Normalize. np.mean, np.std for this.  Add epsilon=1e-10 to std.  (train - mean)/(std+epsilon).  Note we are normalizing across the entire input data, not per-image (question here). Note that same mean and std (from training data) are used on test data.

Model:
Here we used keras.Sequential and Dense

Define model structure:
the model has 2 hidden layers and 1 output layer.
Sequential is the overall CNN comprised of stages.
Dense is a fully-connected single layer.
For the hidden layers are 128 - a fraction of the 784 input.  Activation is 'relu'
For the output layer it is 10 (the number of classes). Activation is softmax (thus giving us a probability).

Compiling model... need:
optimizer = stochastic gradient descent
loss = predicted_cross_entropy
metric = 'accureacy'

Train model...
model.fit(x,y,epochs)

Evaluate model...
loss, accuracy = model.evaluate(x_test, y_test)

Using model to predict...
preds = model.predict(x_test_norm)

Show results 
There is a nice grid with colored label at the end


