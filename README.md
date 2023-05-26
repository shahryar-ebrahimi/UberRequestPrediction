# UberRequest_Prediction
Recurrent Neural Network (RNN) was used to predict Uber requests in 4 major districts of New York during a 1-hour time interval.


Dataset Discription:

borough: NYC's borough.
 pickups: Number of pickups for the period.
 spd: Wind speed in miles/hour.
 vsb: Visibility in Miles to nearest tenth.
 temp: temperature in Fahrenheit.
 dewp: Dew point in Fahrenheit.
 slp: Sea level pressure.
 pcp01: 1-hour liquid precipitation.
 pcp06: 6-hour liquid precipitation.
 pcp24: 24-hour liquid precipitation.
 sd: Snow depth in inches.
 hday: Being a holiday (Y) or not (N).
 hr_sin, hr_cos: Hour of the day
 day_sin, day_cos: Day of the week


We did pre-processing on our dataset before apply it on our proposed network. There are
four columns related to hour and day. These aforementioned columns are in form of sine
and cosine. So, we plotted the dataset and results are shown in fig.1, fig.2.

As you can see in the above figures, we have twenty-four dots in fig.1 and seven dots in
fig.2 that are on a circle. We can consider these dots as twenty-four hour of a day and seven
days of a week respectively. Therefore, we change the original dataset to achieve an accurate
dataset for applying on our network.
We considered “pikcsup”, “temp”, “dewp”, “hday”,” hour”, and “day” as our features
and trained the network according to these.
We consider each six hours as a sample for input of the network and put information of
four boroughs in one row for same hour and day. So, we have 4343 samples now that each
sample contain information about six hours of boroughs.

Firs of all, we will review the recurrent neural network. An RNN works like this; First
words get transformed into machine-readable vectors. Then the RNN processes the
sequence of vectors one by one.
While processing, it passes the previous hidden state to the next step of the sequence.
The hidden state acts as the neural network’s memory. It holds information on previous
data the network has seen before.
In a cell of the RNN we see how we would calculate the hidden state. First, the input
and previous hidden state are combined to form a vector. That vector now has information
on the current input and previous inputs. The vector goes through the tanh activation, and
the output is the new hidden state, or the memory of the network.

Tanh activation
The tanh activation is used to help regulate the values flowing through the network. The
tanh function squishes values to always be between -1 and 1.

When vectors are flowing through a neural network, it undergoes many transformations
due to various math operations. So, imagine a value that continues to be multiplied by let’s
say 3. You can see how some values can explode and become astronomical, causing other
values to seem insignificant.

A tanh function ensures that the values stay between -1 and 1, thus regulating the output
of the neural network. We can see how the same values from above remain between the
boundaries allowed by the tanh function

So that’s an RNN. It has very few operations internally but works pretty well given the
right circumstances (like short sequences). RNNs uses many less computational resources
than it’s evolved variants, LSTMs and GRUs.

An LSTM has a similar control flow as a recurrent neural network. It processes data
passing on information as it propagates forward. The differences are the operations within
the LSTM’s cells.

These operations are used to allow the LSTM to keep or forget information. Now looking
at these operations can get a little overwhelming so we’ll go over this step by step.
Core Concept
The core concept of LSTMs is the cell state, and various gates. The cell state act as a
transport highway that transfers relative information all the way down the sequence chain.
You can think of it as the “memory” of the network. The cell state, in theory, can carry
relevant information throughout the processing of the sequence. So even information from
the earlier time steps can make its way to later time steps, reducing the effects of short-term
memory. As the cell state goes on its journey, information gets added or removed to the cell
state via gates. The gates are different neural networks that decide which information is
allowed on the cell state. The gates can learn what information is relevant to keep or forget
during training.

Forget gate
At first, we have the forget gate. This gate decides what information should be thrown
away or kept. Information from the previous hidden state and information from the current
input is passed through the sigmoid function. Values come out between 0 and 1. The closer
to 0 means to forget, and the closer to 1 means to keep.

Input Gate
To update the cell state, we have the input gate. First, we pass the previous hidden state
and current input into a sigmoid function. That decides which values will be updated by7
transforming the values to be between 0 and 1. 0 means not important, and 1 means
important. You also pass the hidden state and current input into the tanh function to squish
values between -1 and 1 to help regulate the network. Then you multiply the tanh output
with the sigmoid output. The sigmoid output will decide which information is important to
keep from the tanh output

Cell State
Now we should have enough information to calculate the cell state. First, the cell state
gets pointwise multiplied by the forget vector. This has a possibility of dropping values in
the cell state if it gets multiplied by values near 0. Then we take the output from the input
gate and do a pointwise addition which updates the cell state to new values that the neural
network finds relevant. That gives us our new cell state

Output Gate
At the end, we have the output gate. The output gate decides what the next hidden state
should be. Remember that the hidden state contains information on previous inputs. The
hidden state is also used for predictions. First, we pass the previous hidden state and the
current input into a sigmoid function. Then we pass the newly modified cell state to the tanh
function. We multiply the tanh output with the sigmoid output to decide what information
the hidden state should carry. The output is the hidden state. The new cell state and the new
hidden is then carried over to the next time step.

To review, the Forget gate decides what is relevant to keep from prior steps. The input
gate decides what information is relevant to add from the current step. The output gate
determines what the next hidden state should be.

So now we know how an LSTM work, let’s briefly look at the GRU. The GRU is the
newer generation of Recurrent Neural networks and is pretty similar to an LSTM. GRU’s
got rid of the cell state and used the hidden state to transfer information. It also only has two
gates, a reset gate and update gate.

Update Gate
The update gate acts similar to the forget and input gate of an LSTM. It decides what
information to throw away and what new information to add.
Reset Gate
The reset gate is another gate is used to decide how much past information to forget.
GRUs have fewer tensor operations; therefore, they are a little speedier to train then
LSTMs. There isn’t a clear winner which one is better. Researchers and engineers usually
try both to determine which one works better for their use case.

#################
IMPLEMENTATION
#################


To implement an RNN network, features of our network are:
• Number of cells = 30
• Loss function = Mean Square Error
• Optimizer = Adam
• Number of epochs = 50
• Batch size = 5 (We set this to reach best result)
• Training sample size = 3000
• Validation sample size = 1000
• Testing sample size = 337

We implemented the RNN, LSTM, and GRU networks using aforementioned features.
The results are as below. We compared them considering their accuracy and speed

According to fig.13, we can compare the three different networks considering their loss
and accuracy. As you can see in fig.13, LSTM and GRU have better performance than
simple RNN and they are more accurate. Their accuracy after 50 epochs is almost 85%.
In following figures, you can see some more results about these networks. We plotted
true value of pick-ups for each borough versus predicted value.

As you can see in fig.14, the slope of scattered points is near to 1. Therefore, we can say
prediction is done in a good way.

Conclusion and Discussion:
According to results, all three networks have a good performance. As you can see in
figures. 14, 15, and 16, the slope of the scattered line is about 1. Therefore, we can say
predicted value in near to true value. In figures. 17, 18, and 19, we plotted the true and
predicted value in one shot. Considering these results leads us to choose the LSTM, and
GRU as best networks. Because, figure.13 shows us, LSTM and GRU have better results
related to accuracy and loss.
We should speak about training time to compare their speed. Training time for RNN,
LSTM, and GRU is 458.47, 282.16, and 273.47 second respectively. So, LSTM and GRU
are faster than simple RNN. GRU is faster than LSTM, because it has 2 gates despite of
LSTM that have 4 gates to control information.
RNN’s are good for processing sequence data for predictions but suffers from short-term
memory (here we have short sequences, so it does not suffer from short memory). LSTM’s
and GRU’s were created as a method to mitigate short-term memory using mechanisms
called gates. Gates are just neural networks that regulate the flow of information flowing
through the sequence chain.
LSTM and GRU are same, but we choose LSTM for upcoming parts

In this part, we have a LSTM network with two different loss functions. We use MAPE and
MSE as loss function in our network. We are going to compare two networks that are serving
different loss functions. First of all, we present some description about MAPE and MSE.
Mean Squared Error Definition
The mean squared error tells you how close a regression line is to a set of points. It does
this by taking the distances from the points to the regression line (these distances are the
“errors”) and squaring them. The squaring is necessary to remove any negative signs. It also
gives more weight to larger differences. It’s called the mean squared error as you’re finding
the average of a set of errors.

𝑀 = 1/𝑛 ∑(𝑦𝑖 − 𝑦̃𝑖)2

The smaller the means squared error, the closer you are to finding the line of best fit.
Depending on the data, it may be impossible to get a very small value for the mean squared
error. We could try several equations, and the one that gave us the smallest mean squared
error would be the line of best fit.
Mean absolute percentage error (MAPE)
The mean absolute percentage error (MAPE) is a statistical measure of how accurate a
forecast system is. It measures this accuracy as a percentage, and can be calculated as the
average absolute percent error for each time period minus actual values divided by actual
values. Where At is the actual value and Ft is the forecast value, this is given by:

𝑀 = 1/𝑛 ∑|(𝐴𝑡−𝐹𝑡)/𝐴𝑡|

Also, it is the most common measure used to forecast error, and works best if there are no
extremes to the data (and no zeros). The MAPE can only be computed with respect to data
that are guaranteed to be strictly positive. (Note: if the Stat graphics Forecasting procedure
does not display MAPE in its model-fitting results, this usually means that the input variable
contains zeroes or negative numbers, which can happen if it was differenced outside the
forecasting procedure or if it represents a quantity that can honestly be zero in some periods.16


According to the figures. 20, 21, and 22, it is obvious serving MSE as loss function is
more efficient.
In the sequel, there are some more results about comparing true and predicted value for
each borough of city

Conclusion and Discussion:
As it is obvious in above figures, the MSE is more accurate than MAPE. According to
fig. 23 and fig. 24, the network by using MAPE as loss function cannot predict but network
is serving MSE as loss function can predict efficiently.
MAPE cannot be used when percentages make no sense. For example, the Fahrenheit and
Celsius temperature scales have relatively arbitrary zero points, and it makes no sense to talk
about percentages. MAPE also cannot be used when the time series can take zero values.


Optimizer: SGD-RMSProp-Adam
In this part, we use three different optimizers to design our network. We implemented
these networks to find out which one is more accurate. The results are released in below.

Conclusion and Discussion:
As you can see in figure27, RMSProp and Adam are more accurate comparing to SGD.
Accuracy for RMSProp and Adam is much more than SGD. Because of that, serving SGD
in network cannot help to predict efficiently (see Figure28). While we are using SGD in our
network, we will face with gradient vanishing in first layers, so we cannot update weights.
RMSProp and Adam use adaptive learning rate in contrast with SGD. Also, RMSProp
and Adam use all past gradient for next updating step but not SGD.
According to the results, Adam works efficiently compare to RMSProp, because Adam
is using both mean and variance momentum nut RMSProp just is serving variance
momentum.

Drop-out
In this section, we want to show effect of dropout on our results. We design Three RNN,
LSTM, and GRU recurrent networks to identify effect of dropout. 

Conclusion and Discussion:
According to the aforementioned results, we can say that using dropout in our designed
network can help network to work better than when we do not use dropout. Dropout prevents
occurring overfitting. As you can see in figures 34, 39, and 44, when we are using dropout
the accuracy is better than when we are not, and also the loss value is lower.
We served 20% dropout rate for our designed networks



