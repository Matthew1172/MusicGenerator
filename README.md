# MusicGenerator
A deep-learning Music generator based on GoodBoyChan for [Professor J. Zhang's](https://www.ccny.cuny.edu/profiles/jianting-zhang) Senior Design course at [The City College of New York (CCNY)](https://www.ccny.cuny.edu/engineering) for the Spring '22 semester.

## Table of Contents
- [MusicGenerator](#musicgenerator)
  - [Table of Contents](#table-of-contents)
  - [Contributors](#contributors)
  - [Introduction](#introduction)
  - [Background/Motivation/Targeted Application](#backgroundmotivationtargeted-application)
  - [Architecture/Framework](#architectureframework)
  - [Progresses on Components/Steps](#progresses-on-componentssteps)
    - [Dataset](#dataset)
    - [Process the dataset for the learning task](#process-the-dataset-for-the-learning-task)
    - [The Recurrent Neural Network (RNN) model](#the-recurrent-neural-network-rnn-model)
    - [Training the Model](#training-the-model)
    - [Generate music using the RNN model](#generate-music-using-the-rnn-model)
  - [Plan For the Next Two Weeks](#plan-for-the-next-two-weeks)
  - [Sources](#sources)

## Contributors 
[Bhumika Bajracharya](https://github.com/mika-shree), [Liana Hassan](https://github.com/lianahasan), [Gene Lam](https://github.com/genelam26), [Matthew Pecko](https://github.com/Matthew1172), [Daniel Rosenthal](https://github.com/danrose499) 

## Introduction
Music composition has long been a part of the video game experience with game developers spending as much as 15% of their production budget on custom soundtracks (Sweet). Game directors want tailored music that will immerse the player in virtual worlds and enhance their gameplay. We are proposing the development of a music generator that uses the transformer machine learning model to create new music. Previous attempts from our group have used 1. An Optical Music Recognition (OMR) System to classify notes on prepared sheet music 2. Long short-term memory (LSTM) model to generate new sheet music based on output from the OMR system. After obtaining low accuracy on the OMR system, we plan to pivot our project and develop the transformer model, which would generate new music based on prepared sheet music directly. As the need for personalized user experience grows in the gaming industry with investments in the metaverse, we are developing a music generator for the next generation of games. 

## Background/Motivation/Targeted Application
Why a music generator? As it stands, most video games have soundtracks that play as you progress through the game. After spending enough time playing, however, one starts to notice that there are a limited amount of songs that play cyclically, making the game feel limited and repetitive. Imagine, for contrast, a video game that plays a new song every time you play where each song is still thematically linked. Using our music generator, we can feed the model with songs written for a specific game and have the generator create a new song every time you play with a similar feel to the overall soundtrack. Moreover, many games currently have discrete songs for different situations. You can have peaceful music playing as you travel through the world only to have the song suddenly stop for more aggressive music to start playing as you start to fight a boss. Our music generator could be extended to allow game signals to tell the generator to make the song change its theme without stopping it. This will give the game a much greater sense of continuity as the song can change along with your progress instead of having to stop for a new one to start.

Obviously, this is only one application of a music generator, but it is one that demonstrates the creative impacts that it can have. But this project has uses in many different scenarios, such as artists, creators, composers, and producers who are looking for a creative spark when starting a new project. 

Additionally, the music generator makes music creation more accessible. Music theory is a dense field with ubiquitous minutiae that can intimidate and scare away prospective musicians from attempting to create music. Giving a tool to young aspiring artists that allows them to generate music based on their tastes can inspire a new generation of artists.

Finally, there also exist commercial applications of the music generator. We believe that this model has the potential to unlock new trends in the music industry as it can learn from similar aspects used in popular songs that humans may not be aware of. Additionally, many users of services like Spotify have expressed how impressed they were that those services were able to use machine learning to predict which songs they will like and add them to their mixes. With a music generator, we believe that these music services will be able to use their data on listeners’ favorite songs to create entirely new songs that they should enjoy instead of recommending them songs that already exist. 

## Architecture/Framework
The transformer model will be used for the music melody generator project. The transformer is a deep learning model that adopts the mechanism of self-attention, which puts different weights regarding the significance of each section of the input data. It is non-sequential and thus, does not rely on previous hidden states to capture the dependencies. This process allows for more parallelization as well, increasing the computational speed compared to other models such as LSTM or RNNs, and utilizes multi-processing in GPUs.

Transformers process data with positional embeddings, which encode information related to a specific position of a token. The input sequence is fed to the encoder after it is converted into embeddings, which is processed by the stack of encoders and produces an encoded representation of the input sequence. The target sequence is prepended with the start-of-sentence token, converted to positional embeddings, and fed to the decoder, which processes this along with the encoder’s encoded representation to produce an encoded representation of the target sequence. The output layer converts it into word probabilities and the final output sequence. The highest probability part becomes the predicted output for the next section in the output sentence. 

## Progresses on Components/Steps
The initial plan for this project was to use the open-source project called Mozart, which converts sheet music to machine-readable data. Then, to combine the Mozart project with another open-source project, which converts ABC files to MIDI files. After experimenting with Mozart and other Optical Music Recognition (OMR) software, we found that the accuracy is very poor with the most basic melodies. Due to the fact that Mozart and other OMR software have poor accuracy, that portion of the project will be put aside for the time being. Our project is based on a software lab for an introductory deep learning class at MIT. The notebook for the lab can be found here. The project lists the following components:

- Dataset
- Process the dataset for the learning task
    - Vectorize the text
    - Create training examples and targets
- The Recurrent Neural Network (RNN) model
    - Define the RNN model
    - Test out the RNN model
    - Predictions from the untrained model
- Training the model: loss and training operations
- Generate music using the RNN model
    - Restore the latest checkpoint
    - The prediction procedure
    - Play back the generated music!

The following section will explain each component in depth, and how we are going to modify it in our project. It should be explicitly said that we’re publishing this code under the MIT license and referencing [© MIT 6.S191: Introduction to Deep Learning](http://introtodeeplearning.com).

### Dataset
The dataset is a text file of many songs that are in textual ABC notation, separated by new lines. ABC notation also gives the name of the song and other metadata like key signature and tempo.

### Process the dataset for the learning task
ABC is a powerful representation because it allows us to vectorize the characters and pass them into our model. Our model will return a vector of numbers that can be mapped back to the musical notes so that we can write a new ABC file with the generated song in it. We also divide the entire dataset into example sequences that we'll use during training. Each input sequence that we feed into our RNN will contain a predefined length of characters from the text. We'll also need to define a target sequence for each input sequence, which will be used in training the RNN to predict the next character. For each input, the corresponding target will contain the same length of text, except shifted one character to the right. To do this, we'll break the text into chunks of seq_length+1. Suppose seq_length is 4 and our text is "Hello". Then, our input sequence is "Hell" and the target sequence is "ello". The batch method will then let us convert this stream of character indices to sequences of the desired size.

### The Recurrent Neural Network (RNN) model
We will define and train a RNN model on our ABC music dataset, and then use that trained model to generate a new song. We'll train our RNN using batches of song snippets from our dataset, which we generated in the previous section. The model is based on the LSTM architecture, where we use a state vector to maintain information about the temporal relationships between consecutive characters. The final output of the LSTM is then fed into a fully connected Dense layer where we'll output a softmax over each character in the vocabulary, and then sample from this distribution to predict the next character. Here is a description of the layers:

- Embedding: This is the input layer, consisting of a trainable lookup table that maps the numbers of each character to a vector with embedding_dim dimensions.
- LSTM: Our LSTM network, with size units=rnn_units.
- Dense: The output layer, with vocab_size outputs.

Included below is the code from the lab for the layers described above, implemented in Keras.
``` python
def LSTM(rnn_units): 
  return tf.keras.layers.LSTM(
    rnn_units, 
    return_sequences=True, 
    recurrent_initializer='glorot_uniform',
    recurrent_activation='sigmoid',
    stateful=True,
  )
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    # Layer 1: Embedding layer to transform indices into dense vectors 
    #   of a fixed embedding size
    tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),

    # Layer 2: LSTM with `rnn_units` number of units. 
    LSTM(rnn_units),

    # Layer 3: Dense (fully-connected) layer that transforms the LSTM output
    #   into the vocabulary size. 
    tf.keras.layers.Dense(units=vocab_size)
  ])

  return model
```
In the next code snippet below, you can see our PyTorch implementation for the layers described previously. Notice that we had to add an extra “dummy” layer called GetLSTMOutput because the PyTorch LSTM layer returns a tuple and the linear layer requires a tensor. The first element contains the output features from the last layer of the LSTM, for each timestep. We want to pass this element of the tuple into the next layer so we create another layer to retrieve this. The second element of the tuple is another tuple of two tensors, the first containing the final hidden state for each element in the batch; and the second containing the final cell state for each element in the batch. We are not interested in either of these tensors so we discard them.

``` python
class GetLSTMOutput(torch.nn.Module):
   def forward(self, x):
       out, _ = x
       return out

class MusicGenerator(torch.nn.Module):
   def __init__(self, vocab_size, embedding_dim, rnn_units, batch_size, seq_length):
       super().__init__()

       self.lstm_model = torch.nn.Sequential(
           torch.nn.Embedding(batch_size*seq_length, embedding_dim),
           torch.nn.LSTM(embedding_dim, batch_size*embedding_dim, batch_first=True),
           GetLSTMOutput(),
           torch.nn.Linear(rnn_units, vocab_size)
       )

   def forward(self, x):
       x = self.lstm_model(torch.tensor(x))
       return x
```

Then we call to build the Keras model:
``` python
model = build_model(len(vocab), embedding_dim=256, rnn_units=1024, batch_size=4)
```

And then we construct the PyTorch model:
``` python
model = MusicGenerator(len(vocab), embedding_dim=256, rnn_units=1024, batch_size=4, seq_length=100)
```
 The model summary for the Keras and PyTorch implementations are shown below respectively:

 ```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (4, None, 256)            21248     
                                                                 
 lstm (LSTM)                 (4, None, 1024)           5246976   
                                                                 
 dense (Dense)               (4, None, 83)             85075     
                                                                 
=================================================================
Total params: 5,353,299
Trainable params: 5,353,299
Non-trainable params: 0
 ```

 ```
MusicGenerator(
  (lstm_model): Sequential(
    (0): Embedding(400, 256)
    (1): LSTM(256, 1024, batch_first=True)
    (2): GetLSTMOutput()
    (3): Linear(in_features=1024, out_features=83, bias=True)
  )
)
Input shape:       (4, 100)  # (batch_size, sequence_length)
Prediction shape:  torch.Size([4, 100, 83]) # (batch_size, sequence_length, vocab_size)
 ```

### Training the Model
After converting the code we can train our model. The graph below shows the loss for each iteration for the Keras implementation:   
![keras_loss](https://i.imgur.com/IEws8Tz.jpg)  

And the figure below shows the loss for each iteration for our PyTorch implementation:  
![pytorch_loss](https://i.imgur.com/xoQiTcq.jpg)  

As we can see the loss function is not converging for our model as well as it was for the Keras one. We believe that there is a bug in the training step which will be dealt with in the coming days.  

### Generate music using the RNN model
Since our optimizer is converging slowly we’re not able to generate any songs in valid ABC notation. Included below is a generated song in valid ABC notation from the Keras implementation of the model.
```
X:7
T:Blamoll Destay Hont
Z: id:dc-reel-261
M:C
L:1/8
K:G Major
G2|BG dGBG|AGAB cBc:|!
d|Beed Bedc|BBbB BAA2|BedB AdBA|GBAG FAdB|dBAF E2:|!
^c|dgg2 agg^fd|egg2 dgg2|agg^f d2C|]!
Bc|dedB cdef|gdBd gdBF|dedB cAFA|G2B2 c2:|!
d>e|fd ed cd|eg ag/f/g/|fd BA|!
cB BA|c2 d2:|!
ded a^ga|b2a bag|fde ecA|ced f2e|f2g age|fae fdc|!
d3 d2:|!
```
## Plan For the Next Two Weeks
We are currently working on taking the existing Keras code and transforming it into PyTorch code. The reason for this is because we want to use the native transformer layer that PyTorch offers because Keras doesn’t offer one. Rebuilding includes converting each sequential layer from Keras to PyTorch and re-training the model. After converting the model, we also need to convert the Keras loss function into a PyTorch one. Currently, we are using a sparse categorical cross-entropy loss function that Keras offers. PyTorch offers a cross-entropy loss function as well. However, the function call is different and does not exactly follow the same mathematical implementation. Converting the optimizer has been trivial because once we build the model, PyTorch will take care of everything for us after we ask it to use a specific optimizer (in this case Adam). We are also working on converting the training loop to work with our new model, loss, and optimizers. After the conversion is completed, we will work on replacing the LSTM layer with a transformer one.

## Sources
Gaming Music: How to Price Your Composition Work – Berklee Online Take Note. (2022, 
January 03). Retrieved from https://online.berklee.edu/takenote/gaming-music-how-to-price-composition-work  

Transformers Explained Visually (Part 1): Overview of Functionality- Ketan Doshi. (2020, 
December 13). Retrieved from https://towardsdatascience.com/transformers-explained-visually-part-1-overview-of-functionality-95a6dd460452  

Vaswani, Ashish; Shazeer, Noam; Parmar, Niki; Uszkoreit, Jakob; Jones, Llion; Gomez, Aidan 
N.; Kaiser, Lukasz; Polosukhin, Illia (2017-06-12). "Attention Is All You Need". arXiv:1706.03762
