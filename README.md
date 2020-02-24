# Character-based Neural Network Translation

## Introduction
Traditional methods in language modeling involve making an n-th order Markov assumption and estimating n-gram probabilities via counting. The count-based models are simple to train, but due to data sparsity, the probabilities of rare n-grams can be poorly estimated. Neural Language Models (NLM) address the issue of n-gram data sparsity by utilizing word embeddings [3]. These word embeddings derived from NLMs exhibit the property whereby semantically close words are close in the induced vector space. Even though NLMs outperform count-based n-gram language models [4], they are oblivious to subword information (e.g. morphemes). Embeddings of rare words can thus be poorly estimated, leading to high perplexities (Perplexity is the measure of how well a probability distribution predicts a sample) which is especially problematic in morphologically rich languages.

## Character-level Convolutional Neural Network
Let <img src="https://render.githubusercontent.com/render/math?math={\displaystyle C}"> be the vocabulary of characters, <img src="https://render.githubusercontent.com/render/math?math={\displaystyle d}"> be the dimensionality of character embeddings, and <img src="https://render.githubusercontent.com/render/math?math={\displaystyle Q\in R^{d\times |C|}}{\displaystyle Q\in R^{d\times |C|}}"> be the matrix of character embeddings. 
Suppose that word <img src="https://render.githubusercontent.com/render/math?math={\displaystyle k\in V}"> is made up of a sequence of characters <img src="https://render.githubusercontent.com/render/math?math={\displaystyle [c_{1},...,c_{l}]}"> where <img src="https://render.githubusercontent.com/render/math?math={\displaystyle l}"> is the length of word <img src="https://render.githubusercontent.com/render/math?math={\displaystyle k}"> Then the character-level representation of <img src="https://render.githubusercontent.com/render/math?math={\displaystyle k}"> is given by the matrix <img src="https://render.githubusercontent.com/render/math?math={\displaystyle C^{k}\in R^{d\times l}}"> where the <img src="https://render.githubusercontent.com/render/math?math={\displaystyle j}">th column corresponds to the character embedding for <img src="https://render.githubusercontent.com/render/math?math={\displaystyle c_{j}}">.

A convolution between <img src="https://render.githubusercontent.com/render/math?math={\displaystyle C^{k}}">and a filter (or kernel) <img src="https://render.githubusercontent.com/render/math?math={\displaystyle H\in R^{d\times w}}"> of width <img src="https://render.githubusercontent.com/render/math?math={\displaystyle w}"> is applied, after which a bias is added followed by a nonlinearity to obtain a feature map <img src="https://render.githubusercontent.com/render/math?math={\displaystyle f_{k}\in R^{l-w+1}}">. The <img src="https://render.githubusercontent.com/render/math?math={\displaystyle i}">th element of <img src="https://render.githubusercontent.com/render/math?math={\displaystyle f_{k}}"> is:

<img src="https://render.githubusercontent.com/render/math?math={\displaystyle f^{k}[i]}"> = tanh(<img src="https://render.githubusercontent.com/render/math?math={\displaystyle C^{k}[*,i:i+w-1]}"> H) +b,
where <img src="https://render.githubusercontent.com/render/math?math={\displaystyle C^{k}[*,i:i+w-1]}"> is the <img src="https://render.githubusercontent.com/render/math?math={\displaystyle i}">-to-<img src="https://render.githubusercontent.com/render/math?math={\displaystyle (i+w-1)}">-th column of <img src="https://render.githubusercontent.com/render/math?math={\displaystyle C_{k}}"> and <img src="https://render.githubusercontent.com/render/math?math={\displaystyle <A,B>=Tr(AB^{T})}{\displaystyle <A,B>=Tr(AB^{T})}"> is the Frobenius inner product. Finally, take the max-over-time:

<img src="https://render.githubusercontent.com/render/math?math={\displaystyle y^{k}=max_{i}f^{k}[i]}{\displaystyle y^{k}=max_{i}f^{k}[i]}">   as the feature corresponding to the filter <img src="https://render.githubusercontent.com/render/math?math={\displaystyle H}"> (when applied to word <img src="https://render.githubusercontent.com/render/math?math={\displaystyle k})">. The idea, the authors say, is to capture the most important feature for a given filter. "A filter is essentially picking out a character n-gram, where the size of the n-gram corresponds to the filter width". Thus the framework uses multiple filters of varying widths to obtain the feature vector for <img src="https://render.githubusercontent.com/render/math?math={\displaystyle k}">. So if a total of <img src="https://render.githubusercontent.com/render/math?math={\displaystyle h}">  filters  <img src="https://render.githubusercontent.com/render/math?math={\displaystyle H_{1},...,H_{h}}"> are used, then <img src="https://render.githubusercontent.com/render/math?math={\displaystyle yk=[y_{1}^{k},...,y_{h}^{k}]}"> is the input representation of <img src="https://render.githubusercontent.com/render/math?math={\displaystyle k}">.

## Highway network

<img src="https://render.githubusercontent.com/render/math?math={\displaystyle z=t\odot g(W_{H}y+b_{H})+(1-t)\odot y}">
where <img src="https://render.githubusercontent.com/render/math?math={\displaystyle g}"> is a nonlinearity, <img src="https://render.githubusercontent.com/render/math?math={\displaystyle t=\sigma (W_{T}y+b_{T})}">  is called the transform gate, and <img src="https://render.githubusercontent.com/render/math?math={\displaystyle (1-t)}"> is called the carry gate. Similar to the memory cells in LSTM networks, highway layers allow for training of deep networks by carrying some dimensions of the input directly to the output.


Essentially the character level CNN applies convolutions on the character embeddings with multiple filters and max pools from these to get a fixed dimensional representation. This is then fed to the highway layer which helps in encoding semantic features which are not dependent on edit distance alone. The output of the highway layer is then fed into an LSTM that predicts the next word.



## Convolutional Network based Encoder Model

![img2](/img/encoder.png)
1. Convert word to character indices.
2. Padding and embedding lookup
3. Convolutional network. To combine these character embeddings, we'll use 1-dimensional convolutions. The convolutional layer has two hyperparameters:4 the kernel size k (also called window size), which dictates the size of the window used to compute features, and the number of filters f(also called number of output features or number of output channels).
4. Highway layer and dropout


## Character-based LSTM decoder for NMT
The LSTM-based character-level decoder to the NMT system, based on Luong & Manning's paper. The main idea is that when our word-level decoder produces an <UNK> token, run the character-level decoder (which you can think of as a character-level conditional language model) to instead generate the target word one character at a time, as shown in Figure. This will help us to produce rare and out-of-vocabulary target words.
![img2](/img/decoder.png)
  
  
  
