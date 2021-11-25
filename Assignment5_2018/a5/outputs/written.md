* Hard-Copy. https://gitlab.com/vojtamolda/stanford-cs224n-nlp-with-dl

## Written Part
    
### 1. Character-based convolutional encoder for NMT

(a) e_word=256, e_char=50. Why e_word > e_char?

    Answer
    ---
    Size of the character vocabulary is several orders of magnitude smaller (50:50,000) than the word vocabulary. This implies we don't need as many dimensions to represent each character. Smaller character embedding size also makes sense intuitively. A single letter doesn't convey as much information as an entire word.


(b) Total number of parameters in the character-based embedding model. Then do the same for the word-based lookup embedding model.
    Write each answer as a single expression (though you may show working) in terms of 
    
    - e_char 
    - k 
    - e_word 
    - V_word (the size of the word-vocabulary in the lookup embedding model)
    - V_char (the size of the character-vocabulary in the character-based embedding model)
    
    Given that in our code, k = 5, Vword ⇡ 50,000 and Vchar = 96, state which model has more parameters, and by what factor 
    (e.g. twice as many? a thousand times as many?).

    Answer
    ---
    1. The character-based embedding model
        - E_char: e_char * v_char = 50 * 96 = 4.8k 
        - w_conv: e_word * e_char * k = 256 * 50 * 5 = 64k
        - b_conv: e_word = 256
        - W_proj, W_gate = 2 * e_word * e_word = 131k
        - b_proj, b_gate = 2 * e_word = 512
        ----
        total: 267.6k

    2. The word-based lookup embedding model
        - E_word: e_word * V_word = 256*50k = 12.8M

    Character-level model has about 50xless parameters. However it's significant more computationally expensive. Forward pass of each character costs about
    e_char*C_word*K + 2*e_word^2 + 4*e_word ~= 197k Floating point operations. Word level model doesn't do any calculations and only retrieves the embed-vector corresponding to a word index.

(c) Explain one advantage of using a convolutional architecture rather than a recurrent architecture for this purpose, making it clear how the two contrast. Below
    is an example answer; you should give a similar level of detail and choose a different advantage.

    When a 1D convnet computes features for a given window of the input, those features depend on the window only – not any other inputs to the left or right. By contrast, a RNN needs to compute the hidden states sequentially, from left to right (and also right to left, if the RNN is bidirectional). Therefore, unlike a RNN, a convnet’s features can be computed in parallel, which means that convnets are generally faster, especially for long sequences.

    Answer
    ---
    It orders to obtain information from the neighboring characters both on the left and on the right, the character level LSTM needs to be bi-directional. In contrast, when the convolution filter is passing over the sequence context from (k-1)/2 characters on the left and on the right of the center is included in the output calculation.

(d) In lectures we learned about both max-pooling and average-pooling. 
    
    For each pooling method, please explain one advantage in comparison to the other pooling method. For each advantage, make it clear how the two contrast, and write to a similar level of detail as in the example given in the previous question.

    Answer
    ---
    Max pooling \
        extract features that are the most important (i.e. produce the largest activation) in the sequence. The key property is that it's position independent and also robust to modest amount of noise. During back-propagation, gradient flows only to the max element of the sequence and the rest of the features don't receive any update.
    
    Average pooling \
        produces a something like a typical feature representation that can sometimes hide the outliers. It also keeps the location information and the gradient is distributed equally to all features the average is calculated from.

(e) Code, Convert sentene to word to charcter and Add  `start_of_word` character with the `end_of_word` character. 
    
    Answer
    ---
    vocab.words2charindices
    
    Check Method
    ---
    python sanity_check.py 1e

(f) Code, Padding

    Answer
    ---
    utils.pad_sents_chars
    
    Check Method
    ---
    python sanity_check.py 1f

(g) Code, Input Tensor implemenation, output has shape: (max sentence length, batch size, max word length)

    Answer
    ---
    vocab.to_input_tensor_char


(h) Code, highway metohd

    Answer
    ---
    highway.Highway.__init__
    highway.Highway.forward

    test_module: highway.HighwaySanityChecks

(i) Code, CNN

    Answer
    ---
    cnn.CNN.__init__
    cnn.CNN.forward


(j) Code, ModelEmbedding

    Answer
    ---
    model_embedding.ModelEmbeddings.__init__
    model_embedding.ModelEmbeddings.forward

    Check Method:python sanity_check.py 1j


### 2. Character-based LSTM decoder for NMT

(a) Code, CharDecoder structure

    Answer
    ---
    char_decoder.CharDecoder.__init__

    Check Method
    ---
    python sanity_check.py 2a

(b) Code, CharDecoder forward

    Answer
    ---
    char_decoder.CharDecoder.forward

    Check Method
    ---
    python sanity_check.py 2b

(c) Code, CharDecoder train

    Answer
    ---
    char_decoder.CharDecoder.train_forward

    Check Method
    ---
    python sanity_check.py 2c

(d) Code, Greedy coding

    Answer
    ---
    char_decoder.decode_greedy

    Check Method
    ---
    python sanity_check.py 2d

(e) Train/Test, Local train/test

    sh run.sh train_local_q2
    sh run.sh test_local_q2


(f) Train/Test, VM train/test

    pip install -r gpu_requirements.txt
    sh run.sh train
    sh run.sh test

    Result
    ---
    report your test set BLEU score

