### Implementation and written questions

(g) (written) The `generate_sent_masks()` function in nmt model.py produces a tensor called `enc_masks`. It has shape (batch size, max source sentence length) and contains 1s in positions corresponding to ‘pad’ tokens in the input, and 0s for non-pad tokens. Look at how the masks are used during the attention computation in the step() function.

First explain (in around three sentences) what effect the masks have on the entire attention computation. Then explain (in one or two sentences) why it is necessary to use the masks in this way.

1. What effect the masks have on the entire attention computation?
    
    First, the mask operation sets $e_t[src\_len:]$ to negative infinity, i.e. the bit of 'pad' in $e_t$ becomes negative infinity. So we know the effect of mask operation is to make the prob of 'pad' in the attention vector($\alpha_t$ in the PDF) to be zero.
    
2. Why it is necessary to use the masks in this way?
    
    If we don't apply mask operation, the decode will use the information of 'pad' of hidden states, and $O_t$ maybe predicted as 'pad', that's what we don't expect.