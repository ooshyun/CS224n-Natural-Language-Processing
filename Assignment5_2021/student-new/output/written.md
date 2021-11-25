# Written Part for Assignment 5
## 1. Attention exploration

- The reason, referred by [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
    
    1. It expands the model’s ability to focus on different positions. Yes, in the example above, z1 contains a little bit of every other encoding, but it could be dominated by the the actual word itself. It would be useful if we’re translating a sentence like “The animal didn’t cross the street because it was too tired”, we would want to know which word “it” refers to. 

    2. It gives the attention layer multiple “representation subspaces”. As we’ll see next, with multi-headed attention we have not only one, but multiple sets of Query/Key/Value weight matrices (the Transformer uses eight attention heads, so we end up with eight sets for each encoder/decoder). Each of these sets is randomly initialized. Then, after training, each set is used to project the input embeddings (or vectors from lower encoders/decoders) into a different representation subspace.

**Multi-headed self-attention** is the core modeling component of Transformers. In this question, we’ll get some practice working with the self-attention equations, and motivate **why multi-headed self-attention can be preferable to single-headed self-attention.**

(a) Copying in attention: Recall that attention can be viewed as an operation on a query $q ∈ \Reals^d$, a set of value vectors $\{v_1,...,v_n\},v_i ∈ \Reals^d$, and a set of key vectors $\{k_1,...,k_n\},k_i ∈ R_d$, specified as follows:

$c=\displaystyle\sum_{i=1}^{n}v_i\alpha_i$  (1)

$\alpha_i = \dfrac{exp(k_i^Tq)}{\sum_{j=1}^n exp(k_j^Tq)}$ (2)

where $α_i$ are frequently called the “attention weights”, and the output c $∈ \Reals^d$ is a correspondingly weighted average over the value vectors. We’ll first show that it’s particularly simple for attention to “copy” a value vector to the output c. 

Describe (in one sentence) **what properties of the inputs** to the attention operation would result in the output c being approximately equal to $v_j$ for some $j ∈ \{1, . . . , n\}$. Specifically, what must be true about the query q, the values $\{v_1,...,v_n\}$ and/or the keys $\{k_1,...,k_n\}$?

- Answer
    
    $i\not=j, \ k_j^Tq \gg k_i^Tq$   Distribution of data is overweighted to a point.
    

(b) An average of two: Consider a set of key vectors $\{k_1, . . . , k_n\}$ where all key vectors are perpendicular, that is  $k_i ⊥ k_j$ for all $i \not= j$. Let $∥k_i∥ = 1$ for all i. Let $\{v_1,...,v_n\}$ be a set of arbitrary value vectors. Let $v_a, v_b ∈ \{v_1, . . . , v_n\}$ be two of the value vectors. Give an expression for a query vector q such that the output c is approximately equal to the average of $v_a$ and $v_b$, that is, $\frac{1}{2}(v_a +v_b)$.$^1$ Note that you can reference the corresponding key vector of $v_a$ and $v_b$ as $k_a$ and $k_b$.

- Answer
    
    $q=t(k_a+k_b),\ t \gg 0$
    

(c)  Drawbacks of single-headed attention: In the previous part, **we saw how it was possible for single-headed attention to focus equally on two values**. The same concept could easily be extended to any subset of values. In this question we’ll see why it’s not a practical solution. Consider a set of key vectors $\{k_1, . . . , k_n\}$ that are now randomly sampled, $k_i ∼ N(μ_i, Σ_i)$, where **the means $μ_i$ are known to you, but the covariances $Σ_i$ are unknown.** Further, assume that the means $μ_i$ are all perpendicular; $μ^T_i μ_j$ = 0 

if $i \not= j$, and unit norm, $∥μ_i∥ = 1$.

i.  Assume that the covariance matrices are $Σ_i = αI$, for vanishingly small $α$. Design a query in terms of the $μ_i$ such that as before, $c ≈ \frac{1}{2} (v_a + v_b)$, and provide a brief argument as to why it works.

- Answer
    
    $q=t(u_a+u_b),\ t\gg0$
    

ii.  Though single-headed attention is resistant to small perturbations in the keys, **some types of larger perturbations may pose a bigger issue.** Specifically, in some cases, one key vector $k_a$ may be larger or smaller in norm than the others, while still pointing in the same direction as $π_a$ . 

As an example, let us consider a covariance for item a as $Σ_a = αI + \frac{1}{2}(μ_a μ^T_a)$ for vanishingly small α (as shown in figure 1). Further, let $Σi = αI$ for all  $i \not= a$.
When you sample $\{k_1 , . . . , k_n\}$ multiple times, and use the q vector that you defined in part i., what qualitatively do you expect the vector c will look like for different samples?

- Answer
    
    Shortly, if a side of input is biased, then it easily decided to that direction
    
    We got $k_{a} \sim \mathcal{N}\left(\mu_{a}, \alpha I+\frac{1}{2}\left(\mu_{a} \mu_{a}^{\top}\right)\right)$ ,and for vanishingly small $\alpha: k_{a} \approx \epsilon_{a} \mu_{a}, \epsilon_a \sim \mathcal{N}(1, \frac{1}{2}),\ when \ q = t(u_a+u_b), t\gg 0:$
    
    $k_i^Tq \approx 0 \text{ for } i \notin\{a, b\}$
    
    $k_a^Tq \approx \epsilon_a t$
    
    $k_b^Tq \approx \epsilon_b t$
    
    then:
    
    $\begin{aligned}c & \approx \frac{\exp (\epsilon_a t)}{\exp (\epsilon_a t)+\exp (\epsilon_b t)} v_{a}+\frac{\exp (\epsilon_b t)}{\exp (\epsilon_a t)+\exp (\epsilon_b t)} v_{b} \\&=\frac{1}{\exp ((\epsilon_b-\epsilon_a) t)+1} v_{a}+\frac{1}{\exp ((\epsilon_a-\epsilon_b) t)+1} v_{b}\end{aligned}$
    
    Since $\epsilon_a, \epsilon_b \sim \mathcal{N}(1, \frac{1}{2})$, when $\epsilon_a > \epsilon_b,\ c$ will be closer to $v_a$, vice versa.
    
    (i.e. $c$ will be closer to those with larger $||k||$)
    

(d)  Benefits of multi-headed attention: Now we’ll see some of the power of multi-headed attention. We’ll consider a simple version of multi-headed attention which is identical to single- headed self-attention as we’ve presented it in this homework, except two query vectors ($q_1$ and $q_2$) are defined, which leads to a pair of vectors ($c_1$ and $c_2$), each the output of single-headed attention given its respective query vector. The final output of the multi-headed attention is their average,
$\frac{1}{2}(c_1+c_2)$. As in question 1(c), consider a set of key vectors $\{k_1 , . . . , k_n\}$ that are randomly sampled,

* Figure 1, Image is referred to assignment page

$k_i ∼ N(μ_i,Σ_i)$, where the means $μ_i$ are known to you, but the covariances $Σ_i$ are unknown. Also as before, assume that the means $μ_i$ are mutually orthogonal; $μ^T_i μ_j = 0$ if $i\not= j$, and unit norm $∥μ_i∥ = 1$.

i. Assume that the covariance matrices are $Σ_i = αI$, for vanishingly small $\alpha$. Design q1 and q2 such that c is approximately equal to $\frac{1}{2} (v_a + v_b )$.

- Answer
    
    $q_a=t_1\mu_a,\ t_1 \gg 0$
    
    $q_b=t_2\mu_b,\ t_2\gg0$
    

ii. (2 points) Assume that the covariance matrices are $Σ_a = αI + \frac{1}{2}(μ_aμ^⊤_a )$ for vanishingly small $α$, and $Σ_i = αI$  for all $i\not= a$. Take the query vectors q1 and q2 that you designed in part i. What, qualitatively, do you expect the output c to look like across different samples of the key vectors? Please briefly explain why. You can ignore cases in which $q_i^Tk_a < 0$.

- Answer
    
    $k_a^T q = \varepsilon_at_1$
    
    $k_b^T q = \varepsilon_bt_2$
    
    Then,
    
    $c_1 \approx v_a, c_2 \approx v_b$
    
    $c = \frac{1}{2}(c_1+c_2) \approx \frac{1}{2}(v_a+v_b)$

    Because Multi-head states are a portion of filling with the state divided into multiple parts, it is possible to have the number of heads split.
    

(e)[TO-DO] Key-Query-Value self-attention in neural networks: So far, we’ve discussed attention as a function on a set of key vectors, a set of value vectors, and a query vector. In Transformers, we perform self-attention, which roughly means that we draw the keys, values, and queries from the same data. More precisely, let $\{x_1,...,x_n\}$ be a sequence of vectors in $\Reals^d$. Think of each $x_i$ as representing word i in a sentence. One form of self-attention defines keys, queries, and values as follows. Let $V, K, Q ∈ \Reals^{d×d}$ be parameter matrices. Then

$v_i= Vx_i,~i\in\{1,\dots,n\}$

$k_i= Kx_i,~i\in\{1,\dots,n\}$

$q_i= Qx_i,~i\in\{1,\dots,n\}$

Then we get a context vector for each input i; we have $c_i = ∑^n_{j=1} α_{ij}v_j$, where $α_{ij}$ is defined as $α = \dfrac{exp(k_j^Tq_i)}{∑_{l=1}^nexp(k_l^Tq_i)}$  . Note that this is single-headed self-attention.

In this question, we’ll show how key-value-query attention like this allows the network to use different aspects of the input vectors $x_i$ in how it defines keys, queries, and values. **Intuitively, this allows networks to choose different aspects of $x_i$ to be the “content” (value vector) versus what it uses to determine “where to look“ for content (keys and queries.)**

i. First, consider if we didn’t have key-query-value attention. For keys, queries, and values we’ll just use $x_i$; that is, $v_i = q_i = k_i = x_i$. We’ll consider a specific set of $x_i$. In particular, let $u_a, u_b, u_c, u_d$ be **mutually orthogonal vectors** in $\Reals^d$, each with equal norm $∥u_a∥ = ∥u_b∥ = ∥u_c∥ = ∥u_d∥ = β$, where β is very large. Now, let our $x_i$ be:

$x_1 = u_d+u_b$, (6)

$x_2=u_a$, (7)

$x_3=u_c+u_b$, (8)

If we perform self-attention with these vectors, what vector does $c_2$ approximate? Would it be possible for $c_2$ to approximate $u_b$ by adding either $u_d$ or $u_c$ to $x_2$? Explain why or why not (either math or English is fine).

- Answer
    
    $c_2 \approx u_a$
    
    It's impossible for $𝑐_2$ to approximate $u_b$ by adding either $u_d$ or $u_c$ to $x_2$. 
    
    Say, if we add $u_d$ , $\alpha_{21}$ increases, which means the weight of $x_1$ increases, but $u_d$ and $u_b$ will increase equally in $c_2$, that's why $c_2$ can never be approximated to $u_b$
    

ii. Now consider using key-query-value attention as we’ve defined it originally. **Using the same definitions of $x_1, x_2$ and $x_3$ as in part i,** specify matrices K, Q, V such that $c_2 ≈ u_b$, and $c_1 ≈ u_b − u_c$. 

There are many solutions to this problem, so it will be easier for you (and the graders), if you first find V such that $v_1 =u_b$ and $v_3 =u_b−u_c$, then work on Q and K. Some outer product properties may be helpful (as summarized in this footnote)$^2$.

- Answer
    
    $V=u_b u_b^T \odot \dfrac{1}{\lVert
              u_b\rVert^2_2}-u_cu_c^T \odot \dfrac{1}{\lVert
              u_c\rVert^2_2}$
    
    $=u_b u_b^T \odot \dfrac{1}{\lVert
              u_b\rVert^2_2}-u_cu_c^T \odot \dfrac{1}{\lVert
              u_c\rVert^2_2}$
    
    $=(u_bu_b^T - u_cu_c^T)\odot \dfrac{1}{\beta^2}$
    
    $K=I$
    
    $Q=u_d u_a^T \odot \dfrac{1}{\lVert
              u_a\rVert^2_2}+u_cu_d^T \odot \dfrac{1}{\lVert
              u_d\rVert^2_2}$
    
    $=(u_du_a^T + u_cu_d^T)\odot \dfrac{1}{\beta^2}$
    
    Proof.
    
    $v_1=u_b, v_2=0, v_3 = u_b-u_c$
    
    $q_1=u_c,\ q_2=u_d,\ q_3=0$
    
    $k_i=x_i,\ i \in \{1,2,3\}$
    
    So,
    
    $\alpha_1 \approx [0,0,1],\ \alpha_2 \approx [1,0,1]$
    
    $c_1 \approx v_3 = u_b-u_c,\ c_2 \approx v_1 = u_b$