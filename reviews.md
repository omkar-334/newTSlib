# Reveews

## Imputation

1. [LSCD: Lomb--Scargle Conditioned Diffusion for Time series Imputation](https://openreview.net/forum?id=GdYg0Ohx0k) - ICML 25 poster

    - addressing irregularly sampled time series, particularly for regression tasks such as imputation and forecasting

2. [Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models](https://openreview.net/forum?id=hHiIbk7ApW) - TMLR

    - Are all models compatible with backward imputation, and does the SSSD use backward and forward smoothing in all imputation tasks?
    - Did the authors experiment with a setting where there is a mismatch between the missingness ratios during training and inference? For instance, a setting when the model is trained at 20% RM but is evaluated at 30% RM (and vice-versa)? Another possibility could be training for one missingness scenario (maybe 20% RM) and evaluating for MNR or BM? From my understanding of Section 4.1, the authors always train and evaluate across the same setting. It would be interesting to benchmark the generalization capabilities of the imputation model.
    - Diffusion and Training Hyperparameters: It would help to observe the impact of using larger number of timesteps during training and inference on the imputation performance. The authors do mention about this briefly in Appendix A.1 as: "We found that the fewer diffusion steps, the faster the network converges during training, however, at the cost of less accurate results". It would be great if authors can present some quantitative results about the same on a dataset (maybe PTB-XL) since the Inference speed vs performance tradeoff is central to diffusion models.

3. [DiffImp: Efficient Diffusion Model for Probabilistic Time Series Imputation with Bidirectional Mamba Backbone](https://openreview.net/forum?id=j1OucVFZMJ) - ICLR 2025 Reject

    - Although the author mainly emphasizes that this method is better than other methods based on the diffusion model, this is not enough. The author should consider more time series imputation methods such as [1,2,3,4].
        [1] Filling the Gaps: Multivariate Time Series Imputation by Graph Neural Networks.
        [2] Gatgpt: A pre-trained large language model with graph attention network for spatiotemporal imputation.
        [3] Learning to Reconstruct Missing Data from Spatiotemporal Graphs with Sparse Observations.
        [4] Multi-Variate Time Series Forecasting on Variable Subsets.
    - Scalability experiments are necessary to demonstrate the linear complexity of this paper. Therefore, it is suggested to conduct this on additional large real-world datasets.
    - Although the source code has been released, there is not guideline for how to use the codes and how to get the experimental results for each table and figure, leading to the doubts of reproducibility.
    - There is no parameter sensitivity analysis in experiments, which is also suggested to show the stability of models and illustrate how to determine parameters.

4. [Self-attention-based Diffusion Model for Time-series Imputation in Partial Blackout Scenarios](https://openreview.net/forum?id=79AtAA2bVD) - TMLR Reject
   - Results Discrepancies: Some of the results reported in the paper differ from those in previous works (e.g., PriSTI). However, the authors do not provide an explanation for these discrepancies. This raises concerns about whether the differences arise from the training strategy, model implementation, or some other factor.
   - Training Strategy and Result Variations: Could the differences in results from prior works (e.g., PriSTI) be due to different training strategies, or are there other factors that could explain these variations?

5. [Diffusion-TS: Interpretable Diffusion for General Time Series Generation](https://openreview.net/forum?id=4h1apFjO99) - ICLR 24 poster
   - The evaluation on the conditional tasks is limited. The model is only compared against Diffwave and CSDI (which is fairly close to Diffwave) and baselines from time series forecasting literature are missing. It is also unclear how these CSDI and Diffwave baselines were trained.
   - In Figure 6, is MAE computed only over missing data (imputation targets) or over the full time series including existing data? For Diffusion-TS-G, since a soft constraint is used to enforce the conditional generation, how closely do the generated time series for imputation/forecasting match the existing data?
   - How does the number of parameters compare between Diffusion-TS and its competitors?

6. [Frequency-aware Generative Models for Multivariate Time Series Imputation](https://openreview.net/forum?id=UE6CeRMnq3&noteId=7LmUCl1XoB) - NeurIPS 24 poster
   - In the evaluation, only RMSE and MAE are used as metrics. However, it would be better to include additional metrics such as CRPS.
   - [A] Choi and Lee, "Conditional Information Bottleneck Approach for Time Series Imputation, ICLR 2024. [B] Liu et al., "Multivariate Time-series Imputation with Disentangled Temporal Representations," ICLR 2024.
   - Explain how masking ratio/patterns affect model performance

7. [Conditional Information Bottleneck Approach for Time Series Imputation](https://openreview.net/forum?id=K1mcPiDdOJ) - ICLR 24 ppster
   - Can you provide a more comprehensive evaluation, including comparisons with state-of-the-art models like ODE-based models and diffusion models, to demonstrate the performance of your model?
        [1] Multi-Time Attention Networks for Irregularly Sampled Time Series
        [2] Wenjie Du, David Cot́e, and Yan Liu. Saits: Self-attention-based imputation for time series. Expert Systems with Applications, 219:119619, 2023.
        [3] Satya Narayan Shukla and Benjamin Marlin. Multi-time attention networks for irregularly sampled time series. In International Conference on Learning Representations, 2021.
        [4] Yusuke Tashiro, Jiaming Song, Yang Song, and Stefano Ermon. Csdi: Conditional score-based diffusion models for probabilistic time series imputation. Advances in Neural Information Processing Systems, 34:24804–24816, 2021

## Forecasting

1. [Retrieval-Augmented Diffusion Models for Time Series Forecasting](https://openreview.net/forum?id=dRJJt0Ji48&noteId=8wGyyvVUNr) - NeurIPS 24 poster

    - Experimental results do not contain standard deviation, which potentially limits the confidence and significance of the performance superiority.
    - Unknown GPU Memory Usage: There is no information on GPU memory usage and computation efficiency
    - Insufficient Ablation Tests: While Table 3 indicates the importance of choosing good retriever embeddings, there are no systematic ablation tests of the architecture developed (

2. [TimeDiT: General-purpose Diffusion Transformers for Time Series Foundation Model](https://openreview.net/forum?id=FvBTy5Dz9C&noteId=r1QlXBzace) - ICLR 2025 Reject

   - In the anomaly detection task, how sensitive are the results against the choice of 99th percentile threshold shown, did you experiment with others?
   - Which part of the architecture is most responsible for the performance improvement over baselines?
   - Channel Adaptation: Does the model accommodate varying channel counts via padding or another method?

## TimesNet Abstract

Time series analysis is of immense importance in extensive applications, such as
weather forecasting, anomaly detection, and action recognition. This paper focuses
on temporal variation modeling, which is the common key problem of extensive
analysis tasks. Previous methods attempt to accomplish this directly from the 1D
time series, which is extremely challenging due to the intricate temporal patterns.
Based on the observation of multi-periodicity in time series, we ravel out the com-
plex temporal variations into the multiple intraperiod- and interperiod-variations.
To tackle the limitations of 1D time series in representation capability, we extend
the analysis of temporal variations into the 2D space by transforming the 1D time
series into a set of 2D tensors based on multiple periods. This transformation can
embed the intraperiod- and interperiod-variations into the columns and rows of
the 2D tensors respectively, making the 2D-variations to be easily modeled by 2D
kernels. Technically, we propose the TimesNet with TimesBlock as a task-general
backbone for time series analysis. TimesBlock can discover the multi-periodicity
adaptively and extract the complex temporal variations from transformed 2D ten-
sors by a parameter-efficient inception block. Our proposed TimesNet achieves
consistent state-of-the-art in five mainstream time series analysis tasks, including
short- and long-term forecasting, imputation, classification, and anomaly detection.

## CnDiff Abstract

Time-series forecasting finds application across
domains such as finance, climate science, and
energy systems. We introduce the Conditional
Diffusion with Nonlinear Data Transformation
Model (CN-Diff), a generative framework that em-
ploys novel nonlinear transformations and learn-
able conditions in the forward process for time
series forecasting. A new loss formulation for
training is proposed, along with a detailed deriva-
tion of both forward and reverse process. The
new additions improve the diffusion model’s ca-
pacity to capture complex time series patterns,
thus simplifying the reverse process. Our novel
condition facilitates learning an efficient prior dis-
tribution. This also reduces the gap between the
true negative log-likelihood and its variational ap-
proximation. CN-Diff is shown to perform better
than other leading time series models on nine real-
world datasets. Ablation studies are conducted to
elucidate the role of each component of CN-Diff.