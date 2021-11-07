# This is a code implementation of the paper "Data Augmentation for P300-based Brain-Computer Interfaces Using Generative Adversarial Networks"

Available on https://ieeexplore.ieee.org/abstract/document/9385317

In this project, the application of GANs to augment EEG-ERP data is investigated.

To train models, use files starting with 'train_'. Select and hardcode parameters first:

<pre><code>python train_subject_specific.py --gan_type {dcgan (default), wgan_gp, vae}
</code></pre>

<pre><code>python train_subject_independent.py --gan_type {dcgan (default), wgan_gp, vae}
</code></pre>

<pre><code>python train_gan_test.py --gan_type {dcgan (default), wgan_gp, vae}
</code></pre>

<pre><code>python train_without_gans.py --train_type {subject-specific, subject-independent}
</code></pre>
