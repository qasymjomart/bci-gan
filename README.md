# BCI-GAN project
In this project, the application of GANs to augment EEG-ERP data is investigated.

To train models, use files starting with 'train_'. Select and hardcode parameters first:

<pre><code>python train_subject_specific.py --gan_type {dcgan (default), wgan_gp, vae}
</code></pre>

<pre><code>python train_subject_independent.py --gan_type {dcgan (default), wgan_gp, vae}
</code></pre>

<pre><code>python train_gan_test.py --gan_type {dcgan (default), wgan_gp, vae}
</code></pre>

<pre><code>python train_subject_specific.py --train_type {subject-specific, subject-independent}
</code></pre>
