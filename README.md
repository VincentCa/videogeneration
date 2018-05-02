# Videogeneration

### About
In this project, we present a variety of deep learning based setups for text to image and text to image sequence generation. Image sequence generation is a challenging task and an actively researched branch within computer vision, posing special challenges such as temporal coherency. To address this problem, we describe a variety of partial and complete solutions that we developed in three stages: (1) Text to Image Synthesis: using a traditional GAN, we generate a single image from a textual representation. (2) Text + Image to Video Synthesis: using a fully convolutional network, we generate an image sequence given a single frame and a description of the action taking place. Inspired by the recent success of generative adversarial networks, we then also train this architecture in a truly adversarial setting. 

We perform experiments on our synthetic icon and MNIST datasets, as well as the KTH Human Action Dataset for walking. To train our network, we feed in the textual annotation, the current frame and up to 9 preceding frames, if available. The network is then trained to output the next frame. During inference, we just provide an initial frame and the textual description to predict the new frame, and then repeatedly feed in the previous predictions to predict further into the future. In the result section below, we execute this prediction cycle for between five and ten times, but it can be followed an indefinitely number of times.

### Network Architecture 
Inspired by the success of U-Net, an encoder-decoder architecture with skip-connections proposed for biomedical segmentation, we adapt a similar base structure for our problem. We extend U-Net to operate on 3D-volumes instead of single images as input. A 3D-volume here represents previous frames in the GIF sequence. We then incorporate the text captions by embedding a network-in-network: the encoder first transforms the input volume into a compact latent representation. We flatten the resulting volume and feed it into a series of densely connected layers. We also feed in a caption representation at this point. The output of this “network-in-network” is reshaped to a 3D-volume, and subsequently upsampled by the decoder. Our output is a single image which represents the next frame.

<p align="center"><img src="images/4.png" width="700"></p>

We trained this setup on our synthetic symbol dataset, our synthetic MNIST dataset, and the activity dataset. We reached convergence within a few minutes, but found it to be prone to overfitting, which we resolved through cutting down the number of parameters significantly, and improve network regularization. For our experiments, we set h = w = 64 for synthetic data and h = w = 128 for the activity data. We always choose d = 10, which is a reasonable amount of history available for the network. Our sentences are simply represented as one-hot encoded matrices. We show some results below.

<p align="center"><img src="images/1.png"></p>

<p align="center"><img src="images/2.png"></p>

<p align="center"><img src="images/3.png"></p>
