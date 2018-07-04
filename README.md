# Multi-Domain Generative Adversarial Networks (MDGAN)
## The related paper was submitted to International Joint Conference on Artificial Intelligence, 2018.

## Code will be released after the paper is accepted.

## Introduction

  When you are looking at a red T-shirt in eBay, you could easily imagine how you would look like when you wear it: you know well the shape of your body, you have an image of this red T-shirt in your mind, and thus you can wear it in your imaginary world. However, can a learning machine do a job like this? This means that the machine should have the ability to learn from different domains (e.g., people and T-shirts) and extract some specific knowledge from them, respectively (e.g., people’s body shapes and T-shirts’ color style). Then, it is expected to join the specific kinds of knowledge and thereby generate a brand new domain (e.g., imagination on wearing the T-shirt).  In realistic applications, this allows more flexible generation strategies than the conventional generation problem where only one source domain exists and the generated domain is expected to be the same as the source. For instance, as illustrated in the right part in Figure 1, a machine learns to generate images in a new domain (i.e., colorful handbags) with shape style of one domain (black-white handbags) and color style of another domain (colorful shoes) By this way, it helps bring new ideas and provides visualizations for the handbag designers who only have some raw ideas about designing handbags of new color styles.
 ![intro](https://github.com/GuangyuanHao/MDGAN/raw/master/intro.jpg) 
  Unfortunately, as illustrated in the left part in figure above, existing GAN-based methods, e.g., GAN, CoGAN, ACGAN and BEGAN, are restricted to generate new samples similar to the ones from source domains. The style transfer and image-to-image translation models, e.g., CycleGAN, pix2pix, and UNIT, can transfer a single image or samples in one domain to an image with style similar to another image or samples in another domain. However, these models are restricted in that they require certain input images as subjects to be translated, and thus cannot deal with the problems requiring going beyond the available subjects (e.g., designing new shape styles of handbags in the above situation). Therefore, the problem of learning from different domains for generating a new domain still remains an open issue.
  
  To explicitly learn different types of knowledge from different domains, in this work we focus on learning global structural information(e.g., shape of an object like a handbags) from one domain, and learning local structural information (e.g., color style another object like a shoes) from another. Learning global information is straightforward: a simple auto-encoder structure can be leveraged to capture and encode the global information, since the auto-encoder focuses on the whole images. As for learning local structural information, we propose to learn from small patches of the source images. By this way, our model has the ability to additionally focus on the specific local patterns that appear in patches, and thus can also be effective in capturing the local structural information.
  
  We evaluate our model on several tasks. The experimental results show that our model can learn to generate images in a new domain, e.g.,  generating digits whose shape style and color style are learned from two different domains, i.e., the MNIST dataset and the SVHN dataset. The main contributions are as follows:
  
  1.	We propose an unsupervised method to absorb different concepts, i.e., global structural information and local structural information, from different domains.  
  2.	We build a model to learn a distribution to describe a new domain with global structural information of one domain and local structural information of another domain by focusing on learning global and local structural information at the scale of a whole image from one domain and at the scale of small patches from another domain respectively.
  3.	We show our model can successfully learn to generate mixed-style samples with shape style of one dataset and color style of another dataset in several tasks.
  
## Results
  ![digit](https://github.com/GuangyuanHao/MDGAN/raw/master/Results/digit.jpg) 
  ![bag](https://github.com/GuangyuanHao/MDGAN/raw/master/Results/bag.jpg)
## Other Applications
### Learning a Joint Distribution of Two Domains
  Our model can also learn a Joint Distribution of Two Domains successfully as CoGAN did, when two datasets have similar global structural information.
  #### Results
   ![joint1](https://github.com/GuangyuanHao/MDGAN/raw/master/Results/joint1.jpg) 
   ![joint2](https://github.com/GuangyuanHao/MDGAN/raw/master/Results/joint2.jpg) 
### Image-to-Image-Translation
  Our model can also accomplish image-to-image translation tasks successfully as UNIT did, when two datasets have similar global structural information.
  #### Results
   ![tran3](https://github.com/GuangyuanHao/MDGAN/raw/master/Results/tran3.jpg) 
   ![tran4](https://github.com/GuangyuanHao/MDGAN/raw/master/Results/tran4.jpg) 
