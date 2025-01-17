# Patch-Craft
### Basis
The Basis of this algorithm comes from the fact that most diffusion models do not generate even noise across the image. The fore-ground has more noise/detail than the background making the image seem AI Generated. This technique aims to create two images from the original image. One with the low texture patches and the other with high texture patches. Using pre-processing and high-pass filters the texture detail is isolated and the difference between the two images is then passed through a light-weight CNN Classifier which can then distinguish the AI Generated Images. This is approach is more efficient and highly versatile because it does not depend on feature extraction and looking for anomalies in the images but rather exploits the way that images are generated. 
Research-Paper: [https://arxiv.org/abs/2311.12397]
### Approach
The objective of Patch-Crafting is to divide the original image into certain number of small images, classify each patch/small_image as High-Texture or Low-Texture, create the a grid of such poor/rich texture images and then feed it to the classifier
### Implementation
I am not satisfied with the implementation due to the lack of time management on my part. The accuracy on validation is 72% on 10 epochs after which the accuracy degrades. The problem is most likely the dataset. A better approach would be to train this model on a larger dataset like the AiArtDataset
and then freeze all the weights except for the final layer which can be used to fine-tune the model to this dataset
This code was written in one-day and thus is not optimized, but I have complete faith in this approach as a way to classify AI images. Given more time I will completely optimize this code and un-officially share the results of this approach as a proof of concept.
