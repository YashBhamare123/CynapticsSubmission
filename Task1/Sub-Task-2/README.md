### Approach
Progressive GANs are used here. The model is trained progressively by generating and then discriminating the generated images of size 4x4, then 8x8, 16x16 etc. The previous layer generated images are faded into the new layer and gradually the constant for fading is decreased. This is to avoid the load for generation on higher layers. 32x32 images are generated due to time limitation(last epoch(64x64) est.time 10hrs). CelebA dataset is used for training. Wasserstien loss is used along with gradient penalty to enforce the One-lipschwitz-Continuity. Each layer is only trained for one epoch and still the model manages to produce great results which is quite remarkable.
### Images
https://imgur.com/a/nTBC8el
