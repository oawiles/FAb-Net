This is the code for [Self-supervised learning of a facial attribute embedding from video](http://www.robots.ox.ac.uk/~vgg/research/unsup_learn_watch_faces/index.html) in BMVC 2018.

Note that this is a refactored version of the original code, so the numbers resulting from this may not be exactly those given in the paper.
More importantly, this code was run using a version of pytorch compiled from source, so using a standard pytorch may be 
- difficult to load the models and
- give slightly different results (especially as the implementation of the sampler seems to have slightly changed between versions).


**Running demo code**

`FAb-Net/code/demo.ipynb` gives the demo code: i.e. how to load a model and predict various properties from it using a trained model and subsequently trained linear layer as described in the paper.
It is self-contained.
For these regressions, one file stores the original model parameters plus the linear layers. You can try on your own images or train your own linear regressor.

To run the demo code:
- Make sure you satisfy the requirements in requirements.txt
- Download the models from the [project page](http://www.robots.ox.ac.uk/~vgg/research/unsup_learn_watch_faces/fabnet.html).
- Update the model paths in the notebook accordingly

**Training yourself**

The training code is given in `FAb-Net/code/train_attention_curriculum.py`. 

In order to use this training code, it is necessary to download a dataset (e.g. [VoxCeleb1/2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)).
They should then be put into folders as follows and the environment variables in Datasets/config.sh updated appropriately (VOX_CELEB_1 is VoxCeleb1, VOX_CELEB_LOCATION VoxCeleb2).

For our datasets we organised the directories as:

```
IDENTITY
-- VIDEO
-- -- TRACK
-- -- -- frame0001.jpg
-- -- -- frame0002.jpg
-- -- -- ...
-- -- -- frameXXXX.jpg
```


If you arrange the folders/files as illustrated above, then you can generate np split files using `Datasets/generate_large_voxceleb.py` and use our dataloader.
Otherwise, you may have to write your own.

Then you need to update where the model/runs are stored to by setting BASE_LOCATION in config.sh.
Once this has all been done, you can train with: `python train_attention_curriculum.py` and point tensorboard to BASE_LOCATION/code_faces/runs/ to see how training is getting on.
