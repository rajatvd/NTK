# Neural Tangent Kernel

This repo contains code for my blog post on [Understanding the Neural Tangent Kernel](https://rajatvd.github.io/NTK).

## Notebooks

* `ntk.ipynb` has code for generating the gifs and some other visualizations not included in the post.
* `one_layer_asymptotics.ipynb` empirically verifies the asymptotics for the gradient and Hessian derived in the blog post. Play with it in colab here: [colab link](https://colab.research.google.com/drive/1BsqUbINgaEHotlDcYZVVnWnDn_Bk86zk).


## Scripts

* `utils.py` - has some helper functions to create models, run gradient descent, and calculate NTKs.
* `train.py` - a [sacred](https://github.com/IDSIA/sacred) script which trains a simple fully connected network on 1-D data points.
* `grid_run.py` - run a grid of sacred scripts with different configs. For example, I used this to create the runs with different alphas in ellipse gif.
