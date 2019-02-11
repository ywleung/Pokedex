Ultimate Goal: to classify first generation of Pokemon merchandises after only training on images of cartoon images using CNN

Workflow:
1) downloaded 151 first generation of Pokemon from Bing using image_scrape.py with Microsoft Azure free trial account
2) deleted irrelevent images manually and preprocessed images using image_preprocessing.ipynb
3) splited the cartoon images into train, dev, test set
4) trained CNN model and performed error analysis to find out the best model

PS. The images and trained models are not uploaded due to their size.

The best model achieves 0.75 F1-score on test set.
Cartoon images are used to be test set because of lack of merchandise images.

Problems:
1) small datasets for unpopular Pokemons (only 90 images for Marowak, 100 images for Goldeen etc.)
2) intraclass variation (merchandises have different shapes, colours, textures within the same class)
3) scale invariance
4) background colour (colour play an important role in this classification)

To-do list:
1) try to collect more images including cartoon and merchandises)
2) train a model to propose location of Pokemon in images
