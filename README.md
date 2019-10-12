# Flower Recognition - Deep Learning
Link: https://www.hackerearth.com/problem/machine-learning/flower-recognition/

For several years, flower recognition in the wildlife has been an area of great interest among biologists. Recognition of flower in environments such as forests and mountains is necessary to know whether they are extinct or not. While search engines assist in searching for a flower, it lacks robustness because of the intra-class variation among millions of flower species.

The application of deep learning is rapidly growing in the field of computer vision and is helping in building powerful classification and identification models. We can leverage this power of deep learning to build models that can classify and differentiate between different species of flower as well.

We are given a large class of flowers, 102 to be precise. Build a flower classification model which is discriminative between classes but can correctly classify all flower images belonging to the same class. There are a total of 20549 (train + test) images of flowers. Predict the category of the flowers present in the test folder with good accuracy.

The data folder consists of 2 folders and 3 CSV files

* train - Contains 18540 images from 102 categories of flowers
* test  - Contains 2009 images
* train.csv - Contains 2 columns and 18541 rows (including the headers), which consists of image id and the true label for each of the images in the train folder
* test.csv - Contains the image id for the images present in test folder for which the true label needs to be predicted
* sample_submission.csv - Specifies the format for the submission file

**Data Description:**
The image dataset is to be categorized into 102 classes. The names of the categories are as follows in no particular order. 

* Alpine sea holly
* Anthurium
* Artichoke
* Azalea
* Ball Moss
* Balloon Flower
* Barbeton Daisy
* Bearded Iris
* Bee Balm
* Bird of paradise
* Bishop of llandaff
* Blackberry Lily
* Black-eyed Susan
* Blanket flower
* Bolero deep blue
* Bougainvillea
* Bromelia
* Buttercup
* Californian Poppy
* Camellia
* Canna Lily
* Canterbury Bells
* Cape Flower
* Carnation
* Cautleya Spicata
* Clematis
* Colt's Foot
* Columbine
* Common Dandelion
* Corn poppy
* Cyclamen
* Daffodil
* Desert-rose
* English Marigold
* Fire Lily
* Foxglove
* Frangipani
* Fritillary
* Garden Phlox
* Gaura
* Gazania
* Geranium
* Giant white arum lily
* Globe Thistle
* Globe-flower
* Grape Hyacinth
* Great Masterwort
* Hard-leaved pocket orchid
* Hibiscus
* Hippeastrum
* Japanese Anemone
* King Protea
* Lenten Rose    
* Lotus
* Love in the mist
* Magnolia
* Mallow
* Marigold
* Mexican Aster
* Mexican Petunia
* Monkshood
* Moon Orchid
* Morning Glory
* Orange Dahlia
* Osteospermum
* Oxeye Daisy
* Passion Flower
* Pelargonium
* Peruvian Lily
* Petunia
* Pincushion flower
* Pink Primrose
* Pink-yellow Dahlia
* Poinsettia
* Primula
* Prince of wales feathers
* Purple Coneflower
* Red Ginger
* Rose
* Ruby-lipped Cattleya
* Siam Tulip
* Silverbush
* Snapdragon
* Spear Thistle
* Spring Crocus
* Stemless Gentian
* Sunflower
* Sweet pea
* Sweet William
* Sword Lily
* Thorn Apple
* Tiger Lily
* Toad Lily
* Tree Mallow
* Tree Poppy
* Trumpet Creeper
* Wallflower
* Water Lily
* Watercress
* Wild Pansy
* Windflower
* Yellow Iris  

Link to dataset:[https://he-public-data.s3-ap-southeast-1.amazonaws.com/HE_Challenge_data.zip](https://he-public-data.s3-ap-southeast-1.amazonaws.com/HE_Challenge_data.zip "Named link title")
