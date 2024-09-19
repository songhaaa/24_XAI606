# 242R 신경망응용및실습(Applications and Practice in Neural Networks)

Applications and Practice in Neural Networks, Department of Artificial Intelligence, Korea University </br>
(고려대학교 인공지능학과 신경망응용및실습 프로젝트)

### Google Drive Link

<a href="https://drive.google.com/drive/folders/1S1xckDX1waQaXRlF7Ka20ZTUCO5TWlRT?usp=sharing">
  <img src="https://img.shields.io/badge/Google Drive-4285F4?style=flat-square&logo=googledrive&logoColor=white"/>
</a>

### I. Project title
- Cassava Leaf Disease Classification: https://www.kaggle.com/competitions/cassava-leaf-disease-classification
### II. Project introduction
- As the second-largest provider of carbohydrates in Africa, cassava is a key food security crop grown by smallholder farmers because it can withstand harsh conditions. At least 80% of household farms in Sub-Saharan Africa grow this starchy root, but viral diseases are major sources of poor yields. With the help of data science, it may be possible to identify common diseases so they can be treated. </br></br>Existing methods of disease detection require farmers to solicit the help of government-funded agricultural experts to visually inspect and diagnose the plants. This suffers from being labor-intensive, low-supply and costly. As an added challenge, effective solutions for farmers must perform well under significant constraints, since African farmers may only have access to mobile-quality cameras with low-bandwidth.</br></br>In this competition, we introduce a dataset of 21,367 labeled images collected during a regular survey in Uganda. Most images were crowdsourced from farmers taking photos of their gardens, and annotated by experts at the National Crops Resources Research Institute (NaCRRI) in collaboration with the AI lab at Makerere University, Kampala. This is in a format that most realistically represents what farmers would need to diagnose in real life.</br></br><b><i>Your task is to classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.</i></b> With your help, farmers may be able to quickly identify diseased plants, potentially saving their crops before they inflict irreparable damage.

- class</br>"0":string"Cassava Bacterial Blight (CBB)"</br>"1":string"Cassava Brown Streak Disease (CBSD)"</br>"2":string"Cassava Green Mottle (CGM)"</br>"3":string"Cassava Mosaic Disease (CMD)"</br>"4":string"Healthy"
### III. Dataset description (need details)
- Can you identify a problem with a cassava plant using a photo from a relatively inexpensive camera? This competition will challenge you to distinguish between several diseases that cause material harm to the food supply of many African countries. In some cases the main remedy is to burn the infected plants to prevent further spread, which can make a rapid automated turnaround quite useful to the farmers.

#### Files
- [train/test]_images the image files. The full set of test images will only be available to your notebook when it is submitted for scoring. Expect to see roughly 15,000 images in the test set.

- train.csv

  - image_id the image file name.
  - label the ID code for the disease.

- sample_submission.csv A properly formatted sample submission, given the disclosed test set content.
  - image_id the image file name.
  - label the predicted ID code for the disease.

- [train/test]_tfrecords the image files in tfrecord format.

- label_num_to_disease_map.json The mapping between each disease code and the real disease name.
