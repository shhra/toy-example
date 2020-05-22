### Emotion Detection 
A emotion recognition project assigned at training. This is NLP problem where emotion should classify through a text data. 

### Project Structure

	
		├── README.md          <- README file.
		├── app                <- APIs to interact with the inference model.
		│   ├── __init__.py
		│   |__ serve.py
		|   |__ settings.py
		|              
		├── ml		     <- Source code for the use in this project.
		│   ├── __init__.py
		│   │
		│   ├── src	          <- Scripts to ML model
		│   │   └── __init__.py	 
		│   │   └── models.py
		|   |   |___process.py
		|   |   |___train.py
		|   |   |___utils.py	<- Collection of various utility functions.
		|   |
		|   |
		│__notebooks      		<- Collection of notebooks .
		│   │   └── 05_SM_V0.ipynb	<- For Emotion data
		|   |   |__ Car_Data_EDA.ipynb	 <- For Car Ads data
		│   │
		|   |
		├── saved_models      <- Scripts to process the data.
		│   |  └── encoder.sav
		|   |  |__ sentence_classifier.sav
		│   │  |__ vectorizer.sav
		│   |
		|   |
		│   ├── docker-compose.yml     <-docker file   
		|   |   
		|__ instance
		|      |__config
		|           |___ config.json        <-Database Configuration file
		|      
		│── flask      
		│── requirements.txt          <-Pip generated requirements file for the project.
		│── wsgi.py     <- flask end point added

### Getting Started

##### Installation

**Cloning the repository For code**:

		#Get the code 		
		git clone https://github.com/shhra/toy-example.git

**Create the Virtual Environment to avoid any conflicts**:

		#Creating Virtual Env
		virtualenv -p python3 .venv
		#Activating virtual env
		source .venv/bin/activate

**Install Dependencies**:

		pip install -r requirement.txt 

## OR by using docker
Docker file is also added for reproducibility

**Build docker Image**:

		sudo docker-compose up --build

**End-points**

 * `/` - initial API page
 * `/classify` - [GET] Type input sentence for classification
* `/classify` - [POST] Renders the predicted emotion for the sentence


Note: To see the result you have to run the url in postman. For postman you can either download in your local machine and paste the url to view result or simply you can browse postman in your browser.[Highly recommend -Chrome:] 

Also, You can tweak the `DB_NAME` in  the config file. Please keep the config file in order directory as shown above. i.e 
			
			instance/config/config.json
			
**Enjoy the Project !!**


