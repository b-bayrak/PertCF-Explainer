# PertCF-Explainer

This repository provides reproducible benchmarking experiments of '_**PertCF: A Perturbation-Based Counterfactual Generation Approach**_' paper and open-source implementation of the proposed PertCF method.

Todo: Add method descr. and image

Todo: Add the paper and citation info

## Instructions to run the contribution
**Step 1:** Clone the repository

**Step 2:** Build myCBR API

This project requires to run the myCBR Rest API. Therefore you will need to have Java 8 JRE on your computer and download the following jar: https://folk.idi.ntnu.no/kerstinb/mycbr/mycbr-rest/mycbr-rest-2.0.jar
Make sure to add the jar file to the myCBR folder.

This jar is built using this repository https://github.com/ntnu-ai-lab/mycbr-rest

**Step 3:** Starting the API 

We assume that you have cloned the GitHub repository and added the mycbr-rest-2.0.jar into the mycbr folder of this repository. Then in the terminal, navigate to the cloned project folder. 

Then go into the mycbr folder and run the following command:
```
java -DMYCBR.PROJECT.FILE=PROJECT_FILE.prj -jar mycbr-rest-2.0.jar
```
This command will start a local webserver, deploy the myCBR project and expose it's endpoints via a Rest API. We assume that this server is running in the background while the experiments notebooks are run.

**Step 4:** Accesing API

Once the installation is done and the API is running, the API will be accessibleat http://localhost:8080/swagger-ui.html#. To verify that the API is up and running, please check that you can see the swagger documentation page for the API.

![image](https://user-images.githubusercontent.com/22470440/186938749-544d7a95-c8dc-4b6c-be60-62d1de45b03b.png)


**Step 5:** Run the experiments

**Necessary files to run the experiments**

To run the experiments we provide 'SouhtGermanCredit' and 'UserKnowledgeModeling' folders with a number of files under the 'Experiments' folder. 
The folders consist of experiments and the following folders to reproduce the experiments:

- Data/: contains .csv file/s needed for the experiments. 

- Model/: contains the .pkl files for the BB models used in the experiments.

- Output/: contains result files of the experiments.

- temp_files/: contains all other files that are generated and/or used during experimenting.

'SouhtGermanCredit' and 'UserKnowledgeModeling' folders also include reproducible and adaptable ordered set of experiments as ipython notebooks. 

This repository also includes followings:

- myCBR_projects/: contains myCBR project files that are modelled for the experiments. 

- mycbr_py_api.py: contains myCBR-rest API connection and calls

- PertCF.py: contains Explainer Class that implements PertCF including experimental setup