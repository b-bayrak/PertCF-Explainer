# PertCF-Explainer

This repository provides reproducible benchmarking experiments of '_**PertCF: A Perturbation-Based Counterfactual Generation Approach**_' paper and open-source implementation of the proposed PertCF method.


## Instructions to run the contribution
**Step 1:** clone the repository

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


### Running the experiements

#### Necessary files to run the Experiments

[//]: # ()
[//]: # (To run our experiments we provide 'WineQuality' and 'Depression' folders with a number of files. The folders consists of experiments and the following folders to reproduce the experiments:)

[//]: # (- project/: contains project files that are modelled for the experiments. )

[//]: # (- data/: contains the train and test .csv files needed for the experiments. Those are created based on the given datasets. They are the results after we conducted the first set of experiments to find a good split of the provided data. )

[//]: # (- model/: contains the .pkl files for the BB models used in the experiments.)

[//]: # (- shap/: contains the shap values per class .csv files needed to create the global similarity functions of the CBR agents.)
