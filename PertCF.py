"""
    Explainer Class implements PertCF including experimental setup
"""
from mycbr_py_api import MyCBRRestApi as mycbr
from statistics import mean
import pandas as pd
import numpy as np
import time as t
import requests
import shap
import json

class Explainer:
    def __init__(self, model, X_train, X_test, y_train, y_test, label, categoric_features, concept, shap_df=False, maps={}, inv_maps={}, num_iter=10):
        
        self.model = model
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.num_iter = num_iter
        self.concept = concept
        self.label = label
        
        self.train = self.X_train.copy()
        self.train[self.label] = y_train
        
        self.test = self.X_test.copy()
        self.test[self.label] = y_test
        
        self.cat_col_names = categoric_features
        self.maps = maps
        self.inv_maps = inv_maps
        
        
        # Take ordered column and class names from model for shap values 
        self.col_names = model.feature_names_in_
        self.col_names_l = self.col_names.tolist() + [label]
        self.class_names = model.classes_.astype('str')
        self.num_class = len(self.class_names)

        
        if type(shap_df) == bool:
            self._shapValues()
            print('Shap values calculated and used.\n')
        else:
            self.shap_df = shap_df
            print('Given Shap values used.\n')

        
        self._APIConnection()
        print('API connection completed.\n')

        concepts = self.obj.getAllConcepts()
        print('Concept names: ', concepts, ' Is correct: ',(concepts[0]==self.concept) , '\n')
        
        
        self._addCasebases()
        print('Casebase names: ', self.getCasebases(), '\n')
        
        # Delete all cases
        self._deleteInstancesFromConcept()
        
        self._importCases()
        print('\nCases imported to the casebases. \n')
        
        self._setAmalgamationFunctions()
        print('Amalgamation functions created for every class using shap values. \n')


    def _APIConnection(self, server: str = 'localhost', port: str = '8080'):
        """
                Connects to API server

                Parameters
                ----------
                    :param server : IP (Default: 'localhost')
                    :param port : Port number (Default: '8080')
        """
        self.server = server
        self.port = port
        
        self.base_url = 'http://' + self.server + ':' + self.port + '/'
        self.headers = {'Content-type':'application/json'}
        self.obj = mycbr(self.base_url)


    def _shapValues(self, vis: bool = False):
        """
                Calculates SHAP value for each class

                Parameters
                ----------
                    :param vis : visualize the SHAP values with a bar plot (Default: False)
        """
        if len(self.X_train)>300:
            df = self.X_train.sample(300)
        else:
            df = self.X_train.copy()
        
        for column in df.columns:
            if df[column].dtype not in ['int64']:
                df[column] = df[column].astype('category').cat.codes

        # Create shap kernel explainer using model and training data
        explainer = shap.KernelExplainer(self.model.predict_proba, df)

        # Shap values calculated by explainer
        self.shap_values = explainer.shap_values(df)

        # Bar plot of calculated shap values (Each color implies a class)
        if vis:
            shap.summary_plot(shap_values = self.shap_values,features = self.X_train, 
                              feature_names=self.col_names,  plot_type='bar', 
                              class_names=self.class_names) 

        # Create df from mean of shap values (map the order of features and classes)
        mean_classes = []
        for i in range(len(self.shap_values)):
            mean_classes.append(np.mean(np.abs(self.shap_values[i]), axis=0))

        self.shap_df = pd.DataFrame(mean_classes, index=self.class_names, 
                                    columns=self.X_train.columns.tolist())


        # Wheight of label = 0 
        self.shap_df[self.label] = np.zeros(self.num_class)
        
        # normalize shap values
        self.shap_df =  self.shap_df.div(self.shap_df.sum(axis=1), axis=0)           


    
    # "_importCases": Imports cases from train set to the proper casebase

    def _importCases(self):
        """
                Group cases by classes and add to casebases
        """
        for i in range(self.num_class):
            self.addCasesFromDf(self.train.loc[self.train[self.label] == str(self.class_names[i])], 'cb_class'+str(i))    

 
    # "_setAmalgamationFunctions": Sets amalgamation functions from 'shap_df'

    def _setAmalgamationFunctions(self):
        """
                Add an amalgamation function for every class use shap values as weights
        """
        for i in range(self.num_class):
            self.newAmalgamationFunc('amal_func_class'+str(i), 'WEIGHTED_SUM', str(self.shap_df.iloc[i].to_dict()).replace("'",'"'))

    def _deleteInstancesFromConcept(self):
        """
                Delete all cases from 'concept'
        """
        response = requests.delete(self.base_url+'concepts/'+self.concept+'/cases')
        print("Cases deleted from concept '" + self.concept + "': " + str(response.ok))

    def _addCasebases(self):
        """
                Add casebases to the project in 'cb_classX' format
        """
        for i in range(self.num_class):
            cb_name = 'cb_class' + str(i)
            if [cb_name] not in self.getCasebases():
                res = self._putNewCb(cb_name)
                if res:
                    print('cb_class' + str(i) + ' added:', res)
    

    def _putNewCb(self, newCbName: str):
        """
                Add a new casebase with newCbName name

                Parameters
                ----------
                :param newCbName : Casebase name
        """
        res = requests.put(self.base_url + 'casebases/' + newCbName)
        return res.ok


    def getCasebases(self):
        """
                Retrieves the list of casebases

                Returns
                -------
                List: Casebase names
        """
        raw = pd.DataFrame(requests.get(self.base_url + 'casebases/').json())
        casebases = pd.DataFrame.from_records(raw).values.tolist()
        return casebases

                       
    def _addRowAsCase(self, x: pd.DataFrame, concept: str, cb: str):
        """
                Add a row (of df) to casebase 'cb'

                Parameters
                ----------
                :param x: a slice of dataframe
                :param concept: concept name
                :param cb: casebase name
        """
        case_id = 'case_' + str(x['index'])
        x = x.drop(['index'])
        requests.post(self.base_url + 'concepts/' + concept + '/casebases/' 
                      + cb +'/cases/' + case_id, data = str(x.to_json()), 
                      headers=self.headers)

    
    def addCasesFromDf(self, df: pd.DataFrame, cb: str):
        """
                Add cases to casebase 'cb' from dataframe 'df'

                Parameters
                ----------
                :param df: a dataframe
                :param cb: casebase name
        """
        tmp = df.copy(deep=True)
        tmp.reset_index(inplace=True)
        tmp.apply(self._addRowAsCase, args=(self.concept, cb), axis=1)
    
    def newAmalgamationFunc(self, amalFuncID: str, amalFuncType: str, json: str):
        """
                Add new amalgamation function

                Parameters
                ----------
                :param amalFuncID: amalgamation function name
                :param amalFuncType: amalgamation function type
                :param json:  to_json(dictionary of weights)
        """
        return requests.put(self.base_url + 'concepts/' + self.concept 
                            + '/amalgamationFunctions?amalgamationFunctionID=' + amalFuncID 
                            + '&amalgamationFunctionType=' + amalFuncType 
                            + '&attributeWeightsJSON=' + json)

                       

    def querySimilarCasesFromCb(self, casebase: str, amalgamationFct: str, queryJSON: str, k: int = -1):
        """
                Retrieve similar cases to 'queryJSON' from 'casebase' using 'amalgamationFct'
        """
        raw = requests.post(self.base_url + 'concepts/' + self.concept 
                            + '/casebases/' + casebase 
                            + '/amalgamationFunctions/' + amalgamationFct 
                            +'/retrievalByMultipleAttributes?k=' + str(k), 
                            data=str(queryJSON),
                            headers = self.headers)
        results = pd.DataFrame.from_dict(raw.json())
        results = results.apply(pd.to_numeric, errors='coerce').fillna(results).sort_values(by='similarity', ascending=False)
        return results



    def querySimilarCases(self, query: pd.DataFrame, k: int):
        """
                Return a dataframe of similar cases from all casebases
        """
        res = pd.DataFrame()
        for i in range(self.num_class):
            temp = self.querySimilarCasesFromCb('cb_class'+str(i), 'amal_func_class'+str(i), query.to_json(), k)
            res = pd.concat([temp, res])

        return res
                       
    def queryNN(self, query: pd.DataFrame, k: int):
        """
                Retrieve Nearest Neighbours from all casebases
        """
        res = self.querySimilarCases(query, k)
        return (res[res.similarity == res.similarity.max()]) 
    
    def _findSimilarity(self,p1: pd.Series, p2: pd.Series):
        """
                Calculate similarity between p1 and p2
        """

        # Create a CB for testing
        if ['cb_temp'] not in self.getCasebases():
            self._putNewCb('cb_temp')

        # Add testCase_1 and testCase_2
        case_id1 = 'testCase_1'
        case_id2 = 'testCase_2' 
    
        # add 'case_id1' to 'cb_temp'
        requests.post(self.base_url + 'concepts/' + self.concept 
                      + '/casebases/' + 'cb_temp' 
                      +'/cases/' + case_id1, 
                      data = str(p1.to_json()), 
                      headers=self.headers)

        # add 'case_id2' to 'cb_temp'
        requests.post(self.base_url + 'concepts/' + self.concept 
                      + '/casebases/' + 'cb_temp' 
                      +'/cases/' + case_id2, 
                      data = str(p2.to_json()), 
                      headers=self.headers)

        # add 'case_id1' to 'cb_temp'
        requests.put(self.base_url + 'concepts/' + self.concept 
                      + '/casebases/' + 'cb_temp' 
                      +'/cases/' + case_id1, 
                      data = str(p1.to_json()), 
                      headers=self.headers)

        # add 'case_id2' to 'cb_temp'
        requests.put(self.base_url + 'concepts/' + self.concept 
                      + '/casebases/' + 'cb_temp' 
                      +'/cases/' + case_id2, 
                      data = str(p2.to_json()), 
                      headers=self.headers)

        raw = requests.get(self.base_url + 'analytics/' 
                           + 'concepts/' + self.concept 
                           + '/amalgamationFunctions/{amalgamationFunctionID}' 
                           + '/localSimComparison?amalgamationFunctionID=' 
                           + 'amal_func_class' + str(self.exp_class) 
                           + '&caseID_1=' + case_id1 
                           + '&caseID_2=' + case_id2, 
                           headers=self.headers)
        
        raw = raw.text[1:-1].replace('},{',',')
        return json.loads(raw)

    def _findDist(self,p1: pd.Series, p2: pd.Series):
        """
                Calculate the distance between p1 and p2
        """
        res = self._findSimilarity(p1,p2)
        res.pop(self.label)
        res = mean(res.values())
        return 1 - res
    
                    
    def findNUNs(self, query: pd.DataFrame, unwanted: list):
        """
                Return the Nearest Unlike Neighbours (NUNs) from
                each class (except query_class) as a DF
        """
        k = 1 # retrieve the nearest one
        res = pd.DataFrame()

        # From each class retrieve NUNs
        for i in range(self.num_class):
            if str(i) not in unwanted:
                nun = self.querySimilarCasesFromCb('cb_class' + str(i), 'amal_func_class' + str(i), query.to_json(), k)
                if str(nun[self.label][0]) != str(query[self.label]):
                    res = pd.concat([nun,res])

        return res

    
    def findNUN(self, query: pd.DataFrame, unwanted: list = []):
        """
                Return the Nearest Unlike Neighbour (NUN)
        """
        res = self.findNUNs(query, unwanted)
        if len(res):
            return res[res.similarity == res.similarity.max()]
        else:
            return res

       


    def generateCF(self, p1: list, p1_class: str, p2: list, p2_class: str, thresh: float, cnt: int = 0):

        """
                Generates a CF between 'p1' and 'p2'
                'p1', 'p2' (list): samples
                'p1_class', 'p2_class' (str/categoric): Class labels for 'p1' and 'p2'
                'thresh' (float): To terminate the iterations min distance diff between old and new candidate
                'cnt' (int): iteration counter for recursion
                - A recursive function
        """
        # from categoric features to numeric features with maps
        # from numeric features to categoric features with inv_maps
        def _encode(df, maps):
            for i in self.cat_col_names:
                df[i] = df[i].astype('string').map(maps[i])
            return df
        
        # If this is the first iteration initialize necessary variables
        if cnt == 0:
            self.nun = p2.copy()
            self.nun[self.label] = p2_class
            self.q_class = p1_class
            self.old_cnd = pd.DataFrame([], columns = self.col_names)
            self.cnd_list = pd.DataFrame([], columns = self.col_names)
            self.exp_class = p2_class # NUN class
            self.used_classes = [self.q_class, self.exp_class] 
            # Normalize the weights
            self.weights_norm  = self.shap_df.loc[str(self.exp_class)].to_list()[:-1]

            
        
        # If iteration value is (-1) it means in the first search process a CF couldnt be generated,
        # re-set necessary variables and start a CF generation process to generate CF from last candidates class.    
        if cnt == -1:
            self.old_cnd = pd.DataFrame([], columns = self.col_names)
            self.cnd_list = pd.DataFrame([], columns = self.col_names)
            self.exp_class = p2_class #TODO: Change it with p1
            self.used_classes.append(p1_class)
            self.used_classes.append(p2_class)
            
            # Normalize the weights
            self.weights_norm = self.shap_df.loc[str(self.exp_class)].to_list()[:-1]
            cnt = 0
        
        diff_dict = self._findSimilarity(p1,p2)    
        candidate = []
        i = 0
        for f in self.col_names:
            if f in self.cat_col_names:
                if (1 - diff_dict[f]) > 0.5:
                    cnd_i = p2[f]
                else:
                    cnd_i = p1[f]
                
            else:
                diff = p2[f]-p1[f]
                cnd_i = p1[f] + self.weights_norm[i] * diff 
            
            candidate.append(cnd_i)
            i += 1 
        
        candidate_df = pd.DataFrame([candidate], columns = self.col_names)
        
        candidate_inv = _encode(candidate_df.copy(), self.maps).iloc[0]

        # Class of the generated candidate
        cnd_class =  self.model.predict([candidate_inv.to_list()])[0]

        # If 'candidate' is from expected class 
        if cnd_class == self.exp_class:
            # add 'candidate' to 'cnd_list'
            cf = candidate_df.copy().iloc[0]
            cf[self.label]  = cnd_class
            self.cnd_list = self.cnd_list.append(cf)
            
        if cnt != self.num_iter:
            # If 'candidate' is from expected class 
            if cnd_class == self.exp_class:

                # If the distance between previous cand. and 'candidate' (step size) smaller then threshold
                if len(self.old_cnd) > 0: #not self.old_cnd.empty():
                    if self._findDist(self.old_cnd.iloc[0],candidate_df.iloc[0]) <= thresh:
                        #print('STEPSIZE: ', self._findDist(self.old_cnd.iloc[0],candidate_df.iloc[0]), 'THRSH:', thresh)
                        #print('-- If candidate is from expected class ve (step size) smaller then threshold --')
                        # Return 'candidate' as CF
                        cf = candidate_df.copy().iloc[0]
                        cf[self.label]  = cnd_class
                        return cf
                    else:
                        #print('-- If candidate is from expected class and stepsize is not smaller than threshold --')
                        self.old_cnd = candidate_df.copy() # set 'candidate' as 'old_cnd' for the next step
                        #  Find a new cf candidate between 'candidate' and 'p1' (Try to approach to cnd_class)
                        return self.generateCF(candidate_df.iloc[0], cnd_class, p1, p1_class, thresh, cnt+1) 
                else:
                    #print('-- If candidate is from expected class and there is no old cf --')
                    self.old_cnd = candidate_df.copy() # set 'candidate' as 'old_cnd' for the next step
                    #  Find a new cf candidate between 'candidate' and 'p1' (Try to approach to cnd_class)
                    return self.generateCF(candidate_df.iloc[0], cnd_class, p1, p1_class, thresh, cnt+1) 
            
            elif cnd_class != self.q_class:
                #print('-- If candidate is from comp diff class --')
                return self.generateCF(candidate_df.iloc[0], cnd_class, p1, p1_class, thresh, cnt+1) 

        
            # If 'candidate' is not from expected class 
            #elif cnd_class != self.exp_class:
            else:
                #print('-- If candidate is not from expected class --')
                #  Find a new cf candidate between 'candidate' and 'p2' (Try to approach to cnd_class)
                return self.generateCF(candidate_df.iloc[0], cnd_class, p2, p2_class, thresh, cnt+1)

        
        # If it reaches the iteration limit
        #elif cnt == self.num_iter:
        else:
            if len(self.cnd_list) > 0: # if there is at least one candidate in the list
                #print('--If it reaches the iteration limit & if there is at least one candidate in the list--')
                return self.cnd_list.iloc[-1] # return the latest founded CF candidate
            else:
                # Search process for a CF is not succesfull,
                # Start a CF generation process to generate CF from last candidates class.   
                a = candidate_df.iloc[0].copy()
                a[self.label] = cnd_class
                new_nun = self.findNUN(a, unwanted=self.used_classes)
                if len(new_nun):
                    #print('--If it reaches the iteration limit & couldnt find & CF from last candidates class--')
                    new_nun_class =  str(new_nun[self.label][0])
                    new_nun = new_nun.drop([self.label,'caseID','similarity'], axis=1).loc[0]
                    return self.generateCF(candidate_df.iloc[0], cnd_class, new_nun, new_nun_class, thresh, -1) 
                else:
                    #print('--If return nun --')
                    return self.nun.copy()


    # --------------- Experimenting ---------------

    def testGenerateCF(self, n: int = None, coef_thresh: float = 128):
        """
            Test for 'generateCF' method with 'test' data

            'n' (int): number of samples to test 'generateCF' method (if n=5: use the first 5 instances of 'test')
            'coef_thresh' (int/float): Coefficent to calculate threshold for step size
        """
        
        self.coef_thresh = coef_thresh
        
        cf_list = []
        dissimilarity = []
        sparsity = []
        instability = []
        times = []
        
        if n == None:
            n = len(self.y_test)
        
        for i in range(n): 
            q = self.test.iloc[i]

            nun = self.findNUN(q)
            nun_class = str(nun[self.label][0])
            nun = nun.drop([self.label,'caseID','similarity'], axis=1).loc[0]

            q_class = q[self.label]
            q = q.drop([self.label])
            
            self.exp_class = nun_class
            thresh = self._findDist(q,nun)/self.coef_thresh # diff between q and nun / x(2,8,128,...)
            
            start = t.time()
            cf = self.generateCF(q,q_class,nun,nun_class, thresh, 0)
            end = t.time()

            # Metrics
            cf_list.append(cf)
            dissimilarity.append(self.dissimilarity(q,cf.drop([self.label])))
            sparsity.append(self.sparsity(q,cf.drop([self.label])))
            instability.append(self.instability(q,cf))
            times.append(end - start)

        dissimilarity = mean(dissimilarity)
        sparsity = mean(sparsity) 
        instability = mean(instability) 
        times = mean(times)

        
        return cf_list, [dissimilarity, sparsity, instability, times]


    def dissimilarity(self, p1, p2):
        """
                Dissimilarity: (The lower the better) mean(d(x,x_i'))
        """
        return self._findDist(p1,p2)


    def sparsity(self, p1,p2):
        """
                Sparsity: L0 norm between query and counterfactual
                measures how many features were changed to go from x to x'
        """
        spar = 0
        for i in self.col_names:
            if p1[i] == p2[i]:
                spar += 1 
        return(spar/len(self.col_names))
    

    def instability(self, query, cf, pert = 0.01):
        """
                Instability: If x and y are very close to each other,
                the system should generate very close CFs x', y'
                dist(x',y') (The lower the better)
        """
        def _encode(df, maps):
            for i in self.cat_col_names:
                df[i] = df[i].map(maps[i])
            return df
        cf_l = cf.values.tolist()
        query_l = query.values.tolist()
        
        pert_coef = 1 + pert  
        
        i = 0
        query_pert = []

        for f in self.col_names:
            if f not in self.cat_col_names:

                cnd_i = query_l[i] * pert_coef
                query_pert.append(float(cnd_i))
            else:
                if pert > 0.5:
                    query_pert.append(str(cf_l[i]))
                else:
                    query_pert.append(str(query_l[i]))
            i += 1

        query_pert_df = pd.DataFrame([query_pert],columns=self.col_names)
        q_p_class = self.model.predict([_encode(query_pert_df.copy(), self.maps).iloc[0]])[0]
        query_pert_df = pd.DataFrame([query_pert +[q_p_class]],columns=self.col_names_l)

        
        nun = self.findNUN(query_pert_df.iloc[0])
        nun_class = str(nun[self.label][0])
        nun = nun.drop([self.label,'caseID','similarity'], axis=1).iloc[0]

        thresh = self._findDist(query_pert_df.iloc[0],nun)/self.coef_thresh # diff between q and nun / x(2,8,128,...)        
        cf_pert = self.generateCF(query_pert_df.iloc[0],q_p_class,nun,nun_class,thresh, 0)  
        return self._findDist(cf, cf_pert.iloc[0])#.drop([self.label]))



    def proximity(self, normal_x,normal_cf):
        cf_proximity = np.round_(np.linalg.norm(normal_x - normal_cf),3)
        return cf_proximity
