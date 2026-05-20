# %%
import pandas as pd
import numpy as np

# %%
class GIRR:
    
    #Risk weight table for Delta
    risk_weights = {0: 0.016, 0.25: 0.017, 0.5: 0.017, 1: 0.016, 2: 0.013, 3: 0.012, 5: 0.011, 10: 0.011, 15: 0.011, 20: 0.011, 30: 0.011}
    
    def __init__(self, data, reporting_currency):
        #Assume the data has already been cleansed, filtered and grouped
        self.original_data = data
        self.data = data.copy()
        self.prescribed_currencies = ["AUD", "CAD", "EUR", "GBP", "JPY", "SEK", "USD"]
        if reporting_currency not in self.prescribed_currencies:
            self.prescribed_currencies.append(reporting_currency) #These are the currencies that enjoy lower risk weights by scale of 1/sqrt(2)
        
    def weight_sensitivities(self):
        self.data["Weighted Sensitivities"] = self.data["Sensitivity (reporting currency equiv.)"] * np.select([self.data["Sensi Type"] == "Delta"], [self.data["Tenor"].map(GIRR.risk_weights) * self.data["SA Bucket"].isin(self.prescribed_currencies).map({True: 1/np.sqrt(2), False: 1})], default=1)

    def perform_intra_bucket_aggregation(self, sensi_type):
        #Filter data to only include targeted sensitivity type
        filtered_data = self.data[self.data.iloc[:, 0] == sensi_type]
        
        #Aggregate data of the same sensitivities
        if sensi_type == "Delta":
            cols_to_grp = ["SA Bucket", "Curve Name", "Curve Type", "Tenor"]
        elif sensi_type == "Vega":
            cols_to_grp = ["SA Bucket", "Tenor", "Underlying Tenor"]
        else:
            cols_to_grp = ["SA Bucket", "CVR+/CVR-"]
            
        filtered_data = filtered_data.groupby(cols_to_grp)["Weighted Sensitivities"].sum().reset_index()

        #Create a list of unique buckets
        bucket_list = filtered_data["SA Bucket"].drop_duplicates()

        #Initiate variables to store correlations and K for each bucket
        temp_Kb_by_bucket = [[], [], []]   #[[medium], [high], [low]]
        corr_matrix_by_bucket = dict.fromkeys(bucket_list, None)
        K_by_bucket = []
        
        #For each bucket in the list, calculate the correlation than the Ks (only applicable to Delta and Vega)   
        if sensi_type != "Curvature":
            for bucket in bucket_list:
                #Filter data for each bucket
                bucket_data = filtered_data[filtered_data["SA Bucket"] == bucket].reset_index(drop=True)
                weighted_sensi = np.asarray(bucket_data["Weighted Sensitivities"])
                
                n = len(bucket_data)
                temp_corr_matrix = np.identity(n)
                
                for i in range(n):
                    for j in range(n):
                        temp_corr = 1
                        
                        if i != j:
                            if sensi_type == "Delta":
                                
                                #Check curve type dimension
                                #Cross currency basis curves
                                if bucket_data.loc[i, "Curve Type"] == "XCCY" or bucket_data.loc[j, "Curve Type"] == "XCCY":
                                    temp_corr = 0.
                                #Inflation curves (excluding XCCY curves, the only scenario if which temp_corr = 0.4 if when inflation sensi vs yield curve sensi)
                                elif bucket_data.loc[i, "Curve Type"] != bucket_data.loc[j, "Curve Type"]:
                                    temp_corr = 0.4
                                # Yield curves
                                else:
                                    #If two sensitivities are from different curves
                                    if bucket_data.loc[i, "Curve Name"] != bucket_data.loc[j, "Curve Name"]:
                                        temp_corr *= 0.999
                                        
                                    #If two sensitivities are from different tenors
                                    if bucket_data.loc[i, "Tenor"] != bucket_data.loc[j, "Tenor"]:
                                        temp_corr *= np.maximum(np.exp(-0.03 * np.abs(bucket_data.loc[i, "Tenor"] - bucket_data.loc[j, "Tenor"]) / np.minimum(bucket_data.loc[i, "Tenor"], bucket_data.loc[j, "Tenor"])), 0.4)
                                    
                            else:
                                #Calculate correlation based on option tenor
                                temp_corr *= np.exp(-0.01 * np.abs(bucket_data.loc[i, "Tenor"] - bucket_data.loc[j, "Tenor"]) / np.minimum(bucket_data.loc[i, "Tenor"], bucket_data.loc[j, "Tenor"]))
                                
                                #Calculate correlation based on underlying tenor
                                temp_corr *= np.exp(-0.01 * np.abs(bucket_data.loc[i, "Underlying Tenor"] - bucket_data.loc[j, "Underlying Tenor"]) / np.minimum(bucket_data.loc[i, "Underlying Tenor"], bucket_data.loc[j, "Underlying Tenor"]))
                                
                                temp_corr = np.minimum(temp_corr, 1.)
                            
                            #medium scenario result
                            temp_corr_matrix[i, j] = temp_corr

                #high and low scenario results
                temp_high_corr_matrix = np.minimum(1.25 * temp_corr_matrix, 1)
                temp_low_corr_matrix = np.maximum(2 * temp_corr_matrix - 1, 0.75 * temp_corr_matrix)
                
                #store results
                corr_matrix_by_bucket[bucket] = [temp_corr_matrix, temp_high_corr_matrix, temp_low_corr_matrix]
            
                #Calculate and store result of Kb for each bucket 
                for i in range(3):
                    temp_Kb_by_bucket[i].append(np.sqrt(np.maximum(0, np.dot(np.transpose(weighted_sensi),np.dot(weighted_sensi, corr_matrix_by_bucket[bucket][i])))))

            for i in range(3):
                K_by_bucket.append(pd.DataFrame({"SA Bucket": bucket_list, "K": temp_Kb_by_bucket[i]}).reset_index(drop=True))
        else:
            #Compare whether upward or downward scenarios are applicable and calculate their values
            K_by_bucket = filtered_data.pivot_table(index="SA Bucket", columns="CVR+/CVR-", values="Weighted Sensitivities", aggfunc="sum").reset_index()
            K_by_bucket["K+"] = np.sqrt(np.maximum(K_by_bucket["CVR+"], 0) ** 2)
            K_by_bucket["K-"] = np.sqrt(np.maximum(K_by_bucket["CVR-"], 0) ** 2)
            K_by_bucket["K"] = np.maximum(K_by_bucket["K+"], K_by_bucket["K-"])
    
        return K_by_bucket
        
    def perform_across_bucket_aggregation(self, sensi_type, intra_agg_result):
        
        capital_charge = []
        
        if sensi_type != "Curvature":
            
            filtered_data = self.data[self.data["Sensi Type"] == sensi_type]
            
            #Calculate Sb according to formulation in MAR21.4(5)
            Sb_by_bucket = filtered_data.groupby("SA Bucket")["Weighted Sensitivities"].sum().reset_index()
            
            #Generate the correlation matrix
            n = len(Sb_by_bucket)
            medium_corr_matrix = np.full((n, n), 0.5)
            np.fill_diagonal(medium_corr_matrix, 0)
            
            high_corr_matrix = np.minimum(1.25 * medium_corr_matrix, 1)
            low_corr_matrix = np.maximum(2 * medium_corr_matrix - 1, 0.75 * medium_corr_matrix)
            
            corr_matrices = [medium_corr_matrix, high_corr_matrix, low_corr_matrix]
            
            #Calculate the capital charge under each scenario [medium, high, low]
            for i in range(3):
                Kb = intra_agg_result[i]["K"]
                Sb = Sb_by_bucket["Weighted Sensitivities"]
                capital_charge.append(np.sqrt(
                                    np.square(Kb).sum() 
                                    + np.dot(np.transpose(Sb), np.dot(Sb, corr_matrices[i]))
                                    ))
                
                #For each scenario, we have to check whether an alternative formulation for Sb is required as stipulated in MAR21.4(5)
                if capital_charge[i] < 0:
                    Sb_Alt = np.maximum(np.minimum(Sb, Kb), -1 * Kb)
                    capital_charge[i] = (np.sqrt(np.square(Kb).sum() + np.dot(np.transpose(Sb_Alt),np.dot(Sb_Alt, corr_matrices[i]))))

        else:
            
            n = len(intra_agg_result)
            
            #Generate the correlation matrix
            medium_corr_matrix = np.full((n, n), 0.5 ** 2)
            np.fill_diagonal(medium_corr_matrix, 0)
            
            high_corr_matrix = np.minimum(1.25 * medium_corr_matrix, 1)
            low_corr_matrix = np.maximum(2 * medium_corr_matrix - 1, 0.75 * medium_corr_matrix)
            
            corr_matrices = [medium_corr_matrix, high_corr_matrix, low_corr_matrix]
            
            Sb = []    
            for i in range(n):                
                #Calculate Sb according to formulation in MAR21.5(4)
                if intra_agg_result["K"][i] == intra_agg_result["K+"][i]:
                    Sb.append(intra_agg_result["CVR+"][i])
                elif intra_agg_result["K"][i] == intra_agg_result["K-"][i]:
                    Sb.append(intra_agg_result["CVR-"][i])
                elif intra_agg_result["CVR+"][i] > intra_agg_result["CVR-"][i]:
                    Sb.append(intra_agg_result["CVR+"][i])
                else:
                    Sb.append(intra_agg_result["CVR-"][i])
            
            #Generate the psi matrix according to MAR21.5(4)(b)
            psi_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if Sb[i] >= 0 or Sb[j] >= 0:
                        psi_matrix[i, j] = 1
            
            #Store results                
            intra_agg_result["Sb"] = Sb
            
            #Calculate capital charge
            for i in range(3):
                capital_charge.append(np.sqrt(
                                        np.maximum(0, 
                                        np.square(intra_agg_result["K"]).sum() + 
                                        np.dot(Sb, np.dot(np.transpose(Sb), np.multiply(corr_matrices[i], psi_matrix)))
                                        )))        
    
        result = pd.DataFrame({"Medium": [capital_charge[0]], "High": [capital_charge[1]], "Low": [capital_charge[2]]}, index=[sensi_type])

        return result
            

# %%
class CSR_non:
    
    #Risk weight table for Delta
    risk_weights = {1: 0.005, 2: 0.01, 3: 0.05, 4: 0.03, 5: 0.03, 6: 0.02, 7: 0.015, 8: 0.025, 9: 0.02, 10: 0.04, 11: 0.12, 12: 0.07, 13: 0.085, 14: 0.055, 15: 0.05, 16: 0.12, 17: 0.015, 18: 0.05}
    
    #Across bucket correlations which is static
    #For rating - MAR21.57(1)
    rating_medium_corr = np.ones((18,18))
    rows, cols = np.indices(rating_medium_corr.shape)

    mask = ((rows <= 8) & (cols > 8) & (cols <= 15)) | ((rows > 8) & (rows <= 15) & (cols <= 8))

    rating_medium_corr[mask] = 0.5
    
    #For sector - MAR21.57(2)
    sector_medium_corr = np.asarray([[1.0, 0.75, 0.1, 0.2, 0.25, 0.2, 0.15, 0.1, 0.5, 0.75, 0.1, 0.2, 0.25, 0.2, 0.15, 0.0, 0.45, 0.45], 
                                    [0.75, 1.0, 0.05, 0.15, 0.2, 0.15, 0.1, 0.1, 0.375, 1.0, 0.05, 0.15, 0.2, 0.15, 0.1, 0.0, 0.45, 0.45], 
                                    [0.1, 0.05, 1.0, 0.05, 0.15, 0.2, 0.05, 0.2, 0.05, 0.05, 1.0, 0.05, 0.15, 0.2, 0.05, 0.0, 0.45, 0.45], 
                                    [0.2, 0.15, 0.05, 1.0, 0.2, 0.25, 0.05, 0.05, 0.1, 0.15, 0.05, 1.0, 0.2, 0.25, 0.05, 0.0, 0.45, 0.45], 
                                    [0.25, 0.2, 0.15, 0.2, 1.0, 0.25, 0.05, 0.15, 0.125, 0.2, 0.15, 0.2, 1.0, 0.25, 0.05, 0.0, 0.45, 0.45], 
                                    [0.2, 0.15, 0.2, 0.25, 0.25, 1.0, 0.05, 0.2, 0.1, 0.15, 0.2, 0.25, 0.25, 1.0, 0.05, 0.0, 0.45, 0.45], 
                                    [0.15, 0.1, 0.05, 0.05, 0.05, 0.05, 1.0, 0.05, 0.075, 0.1, 0.05, 0.05, 0.05, 0.05, 1.0, 0.0, 0.45, 0.45], 
                                    [0.1, 0.1, 0.2, 0.05, 0.15, 0.2, 0.05, 1.0, 0.05, 0.1, 0.2, 0.05, 0.15, 0.2, 0.05, 0.0, 0.45, 0.45], 
                                    [0.5, 0.375, 0.05, 0.1, 0.125, 0.1, 0.075, 0.05, 1.0, 1.5, 0.2, 0.4, 0.5, 0.4, 0.3, 0.0, 0.45, 0.45], 
                                    [0.75, 1.0, 0.05, 0.15, 0.2, 0.15, 0.1, 0.1, 1.5, 1.0, 0.05, 0.15, 0.2, 0.15, 0.1, 0.0, 0.45, 0.45], 
                                    [0.1, 0.05, 1.0, 0.05, 0.15, 0.2, 0.05, 0.2, 0.2, 0.05, 1.0, 0.05, 0.15, 0.2, 0.05, 0.0, 0.45, 0.45], 
                                    [0.2, 0.15, 0.05, 1.0, 0.2, 0.25, 0.05, 0.05, 0.4, 0.15, 0.05, 1.0, 0.2, 0.25, 0.05, 0.0, 0.45, 0.45], 
                                    [0.25, 0.2, 0.15, 0.2, 1.0, 0.25, 0.05, 0.15, 0.5, 0.2, 0.15, 0.2, 1.0, 0.25, 0.05, 0.0, 0.45, 0.45], 
                                    [0.2, 0.15, 0.2, 0.25, 0.25, 1.0, 0.05, 0.2, 0.4, 0.15, 0.2, 0.25, 0.25, 1.0, 0.05, 0.0, 0.45, 0.45], 
                                    [0.15, 0.1, 0.05, 0.05, 0.05, 0.05, 1.0, 0.05, 0.3, 0.1, 0.05, 0.05, 0.05, 0.05, 1.0, 0.0, 0.45, 0.45], 
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 
                                    [0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.0, 1.0, 0.75], 
                                    [0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.0, 0.75, 1.0]])

    across_bucket_corr = np.multiply(rating_medium_corr, sector_medium_corr)
    
    def __init__(self, data, safe_cover_bond_ratings=["AAA", "AA+", "AA", "AA-"]):
        #Assume the data has already been cleansed, filtered and grouped
        self.original_data = data
        self.data = data.copy()
        self.safe_cover_bond_ratings = safe_cover_bond_ratings #Cover bonds that have these ratings can enjoy a lower risk weights of 1.5%
        
    def weight_sensitivities(self):
        self.data["Weighted Sensitivities"] = self.data["Sensitivity (reporting currency equiv.)"] * np.select(
                                                [
                                                (self.data["Sensi Type"] == "Delta") & (self.data["SA Bucket"] != 8), 
                                                (self.data["Sensi Type"] == "Delta") & (self.data["SA Bucket"] == 8)
                                                ], 
                                                [
                                                self.data["SA Bucket"].map(CSR_non.risk_weights), 
                                                self.data["Credit Rating (Optional)"].isin(self.safe_cover_bond_ratings).map({True: 0.015, False: 0.025})
                                                ], default=1)
    
    def perform_intra_bucket_aggregation(self, sensi_type):
        #Filter data to only include targeted sensitivity type
        filtered_data = self.data[self.data["Sensi Type"] == sensi_type]
        
        #Aggregate data of the same sensitivities
        if sensi_type == "Delta":
            cols_to_grp = ["SA Bucket", "Issuer Name", "Tenor", "Curve Type"]
        elif sensi_type == "Vega":
            cols_to_grp = ["SA Bucket", "Issuer Name", "Tenor"]
        else:
            cols_to_grp = ["SA Bucket", "Issuer Name", "CVR+/CVR-"]
        
        filtered_data = filtered_data.groupby(cols_to_grp)["Weighted Sensitivities"].sum().reset_index()
        
        #Pivot in case of CVR to have separate columns CVR+ and CVR-
        if sensi_type == "Curvature":
            filtered_data = filtered_data.pivot_table(index=["SA Bucket", "Issuer Name"], columns="CVR+/CVR-", values="Weighted Sensitivities", aggfunc="sum").reset_index()
            
            #Initiate variables to store CVR and K for each bucket for curvature
            temp_CVR_plus_by_bucket = [[], [], []]   #[[medium], [high], [low]]
            temp_CVR_minus_by_bucket = [[], [], []]   #[[medium], [high], [low]]
            temp_Kb_minus_by_bucket = [[], [], []]   #[[medium], [high], [low]]
            
        #Initiate variables to store K for each bucket
        temp_Kb_by_bucket = [[], [], []]   #[[medium], [high], [low]]
        K_by_bucket = []
            
        #Create a list of unique buckets
        bucket_list = filtered_data["SA Bucket"].drop_duplicates()
        
        for bucket in bucket_list:
            #Filter data for each bucket
            bucket_data = filtered_data[filtered_data["SA Bucket"] == bucket].reset_index(drop=True)

            #Initiate variables to store correlation matrix for each bucket
            corr_matrix_by_bucket = dict.fromkeys(bucket_list, None)
            
            n = len(bucket_data)
            
            #Generate correlation matrix for each bucket
            if sensi_type != "Curvature":
                
                #Aggregate directly for bucket 16 MAR21.69(1)
                if bucket == 16:
                    for i in range(3):
                        temp_Kb_by_bucket[i].append(np.absolute(bucket_data["Weighted Sensitivities"]).sum())
                    continue
                
                temp_corr_matrix = np.identity(n)
                
                for i in range(n):
                        for j in range(n):
                            temp_corr = 1
                            
                            if i != j:
                                #Check Issuer Names
                                temp_corr *= np.select([bucket_data.loc[i, "Issuer Name"] != bucket_data.loc[j, "Issuer Name"]], [np.select([bucket <= 15, bucket >= 17, bucket == 16], [0.35, 0.8, 0])], default=1)

                                if sensi_type == "Delta":
                                    #Check tenor
                                    if bucket_data.loc[i, "Tenor"] != bucket_data.loc[j, "Tenor"]:
                                        temp_corr *= 0.65
                                    
                                    #Check curve type (bond vs CDS)
                                    if bucket_data.loc[i, "Curve Type"] != bucket_data.loc[j, "Curve Type"]:
                                        temp_corr *= 0.999

                                else:
                                    #Check option tenor
                                    temp_corr *= np.exp(
                                                -0.01 * np.abs(bucket_data.loc[i, "Tenor"] - bucket_data.loc[j, "Tenor"]) / 
                                                np.minimum(bucket_data.loc[i, "Tenor"], bucket_data.loc[j, "Tenor"]))
                                    
                                    temp_corr = np.minimum(temp_corr, 1)
                                    
                                temp_corr_matrix[i, j] = temp_corr

                #High and low scenario
                temp_high_corr_matrix = np.minimum(1.25 * temp_corr_matrix, 1)
                temp_low_corr_matrix = np.maximum(2 * temp_corr_matrix - 1, 0.75 * temp_corr_matrix)
            
                #Store results
                corr_matrix_by_bucket[bucket] = [temp_corr_matrix, temp_high_corr_matrix, temp_low_corr_matrix]
                
                #Calculate sum of weighted sensitivities of the bucket
                weighted_sensi = np.asarray(bucket_data["Weighted Sensitivities"])
                
                #Aggregate
                for i in range(3):
                    temp_Kb_by_bucket[i].append(np.sqrt(np.maximum(0, np.dot(np.transpose(weighted_sensi),np.dot(weighted_sensi, corr_matrix_by_bucket[bucket][i])))))
                                
            else:
                
                #Aggregate directly for bucket 16 MAR21.69(2)
                if bucket == 16:
                    for i in range(3):
                        temp_CVR_plus_by_bucket[i].append(bucket_data["CVR+"].sum())
                        temp_CVR_minus_by_bucket[i].append(bucket_data["CVR-"].sum())
                        temp_Kb_by_bucket[i].append(np.maximum(bucket_data["CVR+"], 0).sum())
                        temp_Kb_minus_by_bucket[i].append(np.maximum(bucket_data["CVR-"], 0).sum())
                    continue
                
                temp_corr_matrix = np.zeros((n,n))
                psi_matrix_CVR_plus = np.zeros((n, n))
                psi_matrix_CVR_minus = np.zeros((n, n))                
                
                for i in range(n):
                        for j in range(n):
                            temp_corr = 1
                            
                            if i != j:
                                #Check Issuer Names
                                temp_corr *= np.select([bucket_data.loc[i, "Issuer Name"] != bucket_data.loc[j, "Issuer Name"]], [np.select([bucket <= 15, bucket >= 17, bucket == 16], [0.35, 0.8, 0])], default=1)                
                                temp_corr = temp_corr ** 2
                                
                                #Calculate psi matrix for CVR+ and CVR-
                                if bucket_data.loc[i, "CVR+"] >= 0 or bucket_data.loc[j, "CVR+"] >= 0:
                                    psi_matrix_CVR_plus[i, j] = 1
                                
                                if bucket_data.loc[i, "CVR-"] >= 0 or bucket_data.loc[j, "CVR-"] >= 0:
                                    psi_matrix_CVR_minus[i, j] = 1
                            
                            temp_corr_matrix[i, j] = temp_corr
                            
                #High and low scenario
                temp_high_corr_matrix = np.minimum(1.25 * temp_corr_matrix, 1)
                temp_low_corr_matrix = np.maximum(2 * temp_corr_matrix - 1, 0.75 * temp_corr_matrix)
                
                #Store results
                corr_matrix_by_bucket[bucket] = [temp_corr_matrix, temp_high_corr_matrix, temp_low_corr_matrix]

                #Calculate K for the bucket
                for i in range(3):
                    temp_CVR_plus_by_bucket[i].append(bucket_data["CVR+"].sum())
                    
                    temp_CVR_minus_by_bucket[i].append(bucket_data["CVR-"].sum())
                    
                    temp_Kb_by_bucket[i].append(np.sqrt(
                                                np.maximum(0, 
                                                np.square(np.maximum(0, bucket_data["CVR+"])).sum() + 
                                                np.dot(bucket_data["CVR+"], np.dot(np.transpose(bucket_data["CVR+"]), np.multiply(corr_matrix_by_bucket[bucket][i], psi_matrix_CVR_plus)))
                                                )))
                    
                    temp_Kb_minus_by_bucket[i].append(np.sqrt(
                                                np.maximum(0, 
                                                np.square(np.maximum(0, bucket_data["CVR-"])).sum() + 
                                                np.dot(bucket_data["CVR-"], np.dot(np.transpose(bucket_data["CVR-"]), np.multiply(corr_matrix_by_bucket[bucket][i], psi_matrix_CVR_minus)))
                                                )))
                
        if sensi_type != "Curvature":
            
            for i in range(3):
                K_by_bucket.append(pd.DataFrame({"SA Bucket": bucket_list, "K": temp_Kb_by_bucket[i]}).reset_index(drop=True))
                
        else:
            
            for i in range(3):
                K_by_bucket.append(pd.DataFrame({"SA Bucket": bucket_list,
                                                "CVR+": temp_CVR_plus_by_bucket[i],
                                                "CVR-": temp_CVR_minus_by_bucket[i],
                                                "K+": temp_Kb_by_bucket[i], 
                                                "K-": temp_Kb_minus_by_bucket[i], 
                                                "K": np.maximum(temp_Kb_by_bucket[i], temp_Kb_minus_by_bucket[i])})
                                                .reset_index(drop=True))
        
        return K_by_bucket

    def perform_across_bucket_aggregation(self, sensi_type, intra_agg_result):
        
        capital_charge = []
        
        if sensi_type != "Curvature":
            
            filtered_data = self.data[self.data["Sensi Type"] == sensi_type]
            
            #Calculate Sb according to formulation in MAR21.4(5)
            Sb_by_bucket = filtered_data.groupby("SA Bucket")["Weighted Sensitivities"].sum().reset_index()
            
            #Generate the correlation matrix
            n = len(Sb_by_bucket)
            medium_corr_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        medium_corr_matrix = CSR_non.across_bucket_corr[Sb_by_bucket.loc[i, "SA Bucket"] - 1, Sb_by_bucket.loc[j, "SA Bucket"] - 1] #Bucket number -1 since bucket numbers start with 1 and python start counts from 0
            
            high_corr_matrix = np.minimum(1.25 * medium_corr_matrix, 1)
            low_corr_matrix = np.maximum(2 * medium_corr_matrix - 1, 0.75 * medium_corr_matrix)
            
            corr_matrices = [medium_corr_matrix, high_corr_matrix, low_corr_matrix]
            
            #Calculate the capital charge under each scenario [medium, high, low]
            for i in range(3):
                Kb = intra_agg_result[i]["K"]
                Sb = Sb_by_bucket["Weighted Sensitivities"]
                capital_charge.append(np.sqrt(
                                        np.square(Kb).sum() + 
                                        np.dot(np.transpose(Sb), np.dot(Sb, corr_matrices[i]))
                                        ))
                
                #For each scenario, we have to check whether an alternative formulation for Sb is required as stipulated in MAR21.4(5)
                if capital_charge[i] < 0:
                    Sb_Alt = np.maximum(np.minimum(Sb, Kb), -1 * Kb)
                    capital_charge[i] = (np.sqrt(
                                            np.square(Kb).sum() + 
                                            np.dot(np.transpose(Sb_Alt),np.dot(Sb_Alt, corr_matrices[i]))
                                            ))
                    
        else:
            
            temp_K_by_bucket = intra_agg_result[0]
            n = len(temp_K_by_bucket)
            
            #Generate the correlation matrix
            medium_corr_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        medium_corr_matrix = CSR_non.across_bucket_corr[temp_K_by_bucket.loc[i, "SA Bucket"] - 1, temp_K_by_bucket.loc[j, "SA Bucket"] - 1] ** 2 #Bucket number -1 since bucket numbers start with 1 and python start counts from 0
            
            high_corr_matrix = np.minimum(1.25 * medium_corr_matrix, 1)
            low_corr_matrix = np.maximum(2 * medium_corr_matrix - 1, 0.75 * medium_corr_matrix)
            
            corr_matrices = [medium_corr_matrix, high_corr_matrix, low_corr_matrix]
            
            Sb = [[], [], []] #[[medium], [high], [low]]
            psi_matrix = [[], [], []] #[[medium], [high], [low]]
            
            for i in range(3):
                temp_K_by_bucket = intra_agg_result[i]
                
                for j in range(n):                
                    #Calculate Sb according to formulation in MAR21.5(4)
                    if temp_K_by_bucket["K"][j] == temp_K_by_bucket["K+"][j]:
                        Sb[i].append(temp_K_by_bucket["CVR+"][j])
                    elif temp_K_by_bucket["K"][j] == temp_K_by_bucket["K-"][j]:
                        Sb[i].append(temp_K_by_bucket["CVR-"][j])
                    elif temp_K_by_bucket["CVR+"][j] > temp_K_by_bucket["CVR-"][j]:
                        Sb[i].append(temp_K_by_bucket["CVR+"][j])
                    else:
                        Sb[i].append(temp_K_by_bucket["CVR-"][j])
            
                #Generate the psi matrix according to MAR21.5(4)(b)
                psi_matrix[i] = np.zeros((n, n))
                for k in range(n):
                    for l in range(n):
                        if Sb[i][k] >= 0 or Sb[i][l] >= 0:
                            psi_matrix[i][k, l] = 1
                
                #Store results                
                intra_agg_result[i]["Sb"] = Sb[i]
            
                #Calculate capital charge
                capital_charge.append(np.sqrt(
                                        np.maximum(0, 
                                        np.square(intra_agg_result[i]["K"]).sum() + 
                                        np.dot(Sb[i], np.dot(np.transpose(Sb[i]), np.multiply(corr_matrices[i], psi_matrix[i])))
                                        )))
        
        result = pd.DataFrame({"Medium": [capital_charge[0]], "High": [capital_charge[1]], "Low": [capital_charge[2]]}, index=[sensi_type])
        
        return result
    

# %%
class Equity:
    
    #Risk weight table for Delta
    delta_risk_weights_spot = {1: 0.55, 2: 0.6, 3: 0.45, 4: 0.55, 5: 0.3, 6: 0.35, 7: 0.4, 8: 0.5, 9: 0.7, 10: 0.5, 11: 0.7, 12: 0.15, 13: 0.25}
    delta_risk_weights_repo = {keys: values / 100 for keys, values in delta_risk_weights_spot.items()}
    
    #Within bucket correlation mapped by bucket number - delta
    intra_corr_bucket_delta = {1: 0.15, 2: 0.15, 3: 0.15, 4: 0.15, 5: 0.25, 6: 0.25, 7: 0.25, 8: 0.25, 9: 0.075, 10: 0.125, 11: 0., 12: 0.8, 13: 0.8}
    
    #Across bucket correlation mapped by bucket number - delta
    across_corr_delta = np.full((13, 13), 0.45)
    
    rows, cols = np.indices(across_corr_delta.shape)

    #index = bucket number - 1
    across_corr_delta[(rows <= 9) & (cols <= 9)] = 0.15
    across_corr_delta[(rows == 10) | (cols == 10)] = 0.
    across_corr_delta[((rows == 11) & (cols == 12)) | ((rows == 12) & (cols == 11))] = 0.75
    
    def __init__(self, data):
        #Assume the data has already been cleansed, filtered and grouped
        self.original_data = data 
        self.data = data.copy()
        
    def weight_sensitivities(self):
        
        temp_risk_weights = np.select(  [
                                        self.data["Sensi Type"] == "Delta", 
                                        self.data["Sensi Type"] == "Vega"
                                        ], 
                                        [
                                        np.select([ self.data["Price Type"] == "Spot", 
                                                    self.data["Price Type"] == "Repo"
                                                    ],
                                                    [
                                                    self.data["SA Bucket"].map(Equity.delta_risk_weights_spot), 
                                                    self.data["SA Bucket"].map(Equity.delta_risk_weights_repo)
                                                    ]), 
                                        self.data["SA Bucket"].isin([9, 10, 11]).map(   {
                                                                                        True: np.minimum(0.55 * np.sqrt(60/10), 1), 
                                                                                        False: np.minimum(0.55 * np.sqrt(20/10), 1)
                                                                                        })],
                                        default=1)
        
        self.data["Weighted Sensitivities"] = self.data["Sensitivity (reporting currency equiv.)"] * temp_risk_weights
        
    def perform_intra_bucket_aggregation(self, sensi_type):
        #Filter data to only include targeted sensitivity type
        filtered_data = self.data[self.data["Sensi Type"] == sensi_type]
        
        #Aggregate data of the same sensitivities
        if sensi_type == "Delta":
            cols_to_grp = ["SA Bucket", "Issuer", "Price Type"]
        elif sensi_type == "Vega":
            cols_to_grp = ["SA Bucket", "Tenor"]
        else:
            cols_to_grp = ["SA Bucket", "Issuer", "CVR+/CVR-"]
        
        filtered_data = filtered_data.groupby(cols_to_grp)["Weighted Sensitivities"].sum().reset_index()
        
        #Pivot in case of CVR to have separate columns CVR+ and CVR-
        if sensi_type == "Curvature":
            filtered_data = filtered_data.pivot_table(index=["SA Bucket", "Issuer"], columns="CVR+/CVR-", values="Weighted Sensitivities", aggfunc="sum").reset_index()
            
            #Initiate variables to store CVR and K for each bucket for curvature
            temp_CVR_plus_by_bucket = [[], [], []]   #[[medium], [high], [low]]
            temp_CVR_minus_by_bucket = [[], [], []]   #[[medium], [high], [low]]
            temp_Kb_minus_by_bucket = [[], [], []]   #[[medium], [high], [low]]
            
        #Initiate variables to store K for each bucket
        temp_Kb_by_bucket = [[], [], []]   #[[medium], [high], [low]]
        K_by_bucket = []
            
        #Create a list of unique buckets
        bucket_list = filtered_data["SA Bucket"].drop_duplicates()
        
        for bucket in bucket_list:
            #Filter data for each bucket
            bucket_data = filtered_data[filtered_data["SA Bucket"] == bucket].reset_index(drop=True)

            #Initiate variables to store correlation matrix for each bucket
            corr_matrix_by_bucket = dict.fromkeys(bucket_list, None)
            
            n = len(bucket_data)
            
            #Generate correlation matrix for each bucket
            temp_corr_matrix = np.full((n, n), Equity.intra_corr_bucket_delta.get(bucket))

            if sensi_type != "Curvature":
                
                #Aggregate directly for bucket 11 MAR21.79(1)
                if bucket == 11:
                    for i in range(3):
                        temp_Kb_by_bucket[i].append(np.absolute(bucket_data["Weighted Sensitivities"]).sum())
                    continue
                
                np.fill_diagonal(temp_corr_matrix, 1)
                
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            
                            if sensi_type == "Delta":
                                #Check if one sensi is spot and the other is repo
                                if bucket_data.loc[i, "Price Type"] != bucket_data.loc[j, "Price Type"]:
                                    
                                    #Check if the issuer is the same
                                    if bucket_data.loc[i, "Issuer"] == bucket_data.loc[j, "Issuer"]:
                                        
                                        #MAR21.78(1)
                                        temp_corr_matrix[i, j] = 0.999
                                        
                                    else:
                                        
                                        #MAR21.78(4)
                                        temp_corr_matrix[i, j] *= 0.999

                            else: #Vega
                                
                                #Check option tenor
                                temp_corr_matrix[i, j] *= np.exp(
                                                            -0.01 * np.abs(bucket_data.loc[i, "Tenor"] - bucket_data.loc[j, "Tenor"]) / 
                                                            np.minimum(bucket_data.loc[i, "Tenor"], bucket_data.loc[j, "Tenor"]))
                                                
                                temp_corr_matrix[i, j] = np.minimum(temp_corr_matrix[i, j], 1)
                                    
                #High and low scenario
                temp_high_corr_matrix = np.minimum(1.25 * temp_corr_matrix, 1)
                temp_low_corr_matrix = np.maximum(2 * temp_corr_matrix - 1, 0.75 * temp_corr_matrix)
            
                #Store results
                corr_matrix_by_bucket[bucket] = [temp_corr_matrix, temp_high_corr_matrix, temp_low_corr_matrix]
                
                #Calculate sum of weighted sensitivities of the bucket
                weighted_sensi = np.asarray(bucket_data["Weighted Sensitivities"])
                
                #Aggregate
                for i in range(3):
                    temp_Kb_by_bucket[i].append(np.sqrt(np.maximum(0, np.dot(np.transpose(weighted_sensi),np.dot(weighted_sensi, corr_matrix_by_bucket[bucket][i])))))
            
            else:
                
                #Aggregate directly for bucket 16 MAR21.79(2)
                if bucket == 11:
                    for i in range(3):
                        temp_CVR_plus_by_bucket[i].append(bucket_data["CVR+"].sum())
                        temp_CVR_minus_by_bucket[i].append(bucket_data["CVR-"].sum())
                        temp_Kb_by_bucket[i].append(np.maximum(bucket_data["CVR+"], 0).sum())
                        temp_Kb_minus_by_bucket[i].append(np.maximum(bucket_data["CVR-"], 0).sum())
                    continue
                
                psi_matrix_CVR_plus = np.zeros((n, n))
                psi_matrix_CVR_minus = np.zeros((n, n))                
                
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            #Calculate psi matrix for CVR+ and CVR-
                            if bucket_data.loc[i, "CVR+"] >= 0 or bucket_data.loc[j, "CVR+"] >= 0:
                                psi_matrix_CVR_plus[i, j] = 1
                            
                            if bucket_data.loc[i, "CVR-"] >= 0 or bucket_data.loc[j, "CVR-"] >= 0:
                                psi_matrix_CVR_minus[i, j] = 1
                            
                #correlation matrix = delta matrix ** 2
                np.fill_diagonal(temp_corr_matrix, 0)
                temp_corr_matrix = temp_corr_matrix ** 2
                temp_high_corr_matrix = np.minimum(1.25 * temp_corr_matrix, 1)
                temp_low_corr_matrix = np.maximum(2 * temp_corr_matrix - 1, 0.75 * temp_corr_matrix)
                
                #Store results
                corr_matrix_by_bucket[bucket] = [temp_corr_matrix, temp_high_corr_matrix, temp_low_corr_matrix]

                #Calculate K for the bucket
                for i in range(3):
                    temp_CVR_plus_by_bucket[i].append(bucket_data["CVR+"].sum())
                    
                    temp_CVR_minus_by_bucket[i].append(bucket_data["CVR-"].sum())
                    
                    temp_Kb_by_bucket[i].append(np.sqrt(
                                                np.maximum(0, 
                                                np.square(np.maximum(0, bucket_data["CVR+"])).sum() + 
                                                np.dot(bucket_data["CVR+"], np.dot(np.transpose(bucket_data["CVR+"]), np.multiply(corr_matrix_by_bucket[bucket][i], psi_matrix_CVR_plus)))
                                                )))
                    
                    temp_Kb_minus_by_bucket[i].append(np.sqrt(
                                                np.maximum(0, 
                                                np.square(np.maximum(0, bucket_data["CVR-"])).sum() + 
                                                np.dot(bucket_data["CVR-"], np.dot(np.transpose(bucket_data["CVR-"]), np.multiply(corr_matrix_by_bucket[bucket][i], psi_matrix_CVR_minus)))
                                                )))
                
        if sensi_type != "Curvature":
            
            for i in range(3):
                K_by_bucket.append(pd.DataFrame({"SA Bucket": bucket_list, "K": temp_Kb_by_bucket[i]}).reset_index(drop=True))
                
        else:
            
            for i in range(3):
                K_by_bucket.append(pd.DataFrame({"SA Bucket": bucket_list,
                                                "CVR+": temp_CVR_plus_by_bucket[i],
                                                "CVR-": temp_CVR_minus_by_bucket[i],
                                                "K+": temp_Kb_by_bucket[i], 
                                                "K-": temp_Kb_minus_by_bucket[i], 
                                                "K": np.maximum(temp_Kb_by_bucket[i], temp_Kb_minus_by_bucket[i])})
                                                .reset_index(drop=True))
        
        return K_by_bucket
    
    def perform_across_bucket_aggregation(self, sensi_type, intra_agg_result):
        
        capital_charge = []
        
        if sensi_type != "Curvature":
            
            filtered_data = self.data[self.data["Sensi Type"] == sensi_type]
            
            #Calculate Sb according to formulation in MAR21.4(5)
            Sb_by_bucket = filtered_data.groupby("SA Bucket")["Weighted Sensitivities"].sum().reset_index()
            
            #Generate the correlation matrix
            n = len(Sb_by_bucket)
            medium_corr_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        medium_corr_matrix = Equity.across_corr_delta[Sb_by_bucket.loc[i, "SA Bucket"] - 1, Sb_by_bucket.loc[j, "SA Bucket"] - 1] #Bucket number -1 since bucket numbers start with 1 and python start counts from 0
            
            high_corr_matrix = np.minimum(1.25 * medium_corr_matrix, 1)
            low_corr_matrix = np.maximum(2 * medium_corr_matrix - 1, 0.75 * medium_corr_matrix)
            
            corr_matrices = [medium_corr_matrix, high_corr_matrix, low_corr_matrix]
            
            #Calculate the capital charge under each scenario [medium, high, low]
            for i in range(3):
                Kb = intra_agg_result[i]["K"]
                Sb = Sb_by_bucket["Weighted Sensitivities"]
                capital_charge.append(np.sqrt(
                                        np.square(Kb).sum() + 
                                        np.dot(np.transpose(Sb), np.dot(Sb, corr_matrices[i]))
                                        ))
                
                #For each scenario, we have to check whether an alternative formulation for Sb is required as stipulated in MAR21.4(5)
                if capital_charge[i] < 0:
                    Sb_Alt = np.maximum(np.minimum(Sb, Kb), -1 * Kb)
                    capital_charge[i] = (np.sqrt(
                                            np.square(Kb).sum() + 
                                            np.dot(np.transpose(Sb_Alt),np.dot(Sb_Alt, corr_matrices[i]))
                                            ))
                    
        else:
            
            temp_K_by_bucket = intra_agg_result[0]
            n = len(temp_K_by_bucket)
            
            #Generate the correlation matrix
            medium_corr_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        medium_corr_matrix = Equity.across_corr_delta[temp_K_by_bucket.loc[i, "SA Bucket"] - 1, temp_K_by_bucket.loc[j, "SA Bucket"] - 1] ** 2 #Bucket number -1 since bucket numbers start with 1 and python start counts from 0
            
            high_corr_matrix = np.minimum(1.25 * medium_corr_matrix, 1)
            low_corr_matrix = np.maximum(2 * medium_corr_matrix - 1, 0.75 * medium_corr_matrix)
            
            corr_matrices = [medium_corr_matrix, high_corr_matrix, low_corr_matrix]
            
            Sb = [[], [], []] #[[medium], [high], [low]]
            psi_matrix = [[], [], []] #[[medium], [high], [low]]
            
            for i in range(3):
                temp_K_by_bucket = intra_agg_result[i]
                
                for j in range(n):                
                    #Calculate Sb according to formulation in MAR21.5(4)
                    if temp_K_by_bucket["K"][j] == temp_K_by_bucket["K+"][j]:
                        Sb[i].append(temp_K_by_bucket["CVR+"][j])
                    elif temp_K_by_bucket["K"][j] == temp_K_by_bucket["K-"][j]:
                        Sb[i].append(temp_K_by_bucket["CVR-"][j])
                    elif temp_K_by_bucket["CVR+"][j] > temp_K_by_bucket["CVR-"][j]:
                        Sb[i].append(temp_K_by_bucket["CVR+"][j])
                    else:
                        Sb[i].append(temp_K_by_bucket["CVR-"][j])
            
                #Generate the psi matrix according to MAR21.5(4)(b)
                psi_matrix[i] = np.zeros((n, n))
                for k in range(n):
                    for l in range(n):
                        if Sb[i][k] >= 0 or Sb[i][l] >= 0:
                            psi_matrix[i][k, l] = 1
                
                #Store results                
                intra_agg_result[i]["Sb"] = Sb[i]
            
                #Calculate capital charge
                capital_charge.append(np.sqrt(
                                        np.maximum(0, 
                                        np.square(intra_agg_result[i]["K"]).sum() + 
                                        np.dot(Sb[i], np.dot(np.transpose(Sb[i]), np.multiply(corr_matrices[i], psi_matrix[i])))
                                        )))
        
        result = pd.DataFrame({"Medium": [capital_charge[0]], "High": [capital_charge[1]], "Low": [capital_charge[2]]}, index=[sensi_type])
        
        return result

# %%
class Commodity:
    
    #Risk weight table for Delta
    delta_risk_weights = {1: 0.3, 2: 0.35, 3: 0.6, 4: 0.8, 5: 0.4, 6: 0.45, 7: 0.2, 8: 0.35, 9: 0.25, 10: 0.35, 11: 0.5}
    
    #correlation for commodity MAR21.83(1)
    intra_corr_cty = {1: 0.55, 2: 0.95, 3: 0.4, 4: 0.8, 5: 0.6, 6: 0.65, 7: 0.55, 8: 0.45, 9: 0.15, 10: 0.4, 11: 0.15}
    
    def __init__(self, data):
        #Assume the data has already been cleansed, filtered and grouped
        self.original_data = data 
        self.data = data.copy()
        
    def weight_sensitivities(self):
        self.data["Weighted Sensitivities"] = self.data["Sensitivity (reporting currency equiv.)"] * np.select([self.data["Sensi Type"] == "Delta"], [self.data["SA Bucket"].map(Commodity.delta_risk_weights)], default=1)
        
    def perform_intra_bucket_aggregation(self, sensi_type):
        #Filter data to only include targeted sensitivity type
        filtered_data = self.data[self.data["Sensi Type"] == sensi_type]
        
        #Aggregate data of the same sensitivities
        if sensi_type != "Curvature":
            cols_to_grp = ["SA Bucket", "Commodity", "Tenor", "Location"]
        else:
            cols_to_grp = ["SA Bucket", "Commodity", "CVR+/CVR-"]
        
        filtered_data = filtered_data.groupby(cols_to_grp)["Weighted Sensitivities"].sum().reset_index()
        
        #Pivot in case of CVR to have separate columns CVR+ and CVR-
        if sensi_type == "Curvature":
            filtered_data = filtered_data.pivot_table(index=["SA Bucket", "Commodity"], columns="CVR+/CVR-", values="Weighted Sensitivities", aggfunc="sum").reset_index()
            
            #Initiate variables to store CVR and K for each bucket for curvature
            temp_CVR_plus_by_bucket = [[], [], []]   #[[medium], [high], [low]]
            temp_CVR_minus_by_bucket = [[], [], []]   #[[medium], [high], [low]]
            temp_Kb_minus_by_bucket = [[], [], []]   #[[medium], [high], [low]]
            
        #Initiate variables to store K for each bucket
        temp_Kb_by_bucket = [[], [], []]   #[[medium], [high], [low]]
        K_by_bucket = []
            
        #Create a list of unique buckets
        bucket_list = filtered_data["SA Bucket"].drop_duplicates()
        
        for bucket in bucket_list:
            #Filter data for each bucket
            bucket_data = filtered_data[filtered_data["SA Bucket"] == bucket].reset_index(drop=True)

            #Initiate variables to store correlation matrix for each bucket
            corr_matrix_by_bucket = dict.fromkeys(bucket_list, None)
            
            n = len(bucket_data)
            
            #Generate correlation matrix for each bucket
            if sensi_type != "Curvature":
                
                temp_corr_matrix = np.ones((n, n))
                
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            
                            #Check if commodity is identical
                            if bucket_data.loc[i, "Commodity"] != bucket_data.loc[j, "Commodity"]:
                                temp_corr_matrix[i, j] = Commodity.intra_corr_cty.get(bucket)
                            
                            if sensi_type == "Delta":        
                                #Check if the tenor is the same
                                if bucket_data.loc[i, "Tenor"] != bucket_data.loc[j, "Tenor"]:
                                    temp_corr_matrix[i, j] *= 0.99
                                
                                #Check if the delivery location is the same
                                if bucket_data.loc[i, "Location"] != bucket_data.loc[j, "Location"]:
                                    temp_corr_matrix[i, j] *= 0.999

                            else: #Vega
                                
                                #Check option tenor
                                temp_corr_matrix[i, j] *= np.exp(
                                                            -0.01 * np.abs(bucket_data.loc[i, "Tenor"] - bucket_data.loc[j, "Tenor"]) / 
                                                            np.minimum(bucket_data.loc[i, "Tenor"], bucket_data.loc[j, "Tenor"]))
                                                
                                temp_corr_matrix[i, j] = np.minimum(temp_corr_matrix[i, j], 1)
                                    
                #High and low scenario
                temp_high_corr_matrix = np.minimum(1.25 * temp_corr_matrix, 1)
                temp_low_corr_matrix = np.maximum(2 * temp_corr_matrix - 1, 0.75 * temp_corr_matrix)
                
                #Store results
                corr_matrix_by_bucket[bucket] = [temp_corr_matrix, temp_high_corr_matrix, temp_low_corr_matrix]
                
                #Calculate sum of weighted sensitivities of the bucket
                weighted_sensi = np.asarray(bucket_data["Weighted Sensitivities"])
                
                #Aggregate
                for i in range(3):
                    temp_Kb_by_bucket[i].append(np.sqrt(np.maximum(0, np.dot(np.transpose(weighted_sensi),np.dot(weighted_sensi, corr_matrix_by_bucket[bucket][i])))))
            
            else:
                
                temp_corr_matrix = np.zeros((n, n))
                psi_matrix_CVR_plus = np.zeros((n, n))
                psi_matrix_CVR_minus = np.zeros((n, n))                
                
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            #Check if commodity is identical to map correlation
                            if bucket_data.loc[i, "Commodity"] != bucket_data.loc[j, "Commodity"]:
                                temp_corr_matrix[i, j] = Commodity.intra_corr_cty.get(bucket) ** 2
                            
                            #Calculate psi matrix for CVR+ and CVR-
                            if bucket_data.loc[i, "CVR+"] >= 0 or bucket_data.loc[j, "CVR+"] >= 0:
                                psi_matrix_CVR_plus[i, j] = 1
                            
                            if bucket_data.loc[i, "CVR-"] >= 0 or bucket_data.loc[j, "CVR-"] >= 0:
                                psi_matrix_CVR_minus[i, j] = 1
                            
                #high and low scenario
                temp_high_corr_matrix = np.minimum(1.25 * temp_corr_matrix, 1)
                temp_low_corr_matrix = np.maximum(2 * temp_corr_matrix - 1, 0.75 * temp_corr_matrix)
                
                #Store results
                corr_matrix_by_bucket[bucket] = [temp_corr_matrix, temp_high_corr_matrix, temp_low_corr_matrix]

                #Calculate K for the bucket
                for i in range(3):
                    temp_CVR_plus_by_bucket[i].append(bucket_data["CVR+"].sum())
                    
                    temp_CVR_minus_by_bucket[i].append(bucket_data["CVR-"].sum())
                    
                    temp_Kb_by_bucket[i].append(np.sqrt(
                                                np.maximum(0, 
                                                np.square(np.maximum(0, bucket_data["CVR+"])).sum() + 
                                                np.dot(bucket_data["CVR+"], np.dot(np.transpose(bucket_data["CVR+"]), np.multiply(corr_matrix_by_bucket[bucket][i], psi_matrix_CVR_plus)))
                                                )))
                    
                    temp_Kb_minus_by_bucket[i].append(np.sqrt(
                                                np.maximum(0, 
                                                np.square(np.maximum(0, bucket_data["CVR-"])).sum() + 
                                                np.dot(bucket_data["CVR-"], np.dot(np.transpose(bucket_data["CVR-"]), np.multiply(corr_matrix_by_bucket[bucket][i], psi_matrix_CVR_minus)))
                                                )))
                
        if sensi_type != "Curvature":
            
            for i in range(3):
                K_by_bucket.append(pd.DataFrame({"SA Bucket": bucket_list, "K": temp_Kb_by_bucket[i]}).reset_index(drop=True))
                
        else:
            
            for i in range(3):
                K_by_bucket.append(pd.DataFrame({"SA Bucket": bucket_list,
                                                "CVR+": temp_CVR_plus_by_bucket[i],
                                                "CVR-": temp_CVR_minus_by_bucket[i],
                                                "K+": temp_Kb_by_bucket[i], 
                                                "K-": temp_Kb_minus_by_bucket[i], 
                                                "K": np.maximum(temp_Kb_by_bucket[i], temp_Kb_minus_by_bucket[i])})
                                                .reset_index(drop=True))

        return K_by_bucket
    
    def perform_across_bucket_aggregation(self, sensi_type, intra_agg_result):
        
        capital_charge = []
        
        if sensi_type != "Curvature":
            
            filtered_data = self.data[self.data["Sensi Type"] == sensi_type]
            
            #Calculate Sb according to formulation in MAR21.4(5)
            Sb_by_bucket = filtered_data.groupby("SA Bucket")["Weighted Sensitivities"].sum().reset_index()
            
            #Generate the correlation matrix
            n = len(Sb_by_bucket)
            medium_corr_matrix = np.full((n, n), 0.2)
            for i in range(n):
                for j in range(n):
                    if i == j or (Sb_by_bucket.loc[i, "SA Bucket"] != 11 or Sb_by_bucket.loc[i, "SA Bucket"] != 11):
                        medium_corr_matrix[i, j] = 0
            
            high_corr_matrix = np.minimum(1.25 * medium_corr_matrix, 1)
            low_corr_matrix = np.maximum(2 * medium_corr_matrix - 1, 0.75 * medium_corr_matrix)
            
            corr_matrices = [medium_corr_matrix, high_corr_matrix, low_corr_matrix]
            
            #Calculate the capital charge under each scenario [medium, high, low]
            for i in range(3):
                Kb = intra_agg_result[i]["K"]
                Sb = Sb_by_bucket["Weighted Sensitivities"]
                capital_charge.append(np.sqrt(
                                        np.square(Kb).sum() + 
                                        np.dot(np.transpose(Sb), np.dot(Sb, corr_matrices[i]))
                                        ))
                
                #For each scenario, we have to check whether an alternative formulation for Sb is required as stipulated in MAR21.4(5)
                if capital_charge[i] < 0:
                    Sb_Alt = np.maximum(np.minimum(Sb, Kb), -1 * Kb)
                    capital_charge[i] = (np.sqrt(
                                            np.square(Kb).sum() + 
                                            np.dot(np.transpose(Sb_Alt),np.dot(Sb_Alt, corr_matrices[i]))
                                            ))
                    
        else:
            
            temp_K_by_bucket = intra_agg_result[0]
            n = len(temp_K_by_bucket)
            
            #Generate the correlation matrix
            medium_corr_matrix = np.full((n, n), 0.2 ** 2)
            for i in range(n):
                for j in range(n):
                    if i == j or (Sb_by_bucket.loc[i, "SA Bucket"] != 11 or Sb_by_bucket.loc[i, "SA Bucket"] != 11):
                        medium_corr_matrix[i, j] = 0
                        
            high_corr_matrix = np.minimum(1.25 * medium_corr_matrix, 1)
            low_corr_matrix = np.maximum(2 * medium_corr_matrix - 1, 0.75 * medium_corr_matrix)
            
            corr_matrices = [medium_corr_matrix, high_corr_matrix, low_corr_matrix]
            
            Sb = [[], [], []] #[[medium], [high], [low]]
            psi_matrix = [[], [], []] #[[medium], [high], [low]]
            
            for i in range(3):
                temp_K_by_bucket = intra_agg_result[i]
                
                for j in range(n):                
                    #Calculate Sb according to formulation in MAR21.5(4)
                    if temp_K_by_bucket["K"][j] == temp_K_by_bucket["K+"][j]:
                        Sb[i].append(temp_K_by_bucket["CVR+"][j])
                    elif temp_K_by_bucket["K"][j] == temp_K_by_bucket["K-"][j]:
                        Sb[i].append(temp_K_by_bucket["CVR-"][j])
                    elif temp_K_by_bucket["CVR+"][j] > temp_K_by_bucket["CVR-"][j]:
                        Sb[i].append(temp_K_by_bucket["CVR+"][j])
                    else:
                        Sb[i].append(temp_K_by_bucket["CVR-"][j])
            
                #Generate the psi matrix according to MAR21.5(4)(b)
                psi_matrix[i] = np.zeros((n, n))
                for k in range(n):
                    for l in range(n):
                        if Sb[i][k] >= 0 or Sb[i][l] >= 0:
                            psi_matrix[i][k, l] = 1
                
                #Store results                
                intra_agg_result[i]["Sb"] = Sb[i]
            
                #Calculate capital charge
                capital_charge.append(np.sqrt(
                                        np.maximum(0, 
                                        np.square(intra_agg_result[i]["K"]).sum() + 
                                        np.dot(Sb[i], np.dot(np.transpose(Sb[i]), np.multiply(corr_matrices[i], psi_matrix[i])))
                                        )))
        
        result = pd.DataFrame({"Medium": [capital_charge[0]], "High": [capital_charge[1]], "Low": [capital_charge[2]]}, index=[sensi_type])
        
        return result

# %%
class FX:
    
    #Risk weight table for Delta
    risk_weight = 0.15
    preferential_risk_weight = risk_weight / np.sqrt(2)
    
    def __init__(self, data, extra_prescribed_currency=None):
        #Assume the data has already been cleansed, filtered and grouped
        self.original_data = data
        self.data = data.copy()
        self.prescribed_currencies = ["AUD", "BRL", "CAD", "CHF", "CNY", "EUR", "GBP", "HKD", "INR", "JPY", "KRW", "MXN", "NOK", "NZD", "RUB", "SEK", "SGD", "TRY", "USD", "ZAR"] #These are the currencies that enjoy lower risk weights by scale of 1/sqrt(2)
        if extra_prescribed_currency != None and extra_prescribed_currency not in self.prescribed_currencies:
            self.prescribed_currencies.append(extra_prescribed_currency)
        
    def weight_sensitivities(self):
        self.data["Weighted Sensitivities"] = self.data["Sensitivity (reporting currency equiv.)"] * np.select([self.data["Sensi Type"] == "Delta"], [self.data["SA Bucket"].isin(self.prescribed_currencies).map({True: FX.preferential_risk_weight, False: FX.risk_weight})], default=1)

    def perform_intra_bucket_aggregation(self, sensi_type):
        #Filter data to only include targeted sensitivity type
        filtered_data = self.data[self.data["Sensi Type"] == sensi_type]
        
        #Aggregate data of the same sensitivities
        if sensi_type == "Delta":
            cols_to_grp = ["SA Bucket"]
        elif sensi_type == "Vega":
            cols_to_grp = ["SA Bucket", "Tenor"]
        else:
            cols_to_grp = ["SA Bucket", "CVR+/CVR-"]
        
        filtered_data = filtered_data.groupby(cols_to_grp)["Weighted Sensitivities"].sum().reset_index()
        
        #Create a list of unique buckets
        bucket_list = filtered_data["SA Bucket"].drop_duplicates()
        
        #Initiate variables to store K for each bucket
        temp_Kb_by_bucket = [[], [], []]   #[[medium], [high], [low]]
        K_by_bucket = []
        
        if sensi_type == "Delta":
            
            for i in range(3):
                temp_Kb_by_bucket[i].append(np.sqrt(np.maximum(0, filtered_data["Weighted Sensitivities"] ** 2)))
                K_by_bucket.append(pd.DataFrame({"SA Bucket": bucket_list, "K": temp_Kb_by_bucket[i][0]}))

        elif sensi_type == "Vega":
            
            #Initiate variables to store correlation matrix for each bucket
            corr_matrix_by_bucket = dict.fromkeys(bucket_list, None)
            
            for bucket in bucket_list:
                #Filter data for each bucket
                bucket_data = filtered_data[filtered_data["SA Bucket"] == bucket].reset_index(drop=True)
                weighted_sensi = np.asarray(bucket_data["Weighted Sensitivities"])
                
                n = len(bucket_data)
                temp_corr_matrix = np.identity(n)
                
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            #Calculate correlation based on option tenor under medium scenario
                            temp_corr = np.exp(
                                        -0.01 * np.abs(bucket_data.loc[i, "Tenor"] - bucket_data.loc[j, "Tenor"]) / 
                                        np.minimum(bucket_data.loc[i, "Tenor"], bucket_data.loc[j, "Tenor"]))
                            temp_corr_matrix[i, j] = np.minimum(temp_corr, 1)
                
                #High and low scenario
                temp_high_corr_matrix = np.minimum(1.25 * temp_corr_matrix, 1)
                temp_low_corr_matrix = np.maximum(2 * temp_corr_matrix - 1, 0.75 * temp_corr_matrix)
                
                #Store results
                corr_matrix_by_bucket[bucket] = [temp_corr_matrix, temp_high_corr_matrix, temp_low_corr_matrix]
                
                #Calculate and store result of Kb for each bucket 
                for i in range(3):
                    temp_Kb_by_bucket[i].append(np.sqrt(np.maximum(0, np.dot(np.transpose(weighted_sensi),np.dot(weighted_sensi, corr_matrix_by_bucket[bucket][i])))))
                
            for i in range(3):
                K_by_bucket.append(pd.DataFrame({"SA Bucket": bucket_list, "K": temp_Kb_by_bucket[i]}).reset_index(drop=True))
        
        else:
            
            #Compare whether upward or downward scenarios are applicable and calculate their values
            K_by_bucket = filtered_data.pivot_table(index="SA Bucket", columns="CVR+/CVR-", values="Weighted Sensitivities", aggfunc="sum").reset_index()
            K_by_bucket["K+"] = np.sqrt(np.maximum(K_by_bucket["CVR+"], 0) ** 2)
            K_by_bucket["K-"] = np.sqrt(np.maximum(K_by_bucket["CVR-"], 0) ** 2)
            K_by_bucket["K"] = np.maximum(K_by_bucket["K+"], K_by_bucket["K-"])
            
        return K_by_bucket
    
    def perform_across_bucket_aggregation(self, sensi_type, intra_agg_result):
        
        capital_charge = []
        
        if sensi_type != "Curvature":
            
            filtered_data = self.data[self.data["Sensi Type"] == sensi_type]
            
            #Calculate Sb according to formulation in MAR21.4(5)
            Sb_by_bucket = filtered_data.groupby("SA Bucket")["Weighted Sensitivities"].sum().reset_index()
            
            #Generate the correlation matrix
            n = len(Sb_by_bucket)
            medium_corr_matrix = np.full((n, n), 0.6)
            np.fill_diagonal(medium_corr_matrix, 0)
            
            high_corr_matrix = np.minimum(1.25 * medium_corr_matrix, 1)
            low_corr_matrix = np.maximum(2 * medium_corr_matrix - 1, 0.75 * medium_corr_matrix)
            
            corr_matrices = [medium_corr_matrix, high_corr_matrix, low_corr_matrix]
                            
            #Calculate the capital charge under each scenario [medium, high, low]
            for i in range(3):
                Kb = intra_agg_result[i]["K"]
                Sb = Sb_by_bucket["Weighted Sensitivities"]
                capital_charge.append(np.sqrt(
                                    np.square(Kb).sum() 
                                    + np.dot(np.transpose(Sb), np.dot(Sb, corr_matrices[i]))
                                    ))
                
                #For each scenario, we have to check whether an alternative formulation for Sb is required as stipulated in MAR21.4(5)
                if capital_charge[i] < 0:
                    Sb_Alt = np.maximum(np.minimum(Sb, Kb), -1 * Kb)
                    capital_charge[i] = (np.sqrt(np.square(Kb).sum() + np.dot(np.transpose(Sb_Alt),np.dot(Sb_Alt, corr_matrices[i]))))
                    
        
        else:
            
            n = len(intra_agg_result)
            
            #Generate the correlation matrix
            medium_corr_matrix = np.full((n, n), 0.6 ** 2)
            np.fill_diagonal(medium_corr_matrix, 0)
            
            high_corr_matrix = np.minimum(1.25 * medium_corr_matrix, 1)
            low_corr_matrix = np.maximum(2 * medium_corr_matrix - 1, 0.75 * medium_corr_matrix)
            
            corr_matrices = [medium_corr_matrix, high_corr_matrix, low_corr_matrix]
            
            Sb = []    
            for i in range(n):                
                #Calculate Sb according to formulation in MAR21.5(4)
                if intra_agg_result["K"][i] == intra_agg_result["K+"][i]:
                    Sb.append(intra_agg_result["CVR+"][i])
                elif intra_agg_result["K"][i] == intra_agg_result["K-"][i]:
                    Sb.append(intra_agg_result["CVR-"][i])
                elif intra_agg_result["CVR+"][i] > intra_agg_result["CVR-"][i]:
                    Sb.append(intra_agg_result["CVR+"][i])
                else:
                    Sb.append(intra_agg_result["CVR-"][i])
            
            #Generate the psi matrix according to MAR21.5(4)(b)
            psi_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if Sb[i] >= 0 or Sb[j] >= 0:
                        psi_matrix[i, j] = 1
            
            #Store results                
            intra_agg_result["Sb"] = Sb
            
            #Calculate capital charge
            for i in range(3):
                capital_charge.append(np.sqrt(
                                        np.maximum(0, 
                                        np.square(intra_agg_result["K"]).sum() + 
                                        np.dot(Sb, np.dot(np.transpose(Sb), np.multiply(corr_matrices[i], psi_matrix)))
                                        )))        
    
        result = pd.DataFrame({"Medium": [capital_charge[0]], "High": [capital_charge[1]], "Low": [capital_charge[2]]}, index=[sensi_type])
        
        return result


