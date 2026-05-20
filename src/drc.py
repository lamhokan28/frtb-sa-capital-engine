# %%
import pandas as pd
import numpy as np

# %%
class DRC_non:
    
    #Risk weight table for different credit quality band
    risk_weights = {"Zero": 0, "AAA": 0.005, "AA": 0.02, "A": 0.03, "BBB": 0.06, "BB": 0.15, "B": 0.3, "CCC": 0.5, "NR": 0.15, "Defaulted": 1}
    
    #LGD according to the seniority
    #LGD_map_string = {0: "Default Free", 1: "Covered Bonds", 2: "Senior Debt", 3: "Junior Debt", 4: "Equity"}
    #LGD_map = {"Default Free": 1, "Covered Bonds": 0.25, "Senior Debt": 0.75, "Junior Debt": 1, "Equity": 1}
    LGD_map = {"0": 1, "1": 0.25, "2": 0.75, "3": 1, "4": 1}
    
    def __init__(self, data, reporting_date):
        #Assume the data has already been cleansed, filtered and grouped
        self.original_data = data 
        self.gross_JTD_data = data.copy()
        self.net_JTD_data = pd.DataFrame
        self.reporting_date = reporting_date
        self.seniority_map = self.gross_JTD_data["Seniority Level"].drop_duplicates().to_list()
        self.seniority_map .sort()
        
    def calculate_gross_JTD(self):
        time_to_maturity = (self.gross_JTD_data["Maturity"] - self.reporting_date).dt.days / 365
        self.gross_JTD_data["Maturity Factor"] = np.where(self.gross_JTD_data["Instrument"] == "Equity", 0.25, np.clip(time_to_maturity, 0.25, 1))
        self.gross_JTD_data["LGD"] = self.gross_JTD_data["Seniority Level"].map(DRC_non.LGD_map)
        self.gross_JTD_data["Gross JTD (Long)"] = np.where(self.gross_JTD_data["Credit Exposure"] == "Long", np.maximum(self.gross_JTD_data["LGD"] * self.gross_JTD_data["Notional Amount"] + self.gross_JTD_data["P&L"], 0), 0) #* self.gross_JTD_data["Maturity Factor"]
        self.gross_JTD_data["Gross JTD (Short)"] = np.where(self.gross_JTD_data["Credit Exposure"] == "Short", np.minimum(self.gross_JTD_data["LGD"] * (-1 *self.gross_JTD_data["Notional Amount"]) + self.gross_JTD_data["P&L"], 0), 0) #* self.gross_JTD_data["Maturity Factor"]
    
    def calculate_net_JTD(self):
        #Organize data by each obligator by different seniority
        temp_data = pd.pivot_table(self.gross_JTD_data, index=["Obligator", "Bucket", "Credit Quality"], columns="Seniority Level", values=["Gross JTD (Long)", "Gross JTD (Short)"], aggfunc="sum").reset_index()
        temp_data.fillna(0, inplace=True)
        temp_data.columns = [col[0] if col[1] == "" else "_".join(col) for col in temp_data.columns]
        
        #map risk weights for each obligator
        temp_data.insert(temp_data.columns.get_loc("Credit Quality") + 1, "Risk Weights", temp_data["Credit Quality"].map(DRC_non.risk_weights))
        
        #calculate net JTD for each obligator
        #1. creates lists of gross JTDs for further handling
        m = len(self.seniority_map)
        temp_long_gross_JTD_list = temp_data[temp_data.columns[4: 4+m]].values.tolist()
        temp_short_gross_JTD_list = temp_data[temp_data.columns[4+m:]].values.tolist()
        
        #2. calculate eligible offset amount by applying rule: only short gross JTD of lower seniority can offset longs gross JTD. REF: MAR22.19(1)
        eligible_offset = []
        n = len(temp_long_gross_JTD_list)
        for i in range(n):
            temp_gross_JTD_list = np.add(temp_long_gross_JTD_list[i], temp_short_gross_JTD_list[i])
            temp_long_gross_JTD_list[i] = sum(temp_long_gross_JTD_list[i])
            temp_short_gross_JTD_list[i] = sum(temp_short_gross_JTD_list[i])
            counter = 1
            offset = 0
            for j in range(m):
                if temp_gross_JTD_list[j] > 0:
                    offset += np.minimum(temp_gross_JTD_list[j], np.abs(sum(x for x in temp_gross_JTD_list[counter:] if x < 0)))
                counter += 1
            eligible_offset.append(offset)
        
        #3. Calculate net JTD
        temp_data["Eligible Offset"] = eligible_offset
        temp_data["Net JTD (Long)"] = np.subtract(temp_long_gross_JTD_list, eligible_offset)
        temp_data["Net JTD (Short)"] = np.add(temp_short_gross_JTD_list, eligible_offset)
        temp_data["Weighted Net JTD (Long)"] = temp_data["Net JTD (Long)"] * temp_data["Risk Weights"]
        temp_data["Weighted Net JTD (Short)"] = temp_data["Net JTD (Short)"] * temp_data["Risk Weights"]
        
        self.net_JTD_data = temp_data
        
    def calculate_DRC_charge(self):
        bucket_list = ["CORPORATE", "SOVEREIGN", "MUNICIPAL"]
        HBR = []
        capital = []
        
        for bucket in bucket_list:
            temp_long_bucket_JTD = self.net_JTD_data.loc[self.net_JTD_data["Bucket"] == bucket, "Weighted Net JTD (Long)"].sum()
            temp_short_bucket_JTD = self.net_JTD_data.loc[self.net_JTD_data["Bucket"] == bucket, "Weighted Net JTD (Short)"].abs().sum()
            
            if temp_long_bucket_JTD == 0 and temp_short_bucket_JTD == 0:
                temp_HRB = 0
            else:
                temp_HRB = temp_long_bucket_JTD / (temp_long_bucket_JTD + temp_short_bucket_JTD)
            HBR.append(temp_HRB)
            
            capital.append(np.maximum(temp_long_bucket_JTD - temp_HRB * temp_short_bucket_JTD, 0))
        
        result = pd.DataFrame(data=capital, index=bucket_list, columns=["SA_DRC"])
            
        return result