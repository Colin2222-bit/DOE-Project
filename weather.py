import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('Agg')
import re
import statsmodels.api as sm
from statsmodels.formula.api import poisson
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
reader = pd.read_csv("orps.csv")
hq = reader["HQ Summary"]



#snowyvalues = reader.loc[reader.astype(str).apply(lambda row: row.str.contains("snow|cold", case=False, na=False).any(), axis=1)]
idavalues=reader.loc[(reader["Keywords"].str.contains("11D",flags=re.IGNORECASE,regex=True)) & 
(reader["HQ Summary"].str.contains("cold|snow|freez|rain|ice",flags=re.IGNORECASE,regex=True)) & 
(reader["Site"].isin(["Idaho National Laboratory"]))].copy()


idavalues.to_csv("ida.csv")


idavalues["Occurrence Date"] = pd.to_datetime(reader["Occurrence Date"], format='%m-%d-%Y')
idavalues['year'] = idavalues["Occurrence Date"].dt.year
idavalues = idavalues[idavalues['year'].between(2010, 2024) & (idavalues['year'] != 2013)]

years = [y for y in range(2010, 2025) if y != 2013]


yearly_counts = idavalues["year"].value_counts().reindex(years, fill_value=0).sort_index()





plt.figure(figsize=(12,6))
plt.plot(years, yearly_counts, marker="o", linestyle="-")
plt.xlabel("Year")
plt.ylabel("Number of Cold Weather Reports")
plt.title("Cold Weather Incidence Reports Per Year")
plt.grid(True)
plt.savefig("idaplot.png")


santafevalues=reader.loc[(reader["Keywords"].str.contains("11D",flags=re.IGNORECASE,regex=True)) & 
(reader["HQ Summary"].str.contains("cold|snow|freez|rain|ice",flags=re.IGNORECASE,regex=True)) & 
(reader["Site"].isin(["Los Alamos National Laboratory"]))].copy()
santafevalues.to_csv("santafe.csv")
santafevalues["Occurrence Date"] = pd.to_datetime(reader["Occurrence Date"], format='%m-%d-%Y')
santafevalues['year'] = santafevalues["Occurrence Date"].dt.year
santafevalues = santafevalues[santafevalues['year'].between(2010, 2024)]
nmyears = [y for y in range(2010, 2025)]
nmyc = santafevalues["year"].value_counts().reindex(nmyears, fill_value=0).sort_index()





plt.figure(figsize=(12,6))
plt.plot(nmyears, nmyc, marker="o", linestyle="-")

plt.xlabel("Year")
plt.ylabel("Number of Cold Weather Reports")
plt.title("Cold Weather Incidence Reports Per Year in Los Alamos Laboratory")
plt.grid(True)
plt.savefig("santafeplot.png")

'''
vectorizer = TfidfVectorizer(stop_words='english')

hq = idavalues["HQ Summary"].astype(str)
tfidf_matrix = vectorizer.fit_transform(hq)
query = "slip ice snow freezing icy cold winter sidewalk"
query_vector = vectorizer.transform([query])
cos_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
top_n = 15
top_indices = cos_similarities.argsort()[-top_n:][::-1]
#print("Top descriptions similar to summary':\n")
#print(hq.iloc[top_indices])

'''








nm = pd.read_csv('nmdata.csv')

nmavg = nm['TAVG']
nmmin = nm ['TMIN']
plt.figure(figsize=(12,6))
plt.plot(nm['DATE'], nm['TAVG'], label='Average Temperature (TAVG)', color='orange')
plt.plot(nm['DATE'], nm['TMIN'], label='Minimum Temperature (TMIN)', color='blue')

plt.xlabel('Year')
plt.ylabel('Temperature (°F)')
plt.title('Average and Minimum Temperatures Over Time in Los Alamos National Laboratory')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("SantaFe.png")

ida = pd.read_csv('idahodata.csv')

idavg = ida['TAVG']
idmin = ida ['TMIN']
idfreeze=ida['DX32']
plt.figure(figsize=(12,6))
plt.plot(ida['DATE'], ida['TAVG'], label='Average Temperature (TAVG)', color='orange')
plt.plot(ida['DATE'], ida['TMIN'], label='Minimum Temperature (TMIN)', color='blue')

plt.xlabel('Year')
plt.ylabel('Temperature (°F)')
plt.title('Average and Minimum Temperatures Over Time in Idaho')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("idaho.png")

mask = ~np.isnan(idavg)
minmask = ~np.isnan(idmin)
freezemask = ~np.isnan(idfreeze)
idavgmask = idavg[mask]
idminmask=idmin[mask]
idfreezemask = idfreeze[freezemask]

model = LinearRegression()
X = nmyc.values.reshape(-1, 1)
model.fit(X, nmmin)
slope = model.coef_[0]
intercept = model.intercept_
print(f"Slope: {slope}, Intercept: {intercept}")
r_sq = model.score(X, nmmin)
print(f"R-squared: {r_sq}")

pearsoncorrnm, pearsonpnm = pearsonr(yearly_counts, idavgmask)
print(f"pearson for avg: {pearsoncorrnm}, p-value avg: {pearsonpnm}")

pearsoncorrnm, pearsonpnm = pearsonr(yearly_counts, idminmask)
print(f"pearson for min: {pearsoncorrnm}, p-value min : {pearsonpnm}")

pearsoncorrnm, pearsonpnm = pearsonr(yearly_counts, idfreezemask)
print(f"pearson for temps <32: {pearsoncorrnm}, p-value <32: {pearsonpnm}")


spearmancorr, spearmanp = spearmanr(yearly_counts, idavgmask)
print(f"Spearman correlation: {spearmancorr:.3f}, p-value: {spearmanp:.3f}")


spearmancorr, spearmanp = spearmanr(yearly_counts, idminmask)
print(f"Spearman correlation: {spearmancorr:.3f}, p-value: {spearmanp:.3f}")

spearmancorr, spearmanp = spearmanr(yearly_counts, idfreezemask)
print(f"Spearman correlation: {spearmancorr:.3f}, p-value: {spearmanp:.3f}")









pearsoncorr, pearsonp = pearsonr(nmyc, nmavg)
print(f"pearson for avg: {pearsoncorr}, p-value avg: {pearsonp}")

pearsoncorr, pearsonp = pearsonr(nmyc, nmmin)
print(f"pearson for min: {pearsoncorr}, p-value min : {pearsonp}")

pearsoncorr, pearsonp = pearsonr(nmyc, nm['DX32'])
print(f"pearson for temps <32: {pearsoncorr}, p-value <32: {pearsonp}")


spearmancorr, spearmanp = spearmanr(nmyc, nmavg)
print(f"Spearman correlation: {spearmancorr:.3f}, p-value: {spearmanp:.3f}")


spearmancorr, spearmanp = spearmanr(nmyc, nmmin)
print(f"Spearman correlation: {spearmancorr:.3f}, p-value: {spearmanp:.3f}")

spearmancorr, spearmanp = spearmanr(nmyc, nm['DX32'])
print(f"Spearman correlation: {spearmancorr:.3f}, p-value: {spearmanp:.3f}")


'''
data = pd.DataFrame({
    'incident_reports': nmyc,
    'avg_temp': nmavg,
    'min_temp': nmmin
})
print(data)

# Step 4: Fit Poisson regression models to find the relationship
# For average temperature correlation
poisson_model_avg = poisson("incident_reports ~ avg_temp", data).fit()

# For minimum temperature correlation
poisson_model_min = poisson("incident_reports ~ min_temp", data).fit()

# Step 5: Output the results
print("Poisson Regression Result for Average Temperature:")
print(poisson_model_avg.summary())

print("\nPoisson Regression Result for Minimum Temperature:")
print(poisson_model_min.summary())
'''


print("end")