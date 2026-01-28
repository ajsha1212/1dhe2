# Recommendation System Project –(Model Lead)

Ky modul është përgjegjës për trajnimin e modelit të Machine Learning (SVD) duke përdorur të dhënat e përpunuara .

Kodi:
- lexon dataset-in e përpunuar (ratings.csv)
- ndan të dhënat në train dhe test
- trajnon modelin SVD (Singular Value Decomposition)
- llogarit performancën me RMSE
- ruan modelin dhe metrikat në file

---

## 📥 Input 

File i përdorur:

Kolonat e nevojshme:
- user_id
- movie_id
- rating  
(timestamp injorohet)

---

## 📤 Output (nga Person 2)

Kodi gjeneron këto file:


Kolonat e nevojshme:
- user_id
- movie_id
- rating  
(timestamp injorohet)

---

## 📤 Output (nga Person 2)

Kodi gjeneron këto file:

model_lead/models/svd_model_long.pkl  
model_lead/models/metrics_long.json  

Ku:
- svd_model_long.pkl = modeli i trajnuar
- metrics_long.json = rezultatet e vlerësimit (RMSE)
