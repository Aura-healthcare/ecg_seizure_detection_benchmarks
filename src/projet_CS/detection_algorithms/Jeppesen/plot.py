from Jeppesen import RRfetch,calcul_features, calcul_seuils, noseizurejson
import matplotlib.pyplot as plt
from custom_features import filter_median

rr_intervals_list = RRfetch("C:/Users/thibault/Documents/CS 1A/Pôle projet IA/Github P25/Projet-Epilepsie/database_processed/train/00001006/rr_00001006_s001_t001.csv")
rr_intervals_list_seuil=noseizurejson("C:/Users/thibault/Documents/CS 1A/Pôle projet IA/Github P25/Projet-Epilepsie/database_processed/train/00001006/00001006_s001_t001.json",
"C:/Users/thibault/Documents/CS 1A/Pôle projet IA/Github P25/Projet-Epilepsie/database_processed/train/00001006/rr_00001006_s001_t001.csv")
x=[]
for i in range(0,len(rr_intervals_list)):
    x.append(i)

plt.plot(x,filter_median(rr_intervals_list,7))
plt.plot(x,calcul_features(filter_median(rr_intervals_list,7))[-2][:])
seuil = 1.05*calcul_seuils(filter_median(rr_intervals_list_seuil,7))[-2]
listeseuil=[seuil for i in range(0,len(rr_intervals_list))]
plt.plot(x,listeseuil)
plt.xlabel("Numéro Intervalle RR")
plt.legend(["Durée intervalle RR (en ms)","Modified_CSi*Slope","Valeur seuil"])
plt.yscale('log')
plt.show()
