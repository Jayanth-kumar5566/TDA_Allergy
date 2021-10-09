b_data=read.csv("./../Results/bronchieactasis_data.csv",row.names=1)

c_data=read.csv("./../Results/COPD.csv",row.names=1)

data=rbind(b_data,c_data)

allergens=colnames(data)

labels_b=read.csv("./../../Bronchiectasis_Clinical_Metadata_V3.4.csv",row.names=1)
labels_c=read.csv("./../../COPD_Clinical_Metadata_V3.3_tpyxlsx.csv",row.names=1)
labels_b=labels_b[row.names(b_data),]
labels_c=labels_c[row.names(c_data),]
#Differntial allergens for Bronchieactasis, COPD and BCOS

labels_b=labels_b[["BCOS"]]
levels(labels_b)<-c(0,1)
labels_c=labels_c[["COPD_Comorbidities_CoExisting_Bronchiectasis"]]
levels(labels_c)<-c(2,1)
#Create new column names and concat
labels=unlist(list(labels_b,labels_c))
#O - bronchieactasis, 1- BCOS and 2 - COPD

#======Statistical testing=====
p_val<-c()
for (i in allergens){t<-kruskal.test(x=data[[i]],g=labels);p_val<-c(p_val,t$p.value)}

#Multiple testing correction
corrected_pval<-p.adjust(p_val,method='BH')

cut_off=0.1

selected_data<-data[,corrected_pval<=cut_off]
selected_data[["labels"]]<-labels
write.csv(selected_data,"./../Results/feat_sel_data.csv")
