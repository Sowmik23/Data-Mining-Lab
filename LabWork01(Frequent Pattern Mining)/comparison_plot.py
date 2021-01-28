import matplotlib.pyplot as plt 


# ******************Time Comparison***********************
# #Dataset: Kosarak(Sparse)
# min_support = [10,15, 20,50]
# time_apriori = [4.29819280599986, 2.97823431500001, 2.49032256700229, 7300138071.76141]
# time_fp_growth = [3.60179459799838, 3.37873382100224, 3.04892728100094, 2.22322553699996]


# #Dataset: Mashroom(dense)
# min_support = [30, 35, 40, 50]
# time_apriori = [12.5367931170003, 4.9370388719999, 2.19332706599926, 0.550905254000099]
# time_fp_growth = [8.42072265299976, 4.1854994309997, 2.29925594199995, 0.80422986800022]

# #Dataset: Accidents(Sparse)
# min_support = [60, 65, 70, 75]
# time_apriori = [435.132293744999, 263.693053963998, 125.629856619998, 83.2758132620002]
# time_fp_growth = [726.919775523002, 413.254671626997, 246.179431601002, 148.587553655001]



# #Dataset: Chess(Dense)
# min_support = [60, 65, 70, 75]
# time_apriori = [8046.45657767, 4032.543543, 2072.356681001, 272.728017232999]
# time_fp_growth = [639.250432118, 314.17196836, 155.412055985, 60.0156165509998]


# #Dataset: Pumsb_star(Dense+Large)
# min_support = [50, 60, 65, 70]
# time_apriori = [16.3935728650004, 4.92596917799983, 2.85134468300021, 1.35669185400002]
# time_fp_growth = [22.0244957970008, 7.58184116099983, 4.82421185800013, 1.4732617790005]



# # plotting the Apriori points 
# plt.plot(min_support, time_apriori, label = "Apriori", marker='o') 

# # plotting the FP-Growth points 
# plt.plot(min_support, time_fp_growth, label = "FP-Growth", marker='o') 

# # naming the axis 
# plt.xlabel('Minimum Support Threshold')  
# plt.ylabel('Time(sec.)') 

# plt.title('Dataset: Pumsb_star')  
# plt.legend() 
# plt.grid()
# plt.show() 




# ******************Memory Comparison***********************
# # Dataset: Kosarak
# min_support = [10,15, 20,50]
# memory_apriori = [0, 0, 0, 0]
# memory_fp_growth = [10842112, 10862592, 8433664, 0]


# # Dataset: Mashroom
# min_support = [30, 35, 40, 50]
# memory_apriori = [3702784, 1982464, 532480, 0]
# memory_fp_growth = [2547712, 1712128, 1531904, 360448]

# # Dataset: Accidents
# min_support = [60, 65, 70, 75]
# memory_apriori = [3624960, 1482752, 262144, 0]
# memory_fp_growth = [20787200, 9420800, 2347008, 44376064]


# # Dataset: Chess
# min_support = [60, 65, 70, 75]
# memory_apriori = [853534543, 483543543, 31993856, 16371712]
# memory_fp_growth = [16834560, 10588160, 5787648, 4280320]


# Dataset: Pumsb_star
min_support = [50, 60, 65, 70]
memory_apriori = [593920, 0, 0, 0]
memory_fp_growth = [8564736, 540672, 126976, 212992]


# plotting the Apriori points 
plt.plot(min_support, memory_apriori, label = "Apriori", marker='o') 

# plotting the FP-Growth points 
plt.plot(min_support, memory_fp_growth, label = "FP-Growth", marker='o') 

# naming the axis 
plt.xlabel('Minimum Support Threshold')  
plt.ylabel('Memory(bytes)') 

plt.title('Dataset: Pumsb_star')  
plt.legend() 
plt.grid()
plt.show() 
