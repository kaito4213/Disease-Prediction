from tkinter import *
from control import *
import pandas as pd
import numpy as np
from tkinter import messagebox
from Model import *

def show_result():
	BENE_SEX_IDENT_CD_1, BENE_SEX_IDENT_CD_2 = set_gender(gender.get())
	BENE_RACE_CD_1,BENE_RACE_CD_2,BENE_RACE_CD_3,BENE_RACE_CD_5 = set_race(race.get())
	BENE_ESRD_IND_0,BENE_ESRD_IND_Y = set_ESRD(ESRD.get())
	statelist = set_state(state.get())

	ALZHDMTA = set_disease(SP_ALZHDMTA.get())
	CHF = set_disease(SP_CHF.get())
	CHRNKIDN = set_disease(SP_CHRNKIDN.get())
	CNCR = set_disease(SP_CNCR.get())
	COPD = set_disease(SP_COPD.get())
	DEPRESSN = set_disease(SP_DEPRESSN.get())
	DIABETES = set_disease(SP_DIABETES.get())
	ISCHMCHT = set_disease(SP_ISCHMCHT.get())
	OSTEOPRS = set_disease(SP_OSTEOPRS.get())
	RA_OA = set_disease(SP_RA_OA.get())
	STRKETIA = set_disease(SP_STRKETIA.get())

	SP_STATE_CODE_1 = statelist[0]
	SP_STATE_CODE_2 = statelist[1]
	SP_STATE_CODE_3 = statelist[2]
	SP_STATE_CODE_4 = statelist[3]
	SP_STATE_CODE_5 = statelist[4]
	SP_STATE_CODE_6 = statelist[5]
	SP_STATE_CODE_7 = statelist[6]
	SP_STATE_CODE_8 = statelist[7]
	SP_STATE_CODE_9 = statelist[8]
	SP_STATE_CODE_10 = statelist[9]
	SP_STATE_CODE_11 = statelist[10]
	SP_STATE_CODE_12 = statelist[11]
	SP_STATE_CODE_13 = statelist[12]
	SP_STATE_CODE_14 = statelist[13]
	SP_STATE_CODE_15 = statelist[14]
	SP_STATE_CODE_16 = statelist[15]
	SP_STATE_CODE_17 = statelist[16]
	SP_STATE_CODE_18 = statelist[17]
	SP_STATE_CODE_19 = statelist[18]
	SP_STATE_CODE_20 = statelist[19]
	SP_STATE_CODE_21 = statelist[20]
	SP_STATE_CODE_22 = statelist[21]
	SP_STATE_CODE_23 = statelist[22]
	SP_STATE_CODE_24 = statelist[23]
	SP_STATE_CODE_25 = statelist[24]
	SP_STATE_CODE_26 = statelist[25]
	SP_STATE_CODE_27 = statelist[26]
	SP_STATE_CODE_28 = statelist[27]
	SP_STATE_CODE_29 = statelist[28]
	SP_STATE_CODE_30 = statelist[29]
	SP_STATE_CODE_31 = statelist[30]
	SP_STATE_CODE_32 = statelist[31]
	SP_STATE_CODE_33 = statelist[32]
	SP_STATE_CODE_34 = statelist[33]
	SP_STATE_CODE_35 = statelist[34]
	SP_STATE_CODE_36 = statelist[35]
	SP_STATE_CODE_37 = statelist[36]
	SP_STATE_CODE_38 = statelist[37]
	SP_STATE_CODE_39 = statelist[38]
	SP_STATE_CODE_41 = statelist[39]
	SP_STATE_CODE_42 = statelist[40]
	SP_STATE_CODE_43 = statelist[41]
	SP_STATE_CODE_44 = statelist[42]
	SP_STATE_CODE_45 = statelist[43]
	SP_STATE_CODE_46 = statelist[44]
	SP_STATE_CODE_47 = statelist[45]
	SP_STATE_CODE_49 = statelist[46]
	SP_STATE_CODE_50 = statelist[47]
	SP_STATE_CODE_51 = statelist[48]
	SP_STATE_CODE_52 = statelist[49]
	SP_STATE_CODE_53 = statelist[50]
	SP_STATE_CODE_54 = statelist[51]

	test_X = [visiting_time_before.get(),ALZHDMTA,CHF,CHRNKIDN,CNCR,COPD, DEPRESSN,DIABETES,ISCHMCHT,OSTEOPRS,RA_OA,STRKETIA, \
		AGE.get(), BENE_SEX_IDENT_CD_1,BENE_SEX_IDENT_CD_2,BENE_RACE_CD_1,BENE_RACE_CD_2,BENE_RACE_CD_3,BENE_RACE_CD_5,BENE_ESRD_IND_0,BENE_ESRD_IND_Y,\
		SP_STATE_CODE_1,SP_STATE_CODE_2,SP_STATE_CODE_3,SP_STATE_CODE_4,SP_STATE_CODE_5,SP_STATE_CODE_6,\
		SP_STATE_CODE_7,SP_STATE_CODE_8,SP_STATE_CODE_9,SP_STATE_CODE_10,SP_STATE_CODE_11,SP_STATE_CODE_12,\
		SP_STATE_CODE_13,SP_STATE_CODE_14,SP_STATE_CODE_15,SP_STATE_CODE_16,SP_STATE_CODE_17,SP_STATE_CODE_18,\
		SP_STATE_CODE_19,SP_STATE_CODE_20,SP_STATE_CODE_21,SP_STATE_CODE_22,SP_STATE_CODE_23,SP_STATE_CODE_24,\
		SP_STATE_CODE_25,SP_STATE_CODE_26,SP_STATE_CODE_27,SP_STATE_CODE_28,SP_STATE_CODE_29,SP_STATE_CODE_30,\
		SP_STATE_CODE_31,SP_STATE_CODE_32,SP_STATE_CODE_33,SP_STATE_CODE_34,SP_STATE_CODE_35,SP_STATE_CODE_36,\
		SP_STATE_CODE_37,SP_STATE_CODE_38,SP_STATE_CODE_39,SP_STATE_CODE_41,SP_STATE_CODE_42,SP_STATE_CODE_43,\
		SP_STATE_CODE_44,SP_STATE_CODE_45,SP_STATE_CODE_46,SP_STATE_CODE_47,SP_STATE_CODE_49,SP_STATE_CODE_50,\
		SP_STATE_CODE_51,SP_STATE_CODE_52,SP_STATE_CODE_53,SP_STATE_CODE_54]

	test_X = np.array(test_X).reshape(1,-1)
	result = myControl.prediction(test_X)
	messagebox.showinfo("Diagnose",result)
	

def set_gender(gender):
	if gender == "male":
		return 1,0
	else:
		return 0,1

def set_race(race):
	if race == "American Indian or Alaska Native":
		return 1,0,0,0
	elif race == "Asian":
		return 0,1,0,0
	elif race == "Black or African American":
		return 0,0,1,0
	elif race == "White":
		return 0,0,0,1

def set_ESRD(ESRD):
	if ESRD == 0:
		return 1,0
	else:
		return 0,1

def set_state(state):
	state_map = {"AK" : 0,"AL" : 1,"AR": 2,"AZ" : 3,"CA" : 4,"CO" : 5,"CT" : 6,"DE" : 7,"FL" : 8,\
	"GA" : 9,"GU" : 10,"HI" : 11,"IA" : 12,"ID" : 13,"IL" : 14,"IN" : 15,"KS" : 16,"KY" : 17,\
	"LA" : 18,"MA" : 19,"MD" : 20,"ME" : 21,"MI" : 22,"MN" : 23,"MO" : 24,"MP" : 25,"MS" : 26,"MT" : 27,\
	"NC" : 28,"ND" : 29,"NE" : 30,"NH" : 31,"NJ" : 32,"NM": 33,"NV" :34,"NY" : 35, "OH" : 36,"OK" : 37,\
	"OR" : 38,"PA" : 39,"PR": 40,"RI" : 41,"SC" : 42,"SD" : 43,"TN" : 44,"TX" : 45,"UT" : 45,"VA" : 46,\
	"VI" : 47,"VT" : 48,"WA": 49,"WI": 50,"WV" : 51,"WY" : 52}

	statelist = list(0 for i in range(52))
	statelist[state_map[state]] = 1
	return statelist

def set_disease(sp):
	if sp == 0:
		return 1
	else:
		return 2

data = np.load('FC_data.npy')
myControl = control(data)


master = Tk()
master.title("Prediction of potential disease")
Label(master, text="Personal Information").grid(row=0,sticky=W)

Label(master, text="First Name").grid(row=1,sticky=W)
firstname = Entry(master).grid(row=1)

Label(master, text="Last Name").grid(row=2, sticky=W)
lastname = Entry(master).grid(row=2)

Label(master, text="Age").grid(row=3, sticky=W)
AGE = Entry(master)
AGE.grid(row=3)

Label(master, text="Race").grid(row=4, sticky=W)
race = StringVar(master)
race.set("American Indian or Alaska Native") # default value
OptionMenu(master,race, "American Indian or Alaska Native", "Asian","Black or African American", \
	"Native Hawaiian or Other Pacific Islander","White").grid(row = 4)

Label(master, text="Gender").grid(row=5, sticky=W)
gender = StringVar(master)
gender.set("male") # default value
OptionMenu(master,gender, "male", "female").grid(row = 5)

Label(master, text="State").grid(row=6, sticky=W)
state = StringVar(master)
state.set("AK") # default value
OptionMenu(master,state, \
"AK","AL","AR","AZ","CA","CO","CT","DE","FL","GA","GU","HI","IA","ID","IL","IN","KS","KY", \
"LA","MA","MD","ME","MI","MN","MO","MS","MT","NC","ND","NE","NH","NJ","NM","NV","NY",\
"OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VA","VI","VT","WA","WI","WV","WY").grid(row = 6)

Label(master, text="How many claims have you made in this year? ").grid(row=7, sticky=W)
visiting_time_before = Entry(master)
visiting_time_before.grid(row=8, sticky=W)

Label(master, text="Check all the symptoms that you're currently experiencing:").grid(row=13, sticky=W)
SP_ALZHDMTA = IntVar()
Checkbutton(master, text="Alzheimer or related disorders or senile", variable=SP_ALZHDMTA).grid(row=14,sticky=W)
SP_CHF = IntVar()
Checkbutton(master, text="Heart Failure", variable=SP_CHF).grid(row=15, sticky=W)
SP_CHRNKIDN = IntVar()
Checkbutton(master, text="Chronic Kidney Disease", variable=SP_CHRNKIDN).grid(row=16, sticky=W)
SP_CNCR = IntVar()
Checkbutton(master, text="Cancer", variable=SP_CNCR).grid(row=17, sticky=W)
SP_COPD = IntVar()
Checkbutton(master, text="Chronic Obstructive Pulmonary Disease", variable=SP_COPD).grid(row=18, sticky=W)
SP_DEPRESSN = IntVar()
Checkbutton(master, text="Depression", variable=SP_DEPRESSN).grid(row=19, sticky=W)
SP_DIABETES = IntVar()
Checkbutton(master, text="Diabetes", variable=SP_DIABETES).grid(row=20, sticky=W)
SP_ISCHMCHT = IntVar()
Checkbutton(master, text="Ischemic Heart Disease", variable=SP_ISCHMCHT).grid(row=21, sticky=W)
SP_OSTEOPRS = IntVar()
Checkbutton(master, text="Osteoporosis", variable=SP_OSTEOPRS).grid(row=22, sticky=W)
SP_RA_OA = IntVar()
Checkbutton(master, text="Rheumatoid arthritis and osteoarthritis (RA/OA)", variable=SP_RA_OA).grid(row=23, sticky=W)
SP_STRKETIA = IntVar()
Checkbutton(master, text="Stroke/transient Ischemic Attack", variable=SP_STRKETIA).grid(row=24, sticky=W)
ESRD = IntVar()
Checkbutton(master, text="End stage renal disease", variable=ESRD).grid(row=25, sticky=W)

Button(master, text='submit', command=show_result).grid(row=26, sticky=W, pady=4)
Button(master, text='quit', command=master.quit).grid(row=26, pady=4)

mainloop( )