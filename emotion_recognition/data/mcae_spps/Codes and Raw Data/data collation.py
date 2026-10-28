import pandas as pd


df1=pd.read_csv("raw_data_actor1.csv")
df2=pd.read_csv("raw_data_actor2.csv")
df3=pd.read_csv("raw_data_actor3.csv")
df4=pd.read_csv("raw_data_actor4.csv")
df5=pd.read_csv("raw_data_actor5.csv")
df6=pd.read_csv("raw_data_actor6.csv")

#merge six actors' raw data
df1 = pd.concat([df1, df2, df3, df4, df5,df6], axis=0)

#column name illustration:

# username: Participants' code, range 20230001 ~ 20230720;
# filename: "Filename" + ".wav" are audio file names;
# emotionresult: participant's emotion type selection result of each audio
# intensity: participant's intensity rating result of each audio, range 1~9


#extract audio emotion type from file name
df1['filename'] = df1['filename'].astype(str)
df1['emotioncode'] =df1['filename'].apply(lambda x: x[1:2])
df1['emotioncode'] = df1['emotioncode'].replace({"1": "neutral", "2": "happiness", "3": "anger","4": "fear"
                                                                    ,"5": "sadness","6": "disgust","7": "surprise"})
#extract sentence subject information
df1['sentencecode'] =df1['filename'].apply(lambda x: x[2:5])
df1['sentencecode'] = pd.to_numeric(df1['sentencecode'], errors='coerce')
column_data_type = df1['sentencecode'].dtype
bins = [0, 41, 81, 121,161,181,201]
labels = ['I', 'We', 'You','Yous','He','They']
df1['subjectname'] = pd.cut(df1['sentencecode'], bins=bins, labels=labels, right=False)

#extract actor code from file name
df1['actress'] =df1['filename'].apply(lambda x: x[0:1])

#match the participant's emotion type selection results and the original audio emotion type, 1 matched/correct, 0 unmatched/uncorrect
df1['correct'] = (df1['emotionresult'] == df1['emotioncode']).astype(int)
df1.to_csv("原始识别数据.csv", index=False)

#caculate orininal accuracy of each emotion type and each actor, see results in tabel 1
correctresult1 = df1.groupby(['emotioncode','actress'])['correct'].mean().reset_index()
correctresult1.columns = ['emotioncode', 'actress','accuracy']
print(correctresult1)

#caculate accuracy of each audio
audiocorrect1 = df1.groupby(['filename','actress'])['correct'].mean().reset_index()
audiocorrect1.columns = ['filename','actress', 'accuracy']
audiocorrect1_1=pd.merge(audiocorrect1,df1[['filename','emotioncode']],left_on='filename',right_on='filename',how='left')
audiocorrect1_1 = audiocorrect1_1.drop_duplicates(subset='filename') #delete duplicated rows

#caculate std of orininal accuracy of each emotion type and each actor, see results in tabel 1
correctresult2 = audiocorrect1_1.groupby(['emotioncode','actress'])['accuracy'].std().reset_index()
correctresult2.columns = ['emotioncode', 'actress','accuracy_std']
print(correctresult2)

#caculate Table 2: Confusion matrix
df1['emotionHucode'] = df1['emotioncode'].str.cat(df1['emotionresult'], sep='')
Huresult1 = df1.groupby(['actress'])['emotionHucode'].value_counts()
Huresult1.columns = ['actress','emotionHucode','counts']
print(Huresult1) #this is the results of table 2 Confusion matrix data

#according to the audio inclusion criteria to include high accuracy audio

    # (1) a recognition accuracy of at least 43% for the target emotion—equivalent to three times the chance level in a seven-alternative forced-choice task;
    # (2) less than 43% of responses attributed to any non-target emotion category.

audiocorrect_all = df1.groupby('filename')['correct'].mean().reset_index()
audiocorrect_all.columns = ['filename', 'accuracy']
audiocorrect1_1 = audiocorrect_all.merge(
    df1[['filename','emotioncode','actress']].drop_duplicates('filename'),
    on='filename',
    how='left'
)
df_counts = df1.groupby(['filename', 'emotionresult']).size().reset_index(name='count')
df_total = df_counts.groupby('filename')['count'].sum().reset_index(name='total')
df_counts = pd.merge(df_counts, df_total, on='filename', how='left')
df_counts['rate'] = df_counts['count'] / df_counts['total']
tmp = df_counts.merge(
    audiocorrect1_1[['filename', 'accuracy', 'emotioncode']],
    on='filename',
    how='left'
)
tmp['is_other'] = tmp['emotionresult'] != tmp['emotioncode']

other_max = tmp[tmp['is_other']].groupby('filename')['rate'].max().reset_index()
other_max.rename(columns={'rate': 'other_max_rate'}, inplace=True)

final_check = audiocorrect1_1[['filename', 'accuracy']].merge(
    other_max, on='filename', how='left'
)
final_check['other_max_rate'] = final_check['other_max_rate'].fillna(0)

final_keep = final_check[
    (final_check['accuracy'] > 0.45) &
    (final_check['other_max_rate'] < 0.45)
]
final_corpus = df1[df1['filename'].isin(final_keep['filename'])] #this is the final corpus
df_word_1correct=final_corpus #this is the final included corpus data,267000 rows/40= 6675 audios

#list of delete audios
all_files = set(audiocorrect1_1['filename'])
keep_files = set(final_keep['filename'])
delete_files = all_files - keep_files
sent_delete = audiocorrect1_1[audiocorrect1_1['filename'].isin(delete_files)].copy()#This is delete audio list, n = 1704
print("number of delete audios：", len(sent_delete))
print(sent_delete.head())

word_delete = pd.merge(
    df1,
    sent_delete[['filename', 'emotioncode']],
    on='filename',
    how='right'
)
word_delete = word_delete.groupby(
    ['actress', 'emotioncode_x']
)['emotionresult'].count().reset_index()
word_delete.columns = ['actress', 'emotioncode', 'delete_count']
# word_delete.to_excel('delete_count.xlsx') #This is used to count the delete number of audios in table 3

finalcorpuscount = final_corpus.groupby(
    ['actress', 'emotioncode']
)['emotionresult'].count().reset_index()
finalcorpuscount.columns = ['actress', 'emotioncode_y', 'delete_count']
finalcorpuscount.to_excel('finalcorpuscount.xlsx')#This is used to count the final corpus number of audios in table 3
print(word_delete)

#match participants' group information
#720 participants randamly divided into 18 groups, each actor's audio was divided into 3 parts, the "group informaiton.xlsx" is the specific group of each audio
groupinfor=pd.read_excel("group information.xlsx",sheet_name="Sheet2")
groupinfor['filename'] = groupinfor['filename'].astype(str)
df_word_1correct=pd.merge(df_word_1correct,groupinfor,left_on="filename", right_on="filename", how="left")
print(df_word_1correct)
# df_word_1correct.to_csv("df_word_1correct.csv")#this is the final included corpus data,267000 rows/40= 6675 audios

#intensity comparison between different emotion type
intensity_word=df_word_1correct.groupby(['username','emotioncode'])['intensity'].mean().reset_index()

#final audio information excel preparation
df2=df_word_1correct
word_lastversion_1=df2.groupby(['filename','actress','emotioncode'])['intensity'].mean().reset_index()
word_lastversion_2=df2.groupby(['filename','actress','emotioncode'])['correct'].mean().reset_index()
word_lastversion=pd.merge(word_lastversion_1, word_lastversion_2[['filename','correct']], left_on="filename", right_on="filename", how="left")
word_lastversion=pd.merge(word_lastversion,df_word_1correct[['filename','groupnumber']],left_on="filename", right_on="filename", how="left")
word_lastversion = word_lastversion.drop_duplicates(subset='filename') #删除重复的行
print(word_lastversion)
word_lastversion['sentencecode']= word_lastversion['filename'].apply(lambda x: x[2:5])
word_lastversion['sentencecode'] = pd.to_numeric(word_lastversion['sentencecode'], errors='coerce') #转化为数值，方便根据数值进行分类
column_data_type = word_lastversion['sentencecode'].dtype
print("Data Type of NumericColumn:", column_data_type)
bins = [0, 41, 81, 121,161,181,201]
labels = ['I', 'We', 'You','Yous','He','They']
word_lastversion['subjectname'] = pd.cut(word_lastversion['sentencecode'], bins=bins, labels=labels, right=False)
sentence_content=pd.read_excel("Appendix1-Sentence material.xlsx")
word_lastversion=pd.merge(word_lastversion,sentence_content[['sentencecode','sentence_content']],left_on="sentencecode", right_on="sentencecode", how="left")

  #the count of selection of each type of emotion of each audio
emotioncat = pd.crosstab(index=df_word_1correct['filename'], columns=df_word_1correct['emotionresult'])
word_lastversion=pd.merge(word_lastversion, emotioncat, left_on="filename", right_on="filename", how="left")

# word_lastversion.to_excel("audio information.xlsx")