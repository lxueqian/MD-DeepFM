import os
import sys
import click
import random
import pandas as pd
import collections
import json

text_vector = 50
continous_features = [i for i in range(26+text_vector)]
categorial_features = [i for i in range(26+text_vector,36+text_vector)]

# Clip integer features. The clip point for each integer feature
# is derived from the 95% quantile of the total values in each feature
# continous_clip = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]


class CategoryDictGenerator:
    """
    Generate dictionary for each of the categorical features
    """

    def __init__(self, num_feature):
        self.dicts = []
        self.num_feature = num_feature
        for i in range(0, num_feature):
            self.dicts.append(collections.defaultdict(int))

    def build(self, datafile, categorial_features, cutoff=-100000):
        with open(datafile, 'r') as f:
            flag = 0
            for line in f:
                features = line.rstrip('\n').split(',')
                if not flag: 
                    flag=1
                    continue
                for i in range(0, self.num_feature):
                    if features[categorial_features[i]] != '':
                        self.dicts[i][features[categorial_features[i]]] += 1
                        
        for i in range(0, self.num_feature):
            self.dicts[i] = filter(lambda x: x[1] >= cutoff,
                                   self.dicts[i].items())
            # print('begin...',self.dicts[i])
            self.dicts[i] = sorted(self.dicts[i], key=lambda x: (-x[1], x[0]))
            # print('ing...',self.dicts[i])
            vocabs, _ = list(zip(*self.dicts[i]))
            # print('vocabs...',vocabs)
            self.dicts[i] = dict(zip(vocabs, range(1, len(vocabs) + 1)))
            self.dicts[i]['<unk>'] = 0
            # break

    def gen(self, idx, key):
        if key not in self.dicts[idx]:
            res = self.dicts[idx]['<unk>']
        else:
            res = self.dicts[idx][key]
        return res

    def dicts_sizes(self):
        return [len(self.dicts[idx]) for idx in range(0, self.num_feature)]


class ContinuousFeatureGenerator:
    """
    Clip continuous features.
    """

    def __init__(self, num_feature):
        self.num_feature = num_feature

    def build(self, datafile, continous_features):
        # features = pd.read_csv(datafile)
        with open(datafile, 'r') as f:
            flag = 0
            for line in f:
                features = line.rstrip('\n').split(',')
                if not flag: 
                    flag=1
                    continue
                for i in range(0, self.num_feature):
                    # print(continous_features[i],features[continous_features[i]])
                    val = features[continous_features[i]]
                    if val != '':
                        val = float(val)
                        # if val > continous_clip[i]:
                        #     val = continous_clip[i]

    def gen(self, idx, val):
        if val == '':
            return 0.0
        val = float(val)
        return val


# @click.command("preprocess")
# @click.option("--datadir", type=str, help="Path to raw criteo dataset")
# @click.option("--outdir", type=str, help="Path to save the processed data")
def preprocess(datadir, outdir, num_train_sample = 20000, num_test_sample = 10000):
    """
    All the 13 integer features are normalzied to continous values and these
    continous features are combined into one vecotr with dimension 13.
    Each of the 26 categorical features are one-hot encoded and all the one-hot
    vectors are combined into one sparse binary vector.
    """
    dists = ContinuousFeatureGenerator(len(continous_features))
    dists.build(datadir, continous_features)

    dicts = CategoryDictGenerator(len(categorial_features))
    dicts.build(datadir, categorial_features, cutoff=-1000000)

    dict_sizes = dicts.dicts_sizes()
    # with open(os.path.join(outdir, 'feature_sizes.txt'), 'w') as feature_sizes:
    #     sizes = [1] * len(continous_features) + dict_sizes
    #     sizes = [str(i) for i in sizes]
    #     feature_sizes.write(','.join(sizes))

    # random.seed(0)

    # Saving the data used for training.
    train_list = []
    label_list = []
    with open(os.path.join(outdir, 'train_0904_addtext50.txt'), 'w') as out_train:
        with open(datadir, 'r') as f:
            flag=0

            for ii,line in enumerate(f.readlines()):
                features = line.rstrip('\n').split(',')
                if not flag: 
                    flag=1
                    continous_vals = [features[i] for i in continous_features]
                    categorial_vals = [features[i] for i in categorial_features]
                    continous_vals = ','.join(continous_vals)
                    categorial_vals = ','.join(categorial_vals)
                    index = 'label'
                    out_train.write(','.join([continous_vals, categorial_vals, index]) + '\n')
                else:
                    continous_vals = []
                    for i in range(0, len(continous_features)):
                        val = dists.gen(i, features[continous_features[i]])
                        continous_vals.append("{0:.9f}".format(val).rstrip('0')
                                                .rstrip('.'))
                    categorial_vals = []
                    for i in range(0, len(categorial_features)):
                        val = dicts.gen(i, features[categorial_features[i]])
                        categorial_vals.append(str(val))

                    continous_vals = ','.join(continous_vals)
                    categorial_vals = ','.join(categorial_vals)
                    index = features[-1]
                    out_train.write(','.join([continous_vals, categorial_vals, index]) + '\n')
                    train_list.append([continous_vals, categorial_vals, index])
                    label_list.append(index)

            print('有标签数据集合总条数：',ii)

    # from sklearn.model_selection import train_test_split
    # import numpy as np

    # X_train, X_test, Y_train, Y_test = train_test_split(np.array(train_list),np.array(label_list), test_size=0.1, 
    #                                         random_state=2018, stratify=np.array(label_list))
    # X_train, X_test, Y_train, Y_test = train_test_split(np.array(train_list),np.array(label_list), test_size=0.1, 
    #                                         stratify=np.array(label_list))
    # np.savetxt('train_0904.txt',X_train, fmt = '%s', delimiter=",")
    # np.savetxt('test_0904.txt',X_test, fmt = '%s', delimiter=",")


    # print('X_train:',len(X_train))
    # print('X_test:',len(X_test))



def get_csv(user_textvec_name):
    """
    构造特征 连续特征+离散特征+label 构造输入样本
    输出:所有样本的csv
    """
    # with open('data/raw/user_2w_live_avatar.json','r') as f:
    #     content = json.loads(f.read())
    with open('../../data/raw/user_12714_live_avatar_ques.json','r') as f:
        content = json.loads(f.read())
    with open(user_textvec_name,'r') as f:
        user_textvec = json.loads(f.read())

    live = pd.read_csv('../../data/raw/live.csv')

    columns = ['following_count','included_articles_count','favorite_count','voteup_count','live_count',
    'following_columns_count','participated_live_count','following_favlists_count','favorited_count','follower_count',
    'following_topic_count','answer_count','question_count','articles_count','included_answers_count',
    'following_question_count','thanked_count','hosted_live_count','original_price','speaker_audio_message_count',
    'attachment_count','feedback_score','review_count','reply_message_count','liked_num','people_count',

    'is_vip','is_advertiser','is_active','gender','in_promotion','is_refundable','purchasable','has_audition',
    'tag','speaker','label']

    sparse_features = ['C' + str(i) for i in range(1, 11)]
    dense_features = ['I' + str(i) for i in range(1, 27+text_vector)]
    columns = dense_features+sparse_features+['label']
    print(len(columns))

    ans = []

    user_id_list = []
    for k,v in content.items():
        print(k,'.......')
        user_id_list.append(k)
        temp  = []
        temp.extend(user_textvec[k])
        temp.append(v['following_count'])
        temp.append(v['included_articles_count'])
        temp.append(v['favorite_count'])
        temp.append(v['voteup_count'])
        temp.append(v['live_count'])
        temp.append(v['following_columns_count'])
        temp.append(v['participated_live_count'])
        temp.append(v['following_favlists_count'])     
        temp.append(v['favorited_count'])
        temp.append(v['follower_count'])
        temp.append(v['following_topic_count'])
        temp.append(v['answer_count'])
        temp.append(v['question_count'])
        temp.append(v['articles_count'])
        temp.append(v['included_answers_count'])
        temp.append(v['following_question_count'])
        temp.append(v['thanked_count'])
        temp.append(v['hosted_live_count'])

        indx = []
        for i in v['lives']:
            item = live[live['id']==int(i)]
            if item.empty:
                continue

            indx.append(item['index'].iloc[0])

            tmp = temp.copy()
            tmp.append(item['original_price'].iloc[0])
            tmp.append(item['speaker_audio_message_count'].iloc[0])
            tmp.append(item['attachment_count'].iloc[0])
            tmp.append(item['feedback_score'].iloc[0])
            tmp.append(item['review_count'].iloc[0])
            tmp.append(item['reply_message_count'].iloc[0])
            tmp.append(item['liked_num'].iloc[0])
            tmp.append(item['people_count'].iloc[0])

            tmp.append(v['is_vip'])
            tmp.append(v['is_advertiser'])
            tmp.append(v['is_active'])
            tmp.append(v['gender'])
            tmp.append(item['in_promotion'].iloc[0])
            tmp.append(item['is_refundable'].iloc[0])
            tmp.append(item['purchasable'].iloc[0])
            tmp.append(item['has_audition'].iloc[0])
            tmp.append(item['tags'].iloc[0])
            tmp.append(item['speaker'].iloc[0])
            # tmp.append(item['created_at'])
            tmp.append(1)
            ans.append(tmp)
            # print(ans)

            
        count=0
        # print(indx)
        while count<min(len(indx),20):
            j = random.randint(1,5483)
            while j in indx:
                j = random.randint(1,5483)

            tmp = temp.copy()
            item = live[live['index']==j]

            tmp.append(item['original_price'].iloc[0])
            tmp.append(item['speaker_audio_message_count'].iloc[0])
            tmp.append(item['attachment_count'].iloc[0])
            tmp.append(item['feedback_score'].iloc[0])
            tmp.append(item['review_count'].iloc[0])
            tmp.append(item['reply_message_count'].iloc[0])
            tmp.append(item['liked_num'].iloc[0])
            tmp.append(item['people_count'].iloc[0])

            tmp.append(v['is_vip'])
            tmp.append(v['is_advertiser'])
            tmp.append(v['is_active'])
            tmp.append(v['gender'])
            tmp.append(item['in_promotion'].iloc[0])
            tmp.append(item['is_refundable'].iloc[0])
            tmp.append(item['purchasable'].iloc[0])
            tmp.append(item['has_audition'].iloc[0])
            tmp.append(item['tags'].iloc[0])
            tmp.append(item['speaker'].iloc[0])
            # tmp.append(item['created_at'])
            tmp.append(0)

            ans.append(tmp)

            count+=1

        # break

    # with open('user_id_list.json','w') as f:
    #     f.write(json.dumps(user_id_list))

    df = pd.DataFrame(ans,columns=columns)
    df.to_csv('all_train_data_12714_addtext50.csv',index=False)
    # print(df.head(10))



if __name__ == "__main__":
    # preprocess('all_train_data_12714.csv', 'data')
    # split_train_test()
    get_csv('user_textvec_50.json')
    preprocess('all_train_data_12714_addtext50.csv', 'data')
    