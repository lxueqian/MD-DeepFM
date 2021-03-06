# MD-DeepFM

哈喽！这里是我的毕业设计论文《知识付费产品的个性化推荐方法研究》部分实验代码，是在DeepFM和BPR基础推荐方法上的改进，使之适用于知识付费产品推荐！
欢迎star、一起讨论一起进步！

    互联网本是“免费”内容的天下，然而在近几年，知识付费成为互联网经济的后起之秀。对于知识付费产品这一线上内容产品来说，内容的精准分发和内容的价值一样重要。如果直接把用户暴露在一个“信息过载”的环境下，不管内容多优秀，用户流失还是不可避免的。所以通过推荐算法实现精准匹配帮用户“减负”，是知识付费平台的一门必修课。
    知识付费产品有着自己区别于其他电商产品的特点，在推荐时也应该考虑到这些特点进行个性化推荐，而目前对于知识付费产品的个性化推荐研究并不多，本研究弥补了这方面的不足，通过获取知乎直播数据，以此为入口，探究知识付费产品的个性化推荐方法。
    本研究探索总结了知识付费产品的典型特点，提出了增强知识付费产品推荐效果的两个可能途径。并在此基础上设计了两个算法：基于内容增强的贝叶斯个性化排序算法（C-BPR）和基于多领域信息表征的深度因子分解机算法（MD-DeepFM）。C-BPR 算法通过改进负采样策略对原有贝叶斯个性化排序算法进行优化，把内容语义相似度较低的样本定位为负样本，巧妙地融入了文本表征信息。MD-DeepFM 算法实现了用户和物品向量表征学习层面的优化，把用户其他领域的行为通过 Bert 编码添加为用户特征的一部分，与本领域提取的用户行为特征共同作为模型输入进行用户向量的表征学习，实现用户向量的跨领域表征学习。
    
    本文在收集整理的知乎直播数据集上进行了实验，结果表明所提出的模型都可以获得更好的推荐效果，C-BPR 算法的 Precision@10 和 NDCG@10 指标都相对提高了 4％以上，MD-DeepFM 算法中 AUC 指标相比基准方法提高了 1.8%以上。另外，通过冷启动实验分析可以发现 MD-DeepFM 算法对新用户的推荐效果更好，AUC 比之前提高了 2.68%，可以有效缓解冷启动问题。本研究证实了在知识付费产品推荐场景下语义附加信息的加入和跨领域特征的融合对推荐效果的增强作用，所提出的两个模型对于知识付费产品的推荐具有一定的理论与实际应用意义。
    
关键词：推荐系统，知识付费，文本表征，深度学习，冷启动

1. dataprocess.py FM的数据预处理
2. deep_fm_v2.py MD-DeepFM的pytorch实现（loss函数和输入数据集有优化）
3. DeepFM.py 原模型实现
4. performanceCompare.py 比较结果
5. data文件夹是数据：其中train.txt里面只添加了部分训练数据，原文件太大无法上传

使用方法建议：
    1. 想要学习DeepFM的伙伴可以认真看一遍DeepFM.py代码，这是我在网上找到的输入我的数据以后能够正常运行的代码，借助代码能够更好地了解DeepFM这个模型
    2. 仔细看下模型输入的数据结构，建议看下模型训练集和测试集以及特征大小的文件


