import operator
import random
import time
import math
import numpy
import pandas
from concurrent.futures import ProcessPoolExecutor

# 以下两个包的import报错可以无视，只要相同目录下有slim.cp36-win_amd64.pyd和lfm.cp36-win_amd64.pyd这两个文件即可
import lfm
import slim


class Data:
    def __init__(self, dataset='ml-100k'):
        """
        无上下文信息的隐性反馈数据集。
        :param dataset: 使用的数据集名字，当前有'ml-100k','ml-1m'
        """

        path = None
        separator = None
        if dataset == 'ml-100k':
            # 共100000条数据，只有943个用户看过电影，只有1682个电影被看过。userID范围[1, 943]，movieID范围[1, 1682]。
            path = 'data/ml-100k/u.data'
            separator = '\t'
        elif dataset == 'ml-1m':
            # 共1000209条数据，只有6040个用户看过电影，只有3706个电影被看过。userID范围[1, 6040]，movieID范围[1, 3952]。
            path = 'data/ml-1m/ratings.dat'
            separator = '::'

        print('开始读取数据')

        # 从源文件读数据
        self.data = []
        for line in open(path, 'r'):
            data_line = line.split(separator)
            userID = int(data_line[0])
            movieID = int(data_line[1])
            # 无上下文信息的隐性反馈数据不需要评分和时间截
            #rating = int(data_line[2])
            #timestamp = int(data_line[3])
            self.data.append([userID, movieID])

        def compress(data, col):
            """
            压缩数据data第col列的数据。保证此列数字会从0开始连续出现，中间不会有一个不存在此列的数字。

            :param data: 二维列表数据
            :param col: 要压缩的列
            :return: 此列不同的数字个数（即此列最大数字加1）
            """
            e_rows = dict()  # 键是data数据第col列数据，值是一个存放键出现在的每一个行号的列表
            for i in range(len(data)):
                e = data[i][col]
                if e not in e_rows:
                    e_rows[e] = []
                e_rows[e].append(i)

            for rows, i in zip(e_rows.values(), range(len(e_rows))):
                for row in rows:
                    data[row][col] = i

            return len(e_rows)

        self.num_user = compress(self.data, 0)
        self.num_item = compress(self.data, 1)

        # 训练集和测试集
        self.train, self.test = self.__split_data()
        print('总共有{}条数据，训练集{}，测试集{}，用户{}，物品{}'.format(len(self.data), len(self.train), len(self.test), self.num_user, self.num_item))

    def __split_data(self):
        """
        将数据随机分成8份，1份作为测试集，7份作为训练集

        :return: 训练集和测试集
        """
        test = []
        train = []
        for user, item in self.data:
            if random.randint(1, 8) == 1:
                test.append([user, item])
            else:
                train.append([user, item])
        return train, test


class UserCF:
    def __init__(self, data):
        """
        基于用户的协同过滤算法。

        :param data: 无上下文信息的隐性反馈数据集，包括训练集，测试集等
        """
        self.data = data

        print('基于用户的协同过滤算法')
        print('开始计算用户相似度矩阵')
        self.W = self.__user_similarity()

        self.K = None  # 推荐时选择与用户最相似的用户个数
        self.N = None  # 每个用户最多推荐物品数量
        self.recommendation = None

    def compute_recommendation(self, K=80, N=10):
        """
        开始计算推荐列表

        :param K: 推荐时选择与物品最相似的物品个数
        :param N: 每个用户最多推荐物品数量
        """
        self.K = K
        self.N = N

        print('开始计算推荐列表（K=' + str(self.K) + ', N=' + str(self.N) + '）')
        self.recommendation = self.__get_recommendation()

    def __user_similarity(self):
        """
        计算训练集的用户相似度

        :return: 存放每两个用户之间相似度的二维列表
        """
        train_item_users = [[] for i in range(self.data.num_item)]  # train_item_users[i]是对物品i有过正反馈的所有用户列表
        for user, item in self.data.train:
            train_item_users[item].append(user)

        print('统计每两个用户之间的共同正反馈物品数量和每个用户有过正反馈物品的总量')
        W = [[0 for j in range(self.data.num_user)] for i in
             range(self.data.num_user)]  # W[u][v]是用户u和v的共同有正反馈物品的数量（v>u）
        N = [0 for i in range(self.data.num_user)]  # N[u]是用户u有过正反馈的所有物品的数量
        for users in train_item_users:
            for user in users:
                # 统计N
                N[user] += 1

                # 统计W
                for v in users:
                    if v > user:
                        W[user][v] += 1

        print('计算每两个用户之间的相似度')
        for i in range(self.data.num_user - 1):
            for j in range(i + 1, self.data.num_user):
                if W[i][j] != 0:
                    W[i][j] /= math.sqrt(N[i] * N[j])
                    W[j][i] = W[i][j]
        return W

    def __recommend(self, user, train_user_items):
        """
        对用户user选取最相似的K个用户推荐他们有行为的最多N个物品。

        :param user: 推荐的目标用户
        :param train_user_items: train_user_items[i]是用户i所有有过正反馈的物品集合
        :return: 推荐给用户user的物品列表
        """
        Wu = dict()
        for v in range(self.data.num_user):
            if self.W[user][v] != 0:
                Wu[v] = self.W[user][v]

        # 计算出用户user对每个物品感兴趣程度
        rank = dict()
        for similar_user, similarity_factor in sorted(Wu.items(), key=operator.itemgetter(1), reverse=True)[:self.K]:
            for item in train_user_items[similar_user] - train_user_items[user]:
                rank[item] = rank.setdefault(item, 0) + similarity_factor

        return [r[0] for r in sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[:self.N]]

    def __get_recommendation(self):
        """
        得到所有用户的推荐物品列表。

        :return: 推荐列表，下标i对应给用户i推荐的物品列表
        """
        # 得到训练集中每个用户所有有过正反馈物品集合
        train_user_items = [set() for u in range(self.data.num_user)]
        for user, item in self.data.train:
            train_user_items[user].add(item)

        recommendation = []
        for user in range(self.data.num_user):
            recommendation.append(self.__recommend(user, train_user_items))
        return recommendation


class ItemCF:
    def __init__(self, data):
        """
        基于物品的协同过滤算法。

        :param data: 无上下文信息的隐性反馈数据集，包括训练集，测试集等
        """
        self.data = data

        print('基于物品的协同过滤算法')
        print('开始计算物品相似度矩阵')
        self.W = self.__item_similarity()

        self.K = None  # 推荐时选择与物品最相似的物品个数
        self.N = None  # 每个用户最多推荐物品数量
        self.recommendation = None

    def compute_recommendation(self, K=10, N=10):
        """
        开始计算推荐列表

        :param K: 推荐时选择与物品最相似的物品个数
        :param N: 每个用户最多推荐物品数量
        """
        self.K = K
        self.N = N

        print('开始计算推荐列表（K=' + str(self.K) + ', N=' + str(self.N) + '）')
        self.recommendation = self.__get_recommendation()

    def __item_similarity(self):
        """
        计算训练集的用户相似度

        :return: 存放每两个用户之间相似度的二维列表
        """
        train_user_items = [[] for u in range(self.data.num_user)]  # train_user_items[i]是用户i有过正反馈的所有物品列表
        for user, item in self.data.train:
            train_user_items[user].append(item)

        print('统计每两个物品之间的共同有过正反馈的用户数量和每个物品被有过正反馈的总量')
        W = [[0 for j in range(self.data.num_item)] for i in range(self.data.num_item)]  # W[i][j]是物品i和j的共同被正反馈的数量（j>i）
        N = [0 for i in range(self.data.num_item)]  # N[i]是物品i被有过正反馈的数量
        for items in train_user_items:
            for item in items:
                # 统计N
                N[item] += 1

                # 统计W
                for j in items:
                    if j > item:
                        W[item][j] += 1

        print('计算每两个物品之间的相似度')
        for i in range(self.data.num_item - 1):
            for j in range(i + 1, self.data.num_item):
                if W[i][j] != 0:
                    W[i][j] /= math.sqrt(N[i] * N[j])
                    W[j][i] = W[i][j]
        return W

    def __recommend(self, user_item_set, k_items):
        """
        对每个用户user没有正反馈的物品选取最相似的K个物品计算兴趣，给用户推荐最多N个物品。

        :param user_item_set: 训练集用户user所有有过正反馈的物品集合
        :param k_items: k_items[i]是与物品i最相似的K个物品集合
        :return: 推荐给用户user的物品列表
        """
        rank = dict()
        for i in user_item_set:
            for j in k_items[i]:
                if j not in user_item_set:
                    rank[j] = rank.setdefault(j, 0) + self.W[i][j]
        # for i in set(range(self.data.num_item)) - user_item_set:  # 计算用户user对物品i的兴趣
        #    for j in user_item_set & k_items[i]:
        #        rank[i] = rank.setdefault(i, 0) + self.W[i][j]

        return [items[0] for items in sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[:self.N]]

    def __get_recommendation(self):
        """
        得到所有用户的推荐物品列表。

        :return: 推荐列表，下标i对应给用户i推荐的物品列表
        """
        # 得到训练集中每个用户所有有过正反馈物品集合
        train_user_items = [set() for u in range(self.data.num_user)]
        for user, item in self.data.train:
            train_user_items[user].add(item)

        print('得到每个物品与其最相似的K个物品集合')
        k_items = []
        for i in range(self.data.num_item):
            Wi = dict()  # Wi[j]是物品i和j之间的相似度
            for j in range(self.data.num_item):
                if self.W[i][j] != 0:
                    Wi[j] = self.W[i][j]

            k_items.append(
                set(items[0] for items in sorted(Wi.items(), key=operator.itemgetter(1), reverse=True)[:self.K]))

        print('计算每个用户的推荐列表')
        recommendation = []
        for user_item_set in train_user_items:
            recommendation.append(self.__recommend(user_item_set, k_items))
        return recommendation


class LFM:
    def __init__(self, data):
        """
        隐语义模型算法。

        :param data: 无上下文信息的隐性反馈数据集，包括训练集，测试集等
        """
        self.data = data

        print('隐语义模型算法')

        self.ratio = None  # 负正样本比例
        self.max_iter = None  # 学习迭代次数
        self.F = None  # 隐类个数
        self.N = None  # 每个用户最多推荐物品数量

        self.P = None  # P[u][k]是用户u和第k个隐类的关系
        self.Q = None  # Q[k][i]是物品i和第k个隐类的关系
        self.recommendation = None

    def compute_recommendation(self, ratio=10, max_iter=30, F=100, N=10):
        """
        开始计算推荐列表

        :param ratio: 负正样本比例
        :param max_iter: 学习迭代次数
        :param F: 隐类个数
        :param N: 每个用户最多推荐物品数量
        """
        self.ratio = ratio
        self.max_iter = max_iter
        self.F = F
        self.N = N

        print('开始计算P,Q矩阵（ratio=' + str(self.ratio) + ', max_iter=' + str(self.max_iter) + ', F=' + str(self.F) + '）')
        self.P, self.Q = self.__latent_factor_model()

        print('开始计算推荐列表（N=' + str(self.N) + '）')
        self.recommendation = self.__get_recommendation()

    def __select_negative_sample(self):
        """
        对每个用户分别进行负样本采集。

        :return: 所有采集结果列表，列表第u项表示用户u样本采集结果，是一个dict，key是物品，value为1表示正样本，为0表示负样本
        """
        train_user_items = [set() for u in range(self.data.num_user)]
        item_pool = []  # 候选物品列表，每个物品出现的次数和其流行度成正比
        for user, item in self.data.train:
            train_user_items[user].add(item)
            item_pool.append(item)

        user_samples = []
        for user in range(self.data.num_user):
            sample = dict()
            for i in train_user_items[user]:  # 设置用户user所有正反馈物品为正样本（值为1）
                sample[i] = 1
            n = 0  # 已取负样本总量
            max_n = int(len(train_user_items[user]) * self.ratio)  # 根据正样本数量和负正样本比例得到负样本目标数量
            for i in range(max_n * 3):
                item = random.choice(item_pool)
                if item in sample:
                    continue
                sample[item] = 0
                n += 1
                if n >= max_n:
                    break
            user_samples.append(sample)
        return user_samples

    def __latent_factor_model(self, alpha=0.02, lam_bda=0.01):
        print('对每个用户采集负样例')
        user_samples = self.__select_negative_sample()
        samples_ui = [[], []]
        samples_r = []
        for user in range(self.data.num_user):
            for item, rui in user_samples[user].items():
                samples_ui[0].append(user)
                samples_ui[1].append(item)
                samples_r.append(rui)
        samples_ui = numpy.array(samples_ui, numpy.int)
        samples_r = numpy.array(samples_r, numpy.double)

        print('随机梯度下降法学习P,Q矩阵')
        k = 1 / math.sqrt(self.F)
        P = numpy.array([[random.random() * k for f in range(self.F)] for u in range(self.data.num_user)])
        Q = numpy.array([[random.random() * k for i in range(self.data.num_item)] for f in range(self.F)])
        return lfm.gradient_decsent(alpha, lam_bda, self.max_iter, P, Q, samples_ui, samples_r)

    def __recommend(self, user_PQ, user_item_set):
        """
        给用户user推荐最多N个物品。

        :param user_PQ: PQ矩阵相乘的第user行
        :param user_item_set: 训练集用户user所有有过正反馈的物品集合
        :return: 推荐给用户user的物品列表
        """
        rank = dict()
        for i in set(range(self.data.num_item)) - user_item_set:
            rank[i] = user_PQ[i]
        return [items[0] for items in sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[:self.N]]

    def __get_recommendation(self):
        """
        得到所有用户的推荐物品列表。

        :return: 推荐列表，下标i对应给用户i推荐的物品列表
        """
        # 得到训练集中每个用户所有有过正反馈物品集合
        train_user_items = [set() for u in range(self.data.num_user)]
        for user, item in self.data.train:
            train_user_items[user].add(item)

        PQ = self.P.dot(self.Q)

        # 对每个用户推荐最多N个物品
        recommendation = []
        for user_PQ, user_item_set in zip(PQ, train_user_items):
            recommendation.append(self.__recommend(user_PQ, user_item_set))
        return recommendation


class SLIM:
    def __init__(self, data):
        """
        稀疏线性算法。

        :param data: 无上下文信息的隐性反馈数据集，包括训练集，测试集等
        """
        self.data = data

        print('稀疏线性算法')
        self.A = self.__user_item_matrix()  # 用户-物品行为矩阵

        self.alpha = None
        self.lam_bda = None
        self.max_iter = None  # 学习最大迭代次数
        self.tol = None  # 学习阈值
        self.N = None  # 每个用户最多推荐物品数量
        self.lambda_is_ratio = None  # lambda参数是否代表比例值

        self.W = None  # 系数集合
        self.recommendation = None

        self.__covariance_dict = None

    def compute_recommendation(self, alpha=0.5, lam_bda=0.02, max_iter=1000, tol=0.0001, N=10, lambda_is_ratio=True):
        """
        开始计算推荐列表

        :param alpha: lasso占比（为0只有ridge-regression，为1只有lasso）
        :param lam_bda: elastic net系数
        :param max_iter: 学习最大迭代次数
        :param tol: 学习阈值
        :param N: 每个用户最多推荐物品数量
        :param lambda_is_ratio: lambda参数是否代表比例值。若为True，则运算时每列lambda单独计算；若为False，则运算时使用单一lambda的值
        """
        self.alpha = alpha
        self.lam_bda = lam_bda
        self.max_iter = max_iter
        self.tol = tol
        self.N = N
        self.lambda_is_ratio = lambda_is_ratio

        print('开始计算W矩阵（alpha=' + str(self.alpha) + ', lambda=' + str(self.lam_bda) + ', max_iter=' + str(
            self.max_iter) + ', tol=' + str(self.tol) + '）')
        self.W = self.__aggregation_coefficients()

        print('开始计算推荐列表（N=' + str(self.N) + '）')
        self.recommendation = self.__get_recommendation()

    def __user_item_matrix(self):
        A = numpy.zeros((self.data.num_user, self.data.num_item))
        for user, item in self.data.train:
            A[user, item] = 1
        return A

    def __aggregation_coefficients(self):
        group_size = 100  # 并行计算每组计算的行/列数
        n = self.data.num_item // group_size  # 并行计算分组个数
        starts = []
        ends = []
        for i in range(n):
            start = i * group_size
            starts.append(start)
            ends.append(start + group_size)
        if self.data.num_item % group_size != 0:
            starts.append(n * group_size)
            ends.append(self.data.num_item)
            n += 1

        print('进行covariance updates的预算')
        covariance_array = None
        with ProcessPoolExecutor() as executor:
            covariance_array = numpy.vstack(executor.map(slim.compute_covariance, [self.A] * n, starts, ends))
        slim.symmetrize_covariance(covariance_array)

        print('坐标下降法学习W矩阵')
        if self.lambda_is_ratio:
            with ProcessPoolExecutor() as executor:
                return numpy.hstack(executor.map(slim.coordinate_descent_lambda_ratio, [self.alpha] * n, [self.lam_bda] * n, [self.max_iter] * n, [self.tol] * n, [self.data.num_user] * n, [self.data.num_item] * n, [covariance_array] * n, starts, ends))
        else:
            with ProcessPoolExecutor() as executor:
                return numpy.hstack(executor.map(slim.coordinate_descent, [self.alpha] * n, [self.lam_bda] * n, [self.max_iter] * n, [self.tol] * n, [self.data.num_user] * n, [self.data.num_item] * n, [covariance_array] * n, starts, ends))

    def __recommend(self, user_AW, user_item_set):
        """
        给用户user推荐最多N个物品。

        :param user_AW: AW矩阵相乘的第user行
        :param user_item_set: 训练集用户user所有有过正反馈的物品集合
        :return: 推荐给本行用户的物品列表
        """
        rank = dict()
        for i in set(range(self.data.num_item)) - user_item_set:
            rank[i] = user_AW[i]
        return [items[0] for items in sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[:self.N]]

    def __get_recommendation(self):
        """
        得到所有用户的推荐物品列表。

        :return: 推荐列表，下标i对应给用户i推荐的物品列表
        """
        # 得到训练集中每个用户所有有过正反馈物品集合
        train_user_items = [set() for u in range(self.data.num_user)]
        for user, item in self.data.train:
            train_user_items[user].add(item)

        AW = self.A.dot(self.W)

        # 对每个用户推荐最多N个物品
        recommendation = []
        for user_AW, user_item_set in zip(AW, train_user_items):
            recommendation.append(self.__recommend(user_AW, user_item_set))
        return recommendation


class Evaluation:
    def __init__(self, recommend_algorithm):
        """
        对推荐算法recommend_algorithm计算各种评测指标。

        :param recommend_algorithm: 推荐算法，包括推荐结果列表，数据集等
        """
        self.rec_alg = recommend_algorithm

        self.precision = None
        self.recall = None
        self.coverage = None
        self.popularity = None

    def evaluate(self):
        """
        评测指标的计算。
        """
        # 准确率和召回率
        self.precision, self.recall = self.__precision_recall()
        print('准确率 = ' + str(self.precision * 100) + "%  召回率 = " + str(self.recall * 100) + '%')

        # 覆盖率
        self.coverage = self.__coverage()
        print('覆盖率 = ' + str(self.coverage * 100) + '%')

        # 流行度
        self.popularity = self.__popularity()
        print('流行度 = ' + str(self.popularity))

    def __precision_recall(self):
        """
        计算准确率和召回率。

        :return: 准确率和召回率
        """
        # 得到测试集用户与其所有有正反馈物品集合的映射
        test_user_items = dict()
        for user, item in self.rec_alg.data.test:
            if user not in test_user_items:
                test_user_items[user] = set()
            test_user_items[user].add(item)

        # 计算准确率和召回率
        hit = 0
        all_ru = 0
        all_tu = 0
        for user, items in test_user_items.items():
            ru = set(self.rec_alg.recommendation[user])
            tu = items

            hit += len(ru & tu)
            all_ru += len(ru)
            all_tu += len(tu)
        return hit / all_ru, hit / all_tu

    def __coverage(self):
        """
        计算覆盖率

        :return: 覆盖率
        """
        recommend_items = set()
        for user in range(self.rec_alg.data.num_user):
            for item in self.rec_alg.recommendation[user]:
                recommend_items.add(item)
        return len(recommend_items) / self.rec_alg.data.num_item

    def __popularity(self):
        """
        计算新颖度（平均流行度）

        :return: 新颖度
        """
        item_popularity = [0 for i in range(self.rec_alg.data.num_item)]
        for user, item in self.rec_alg.data.train:
            item_popularity[item] += 1

        ret = 0
        n = 0
        for user in range(self.rec_alg.data.num_user):
            for item in self.rec_alg.recommendation[user]:
                ret += math.log(1 + item_popularity[item])
                n += 1
        return ret / n


if __name__ == '__main__':
    algorithms = [UserCF, ItemCF, LFM, SLIM]
    precisions = []
    recalls = []
    coverages = []
    popularities = []
    times = []

    data = Data()

    for algorithm in algorithms:
        startTime = time.time()
        recommend = algorithm(data)
        recommend.compute_recommendation()
        eva = Evaluation(recommend)
        eva.evaluate()
        times.append('%.3fs' % (time.time() - startTime))
        precisions.append('%.3f%%' % (eva.precision * 100))
        recalls.append('%.3f%%' % (eva.recall * 100))
        coverages.append('%.3f%%' % (eva.coverage * 100))
        popularities.append(eva.popularity)

    df = pandas.DataFrame()
    df['algorithm'] = [algorithm.__name__ for algorithm in algorithms]
    df['precision'] = precisions
    df['recall'] = recalls
    df['coverage'] = coverages
    df['popularity'] = popularities
    df['time'] = times
    print(df)

    # recommend = SLIM_c(Data())
    # recommend.compute_recommendation()
    # Evaluation(recommend).evaluate()
