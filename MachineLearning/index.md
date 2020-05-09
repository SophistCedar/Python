## 01.01-SimpleLinearRegression
模型选择  
模型实例化  
模型训练  
模型验证  
线性回归模型：from sklearn.linear_model import LinearRegression  
## 01.02-IrisClassification
查看数据集内部的数据关系  
划分训练接和数据集  
准确率计算  
高斯朴素贝叶斯：from sklearn.naive_bayes import GaussianNB
## 01.03-IrisDimensionalityReduction+Clusters
降维  
主成分分析法：from sklearn.decomposition import PCA  
降维到二维：model = PCA(n_components=2) 
集群  
高斯混合模型：from sklearn.mixture import GaussianMixture as GMM  
预先制定集群数量：model = GMM(n_components=3,covariance_type='full')  
## 01.04-ModelValidation+Hyperparameter
应用监督式学习模型四个步骤：  
1 模型选择  
2 模型实例化，使用超参数  
3 模型训练  
4 模型预测  
如何选择模型，以及模型的超参数如何选择很重要，需要一些方法来研究模型和超参数对研究数据的适用性。  
### 模型验证 
正确验证的方法是可以将数据分为训练集和测试集，使用交叉验证。将数据分为训练集和测试集。用一个子集训练，去预测另外一个子集，取得准确率的平均结果。也可以将数据集分成五个子集，每次用其中四个子集训练模型，再预测剩下的那一个子集的数据。还可以每次只用剩下一个点的数据集进行训练，预测剩下那个点的情况。
### 超参数选择
The Bias-variance trade-off权衡方差和偏差  
偏差大的模型，模型对训练集和验证集预测效果相近。  
方差大的模型，模型对训练集预测效果很好，但是对于验证集效果会很差。   
验证曲线：横坐标是模型复杂性，纵坐标是模型分数，在这个坐标系内比较训练分数和验证分数。  
1 训练集分数优于验证集。  
2 偏差大，模型不复杂，训练数据欠拟合，训练集分数和验证集分数都会低。  
3 方差大，模型较复杂，训练数据过拟合，训练集分数高，但是验证集分低。  
4 如何对于方差和偏差权衡？当验证集分数最高的时候即可。  
验证曲线：模型性能与模型复杂度关系  
学习曲线：模型性能与数据集大小关系  
网格搜索：from sklearn.model_selection import GridSearchCV  
指定超参数的集合，遍历使用这些超参数去训练模型和预测数据，找出超参数集合中最优化的超参数的值。
## 01.05-FeaturesEngineering
1. Categorical Features  
将列表字典向量化，使用one-hot编码，将字符串之类的信息转化成0,1数字信息  
from sklearn.feature_extraction import DictVectorizer  
2. Text Features  
用单词计数编码数据  
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.feature_extraction.text import TfidfVectorizer  
3. Image Features  
4. Derived Features  
改造数据的特征矩阵：对数据增加多项式特征  
from sklearn.preprocessing import PolynomialFeatures  
5. Imputation of Missing Data  
使用平均值等策略进行缺失值填充  
from sklearn.preprocessing import Imputer  
6. Feature Pipelines  
可以使用管道将缺失值填充，衍生特征扩展，模型训练等放在一起。
## 01.06-NaiveBayesClassification
Gaussian Naive Bayes  
高斯朴素贝叶斯估计器支持概率预测，即可以给出该sample属于各个label的概率。  
Multinomial Naive Bayes
## 01.07-LinearRegression
Simple Linear Regression：线性回归模型除了可以训练简单的一维线性数据，也可以训练多维的线性模型,例如：
$$
y = a_0 + a_1 x_1 + a_2 x_2 + \cdots
$$
Basis Function Regression: 将线性回归应用于变量之间的非线性关系的一个技巧是根据基函数转换数据。 我们实际上所做的是将一维的 𝑥 值投影到更高的维度中，这样线性拟合就可以拟合 𝑥 和 𝑦 之间更复杂的关系。  Polynomial basis functions  
Gaussian basis functions  
基函数如果增加了过多的衍生特征，有可能会导致过拟合现象出现，可以通过使用正则化来解决这一问题。  
Regularization: Ridge regression ( 𝐿2  Regularization), Lasso regression ( 𝐿1  regularization)  
Example: Predicting Bicycle Traffic
增加与自行车数量相关的因素到特征矩阵中，模型训练后，计算每个系数的影响，以及每个系数遍历一定次数后的标准差。
## 01.08-SupportVectorMachines
既可以用于分类，也可以用于回归。  
1. Fitting a support vector machine   
from sklearn.svm import SVC # "Support vector classifier"，最大化边缘。
model = SVC(kernel='linear', C=1E10)  此时内核为线性，可以通过改变内核去区分非线性的分类，这种改变内核的策略是一种很强大的方法，可以把快速的线性方法转换成快速的非线性方法。可选的内核包括：'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' 
2. Tuning the SVM: Softening Margins  
调整参数C可以软化边缘，参数C的取值可以通过交叉验证的方法得到一个合理的结果。
3. Example: Face Recognition  
PCA降维，SVM分类，网格搜索找到最优参数，建立最优模型预测结果，生成混淆矩阵，或者生成分类报告。
## 01.09-RandomForests
既可以用于分类，也可以用于回归。 
1. Motivating Random Forests: Decision Trees  
from sklearn.tree import DecisionTreeClassifier
决策树有可能发生过拟合，可以使用BaggingClassifier解决此问题。  
2. Ensembles of Estimators: Random Forests  
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
3. Random Forest Regression  
from sklearn.ensemble import RandomForestRegressor，可以适应多周期数据，不需要指定多周期模型。
4. Example: Random Forest for Classifying Digits
使用随机树分类数字，分类准确率较高。
## 01.10-PrincipalComponentAnalysis
1. Introducing Principal Component Analysis
无监督学习问题试图了解x和y值之间的“关系”。 在主成分分析中，通过查找数据中的主轴列表并使用这些轴来描述数据集来量化这种关系。
print(pca.components_)#using the "components" to define the direction of the vector,使用“分量”来定义向量的方向，
print(pca.explained_variance_)#the "explained variance" to define the squared-length of the vector，使用“解释的方差”来定义向量的平方长度。
这些向量表示数据的主轴，而向量的长度表示该轴在描述数据分布时的“重要性”——更准确地说，它是数据在投射到该轴时的方差的度量。 每个数据点在主轴上的投影是数据的“主成分”。 
2. PCA as dimensionality reduction
降维后的数据也可以转换回原有维度，与原有数据进行比较。
3. PCA for visualization: Hand-written digits
可以把64维的手写数据，降到二维，这样就可以画出散点图，看数字的分布分类情况。
4. Choosing the number of components
使用PCA的一个重要部分是能够估计需要多少组件来描述数据。 
这可以通过查看累加的解释方差比作为组件数量的函数来确定: plt.plot(np.cumsum(pca.explained_variance_ratio_))，方差越大表示给定数量的组件保留下来的信息越多。
5. PCA as Noise Filtering
先降维，再转换回原有维度，可以有效去除噪声数据。
6. Example: Eigenfaces
原有人脸数据维度2914，降维到150维度可以保留超过90%解释方差，再转换回2914维度，比较两种情况人脸图像，发现人眼能够明显分辨。也就是说150个组件已经保留了人脸图像的足够信息，这意味着分类算法只需要在150维数据上训练，不需要在2914维度数据上训练，这样只有原有维度的5%，可以更加有效分类。
## 01.11-ManifoldLearning
1. Manifold Learning: "HELLO"
创建形状是HELLO的数据，画出散点图构成HELLO形状。
2. Multidimensional Scaling (MDS)
可以使用旋转矩阵对数据进行旋转，从原有数据和旋转后的数据构造距离矩阵，比较发现这两个距离矩阵是一样的。
使用维度扩展MDS可以从距离矩阵中还原原来的HELLO形状。
3. MDS as Manifold Learning
可以将从原始的HELLO数据投影到三维空间内，可视化变换后HELLO三维数据，是一个线性的HELLO三维形状，这样我们就有三维数据。使用MDS对三维数据进行降维到二维，画出投影后的二维数据和原有的HELLO数据比较，两者很类似。这本质上是流形学习估计器的目标: 给定高维嵌入式数据，它寻找数据的低维表示，以保持数据中的某些关系。对于MDS，保留的量是每对点之间的距离。
4. Nonlinear Embeddings: Where MDS Fails
到目前为止，我们的讨论已经考虑了“线性”嵌入，它本质上是将数据旋转、平移和缩放到高维空间。当嵌入是非线性的，也就是说，当嵌入超出了这个简单的操作集时，MDS就失效了。将原始HELLO数据扭曲成非线性的S形变化，可视化此三维数据，这种非线性嵌入相对来说比较复杂，使用MDS降维算法，无法保留原有的非线性关系，失去了嵌入式流行数据集中的关系。
5. Nonlinear Manifolds: Locally Linear Embedding
我们如何才能向前迈进?回顾一下，我们可以看到问题的根源是MDS在构建嵌入时试图保持遥远点之间的距离。但是，如果我们修改算法，使其只保留附近点之间的距离，结果的嵌入会更接近我们想要的。MDS试图保留数据集内所有点之间的距离，但是没有任何一种方法能保证数据铺平，同时又保证数据点之间的距离，不过LocallyLinearEmbedding只保留附近点的距离，叫局部线性嵌入。将S形的HELLO数据使用局部线性嵌入降维到二维数据，可视化二维数据结果，相比原始流行数据，结果仍然有点扭曲，但是捕获了数据中的基本关系。与PCA相比，流形学习方法唯一明显的优势是它们能够在数据中保持非线性关系; 出于这个原因，我倾向于在使用PCA研究数据之后再使用流行方法来研究数据。
6. Example: Isomap on Faces
Isometric Feature Mapping (Isomap) 等距映射。使用PCA降维需要100个组件，才能够保持90%以上的解释方差比，但是使用Isomap将数据非线性降维到二维，可视化此二维数据，这两个Isomap维度似乎描述了全局图像特征:图像从左到右的整体黑暗或亮度，以及面部从下到上的总体方向。
7. Example: Visualizing Structure in Digits
将784维的手写数字，使用Isomap非线性降维到2维，这样可以画出散点图，可视化分类的结果，不过数据重叠比较多，视觉感官不是很好。可以一次只看一种数字，根据降维的二维数据，画出所有的数字1。这能够帮助我们理解数据，比如从可视化结果可以看到是否有一些异常值出现，这样我们可以在预处理数据之前建立一个分类管道，剔除异常值。
## 01.12-K-Means
聚类算法试图从数据的属性中学习点群的最优划分或离散标记。
1. Introducing k-Means
k-means算法在未标记的多维数据集中搜索预先确定的集群数量。它使用最优聚类的一个简单概念来实现这一点: “聚类中心”是属于该聚类的所有点的算术平均值。每个点离自己的集群中心都比离其他集群中心更近。 这两个假设是k-means模型的基础。
2. k-Means Algorithm: Expectation–Maximization
期望最大化(E-M)是一种强大的算法，它出现在数据科学的各种环境中。k-means是一个特别简单和易于理解的算法应用程序，我们将在这里简要介绍它。
简而言之，这里的期望最大化方法包括以下步骤:
>* 猜一些集群中心
>* 重复直到聚合
>>1. *E-Step*: 更新每个点所属的集群的期望，将点分配到最近的集群中心
>>2. *M-Step*: 将定义集群中心位置的适应度函数最大化，设置集群中心为均值     

在典型的情况下，E-step和M-step的每次重复都将导致对集群特征的更好估计，但是关于期望最大化的也有一些注意事项：
>* The globally optimal result may not be achieved，因此，通常算法会运行多次初试猜测。
>* The number of clusters must be selected beforehand，k-means无法从数据中了解集群的数量。
>* k-means is limited to linear cluster boundaries，如果集群具有复杂的几何形状，该算法通常是无效的。可以使用这个内核化的k-means的版本“SpectralClustering”估计器，解决非线性关系的聚类。
3. Example 1: k-means on digits
使用k-means来尝试在不使用原始标签信息的情况下识别相似的数字，准确率大约80%，这表明，使用k-means，我们基本上可以构建一个没有引用任何已知标签的数字分类器。使用t-分布随机邻接嵌入(t-SNE)算法将数据投影到二维，再对二维数据使用k-means算法，分类准确率约92%。如果使用得当，这就是无监督学习的强大之处: 它可以从数据集中提取难以用手或肉眼完成的信息。
2. Example 2: k-means for color compression
使用MiniBatchKMeans将1600万种颜色聚类到16个集群内，完成颜色压缩，将颜色压缩后的图像转换原始形状，与原始图像相比，会丢失一些细节，但是整体图像还是很容易识别的。
## 01.13-GaussianMixtures
1. Motivating GMM: Weaknesses of k-Means
k-means不支持概率性预测，如果两个集群之间有非常轻微的重叠，我们可能对它们之间的点的集群分配没有完全的信心。可以在每个集群的中心放置一个圆(或者，在更高维度中，一个超球体)，半径由集群中最遥远的点定义，此半径充当训练集内集群分配的硬截止点: 此圆之外的任何点都不被视为集群的成员。但是这可能导致集群分配的混合，产生圆圈重叠。因此k-means在集群形状上缺乏灵活性，并且缺乏概率集群分配。
高斯混合模型增加了两个部分的考虑：
>* 比较每个点到所有集群中心的距离来度量集群分配中的不确定性。
>* 考虑允许集群边界为椭圆形而不是圆形，以便考虑非圆形集群。
2. Generalizing E–M: Gaussian Mixture Models
高斯混合模型(GMM)试图找到一种多维高斯概率分布的混合模型，它可以很好地模拟任何输入数据集。 同时，因为GMM在底层包含一个概率模型，所以在scikit中也可以找到概率集群分配，这是使用predict_proba方法完成的。它返回一个大小为[n_samples, n_clusters]的矩阵，该矩阵度量任意点属于给定集群的概率。也可以将这种不确定性形象化，例如，使每个点的大小与预测的确定性成比例。根据集群形状合理选择covariance_type，可以对扭曲的一些形状进行准确的集群处理。
3. GMM as Density Estimation
虽然GMM经常被归类为一种聚类算法，但它本质上是一种用于密度估计的算法。 也就是说，GMM对某些数据的拟合结果在技术上不是一个聚类模型，而是一个描述数据分布的生成概率模型。  
使用16个Gaussians混合模型并不是为了找到单独的数据集群，而是为了对输入数据的整体分布建模。也就是找到了一个分布的生成模型，这意味着GMM为我们提供了生成新的随机数据的方法。  
如何选择组件/集群数量？GMM是一个生成模型，这一事实为我们提供了一种确定给定数据集的最优组件数量的自然方法。生成模型本质上是数据集的概率分布，因此我们可以简单地评估模型下数据的“可能性”，使用交叉验证来避免过度拟合。纠正过拟合的另一种方法是使用一些分析标准来调整模型的可能性，如Akaike information criterion (AIC)或Bayesian information criterion (BIC)。Scikit-Learn的GMM估计器实际上包含了计算这两个值的内置方法，因此使用这种方法非常容易。集群的最优数量是使AIC或BIC最小的值。请注意重要的一点:组件数量的选择度量了GMM作为密度估计器的工作情况，而不是它作为集群算法的工作情况。 我鼓励您将GMM主要看作是一种密度估计器，并且仅在简单数据集中需要时才将其用于集群。
4. Example: GMM for Generating New Data
加载64维数字数据，GMMs在高维空间不易收敛，因此保留99%解释方差使用PCA，将数据降维到41维且几乎没有信息损失；  
使用41维数据，根据最小化AIC找到110个components比较合适，使用此模型对数据训练并验证是否收敛；
使用之前的GMM，生成新的41维随机数据，使用主成分分析逆变换转换到64维，可视化新生成的一些数字的结果。  
总结：给定一个手写数字的采样，我们已经对该数据的分布建模，这样我们就可以从数据生成全新的数字样本。这种数字生成模型作为贝叶斯生成分类器的一个组成部分是非常有用的。
## 01.14-KernelDensityEstimation
1. Motivating KDE: Histograms
密度估计器是一种试图对生成数据集的概率分布进行建模的算法。对于一维数据，一个简单的密度估计器:直方图。直方图将数据分成离散的箱子，计算每个箱子里的点数，然后以直观的方式将结果可视化。使用plt.hist()进行绘图时，可以指定normed=True得到一个归一化的直方图，使得直方图下面的总面积等于1，其中箱子的高度不反映计数，而是反映概率密度。  
使用直方图作为密度估计器的一个问题是，bin大小和位置的选择可能导致具有不同性质特征的表示。直方图内块堆栈的高度通常不是反映附近点的实际密度，而是反映箱子与数据点对齐的巧合，因此我们可以在x轴上的每个位置添加块所代表的点的贡献，虽然结果有些混乱，但是相比标准的直方图能够更好反应数据的概率分布。我们进一步可以在每个位置使用平滑函数替代块，这样就能行程关于数据分布的更加精确的概念。
2. Kernel Density Estimation in Practice
核密度估计的自由参数是kernel，它指定了分布在每个点上的形状，还有kernel bandwidth，它控制着每个点上的核的大小。  
Selecting the bandwidth via cross-validation：KDE中带宽的选择对于找到合适的密度估计是非常重要的，并且是控制密度估计中的偏差方差权衡的旋钮:太窄的带宽会导致高方差估计(即(如过拟合)，即一个点的存在或不存在会造成很大的差异。带宽太宽会导致高偏差估计(即(非拟合)数据中的结构被广泛的内核所淘汰。可以使用GridSearchCV和LeaveOneOut确定最优带宽。
3. Example: KDE on a Sphere
可视化物种分布
4. Example: Not-So-Naive Bayes
建立KDEClassifier，使用网格搜索确定最优带宽，分类的数字准确率96.7%，使用高斯朴素贝叶斯模型分类，准确率只有81.9%
## 01.15-ImageFeatures
