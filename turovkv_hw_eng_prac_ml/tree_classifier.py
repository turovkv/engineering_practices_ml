from typing import Callable, Optional, NoReturn, List, Dict, Any

import numpy as np

from tree_classifier.tree import DecisionTreeLeaf, DecisionTreeNode


def gini(x: np.ndarray) -> float:
    """
    Считает коэффициент Джини для массива меток x.
    """
    unique, counts = np.unique(x, return_counts=True)
    counts = counts / x.shape[0]
    return (counts * (1 - counts)).sum()


def entropy(x: np.ndarray) -> float:
    """
    Считает энтропию для массива меток x.
    """
    unique, counts = np.unique(x, return_counts=True)
    counts = counts / x.shape[0]
    return -(counts * np.log2(counts)).sum()


def gain(left_y: np.ndarray, right_y: np.ndarray, criterion: Callable) -> float:
    """
    Считает информативность разбиения массива меток.

    Parameters
    ----------
    left_y : np.ndarray
        Левая часть разбиения.
    right_y : np.ndarray
        Правая часть разбиения.
    criterion : Callable
        Критерий разбиения.
    """
    ln, rn = left_y.shape[0], right_y.shape[0]
    cur = (ln + rn) * criterion(np.concatenate([left_y, right_y]))
    l = ln * criterion(left_y)
    r = rn * criterion(right_y)
    return cur - l - r


# ### Задание 2 (1 балл)
# Деревья решений имеют хорошую интерпретируемость, т.к. позволяют не только предсказать класс, но и объяснить, почему мы предсказали именно его. Например, мы можем его нарисовать. Чтобы сделать это, нам необходимо знать, как оно устроено внутри. Реализуйте классы, которые будут задавать структуру дерева.
#
# #### DecisionTreeLeaf
# Поля:
# 1. `y` должно содержать класс, который встречается чаще всего среди элементов листа дерева
#
# #### DecisionTreeNode
# В данной домашней работе мы ограничемся порядковыми и количественными признаками, поэтому достаточно хранить измерение и значение признака, по которому разбиваем обучающую выборку.
#
# Поля:
# 1. `split_dim` измерение, по которому разбиваем выборку
# 2. `split_value` значение, по которому разбираем выборку
# 3. `left` поддерево, отвечающее за случай `x[split_dim] < split_value`. Может быть `DecisionTreeNode` или `DecisionTreeLeaf`
# 4. `right` поддерево, отвечающее за случай `x[split_dim] >= split_value`. Может быть `DecisionTreeNode` или `DecisionTreeLeaf`
#
# __Интерфейс классов можно и нужно менять при необходимости__

# In[72]:


# ### Задание 3 (3 балла)
# Теперь перейдем к самому дереву решений. Реализуйте класс `DecisionTreeClassifier`.
#
# #### Описание методов
# `fit(X, y)` строит дерево решений по обучающей выборке.
#
# `predict_proba(X)` для каждого элемента из `X` возвращает словарь `dict`, состоящий из пар `(класс, вероятность)`
#
# #### Описание параметров конструктора
# `criterion="gini"` - задает критерий, который будет использоваться при построении дерева. Возможные значения: `"gini"`, `"entropy"`.
#
# `max_depth=None` - ограничение глубины дерева. Если `None` - глубина не ограничена
#
# `min_samples_leaf=1` - минимальное количество элементов в каждом листе дерева.
#
# #### Описание полей
# `root` - корень дерева. Может быть `DecisionTreeNode` или `DecisionTreeLeaf`

# In[73]:


class DecisionTreeClassifier:
    """
    Attributes
    ----------
    root : Union[DecisionTreeNode, DecisionTreeLeaf]
        Корень дерева.

    (можете добавлять в класс другие аттрибуты).

    """

    def __init__(self, criterion: str = "gini",
                 max_depth: Optional[int] = None,
                 min_samples_leaf: int = 1):
        """
        Parameters
        ----------
        criterion : str
            Задает критерий, который будет использоваться при построении дерева.
            Возможные значения: "gini", "entropy".
        max_depth : Optional[int]
            Ограничение глубины дерева. Если None - глубина не ограничена.
        min_samples_leaf : int
            Минимальное количество элементов в каждом листе дерева.

        """
        self.root = None
        self.criterion = None
        if criterion == "gini":
            self.criterion = gini
        if criterion == "entropy":
            self.criterion = entropy
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def create_leaf(self, XY: np.ndarray):
        unique, counts = np.unique(XY[:, -1], return_counts=True)
        return DecisionTreeLeaf(
            dict(
                zip(
                    list(map(lambda yi: self.float_label[yi], unique)),
                    counts / XY.shape[0]
                )
            )
        )

    def create_node(self, XY: np.ndarray):
        rn, cn = XY.shape
        maxd = -1
        maxri = -1
        maxig = -1
        for ci in range(cn - 1):
            XY = XY[XY[:, ci].argsort()]
            for ri in range(rn):
                curig = gain(XY[:ri + 1, -1], XY[ri + 1:, -1], self.criterion)
                if maxig < curig:
                    maxig = curig
                    maxd = ci
                    maxri = ri
        XY = XY[XY[:, maxd].argsort()]
        l, r = XY[:maxri + 1], XY[maxri + 1:]
        sep_val = XY[maxri, maxd]
        return DecisionTreeNode(maxd, sep_val), l, r

    def build(self, XY, depth=1):
        if XY.shape[0] // 2 < self.min_samples_leaf or depth >= self.max_depth:
            return self.create_leaf(XY)
        n, l, r = self.create_node(XY)
        n.left = self.build(l, depth + 1)
        n.right = self.build(r, depth + 1)
        return n

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Строит дерево решений по обучающей выборке.

        Parameters
        ----------
        X : np.ndarray
            Обучающая выборка.
        y : np.ndarray
            Вектор меток классов.
        """
        labels = np.unique(y)
        self.label_float = {}
        self.float_label = {}
        for i, label in enumerate(labels):
            self.label_float[label] = float(i)
            self.float_label[i] = label
        y_float = np.array(list(map(lambda yi: self.label_float[yi], y)))
        XY = np.c_[X, y_float]
        self.root = self.build(XY)

    def find_leaf(self, x: np.ndarray, cur_node):
        if isinstance(cur_node, DecisionTreeLeaf):
            return cur_node.d
        if x[cur_node.split_dim] <= cur_node.split_value:
            return self.find_leaf(x, cur_node.left)
        else:
            return self.find_leaf(x, cur_node.right)

    def predict_proba(self, X: np.ndarray) -> List[Dict[Any, float]]:
        """
        Предсказывает вероятность классов для элементов из X.

        Parameters
        ----------
        X : np.ndarray
            Элементы для предсказания.

        Return
        ------
        List[Dict[Any, float]]
            Для каждого элемента из X возвращает словарь
            {метка класса -> вероятность класса}.
        """

        return list(map(lambda x: self.find_leaf(x, self.root), X))

    def predict(self, X: np.ndarray) -> list:
        """
        Предсказывает классы для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Элементы для предсказания.

        Return
        ------
        list
            Вектор предсказанных меток для элементов X.
        """
        proba = self.predict_proba(X)
        return [max(p.keys(), key=lambda k: p[k]) for p in proba]