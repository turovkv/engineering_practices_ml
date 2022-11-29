from typing import Any, Dict, Union


class DecisionTreeLeaf:
    """

    Attributes
    ----------
    y : Тип метки (напр., int или str)
        Метка класса, который встречается чаще всего среди элементов листа дерева
    """

    def __init__(self, d: Dict[Any, float]):
        self.d = d
        self.y = max(d.keys(), key=lambda k: d[k])


class DecisionTreeNode:
    """

    Attributes
    ----------
    split_dim : int
        Измерение, по которому разбиваем выборку.
    split_value : float
        Значение, по которому разбираем выборку.
    left : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] < split_value.
    right : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] >= split_value.
    """

    def __init__(
        self,
        split_dim: int,
        split_value: float,
        left: Union["DecisionTreeNode", DecisionTreeLeaf] = None,
        right: Union["DecisionTreeNode", DecisionTreeLeaf] = None,
    ):
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right
