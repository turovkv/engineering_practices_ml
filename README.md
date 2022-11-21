# ДЗ 2


## установка пакетного менеджера.

```
curl -sSL https://install.python-poetry.org | python3 -
```

## развертывание окружения.
```
poetry install   
```

## сборка пакета:

добавляем pypi-test
```
poetry config repositories.test-pypi https://test.pypi.org/legacy/
```

добавляем токен доступа к нашему аккаунту в test.pypi.org
```
poetry config pypi-token.test-pypi  pypi-*****
```

собираем и публикуем
```
poetry publish --build --repository test-pypi
```

## ссылка на пакет в pypi-test.
```
https://test.pypi.org/project/turovkv-hw-eng-prac-ml/
```

## установка пакета из pypi-test.
```
pip install -i https://test.pypi.org/simple/ turovkv-hw-eng-prac-ml
```
или
```
poetry source add testpypi https://test.pypi.org/simple/ 
poetry add --source testpypi turovkv-hw-eng-prac-ml
```