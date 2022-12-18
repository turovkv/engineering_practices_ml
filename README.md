# ДЗ 5

Инструмент для пайплайнов: DVC

Запустить пайплайн: `dvc repro`

DAG:
можно получить командой `dvc dag`.
можно почитать его описание в `dvc.yaml`

```
+--------------------+
| data/train.csv.dvc |
+--------------------+
           *
           *
           *
   +--------------+
   | prepare_data |
   +--------------+
           *
           *
           *
      +-------+
      | train |
      +-------+
```

# ДЗ 4

запустить эксперимент   `dvc exp run`

посмотреть результаты `dvc exp show`

# ДЗ 3

форматирование: isort, black


flake8 plugins:
- flake8-return
- flake8-use-fstring
- flake8-match
- flake8-simplify
- flake8-unused-arguments


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
