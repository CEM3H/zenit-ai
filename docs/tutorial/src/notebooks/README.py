# -*- coding: utf-8 -*-
# %% [markdown]
# # Разработка и поддержка библиотеки ZenitAI

# %% [markdown]
# ## Общие положения

# %% [markdown]
# 1. ZenitAI - это Python-библиотека
# 2. Версионируется с помомощью git и setuptools, в соответствии с официальными рекомендациями Python
# 3. Проект использует автоматические форматтеры кода и линтеры. Они будут проверять и по возможности форматировать код при попытке отправить файлы в git. Если будут сильно мешать можно отключить, удалив файлы:
#     - `pyproject.toml` - содержит конфиг для автоматического форматтера кода `black`
#     - `setup.cfg` - конфиг для линтера `flake8`
#     - `.pre-commit-config.yaml`

# %% [markdown]
# ### Расположение репозитория

# %% [markdown]
# Основной репозиторий с проектом хранится в общей папке:  
#     
#     K:\ДМиРТ\Управление моделирования\#Zenit.ai\git_repos\zenitai-lib.git
#

# %% [markdown]
# ### Требуемые пакеты Python

# %% [markdown]
# - python
# - numpy
# - pandas
# - matplotlib
# - scikit-learn
# - openpyxl
# - ipython
# - tqdm
# - scipy
# - sphinx

# %% [markdown]
# ### Ссылки на источники

# %% [markdown]
# - Настройка git на сервере: 
#     [ссылка](https://git-scm.com/book/ru/v2/Git-%D0%BD%D0%B0-%D1%81%D0%B5%D1%80%D0%B2%D0%B5%D1%80%D0%B5-%D0%9F%D1%80%D0%BE%D1%82%D0%BE%D0%BA%D0%BE%D0%BB%D1%8B)
# - Блог-пост про git-hooks и автоформаттеры:
#     [ссылка](https://ternaus.blog/tutorial/2020/08/28/Trained-model-what-is-next.html)
#

# %% [markdown]
# ## Структура решения

# %% [markdown]
#
# Папка с проектом на данный момент выглядит так
# ![tree](img/tree.PNG)

# %% [markdown]
# Краткое описание папок:  
# - `build` - содержит файлы, необходимые для сборки пакета
# - `dist` - содержит архивы со всеми собранными версиями библиотеки - из этих архивов можно устанавливать через `pip install`
# - `docs` - содержит файлы, необходимые для автоматического создания документации. Документация генерируется на основе docstrings функциях/классах и этих файлах с помощью библиотеки _Sphinx_
#     - `tutorial` - здесь лежит этот гайд
# - `zenitai` - папка с исходным кодом проекта  
#    - `experiment` - класс Experiment для контроля экспериментов и запусков моделей
#    - `transform` - инструменты для преобразования данных (в частности - WoE-трансформер)
#    - `utils` - функции, которым пока не нашлось отдельного раздела: функции-хелперы, функции для вычисления метрик (вроде расчета Gini на основе ROC AUC и отрисовки самих ROC-кривых)
# - `ZenitAI.egg-info` - служебные данные, которые создаются при установке пакета
#
# Краткое описание файлов:
# - `.gitignore` - список файлов, папок и шаблонов, которые будут игнорироваться гитом при версионировании
# - `pyproject.toml` - содержит конфиг для автоматического форматтера кода `black`
# - `setup.cfg` - конфиг для модуля `setuptools` и для линтера `flake8`
# - `setup.py` - тоже конфиг для модуля `setuptools`
#
# Кроме того, есть еще файлы с конфигами:
# - `.pre-commit-config.yaml` - конфиг для git-хуков (git-hooks). Нужен для автоматического форматирования, линтинга (проверки на соответствие стиля кода) и проверки кода при коммитах в git. При коммите будет предпринята попытка отформатировать код, если она будет неудачной, коммит сделать не получится до устранения замечаний. Это нужно, чтобы поддерживать код в читабельном состоянии и для соблюдения некоторых правил оформления в случае совместной работы 

# %% [markdown]
# ## Начало работы

# %% [markdown]
# Предлагаю организовать работу над библиотекой в таком формате:
# - Разработка и отладка функционала библиотеки локально на рабочем ПК
# - Сборка библиотеки под новой версией 
#     - Автогенерация документации
#     - Сборка самого пакета
# - Сохранение результатов в git-репозитории в общей папке

# %% [markdown]
# **Более подробно:**  
#
# 0. Установить git на рабочий ПК: 
#     [ссылка для скачивания](https://git-scm.com/download/win)
# 1. Склонировать репозиторий на рабочий ПК: в терминале (в командной строке, или в git bash, или в терминале IDE) набрать  
#         git clone "K:\ДМиРТ\Управление моделирования\#Zenit.ai\zenitai" <путь к папке, где будет лежать проект>
#     Например, вот так:
#     - Правой кнопкой на пустое место в Проводнике
#     ![git_bash_0](img/git_bash_0.PNG)  
#     
#     - Вводим команду
#     ![git_clone](img/git_clone.PNG)
# 2. Создать и активировать виртуальное окружение для проекта:
#         conda create -n <environment_name> 
#         conda activate dashboards
#         conda install pandas numpy <...>
#         cd <полный путь к папке>     #перемещение консоли в папку проекта
# 3. Установить пакет в режиме редактирования:
#         pip install -e .
#    Это позволит импортировать подмодули проекта в самих файлах, а также установит в окружение (почти) ве нужные библиотеки
# 4. Внести нужные изменения  
# 5. Проверить, что локально функционал работает так, как задумано
# 6. Изменить номер версии библиотеки в файле `setup.py` в корне папки с библиотекой:
#     ![version](img/version.PNG)
#     
#     
# 7. Сгенерировать документацию с помощью библиотеки [Sphinx](https://www.sphinx-doc.org/en/master/). Документация генерируется в виде сайта на основе специальных файлов разметки формата [reStructuredText](https://ru.wikipedia.org/wiki/ReStructuredText) и docstrings, прописанных в функциях и классах.  
#     Чтобы запустить автоматическую генерацию документации, нужно, находясь в папке с проектом, выполнить в консоли команду 
#         make html
#     ![make_html](img/make.PNG)
#     
#     В результате должна появиться серия сообщений вроде этой:
#     ![make_res](img/make_res.PNG)
#         
# 8. [Если накопилось достаточно изменений для нового релиза], необходимо создать новую версию библиотеки c помощью `setuptools`:
#         python -m setup sdist
#     ![setup](img/setup.PNG)
# 6. Отправить изменения в центральный репозиторий
#         git add .
#         git commit -m '<текст сообщения>'
#         git push origin master
# 7. Изменения в общей папке применятся автоматически.  
#    Чтобы применить изменения вручную, можно сделать следующее:
#     - Открыть в проводнике папку _K:\ДМиРТ\Управление моделирования\\#Zenit.ai\zenitai_
#     - Кликнуть правой кнопкой по пустому месту в папке, выбрать пункт `Git Bash here`
#         ![git_bash](img/git_bash.PNG)
#     - Выполнить команду для получения изменений  
#             git pull origin master
