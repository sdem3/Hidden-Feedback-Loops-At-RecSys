# Руководство по созданию научной презентации в LaTeX Beamer (Overleaf)

Этот документ содержит все правила и шаблоны для создания новой презентации в Overleaf на основе существующего проекта. Используй его как инструкцию при работе с LLM-ассистентом.

---

## 1. Технический стек

- **Формат:** LaTeX + Beamer
- **Компилятор:** XeLaTeX (обязательно, не pdflatex — используется `fontspec`)
- **Редактор:** Overleaf → Menu → Compiler → выбрать **XeLaTeX**
- **Структура проекта в Overleaf:**
  ```
  main.tex        ← главный файл
  refs.bib        ← список литературы
  pictures/       ← папка с изображениями (PNG/PDF)
  ```

---

## 2. Полная преамбула (копировать без изменений)

```latex
\documentclass[aspectratio=169]{beamer}
\usepackage[utf8]{inputenc}
\usepackage{xeCJK}
\usepackage{graphicx}
\usepackage{mathtools}
\usepackage{utopia}
\usetheme{CambridgeUS}
\usecolortheme{dolphin}

% Поддержка русского языка
\usepackage[utf8]{inputenc}
\usepackage[T2A, T1]{fontenc}
\usepackage{fontspec}
\usefonttheme{serif}
\setmainfont{Times New Roman}

% Цветовая схема (фиолетовая палитра)
\definecolor{myNewColorA}{RGB}{126,12,110}
\definecolor{myNewColorB}{RGB}{165,85,154}
\definecolor{myNewColorC}{RGB}{203,158,197}
\setbeamercolor*{palette primary}{bg=myNewColorC}
\setbeamercolor*{palette secondary}{bg=myNewColorB, fg=white}
\setbeamercolor*{palette tertiary}{bg=myNewColorA, fg=white}
\setbeamercolor*{titlelike}{fg=myNewColorA}
\setbeamercolor*{title}{bg=myNewColorA, fg=white}
\setbeamercolor*{item}{fg=myNewColorA}
\setbeamercolor*{caption name}{fg=myNewColorA}
\usefonttheme{professionalfonts}

\usepackage{natbib}
\usepackage{hyperref}

% Размеры шрифтов на титульном слайде
\setbeamerfont{title}{size=\large}
\setbeamerfont{subtitle}{size=\small}
\setbeamerfont{author}{size=\small}
\setbeamerfont{date}{size=\small}
\setbeamerfont{institute}{size=\small}
```

> **Ничего не менять в преамбуле** — пакеты, тема и цвета должны быть именно такими.

---

## 3. Титульный слайд

### Шаблон метаданных (после преамбулы, до `\begin{document}`)

```latex
\title[Краткое название]{Полное название презентации}
\subtitle{Подзаголовок или тип работы}
\author[Фамилия]{Имя Фамилия \and Второй Автор \and Третий Автор}
\institute[email@phystech.edu]{
МФТИ\\
Долгопрудный, Россия}
\date[Семестр год]{Семестр год}
```

### Вставка титульного слайда (первая строка в `\begin{document}`)

```latex
\begin{document}
\frame{\titlepage}
```

---

## 4. Структура документа

```latex
\begin{document}

\frame{\titlepage}                   % Титульный слайд

\section{Название раздела 1}
    \begin{frame}{Заголовок слайда}
        % содержимое
    \end{frame}

\section{Название раздела 2}
    \begin{frame}{Заголовок слайда}
        % содержимое
    \end{frame}

% ... другие разделы ...

\section{References}
    \begin{frame}{References}
        \tiny
        \bibliographystyle{plain}
        \bibliography{refs}
    \end{frame}

\end{document}
```

### Рекомендуемые разделы для научной презентации

1. Introduction / Введение
2. Problem Statement / Постановка задачи
3. Main Results / Основные результаты
4. Experiments / Эксперименты
5. Conclusion / Заключение
6. References

---

## 5. Правила оформления слайдов

### Размер шрифта на слайде

Выбирай в начале `\begin{frame}` в зависимости от объёма текста:

```latex
\begin{frame}{Заголовок}
    \small          % много текста
    % или
    \footnotesize   % очень много текста / формулы
    % или
    \normalsize     % мало текста
```

### Маркированные и нумерованные списки

```latex
% Нумерованный список (предпочтительный в этом стиле)
\begin{enumerate}
    \item Первый пункт
    \item Второй пункт
\end{enumerate}

% Маркированный список
\begin{itemize}
    \item Первый пункт
    \item Второй пункт
\end{itemize}
```

### Два столбца (для сравнения)

```latex
\begin{columns}
    \column{0.5\textwidth}
        Левый столбец
    \column{0.5\textwidth}
        Правый столбец
\end{columns}
```

---

## 6. Теоремы, следствия, определения

Оформляй через `\textcolor` с жирным шрифтом (стандартные beamer-блоки не используются):

```latex
% Теорема
\large{\textcolor{myNewColorA}{\textbf{Теорема 1} (Название теоремы)}}
\normalsize

Формулировка теоремы...

% Следствие
\textcolor{myNewColorA}{\textbf{Следствие 1} (Название)}

Формулировка следствия...

% Лемма / Факт
\large{\textcolor{myNewColorA}{\textbf{Лемма 1} (Факт)}}
\normalsize

Формулировка...
```

> Важно: после `\large{...}` всегда ставить `\normalsize`, чтобы текст теоремы был обычного размера.

---

## 7. Математические формулы

### Ненумерованная формула

```latex
\begin{equation*}
    f_{t+1}(x) = \mathrm{D}(f_t)(x)
\end{equation*}
```

### Нумерованная формула с меткой

```latex
\begin{equation}
    \label{eq:system}
    f_{t+1}(x) = \mathrm{D}_t(f_t)(x), \quad \forall x \in \mathbb{R}^n, \; t \in \mathbb{N}
\end{equation}
```

### Ссылка на нумерованную формулу

```latex
см. уравнение~\eqref{eq:system}
```

### Часто используемые команды

| Что нужно | Команда LaTeX |
|-----------|--------------|
| Вещественные числа | `\mathbb{R}` |
| Натуральные числа | `\mathbb{N}` |
| Положительные вещественные | `\mathbb{R}_+` |
| Оператор D | `\mathrm{D}` или `\text{D}` |
| Норма | `\|f\|_1` |
| Предел при t→∞ | `\underset{t \to \infty}{\longrightarrow}` |
| Дельта-функция | `\delta(x)` |
| Интеграл от -∞ до +∞ | `\int\limits_{-\infty}^{+\infty}` |
| Для всех | `\forall` |
| Существует | `\exists` |
| Стрелка "определяется как" | `\hookrightarrow` |
| Дробь | `\dfrac{числитель}{знаменатель}` |
| Сходимость в L1 | `\overset{l_1}{\longrightarrow}` |
| Множество R (специальное) | `\mathbf{R}` |

---

## 8. Вставка изображений

### Одно изображение по центру

```latex
\begin{figure}[h!]
    \centering
    \includegraphics[width=0.6\linewidth]{pictures/filename.png}
    \caption{Описание рисунка.}
    \label{fig:label}
\end{figure}
```

### Два изображения рядом (50/50)

```latex
\begin{figure}[h!]
    \centering
    \includegraphics[width=0.49\linewidth]{pictures/left.png}
    \includegraphics[width=0.49\linewidth]{pictures/right.png}
    \caption{Левый (слева), правый (справа).}
    \label{fig:two}
\end{figure}
```

### Три изображения рядом (33/33/33)

```latex
\begin{figure}[h!]
    \centering
    \includegraphics[width=0.32\linewidth]{pictures/a.png}
    \includegraphics[width=0.32\linewidth]{pictures/b.png}
    \includegraphics[width=0.32\linewidth]{pictures/c.png}
    \caption{Описание.}
    \label{fig:three}
\end{figure}
```

> **Правило ширины:** сумма всех `width` в строке должна быть ≤ 0.99, иначе перенесётся на другую строку.

### Изображение без подписи (caption как текст)

```latex
\begin{figure}[h!]
    \centering
    \includegraphics[width=0.55\linewidth]{pictures/example.png}
    
    Описание под рисунком обычным текстом.
    \label{fig:example}
\end{figure}
```

> Загрузи изображения в Overleaf в папку `pictures/` через кнопку Upload.

---

## 9. Библиография

### Файл refs.bib — примеры записей

```bibtex
@article{author2023title,
  title={Название статьи},
  author={Author, First and Second, Author},
  journal={Journal Name},
  volume={10},
  pages={1--20},
  year={2023},
  publisher={Publisher}
}

@book{author2020book,
  title={Название книги},
  author={Author, Name},
  year={2020},
  publisher={Publisher},
  address={City}
}

@inproceedings{author2022conf,
  title={Название доклада},
  author={Author, Name},
  booktitle={Conference Name},
  pages={100--110},
  year={2022}
}
```

### Цитирование в тексте

```latex
% Одна ссылка
\cite{author2023title}

% Несколько ссылок через запятую
\cite{author2023title, author2020book}
```

### Слайд с литературой (всегда последний)

```latex
\section{References}
    \begin{frame}{References}
        \tiny
        \bibliographystyle{plain}
        \bibliography{refs}
    \end{frame}
```

---

## 10. Полный минимальный рабочий пример (MWE)

Скопируй этот шаблон в `main.tex` в Overleaf и замени содержимое:

```latex
\documentclass[aspectratio=169]{beamer}
\usepackage[utf8]{inputenc}
\usepackage{xeCJK}
\usepackage{graphicx}
\usepackage{mathtools}
\usepackage{utopia}
\usetheme{CambridgeUS}
\usecolortheme{dolphin}

\usepackage[utf8]{inputenc}
\usepackage[T2A, T1]{fontenc}
\usepackage{fontspec}
\usefonttheme{serif}
\setmainfont{Times New Roman}

\definecolor{myNewColorA}{RGB}{126,12,110}
\definecolor{myNewColorB}{RGB}{165,85,154}
\definecolor{myNewColorC}{RGB}{203,158,197}
\setbeamercolor*{palette primary}{bg=myNewColorC}
\setbeamercolor*{palette secondary}{bg=myNewColorB, fg=white}
\setbeamercolor*{palette tertiary}{bg=myNewColorA, fg=white}
\setbeamercolor*{titlelike}{fg=myNewColorA}
\setbeamercolor*{title}{bg=myNewColorA, fg=white}
\setbeamercolor*{item}{fg=myNewColorA}
\setbeamercolor*{caption name}{fg=myNewColorA}
\usefonttheme{professionalfonts}

\usepackage{natbib}
\usepackage{hyperref}

\setbeamerfont{title}{size=\large}
\setbeamerfont{subtitle}{size=\small}
\setbeamerfont{author}{size=\small}
\setbeamerfont{date}{size=\small}
\setbeamerfont{institute}{size=\small}

\title[Краткое название]{Полное название презентации}
\subtitle{Тип работы / Мероприятие}
\author[Фамилия]{Имя Фамилия \and Второй Соавтор}
\institute[email@phystech.edu]{
МФТИ\\
Долгопрудный, Россия}
\date[Весна 2025]{Весна 2025}

\begin{document}

\frame{\titlepage}

%------------------------------------------------------------
\section{Введение}

    \begin{frame}{Введение и обзор литературы}
        \small

        Мотивация и контекст исследования~\cite{example2020}.

        Проблема встречается в следующих приложениях:

        \begin{enumerate}
            \item Рекомендательные системы~\cite{example2021}
            \item Здравоохранение
            \item Предиктивная аналитика
        \end{enumerate}

        Вклад данной работы:

        \begin{enumerate}
            \item Первый вклад.
            \item Второй вклад.
        \end{enumerate}
    \end{frame}

%------------------------------------------------------------
\section{Постановка задачи}

    \begin{frame}{Постановка задачи}
        \footnotesize

        Рассмотрим множество $\mathbf{R}$ плотностей распределения:

        \begin{equation*}
            \mathbf{R} := \left\{ f : \mathbb{R}^n \to \mathbb{R}_+ \;\text{и}\;
            \int\limits_{\mathbb{R}^n} f(x)\,dx = 1 \right\}
        \end{equation*}

        Дискретная динамическая система:

        \begin{equation}
            \label{eq:system}
            f_{t+1}(x) = \mathrm{D}_t(f_t)(x), \quad \forall x \in \mathbb{R}^n,\; t \in \mathbb{N}
        \end{equation}

        Нас интересует поведение траекторий~\eqref{eq:system} при $t \to \infty$.
    \end{frame}

%------------------------------------------------------------
\section{Основные результаты}

    \begin{frame}{Основная теорема}

        \large{\textcolor{myNewColorA}{\textbf{Теорема 1} (Основной результат)}}
        \normalsize

        Если выполнено условие A и условие B, то

        \begin{equation*}
            f_t(x) \underset{t \to \infty}{\longrightarrow} \delta(x)
        \end{equation*}

        в слабом смысле, то есть для любой непрерывной $\phi$ с компактным носителем:

        \begin{equation*}
            \lim_{t \to +\infty} \int\limits_{-\infty}^{+\infty} f_t(x)\,\phi(x)\,dx = \phi(0)
        \end{equation*}

        \textcolor{myNewColorA}{\textbf{Следствие 1}}

        Формулировка следствия...
    \end{frame}

%------------------------------------------------------------
\section{Эксперименты}

    \begin{frame}{Постановка эксперимента}
        \begin{figure}[h!]
            \centering
            \includegraphics[width=0.49\linewidth]{pictures/setup_left.png}
            \includegraphics[width=0.49\linewidth]{pictures/setup_right.png}
            \caption{Схема первого эксперимента (слева) и второго (справа).}
            \label{fig:setup}
        \end{figure}
    \end{frame}

    \begin{frame}{Результаты эксперимента}
        \small

        Описание результатов.

        \begin{figure}[h!]
            \centering
            \includegraphics[width=0.6\linewidth]{pictures/result.png}
            \caption{График результатов.}
            \label{fig:result}
        \end{figure}

        \begin{enumerate}
            \item Вывод первый.
            \item Вывод второй.
        \end{enumerate}
    \end{frame}

%------------------------------------------------------------
\section{Заключение}

    \begin{frame}{Заключение}
        \begin{enumerate}
            \item Первый итог работы.
            \item Второй итог работы.
            \item Третий итог работы.
        \end{enumerate}
    \end{frame}

%------------------------------------------------------------
\section{References}

    \begin{frame}{References}
        \tiny
        \bibliographystyle{plain}
        \bibliography{refs}
    \end{frame}

\end{document}
```

---

## 11. Настройка Overleaf

1. Создай новый проект: **New Project → Blank Project**
2. Замени содержимое `main.tex` на шаблон выше
3. Создай файл `refs.bib` (кнопка New File)
4. Создай папку `pictures` и загрузи туда изображения
5. Перейди в **Menu → Compiler → XeLaTeX**
6. Нажми **Recompile**

---

## 12. Частые ошибки и решения

| Ошибка | Причина | Решение |
|--------|---------|---------|
| `Font not found` | Нет Times New Roman | Overleaf его поддерживает, убедись что компилятор XeLaTeX |
| `! LaTeX Error: File 'utopia.sty' not found` | Пакет utopia | Закомментируй `\usepackage{utopia}` |
| Рисунок не находится | Неверный путь | Проверь, что файл в папке `pictures/` и имя совпадает |
| Библиография не отображается | Не запущен bibtex | В Overleaf это происходит автоматически при Recompile |
| Текст выходит за слайд | Слишком много текста | Уменьши шрифт: `\small` → `\footnotesize` → `\tiny` |

---

## 13. Промпт для LLM-ассистента

Используй этот промпт когда просишь LLM создать слайды:

```
Создай слайд в LaTeX Beamer по следующим правилам:
- Тема CambridgeUS, coloretheme dolphin
- Цвет акцента: \textcolor{myNewColorA}{...} где myNewColorA = RGB(126,12,110)
- Шрифт: XeLaTeX + fontspec + Times New Roman
- Теоремы оформляй через \large{\textcolor{myNewColorA}{\textbf{Теорема N}}}
- Формулы в equation/equation* окружениях
- Изображения через \includegraphics[width=X\linewidth]{pictures/name.png}
- Списки через enumerate (не itemize)
- Размер шрифта на слайде: \small или \footnotesize

Содержание слайда: [опиши что должно быть на слайде]
```
