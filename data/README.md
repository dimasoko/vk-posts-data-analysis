# Данные

Полные CSV-файлы не хранятся в репозитории.

1. Скачайте исходный файл `vk_filtered_posts_metadata.csv` со страницы датасета Kaggle:
   <https://www.kaggle.com/datasets/nooruzbekt/vk-social-media-posts-metadata>
2. Для основного анализа нужен файл `vk_with_sentiment.csv`.
3. Его можно получить в едином ноутбуке `notebooks/01_vk_full_analysis_colab.ipynb`.
4. Положите готовый файл в эту папку или загрузите его прямо в Colab.

Файл с результатами сентимента содержит исходные поля поста и дополнительные колонки:
`clean_text`, `sent_label`, `sent_score`.

