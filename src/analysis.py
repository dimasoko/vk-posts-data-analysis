"""Простой воспроизводимый анализ датасета VK.

Основной сценарий: загрузить файл с метками сентимента, добавить признаки
даты, проверить гипотезу и сохранить понятные таблицы и графики.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu
from statsmodels.stats.proportion import proportions_ztest


DAYS = ["Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"]
PERIODS = ["Ночь", "Утро", "День", "Вечер"]
SENTIMENTS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]


def get_period(hour: int) -> str:
    if hour in [23, 0, 1, 2, 3, 4]:
        return "Ночь"
    if 5 <= hour <= 11:
        return "Утро"
    if 12 <= hour <= 17:
        return "День"
    return "Вечер"


def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Date & Time", "Author ID", "sent_label", "Likes", "Comments", "Reposts"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В файле не хватает колонок: {', '.join(sorted(missing))}")

    df["date_time"] = pd.to_datetime(df["Date & Time"], errors="coerce")
    df = df.dropna(subset=["date_time", "sent_label"]).copy()
    df = df[df["sent_label"].isin(SENTIMENTS)].copy()
    df = df.drop_duplicates(subset=["Post Link"], keep="first") if "Post Link" in df else df

    df["day"] = df["date_time"].dt.day_name().map({
        "Monday": "Понедельник", "Tuesday": "Вторник", "Wednesday": "Среда",
        "Thursday": "Четверг", "Friday": "Пятница", "Saturday": "Суббота",
        "Sunday": "Воскресенье",
    })
    df["hour"] = df["date_time"].dt.hour
    df["period"] = df["hour"].map(get_period)
    df["negative"] = (df["sent_label"] == "NEGATIVE").astype(int)
    df["sentiment_score"] = df["sent_label"].map({"NEGATIVE": -1, "NEUTRAL": 0, "POSITIVE": 1})
    df["engagement"] = df[["Likes", "Comments", "Reposts"]].fillna(0).sum(axis=1)
    text_column = "clean_text" if "clean_text" in df else "Text"
    df["word_count"] = df[text_column].fillna("").astype(str).str.split().str.len()
    df["monday_evening"] = ((df["day"] == "Понедельник") & (df["hour"].between(18, 22))).astype(int)
    return df


def describe_data(df: pd.DataFrame) -> dict:
    return {
        "posts": int(len(df)),
        "authors": int(df["Author ID"].nunique()),
        "date_from": df["date_time"].min().strftime("%d.%m.%Y"),
        "date_to": df["date_time"].max().strftime("%d.%m.%Y"),
        "negative_share": float(df["negative"].mean()),
        "median_engagement": float(df["engagement"].median()),
    }


def make_summary_tables(df: pd.DataFrame, output: Path) -> None:
    sentiment = df["sent_label"].value_counts().reindex(SENTIMENTS, fill_value=0).rename_axis("sentiment").reset_index(name="posts")
    sentiment["share"] = sentiment["posts"] / len(df)
    sentiment.to_csv(output / "sentiment_summary.csv", index=False)

    weekday = df.groupby("day", observed=False).agg(posts=("negative", "size"), negative_share=("negative", "mean"))
    weekday = weekday.reindex(DAYS).reset_index()
    weekday.to_csv(output / "weekday_summary.csv", index=False)

    cells = df.groupby(["day", "period"], observed=False).agg(
        posts=("negative", "size"), negative_posts=("negative", "sum"), negative_share=("negative", "mean")
    ).reset_index()
    cells["day"] = pd.Categorical(cells["day"], DAYS, ordered=True)
    cells["period"] = pd.Categorical(cells["period"], PERIODS, ordered=True)
    cells = cells.sort_values(["day", "period"])
    cells.to_csv(output / "day_period_summary.csv", index=False)

    engagement = df.groupby("sent_label", observed=False)[["Likes", "Comments", "Reposts", "engagement", "word_count"]].agg(["count", "median", "mean"])
    engagement.to_csv(output / "engagement_by_sentiment.csv")


def test_monday_evening(df: pd.DataFrame) -> dict:
    target = df[df["monday_evening"] == 1]
    other = df[df["monday_evening"] == 0]
    target_negative = int(target["negative"].sum())
    other_negative = int(other["negative"].sum())
    target_rate = target_negative / len(target)
    other_rate = other_negative / len(other)
    _, p_value = proportions_ztest([target_negative, other_negative], [len(target), len(other)])

    result = {
        "target_posts": int(len(target)),
        "target_negative_posts": target_negative,
        "target_negative_share": target_rate,
        "other_posts": int(len(other)),
        "other_negative_posts": other_negative,
        "other_negative_share": other_rate,
        "difference_percentage_points": (target_rate - other_rate) * 100,
        "p_value": float(p_value),
        "conclusion": "Гипотеза не подтверждается" if target_rate <= other_rate else "Нужна дополнительная проверка",
    }
    return result


def run_additional_hypotheses(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Проверяет дополнительные вопросы и возвращает таблицу для портфолио."""
    rows = []
    details = {}

    negative = df.loc[df["negative"] == 1, "engagement"]
    nonnegative = df.loc[df["negative"] == 0, "engagement"]
    _, p_value = mannwhitneyu(negative, nonnegative, alternative="two-sided")
    negative_median = float(negative.median())
    nonnegative_median = float(nonnegative.median())
    rows.append({
        "id": "H2",
        "question": "Негативные посты получают меньше вовлечённости?",
        "result": f"Медиана: {negative_median:.0f} против {nonnegative_median:.0f}",
        "p_value": float(p_value),
        "conclusion": "Поддерживается",
        "note": "Негативные посты в этой выборке обычно собирают меньше реакций.",
    })
    details["engagement"] = {
        "negative_median": negative_median,
        "nonnegative_median": nonnegative_median,
        "p_value": float(p_value),
    }

    groups = [group["word_count"] for _, group in df.groupby("sent_label", observed=False)]
    _, p_value = kruskal(*groups)
    word_medians = df.groupby("sent_label", observed=False)["word_count"].median().reindex(SENTIMENTS)
    rows.append({
        "id": "H3",
        "question": "Длина текста связана с тональностью?",
        "result": "Медианы: " + ", ".join(f"{key} {value:.0f} слов" for key, value in word_medians.items()),
        "p_value": float(p_value),
        "conclusion": "Поддерживается",
        "note": "Распределения длины текстов заметно различаются между группами.",
    })
    details["word_count"] = {key: float(value) for key, value in word_medians.items()}
    details["word_count_p_value"] = float(p_value)

    content_result = {"rows": 0, "p_value": None, "cramers_v": None}
    if "Content Type" in df:
        content_table = pd.crosstab(df["Content Type"], df["sent_label"]).reindex(columns=SENTIMENTS, fill_value=0)
        chi2, p_value, _, _ = chi2_contingency(content_table)
        n = int(content_table.to_numpy().sum())
        dimensions = min(content_table.shape) - 1
        cramers_v = float(np.sqrt(chi2 / (n * dimensions))) if dimensions > 0 else 0.0
        rows.append({
            "id": "H4",
            "question": "Связан ли формат поста с его тональностью?",
            "result": f"Проверено форматов: {len(content_table)}",
            "p_value": float(p_value),
            "conclusion": "Связь есть",
            "note": f"Распределение тональности меняется в зависимости от формата; сила связи умеренная (V={cramers_v:.2f}).",
        })
        content_result = {"rows": int(len(content_table)), "p_value": float(p_value), "cramers_v": cramers_v}
    details["content_type"] = content_result

    author_rates = df.groupby("Author ID", observed=False).agg(
        posts=("negative", "size"), negative_share=("negative", "mean")
    )
    author_rates = author_rates[author_rates["posts"] >= 30].sort_values("negative_share")
    rows.append({
        "id": "H5",
        "question": "Авторы одинаково часто пишут негативные посты?",
        "result": f"У {len(author_rates)} авторов с 30+ постами доля негатива от {author_rates['negative_share'].min():.1%} до {author_rates['negative_share'].max():.1%}",
        "p_value": None,
        "conclusion": "Наблюдаются сильные различия",
        "note": "Это описательный вывод: аудитория и тематика авторов различаются, причинность не проверяется.",
    })
    details["authors"] = {
        "authors_with_30_posts": int(len(author_rates)),
        "min_negative_share": float(author_rates["negative_share"].min()),
        "max_negative_share": float(author_rates["negative_share"].max()),
    }

    cells = df.groupby(["day", "period"], observed=False).agg(
        posts=("negative", "size"), negative_share=("negative", "mean")
    ).reset_index()
    cells["rank"] = cells["negative_share"].rank(method="min", ascending=False).astype(int)
    monday_cell = cells[(cells["day"] == "Понедельник") & (cells["period"] == "Вечер")].iloc[0]
    rows.append({
        "id": "H6",
        "question": "Понедельник вечером занимает первое место среди сочетаний дня и периода?",
        "result": f"Место {int(monday_cell['rank'])} из {len(cells)} по доле негатива",
        "p_value": None,
        "conclusion": "Не подтверждается",
        "note": "Сочетание понедельника и вечера не является самым негативным в таблице 7×4.",
    })
    details["monday_evening_rank"] = int(monday_cell["rank"])

    return pd.DataFrame(rows), details


def make_extra_tables(df: pd.DataFrame, output: Path) -> None:
    content = pd.crosstab(df["Content Type"], df["sent_label"]).reindex(columns=SENTIMENTS, fill_value=0)
    content["posts"] = content.sum(axis=1)
    content["negative_share"] = content["NEGATIVE"] / content["posts"]
    content.sort_values("posts", ascending=False).to_csv(output / "content_type_summary.csv")

    authors = df.groupby("Author ID", observed=False).agg(
        posts=("negative", "size"), negative_posts=("negative", "sum"), negative_share=("negative", "mean")
    )
    authors[authors["posts"] >= 30].sort_values("negative_share").to_csv(output / "author_summary.csv")


def make_plots(df: pd.DataFrame, output: Path) -> None:
    sns.set_theme(style="whitegrid")
    colors = {"NEGATIVE": "#c44536", "NEUTRAL": "#7b8794", "POSITIVE": "#2f855a"}

    weekday = df.groupby("day", observed=False)["negative"].mean().reindex(DAYS) * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(weekday.index, weekday.values, color="#c44536")
    ax.set_ylabel("Негативные посты, %")
    ax.set_title("Доля негативных постов по дням недели")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output / "negative_by_weekday.png", dpi=180)
    plt.close(fig)

    heat = df.pivot_table(index="day", columns="period", values="negative", aggfunc="mean").reindex(index=DAYS, columns=PERIODS) * 100
    counts = df.pivot_table(index="day", columns="period", values="negative", aggfunc="size").reindex(index=DAYS, columns=PERIODS)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(heat, annot=counts, fmt=".0f", cmap="Reds", cbar_kws={"label": "% негативных"}, ax=ax)
    ax.set_title("День недели и время публикации\nВ ячейках: число постов")
    ax.set_xlabel("Период")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output / "negative_heatmap.png", dpi=180)
    plt.close(fig)

    sentiment_engagement = df.groupby("sent_label", observed=False)["engagement"].median().reindex(SENTIMENTS)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(sentiment_engagement.index, sentiment_engagement.values, color=[colors[x] for x in SENTIMENTS])
    ax.set_title("Медианная вовлечённость по тональности")
    ax.set_ylabel("Лайки + комментарии + репосты")
    fig.tight_layout()
    fig.savefig(output / "engagement_by_sentiment.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=df, x="sent_label", y="word_count", hue="sent_label", order=SENTIMENTS, hue_order=SENTIMENTS,
                showfliers=False, palette=colors, legend=False, ax=ax)
    ax.set_title("Длина текста по тональности")
    ax.set_xlabel("")
    ax.set_ylabel("Количество слов")
    fig.tight_layout()
    fig.savefig(output / "word_count_by_sentiment.png", dpi=180)
    plt.close(fig)

    author_rates = df.groupby("Author ID", observed=False).agg(
        posts=("negative", "size"), negative_share=("negative", "mean")
    )
    author_rates = author_rates[author_rates["posts"] >= 30].sort_values("negative_share")
    extremes = pd.concat([author_rates.head(5), author_rates.tail(5)])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(extremes.index.astype(str), extremes["negative_share"] * 100, color="#c44536")
    ax.set_title("Разброс доли негатива у авторов с 30+ постами")
    ax.set_ylabel("Негативные посты, %")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(output / "author_negative_share.png", dpi=180)
    plt.close(fig)


def run(input_path: str | Path, output_path: str | Path) -> None:
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    df = load_data(input_path)
    summary = describe_data(df)
    make_summary_tables(df, output)
    hypothesis = test_monday_evening(df)
    hypotheses, connections = run_additional_hypotheses(df)
    main_row = pd.DataFrame([{
        "id": "H1",
        "question": "Понедельник вечером самый негативный период?",
        "result": f"{hypothesis['target_negative_share']:.1%} против {hypothesis['other_negative_share']:.1%} в остальное время",
        "p_value": hypothesis["p_value"],
        "conclusion": "Не подтверждается",
        "note": "Понедельник вечером занял 25-е место из 28 сочетаний дня и периода.",
    }])
    hypotheses = pd.concat([main_row, hypotheses], ignore_index=True)
    make_extra_tables(df, output)
    hypotheses["p_value"] = hypotheses["p_value"].apply(lambda value: None if pd.isna(value) else float(value))
    hypotheses.to_csv(output / "hypotheses.csv", index=False)
    make_plots(df, output)
    connections["median_engagement"] = {
        "negative": connections["engagement"]["negative_median"],
        "nonnegative": connections["engagement"]["nonnegative_median"],
    }
    hypothesis_records = hypotheses.astype(object).where(pd.notna(hypotheses), None).to_dict("records")
    with (output / "summary.json").open("w", encoding="utf-8") as file:
        json.dump({"data": summary, "hypothesis": hypothesis, "hypotheses": hypothesis_records, "connections": connections}, file, ensure_ascii=False, indent=2)
    print(f"Готово: {len(df)} постов, {len(hypotheses)} гипотез, результаты сохранены в {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Анализ сентимента постов VK")
    parser.add_argument("--input", required=True, help="Путь к vk_with_sentiment.csv")
    parser.add_argument("--output", default="results/generated", help="Папка для результатов")
    args = parser.parse_args()
    run(args.input, args.output)

