import csv
import boto3
import os
from pyspark.sql import SparkSession
from pyspark.sql import Row
from typing import Optional, Tuple

# AWS (for production run)
BUCKET_NAME = "amzn-s3-steam-reviews"
BUCKET_URI = f"s3://{BUCKET_NAME}"

reviewFilePath      = f"{BUCKET_URI}/chunk_*.csv"   # glob reads all chunks
genreFilePath       = f"{BUCKET_URI}/genres.csv"
genreByAppFilePath  = f"{BUCKET_URI}/application_genres_standardized.csv"
OUTPUT_FOLDER       = "output/non_optimized"

spark = SparkSession.builder \
    .appName("Steam Reviews Analysis - Non Optimized") \
    .config('spark.sql.shuffle.partitions', '400') \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .getOrCreate()
sc = spark.sparkContext

sc.setLogLevel("WARN")

class SteamParser:
    @staticmethod
    def parse_application_line(row) -> Optional[Tuple[int, str, int, int]]:
        try:
            parts = list(row)
            appid = int(parts[0]) if parts[0] else 0
            name = parts[1] if parts[1] else ""
            metacritic = int(parts[10]) if parts[10] else 0
            recommendations = int(parts[11]) if parts[11] else 0
            return (appid, name, metacritic, recommendations)
        except Exception as e:
            return None

    @staticmethod
    def parse_review_line(row) -> Optional[Tuple[int, bool, float, float, float]]:
        try:
            parts = list(row)
            appid = int(parts[1].strip())
            playtime_forever = float(parts[6].strip()) if parts[6].strip() else 0.0
            playtime_at_review = float(parts[8].strip()) if parts[8].strip() else 0.0
            voted_up_str = parts[14].strip().lower()
            voted_up = True if voted_up_str in ('true', '1', 'yes') else False
            weight_vote_score = float(parts[17].strip())
            if playtime_at_review < 0:
                return None
            return (appid, voted_up, playtime_forever, playtime_at_review, weight_vote_score)
        except:
            return None

    @staticmethod
    def parse_genre_line(row) -> Optional[Tuple[int, str]]:
        try:
            parts = list(row)
            genre_id = int(parts[0].strip())
            genre_name = parts[1].strip()
            if not genre_name:
                return None
            return (genre_id, genre_name)
        except:
            return None

    @staticmethod
    def parse_app_genre_line(row) -> Optional[Tuple[int, int]]:
        try:
            parts = list(row)
            appid = int(parts[0].strip())
            genre_id = int(parts[1].strip())
            return (appid, genre_id)
        except:
            return None

# =========================================================================
# FUNZIONE DI SALVATAGGIO CON BOTO3 (Sostituisce il vecchio hack JVM)
# =========================================================================
def save_csv_to_s3(rows, headers, filename):
    """Writes a list of tuples to a CSV file locally and uploads to S3."""
    local_path = f"/tmp/{filename}"
    s3_path = f"{OUTPUT_FOLDER}/{filename}"
    
    # Scrive il file in locale sul nodo
    with open(local_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    
    # Carica il file su S3
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(local_path, BUCKET_NAME, s3_path)
        print(f"Saved successfully to s3://{BUCKET_NAME}/{s3_path}")
    except Exception as e:
        print(f"Failed to upload {filename}. Error: {e}")

# =========================================================================
# DATA LOADING
# =========================================================================
df_rev = spark.read.csv(reviewFilePath, header=True, quote='"', escape='"', multiLine=True)
rddReviews = df_rev.rdd.map(lambda row: SteamParser.parse_review_line(row)).filter(lambda x: x is not None)

df_gen = spark.read.csv(genreFilePath, header=True, quote='"', escape='"', multiLine=True)
rddGenres = df_gen.rdd.map(lambda row: SteamParser.parse_genre_line(row)).filter(lambda x: x is not None)

df_app_gen = spark.read.csv(genreByAppFilePath, header=True, quote='"', escape='"', multiLine=True)
rddGenresByApp = df_app_gen.rdd.map(lambda row: SteamParser.parse_app_genre_line(row)).filter(lambda x: x is not None)

num_partitions = 200

# =========================================================================
# CORE LOGIC (Intenzionalmente NON ottimizzata, lasciata intatta)
# =========================================================================
reviews_kv = rddReviews.map(lambda x: (x[0], (x[1], x[3], x[4]))).partitionBy(num_partitions)
genres_by_app_kv = rddGenresByApp.map(lambda x: (x[0], x[1])).partitionBy(num_partitions).cache()
genres_kv = rddGenres.map(lambda x: (x[0], x[1])).cache()

# Le famigerate Join che stresseranno il cluster!
reviews_with_genre_id = reviews_kv.join(genres_by_app_kv)
reviews_by_genre_id = reviews_with_genre_id.map(
    lambda x: (x[1][1], (x[1][0][0], x[1][0][1], x[1][0][2]))
)
reviews_with_genre_name = reviews_by_genre_id.join(genres_kv)

genre_sentiment_pairs = reviews_with_genre_name.map(
    lambda x: ((x[1][1], x[1][0][0]), (x[1][0][1], x[1][0][2]))
)

aggregated = genre_sentiment_pairs \
    .mapValues(lambda v: (v[0] * v[1], v[1], 1)) \
    .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2]))

results = aggregated.map(
    lambda x: (
        x[0][0],  # genre_name
        "Positive" if x[0][1] else "Negative",  # sentiment
        x[1][0] / x[1][1] if x[1][1] > 0 else 0.0,  # weighted_avg_hours
        x[1][2]  # review_count
    )
)

sorted_results = results.sortBy(lambda x: (x[0], x[1]))
final_results = sorted_results.collect()

# =========================================================================
# STAMPA E SALVATAGGIO DEI RISULTATI
# =========================================================================
print("\n" + "="*80)
print("AVERAGE HOURS PLAYED: POSITIVE VS NEGATIVE REVIEWS BY GENRE")
print("="*80)
print(f"{'Genre':<30} {'Sentiment':<12} {'Avg Hours':<15} {'Review Count':<15}")
print("-"*80)
for genre, sentiment, avg_hours, count in final_results:
    print(f"{genre:<30} {sentiment:<12} {avg_hours:<15.2f} {count:<15,}")
print("="*80)

save_csv_to_s3(
    rows=[(genre, sentiment, round(avg_hours, 2), count) for genre, sentiment, avg_hours, count in final_results],
    headers=["Genre", "Sentiment", "Avg_Hours_Played", "Review_Count"],
    filename="avg_hours_by_genre_sentiment.csv"
)

genre_comparison = {}
for genre, sentiment, avg_hours, count in final_results:
    if genre not in genre_comparison:
        genre_comparison[genre] = {}
    genre_comparison[genre][sentiment] = {"avg_hours": avg_hours, "count": count}

print("\n" + "="*106)
print("GENRE COMPARISON: POSITIVE VS NEGATIVE REVIEW HOURS")
print("="*106)
print(f"{'Genre':<30} {'Pos Hrs':<12} {'Neg Hrs':<12} {'Difference':<15} {'Total Reviews':<15} {'Overall Sentiment':<20}")
print("-"*106)

comparison_results = []
for genre, data in genre_comparison.items():
    pos_hours = data.get("Positive", {}).get("avg_hours", 0)
    neg_hours = data.get("Negative", {}).get("avg_hours", 0)
    pos_count = data.get("Positive", {}).get("count", 0)
    neg_count = data.get("Negative", {}).get("count", 0)
    difference = pos_hours - neg_hours
    total_reviews = pos_count + neg_count
    comparison_results.append((genre, pos_hours, neg_hours, difference, total_reviews))

comparison_results.sort(key=lambda x: x[3], reverse=True)

for genre, pos_hrs, neg_hrs, diff, total in comparison_results:
    sentiment = 'Positive' if diff > 0 else 'Negative' if diff < 0 else 'Neutral'
    print(f"{genre:<30} {pos_hrs:<12.2f} {neg_hrs:<12.2f} {diff:<+15.2f} {total:<15,} {sentiment: <20}")
print("="*106)

save_csv_to_s3(
    rows=[
        (genre, round(pos_hrs, 2), round(neg_hrs, 2), round(diff, 2), total,
         'Positive' if diff > 0 else 'Negative' if diff < 0 else 'Neutral')
        for genre, pos_hrs, neg_hrs, diff, total in comparison_results
    ],
    headers=["Genre", "Pos_Avg_Hours", "Neg_Avg_Hours", "Difference", "Total_Reviews", "Overall_Sentiment"],
    filename="genre_comparison.csv"
)

def categorize_playtime(hours):
    if hours < 2: return "000-0h (Refund Window)"
    elif hours < 10: return "002-10h (Early Game)"
    elif hours < 50: return "010-50h (Mid Game)"
    elif hours < 200: return "050-200h (Committed)"
    else: return "200+h (Hardcore)"

playtime_sentiment = reviews_kv.map(
    lambda x: (categorize_playtime(x[1][1]), (1 if x[1][0] else 0, 1))
).reduceByKey(
    lambda a, b: (a[0] + b[0], a[1] + b[1])
).map(
    lambda x: (x[0], x[1][0] / x[1][1] * 100, x[1][1])
).sortBy(lambda x: x[0])

bracket_results = playtime_sentiment.collect()

print("\n" + "="*80)
print("SENTIMENT BY PLAYTIME BRACKET")
print("="*80)
print(f"{'Playtime Bracket':<30} {'% Positive':<15} {'Review Count':<15}")
print("-"*80)
for bracket, pct_positive, count in bracket_results:
    print(f"{bracket:<30} {pct_positive:<15.2f} {count:<15,}")
print("="*80)

save_csv_to_s3(
    rows=[(bracket, round(pct_positive, 2), count) for bracket, pct_positive, count in bracket_results],
    headers=["Playtime_Bracket", "Pct_Positive_Reviews", "Review_Count"],
    filename="sentiment_by_playtime_bracket.csv"
)

spark.stop()