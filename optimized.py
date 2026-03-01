import time
import json
import boto3
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pyspark.sql import SparkSession
from pyspark.sql import Row

class PerformanceMonitor:
    """Monitor and track Spark job performance metrics (Dependency-Free)"""
    
    def __init__(self):
        self.metrics = []
        self.current_operation = None
        self.start_time = None
        
    def start(self, operation_name: str):
        """Start monitoring an operation"""
        self.current_operation = operation_name
        self.start_time = time.time()
        print(f"\n{'='*80}")
        print(f"Starting: {operation_name}")
        print(f"{'='*80}")
        
    def end(self, spark_context=None, record_count: int = None, additional_metrics: Dict = None):
        """End monitoring and record metrics"""
        if not self.current_operation:
            return
            
        end_time = time.time()
        duration = end_time - self.start_time
        
        metric = {
            'operation': self.current_operation,
            'duration_seconds': duration,
            'timestamp': datetime.now().isoformat(),
            'record_count': record_count
        }
        
        # Add Spark-specific metrics if available
        if spark_context:
            try:
                status = spark_context.statusTracker()
                metric['active_jobs'] = len(status.getActiveJobIds())
                metric['active_stages'] = len(status.getActiveStageIds())
            except:
                pass
        
        # Add any additional custom metrics
        if additional_metrics:
            metric.update(additional_metrics)
        
        self.metrics.append(metric)
        
        print(f"Completed: {self.current_operation}")
        print(f"Duration: {duration:.2f}s")
        if record_count:
            print(f"Records Processed: {record_count:,}")
            print(f"Throughput: {record_count/duration:,.0f} records/sec")
        print(f"{'='*80}\n")
        
        self.current_operation = None
        
    def get_summary_string(self) -> str:
        """Get summary of all metrics as a formatted string"""
        if not self.metrics:
            return "No metrics recorded."
            
        header = f"{'Operation':<45} {'Duration (s)':<15} {'Records Processed':<20}"
        lines = [header, "-" * 85]
        
        for m in self.metrics:
            op = m.get('operation', 'N/A')
            dur = f"{m.get('duration_seconds', 0):.2f}"
            rec = str(m.get('record_count')) if m.get('record_count') is not None else "NaN"
            lines.append(f"{op:<45} {dur:<15} {rec:<20}")
            
        return "\n".join(lines)
    
    def save_report(self, filename='performance_report.json'):
        """Save metrics to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Performance report saved to {filename}")


class SteamParser:
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
            # Filter invalid data
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


if __name__ == "__main__":
    # Initialize performance monitor
    perf = PerformanceMonitor()
    print("Performance monitoring initialized!")

    perf.start("Spark Session Initialization")

    spark = SparkSession.builder \
        .appName("EMR Spark Opt with Benchmarks") \
        .config('spark.ui.port', '4040') \
        .config('spark.sql.shuffle.partitions', '200') \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.default.parallelism", "8") \
        .getOrCreate()
    sc = spark.sparkContext

    sc.setLogLevel("WARN")

    perf.end(sc)

    # Percorsi S3 aggiornati con il tuo bucket
    reviewFilePath = "s3://amzn-s3-steam-reviews/chunk_*.csv"
    genreFilePath = "s3://amzn-s3-steam-reviews/genres.csv"
    genreByAppFilePath = "s3://amzn-s3-steam-reviews/application_genres_standardized.csv"

    df_reviews = spark.read.csv(reviewFilePath, header=True, quote='\"', escape='\"', multiLine=True)
    rddReviews = df_reviews.rdd.map(SteamParser.parse_review_line).filter(lambda x: x is not None)

    # 2. Load Genres (Eager for Broadcast)
    df_genres = spark.read.csv(genreFilePath, header=True, quote='\"', escape='\"', multiLine=True)
    rddGenres = df_genres.rdd.map(SteamParser.parse_genre_line).filter(lambda x: x is not None) \
                    .partitionBy(8) \
                    .cache()

    # 3. Load App Genres (Eager for Broadcast)
    df_app_genres = spark.read.csv(genreByAppFilePath, header=True, quote='\"', escape='\"', multiLine=True)
    rddGenresByApp = df_app_genres.rdd.map(SteamParser.parse_app_genre_line).filter(lambda x: x is not None)

    perf.start("Optimization: Setup Broadcast Variables")

    # 1. Collect Genres into a dictionary {genre_id: genre_name}
    genre_lookup = rddGenres.collectAsMap()
    # 2. Collect App-Genres into a dictionary {appid: [genre_id_1, genre_id_2, ...]}
    app_genre_lookup = rddGenresByApp \
        .map(lambda x: (x[0], x[1])) \
        .groupByKey() \
        .mapValues(list) \
        .collectAsMap()

    # 3. Broadcast to cluster
    bc_genre_lookup = sc.broadcast(genre_lookup)
    bc_app_genre_lookup = sc.broadcast(app_genre_lookup)

    perf.end(sc, len(app_genre_lookup), {'type': 'broadcast_creation'})

    num_partitions = 8

    def map_review_to_stats(row):
        # (appid, voted_up, playtime_forever, playtime_at_review, weight_vote_score)
        appid = row[0]
        voted_up = row[1]
        playtime = row[3]
        weight = row[4]

        # .value gives access to the dictionary on the worker node
        app_genres = bc_app_genre_lookup.value.get(appid)

        results = []
        if app_genres:
            sentiment = "Positive" if voted_up else "Negative"
            weighted_hours = playtime * weight

            for genre_id in app_genres:
                genre_name = bc_genre_lookup.value.get(genre_id)
                if genre_name:
                    results.append(((genre_name, sentiment), (weighted_hours, weight, 1)))
        return results

    perf.start("Optimized Pipeline: Map-Side Join + Aggregation")

    final_results = rddReviews \
        .flatMap(map_review_to_stats) \
        .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2])) \
        .map(lambda x: (
            x[0][0],                                    # Genre
            x[0][1],                                    # Sentiment
            x[1][0] / x[1][1] if x[1][1] > 0 else 0.0,  # Avg Weighted Hours
            x[1][2]                                     # Review Count
        )) \
        .sortBy(lambda x: (x[0], x[1])) \
        .collect()

    perf.end(sc, len(final_results), {'type': 'optimized_execution'})

    perf.start("Generate Genre Comparison Analysis")

    genre_comparison = {}
    for genre, sentiment, avg_hours, count in final_results:
        if genre not in genre_comparison:
            genre_comparison[genre] = {}
        genre_comparison[genre][sentiment] = {"avg_hours": avg_hours, "count": count}

    comparison_list = []
    for genre, data in genre_comparison.items():
        pos = data.get("Positive", {}).get("avg_hours", 0)
        neg = data.get("Negative", {}).get("avg_hours", 0)
        total = data.get("Positive", {}).get("count", 0) + data.get("Negative", {}).get("count", 0)
        comparison_list.append((genre, pos, neg, pos - neg, total))

    comparison_list.sort(key=lambda x: x[3], reverse=True)

    perf.end(sc, len(comparison_list), {'analysis_type': 'genre_comparison'})

    perf.start("Sentiment by Playtime Bracket Analysis")

    def categorize_playtime(hours):
        if hours < 2:
            return "000-0h (Refund Window)"
        elif hours < 10:
            return "002-10h (Early Game)"
        elif hours < 50:
            return "010-50h (Mid Game)"
        elif hours < 200:
            return "050-200h (Committed)"
        else:
            return "200+h (Hardcore)"

    bracket_results = rddReviews.map(
        lambda x: (categorize_playtime(x[2]), (1 if x[1] else 0, 1))
    ).reduceByKey(
        lambda a, b: (a[0] + b[0], a[1] + b[1])
    ).map(
        lambda x: (x[0], x[1][0] / x[1][1] * 100, x[1][1])  # (bracket, % positive, count)
    ).sortBy(lambda x: x[0]).collect()

    perf.end(sc, len(bracket_results), {'analysis_type': 'playtime_brackets'})

    # =========================================================================
    # SALVATAGGIO OUTPUT ANALISI SU FILE
    # =========================================================================
    with open("analysis_results.txt", "w") as f_out:
        print("\n" + "="*80, file=f_out)
        print("AVERAGE HOURS PLAYED: POSITIVE VS NEGATIVE REVIEWS BY GENRE", file=f_out)
        print("="*80, file=f_out)
        print(f"{'Genre':<30} {'Sentiment':<12} {'Avg Hours':<15} {'Review Count':<15}", file=f_out)
        print("-"*80, file=f_out)

        for genre, sentiment, avg_hours, count in final_results:
            print(f"{genre:<30} {sentiment:<12} {avg_hours:<15.2f} {count:<15,}", file=f_out)

        print("="*80, file=f_out)

        print("\n" + "="*106, file=f_out)
        print("GENRE COMPARISON SUMMARY", file=f_out)
        print("="*106, file=f_out)
        print(f"{'Genre':<30} {'Pos Hrs':<12} {'Neg Hrs':<12} {'Diff':<15} {'Total Reviews':<15}", file=f_out)
        print("-"*106, file=f_out)

        for genre, pos, neg, diff, total in comparison_list:
            sentiment = 'Positive' if diff > 0 else 'Negative' if diff < 0 else 'Neutral'
            print(f"{genre:<30} {pos:<12.2f} {neg:<12.2f} {diff:<+15.2f} {total:<15,} {sentiment: <20}", file=f_out)
        print("="*106, file=f_out)

        print("\n" + "="*80, file=f_out)
        print("SENTIMENT BY PLAYTIME BRACKET", file=f_out)
        print("="*80, file=f_out)
        print(f"{'Playtime Bracket':<30} {'% Positive':<15} {'Review Count':<15}", file=f_out)
        print("-"*80, file=f_out)
        for bracket, pct_positive, count in bracket_results:
            print(f"{bracket:<30} {pct_positive:<15.2f} {count:<15,}", file=f_out)
        print("="*80, file=f_out)

    # =========================================================================
    # SALVATAGGIO SUMMARY PERFORMANCE SU FILE
    # =========================================================================
    with open("performance_summary.txt", "w") as f_perf:
        print("\n" + "="*105, file=f_perf)
        print("PERFORMANCE SUMMARY", file=f_perf)
        print("="*105, file=f_perf)

        print(perf.get_summary_string(), file=f_perf)

        print("\n" + "="*105, file=f_perf)
        print("AGGREGATE STATISTICS", file=f_perf)
        print("="*105, file=f_perf)
        
        total_duration = sum(m.get('duration_seconds', 0) for m in perf.metrics)
        avg_duration = total_duration / len(perf.metrics) if perf.metrics else 0
        total_records = sum(m.get('record_count', 0) for m in perf.metrics if m.get('record_count') is not None)
        
        print(f"Total Operations: {len(perf.metrics)}", file=f_perf)
        print(f"Total Execution Time: {total_duration:.2f} seconds", file=f_perf)
        print(f"Average Operation Time: {avg_duration:.2f} seconds", file=f_perf)
        print(f"Total Records Processed: {total_records:,.0f}", file=f_perf)
        print("="*105, file=f_perf)

    # Salva report JSON
    perf.save_report('performance_report.json')

    # =========================================================================
    # UPLOAD DEI FILE SU S3
    # =========================================================================
    print("\n" + "="*80)
    print("UPLOADING RESULTS TO S3...")
    print("="*80)
    
    s3_client = boto3.client('s3')
    bucket_name = "amzn-s3-steam-reviews"
    
    files_to_upload = [
        "analysis_results.txt",
        "performance_summary.txt",
        "performance_report.json"
    ]
    
    for file_name in files_to_upload:
        s3_path = f"output/optimized/{file_name}" # Li salva in una cartella output/optimized/
        try:
            s3_client.upload_file(file_name, bucket_name, s3_path)
            print(f"Successfully uploaded {file_name} to s3://{bucket_name}/{s3_path}")
        except Exception as e:
            print(f"Failed to upload {file_name}. Error: {e}")

    spark.stop()