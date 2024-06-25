import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

#adjust spark config based on hardware resources
spark = SparkSession.builder \
    .appName("process_tx") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "32g") \
    .config("spark.driver.maxResultSize", "2g") \
    .config("spark.executor.instances", "2") \
    .config("spark.executor.cores", "4") \
    .config("spark.default.parallelism", "8") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

# config for the transaction database
db_config = {
    'user': 'eth_project',
    'password': 'eth_000!',
    'host': 'localhost',
    'port': '3306',
    'database': 'ETHProjectDB'
}

# config for the scam database
db_config_scams = {
    'user': 'eth_project',
    'password': 'eth_000!',
    'host': 'localhost',
    'port': '3306',
    'database': 'ETHProjectScamDB'
}

url = f"jdbc:mysql://{db_config['host']}:{db_config['port']}/{db_config['database']}?user={db_config['user']}&password={db_config['password']}&permitMysqlScheme=true"
user = db_config['user']
password = db_config['password']

#remove limit
tx_query = "(SELECT * FROM Transactions LIMIT 5000000) AS subset"
transactions_df = spark.read.format('jdbc').option("url", url) \
    .option("dbtable", tx_query) \
    .option("user", user) \
    .option("password", password) \
    .load()

address_query = "(SELECT * FROM Addresses) AS subset"
addresses_df = spark.read.format('jdbc').option("url", url) \
    .option("dbtable", address_query) \
    .option("user", user) \
    .option("password", password) \
    .load()

transactions_df = transactions_df.withColumn(
    'asset_value',
    F.when(transactions_df.category_id == 5, 1).otherwise(transactions_df.asset_value)
)

transactions_df = transactions_df.dropna(subset=['to_id'])
transactions_df = transactions_df.na.fill({'asset_value': 0})

transactions_df = transactions_df.withColumn('asset_value', transactions_df['asset_value'].cast('float'))
transactions_df = transactions_df.withColumn('from_id', transactions_df['from_id'].cast('int'))
transactions_df = transactions_df.withColumn('to_id', transactions_df['to_id'].cast('int'))

transactions_df = transactions_df.filter(~((transactions_df.asset == 'ETH') & (transactions_df.category_id == 3)))

sent_df = transactions_df.select(
    F.col("from_id").alias("node"),
    F.col("asset"),
    F.col("category_id"),
    (-F.col("asset_value")).alias("volume")
)

received_df = transactions_df.select(
    F.col("to_id").alias("node"),
    F.col("asset"),
    F.col("category_id"),
    F.col("asset_value").alias("volume")
)

combined_df = sent_df.union(received_df)

result_df = combined_df.groupBy("node", "asset", "category_id").agg(
    F.sum("volume").alias("volume")
)

total_volume_df = transactions_df.groupBy("asset", "category_id").agg(
    F.sum("asset_value").alias("total_volume")
)

combined_with_total_df = result_df.join(
    total_volume_df,
    on=["asset", "category_id"],
    how="inner"
)

combined_with_total_df = combined_with_total_df.withColumn(
    "normalized_volume",
    F.col("volume") / F.col("total_volume")
)

node_transactions_df = combined_with_total_df.groupBy("node").agg(
    F.avg("normalized_volume").alias("avg_normalized_volume")
)

node_transactions_df = node_transactions_df.join(
    addresses_df,
    on=[node_transactions_df.node == addresses_df.address_id],
    how="inner"
)

url = f"jdbc:mysql://{db_config_scams['host']}:{db_config_scams['port']}/{db_config_scams['database']}?user={db_config_scams['user']}&password={db_config_scams['password']}&permitMysqlScheme=true"
user = db_config_scams['user']
password = db_config_scams['password']

scam_address_query = "(SELECT address, 1 as fraud FROM ScamAddresses) AS subset"
scam_addresses_df = spark.read.format('jdbc').option("url", url) \
    .option("dbtable", scam_address_query) \
    .option("user", user) \
    .option("password", password) \
    .load()

node_transactions_df = node_transactions_df.join(
    scam_addresses_df,
    on=[node_transactions_df.address == scam_addresses_df.address],
    how="left"
)

node_transactions_df = node_transactions_df.fillna({'fraud': 0})
node_transactions_df = node_transactions_df.select("node", "avg_normalized_volume", "fraud")
edge_transactions_df = transactions_df.select("from_id", "to_id")

node_transactions_df = node_transactions_df.toPandas()
edge_transactions_df = edge_transactions_df.toPandas()

node_transactions_df.to_parquet("data/node_transactions.parquet")
edge_transactions_df.to_parquet("data/edge_transactions.parquet")