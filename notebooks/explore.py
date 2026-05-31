import duckdb
con = duckdb.connect('D:/github/nyc311-analytics/data/processed/nyc311.duckdb', read_only=True)
df = con.execute('''
    SELECT 
        COUNT(*) as total,
        COUNT(latitude) as has_lat,
        ROUND(COUNT(latitude) * 100.0 / COUNT(*), 1) as pct_has_coords,
        MIN(latitude) as min_lat,
        MAX(latitude) as max_lat
    FROM mart_ml_features
    WHERE latitude IS NOT NULL
''').df()
print(df)
con.close()