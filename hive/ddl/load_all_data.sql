USE movie_analytics;

-- 加载原始数据
LOAD DATA LOCAL INPATH './tx_video.csv' 
OVERWRITE INTO TABLE movie_reviews_raw;

-- 基础验证
SELECT COUNT(*) as total_raw_records FROM movie_reviews_raw;
SELECT * FROM movie_reviews_raw LIMIT 3;
