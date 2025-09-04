-- 完整的电影数据分析Hive DDL脚本
-- 创建数据库
CREATE DATABASE IF NOT EXISTS movie_analytics;
USE movie_analytics;

-- 1. 原始数据表
DROP TABLE IF EXISTS movie_reviews_raw;
CREATE TABLE movie_reviews_raw (
    id STRING,
    movie_name_en STRING,
    movie_name_cn STRING,
    crawl_date STRING,
    number INT,
    username STRING,
    review_date STRING,
    star DOUBLE,
    comment STRING,
    like_count INT
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
STORED AS TEXTFILE
TBLPROPERTIES ('skip.header.line.count'='1');

-- 2. 清洗数据表
DROP TABLE IF EXISTS movie_reviews_clean;
CREATE TABLE movie_reviews_clean (
    id STRING,
    movie_name_en STRING,
    movie_name_cn STRING,
    crawl_date STRING,
    number INT,
    username STRING,
    review_date STRING,
    star DOUBLE,
    comment STRING,
    like_count INT,
    comment_length INT
)
STORED AS ORC;

-- 3. 情感分析表
DROP TABLE IF EXISTS sentiment_results;
CREATE TABLE sentiment_results (
    id STRING,
    username STRING,
    comment STRING,
    sentiment_score DOUBLE,
    sentiment_label STRING,
    analysis_date STRING
)
STORED AS ORC;

-- 4. 用户统计表
DROP TABLE IF EXISTS user_stats;
CREATE TABLE user_stats (
    username STRING,
    total_reviews INT,
    avg_rating DOUBLE,
    total_likes BIGINT,
    user_type STRING
)
STORED AS ORC;

-- 5. 电影统计表
DROP TABLE IF EXISTS movie_stats;
CREATE TABLE movie_stats (
    movie_name STRING,
    total_reviews INT,
    avg_rating DOUBLE,
    total_likes BIGINT,
    quality_score DOUBLE
)
STORED AS ORC;

SHOW TABLES;
