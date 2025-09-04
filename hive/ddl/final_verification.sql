USE movie_analytics;

SELECT '=== 最终数据验证 ===' as status;
SELECT 'Raw Data' as table_name, COUNT(*) as record_count FROM movie_reviews_raw
UNION ALL
SELECT 'Clean Data', COUNT(*) FROM movie_reviews_clean
UNION ALL  
SELECT 'User Stats', COUNT(*) FROM user_stats
UNION ALL
SELECT 'Movie Stats', COUNT(*) FROM movie_stats
UNION ALL
SELECT 'Sentiment Results', COUNT(*) FROM sentiment_results;

-- 显示分析结果摘要
SELECT '=== 分析结果摘要 ===' as summary;
SELECT AVG(avg_rating) as overall_avg_rating FROM movie_stats;
SELECT user_type, COUNT(*) as user_count FROM user_stats GROUP BY user_type;
SELECT sentiment_label, COUNT(*) as sentiment_count FROM sentiment_results GROUP BY sentiment_label;
