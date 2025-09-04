USE movie_analytics;

-- 简化情感分析
INSERT OVERWRITE TABLE sentiment_results
SELECT 
    id,
    username,
    comment,
    CASE 
        WHEN comment LIKE '%好%' OR comment LIKE '%棒%' OR comment LIKE '%赞%' THEN 0.8
        WHEN comment LIKE '%差%' OR comment LIKE '%烂%' OR comment LIKE '%垃圾%' THEN -0.8
        ELSE 0.0
    END as sentiment_score,
    CASE 
        WHEN comment LIKE '%好%' OR comment LIKE '%棒%' OR comment LIKE '%赞%' THEN 'positive'
        WHEN comment LIKE '%差%' OR comment LIKE '%烂%' OR comment LIKE '%垃圾%' THEN 'negative'
        ELSE 'neutral'
    END as sentiment_label,
    CURRENT_DATE() as analysis_date
FROM movie_reviews_clean
WHERE comment IS NOT NULL;

SELECT COUNT(*) as sentiment_results_created FROM sentiment_results;
SELECT sentiment_label, COUNT(*) as count FROM sentiment_results GROUP BY sentiment_label;
