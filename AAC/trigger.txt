CREATE TRIGGER IF NOT EXISTS set_time_phase
BEFORE INSERT ON Sentences
FOR EACH ROW
BEGIN
    SELECT CASE
        WHEN strftime('%H:%M', 'now', 'localtime') BETWEEN '06:00' AND '09:59' THEN 'MORNING'
        WHEN strftime('%H:%M', 'now', 'localtime') BETWEEN '10:00' AND '13:59' THEN 'MID-DAY'
        WHEN strftime('%H:%M', 'now', 'localtime') BETWEEN '14:00' AND '17:59' THEN 'AFTERNOON'
        WHEN strftime('%H:%M', 'now', 'localtime') BETWEEN '18:00' AND '20:59' THEN 'EVENING'
        ELSE 'NIGHT'
    END