-- todas as datas de fechamento com ano, mes, dia, hora, minuto e total de tickets fechados
SELECT
    d.Year AS ClosedYear,
    d.Month AS ClosedMonth,
    d.Day AS ClosedDay,
    d.Hour AS ClosedHour,
    d.Minute AS ClosedMinute,
    SUM(t.QtTickets) AS TotalTickets
FROM
    dbo.Fact_Tickets t
JOIN
    dbo.Dim_Dates d ON t.ClosedDateKey = d.DateKey
WHERE
    t.ClosedDateKey IS NOT NULL
GROUP BY
    d.Year, d.Month, d.Day, d.Hour, d.Minute
ORDER BY
    d.Year, d.Month, d.Day, d.Hour, d.Minute
